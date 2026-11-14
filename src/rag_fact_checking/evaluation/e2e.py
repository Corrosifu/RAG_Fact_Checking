"""End-to-end evaluation of the whole pipeline on a question set with reference answers.

Unlike the unit tests (which check the code) and the SciFact benchmark (which checks claim
verification), this runs ``question -> retrieval -> answer -> answer check`` and
``claim -> retrieval -> verdict`` on a small golden set and scores every stage:

* **retrieval**: is a gold document retrieved, and how much of the gold evidence?
* **answer**: token F1 and containment against the reference answers;
* **faithfulness**: share of answer sentences the verifier confirms;
* **abstention**: does the system decline questions the corpus cannot answer?
* **claims**: accuracy of the SUPPORTED / REFUTED / NOT_ENOUGH_INFO verdict;
* **latency**: time per stage.

None of these metrics needs an LLM judge, so the run is free and reproducible. The CI-level
run (``tier A``) uses the lightweight defaults and :class:`OverlapVerifier`, a deterministic
stand-in for the NLI model: it protects the plumbing (chunking, retrieval, evidence windows,
aggregation) against regressions but says **nothing about the quality of a real NLI model**.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from rag_fact_checking.generation import NO_EVIDENCE
from rag_fact_checking.schemas import Answer, Premise, Verdict
from rag_fact_checking.textutils import content_terms, split_sentences

ASK_KINDS = ("factual", "multi_doc", "yes_no", "numeric", "unanswerable")
CLAIM_KIND = "claim"

# Chunking used for the golden set: documents are 40 to 100 words long.
GOLDEN_CHUNK_SIZE = 70
GOLDEN_CHUNK_OVERLAP = 25
GOLDEN_TOP_K = 3

# Metrics that gate regressions (name -> where to find it in the summary).
GATED = {
    "retrieval.hit": ("retrieval", "hit"),
    "retrieval.context_recall": ("retrieval", "context_recall"),
    "answer.f1": ("answer", "f1"),
    "answer.contains": ("answer", "contains"),
    "faithfulness": ("faithfulness",),
    "abstention.rate": ("abstention", "rate"),
    "claims.accuracy": ("claims", "accuracy"),
}
SETUP_KEYS = ("embedder", "generator", "verifier", "chunk_size", "chunk_overlap", "top_k", "items")


# ------------------------------------------------------------------------------- data


@dataclass(frozen=True)
class GoldItem:
    id: str
    kind: str  # one of ASK_KINDS or "claim"
    text: str  # the question, or the claim to verify
    references: tuple[str, ...] = ()  # acceptable answers (questions only)
    gold_docs: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()  # sentences that carry the answer
    yes_no: str | None = None  # "yes" / "no" for yes/no questions
    verdict: str | None = None  # expected verdict for claims

    @property
    def is_claim(self) -> bool:
        return self.kind == CLAIM_KIND


def load_items(path: Path | str) -> list[GoldItem]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            items.append(
                GoldItem(
                    id=raw["id"],
                    kind=raw["kind"],
                    text=raw["text"],
                    references=tuple(raw.get("references", ())),
                    gold_docs=tuple(raw.get("gold_docs", ())),
                    evidence=tuple(raw.get("evidence", ())),
                    yes_no=raw.get("yes_no"),
                    verdict=raw.get("verdict"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{n}: missing field {exc}") from exc
    return items


def golden_dir() -> Path:
    """Directory of the bundled golden set (``corpus.jsonl`` + ``questions.jsonl``)."""
    return Path(str(resources.files("rag_fact_checking").joinpath("sample", "golden")))


# --------------------------------------------------------------------------- metrics

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_ARTICLES = {"a", "an", "the"}


def tokens(text: str) -> list[str]:
    """Lower-case word tokens without articles; numbers like ``16.8`` or ``32,000`` stay whole."""
    return [t.replace(",", "") for t in _TOKEN_RE.findall(text.lower()) if t not in _ARTICLES]


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, r) for r in references), default=0.0)


def contains(prediction: str, reference: str) -> bool:
    """True if the reference tokens appear as a contiguous run in the prediction."""
    p, r = tokens(prediction), tokens(reference)
    if not r:
        return False
    return any(p[i : i + len(r)] == r for i in range(len(p) - len(r) + 1))


def contains_any(prediction: str, references: Sequence[str]) -> bool:
    return any(contains(prediction, r) for r in references)


def yes_no_polarity(answer: str) -> str | None:
    """``yes`` or ``no`` when the answer starts with it, else ``None`` (not parseable)."""
    first = tokens(answer)[:1]
    return first[0] if first and first[0] in ("yes", "no") else None


def context_recall(evidence: Sequence[str], chunk_texts: Sequence[str]) -> float | None:
    """Share of gold evidence sentences found, word for word, in some retrieved chunk."""
    if not evidence:
        return None
    return sum(any(contains(c, e) for c in chunk_texts) for e in evidence) / len(evidence)


def context_precision(retrieved_docs: Sequence[str], gold_docs: Sequence[str]) -> float | None:
    """Share of retrieved chunks that come from a gold document."""
    if not gold_docs or not retrieved_docs:
        return None if not gold_docs else 0.0
    return sum(d in gold_docs for d in retrieved_docs) / len(retrieved_docs)


def faithfulness(answer: Answer) -> float | None:
    """Share of answer sentences confirmed by the verifier; ``None`` when nothing was checked."""
    if not answer.checks:
        return None
    return sum(c.status == "supported" for c in answer.checks) / len(answer.checks)


def abstained(answer: Answer) -> bool:
    """The system declined: the fallback sentence, or no answer sentence is confirmed."""
    if answer.text == NO_EVIDENCE:
        return True
    return bool(answer.checks) and not any(c.status == "supported" for c in answer.checks)


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(values) * 1000
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


# ---------------------------------------------------------------- deterministic double

_NEGATION_RE = re.compile(r"(?i)\b(?:not|no|never|cannot|without|neither|nor)\b|n't\b")
# a number, not the digits inside a name such as "BM25" or "k1"
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.,])\d+(?:[.,]\d+)*(?![A-Za-z0-9])")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def _word_terms(text: str) -> set[str]:
    return {t for t in content_terms(text) if not any(ch.isdigit() for ch in t)}


class OverlapVerifier:
    """Deterministic stand-in for an NLI model, for the CI-level end-to-end run.

    A premise "entails" a claim when it covers most of the claim's content words and agrees
    with it on numbers and on negation; it "contradicts" it when it covers them but differs on
    a number or on negation; otherwise it is neutral. It is a test double: it makes the run
    reproducible and says nothing about the quality of a real NLI model.
    """

    name = "overlap-double"

    def __init__(self, coverage: float = 0.6) -> None:
        self.coverage = coverage

    def entailment_probs(self, claim: str, premises: Sequence[Premise]) -> list[dict[str, float]]:
        claim_terms = _word_terms(claim)
        claim_numbers = _numbers(claim)
        claim_negated = bool(_NEGATION_RE.search(claim))
        out = []
        for premise in premises:
            # Judge the premise sentence that covers the claim best: a negation or a number in
            # another sentence of the window says nothing about this claim.
            best, covered = premise.text, 0.0
            for sentence in split_sentences(premise.text):
                score = (
                    len(claim_terms & _word_terms(sentence)) / len(claim_terms)
                    if claim_terms
                    else 0.0
                )
                if score > covered:
                    best, covered = sentence, score
            premise_numbers = _numbers(best)
            number_clash = bool(
                claim_numbers and premise_numbers and not claim_numbers <= premise_numbers
            )
            negation_clash = claim_negated != bool(_NEGATION_RE.search(best))
            if covered >= self.coverage and (number_clash or negation_clash):
                out.append({"entailment": 0.03, "contradiction": 0.92, "neutral": 0.05})
            elif covered >= self.coverage:
                out.append({"entailment": 0.92, "contradiction": 0.03, "neutral": 0.05})
            else:
                out.append({"entailment": 0.08, "contradiction": 0.04, "neutral": 0.88})
        return out


# ------------------------------------------------------------------------------ runner


def evaluate_item(pipeline: Any, item: GoldItem, embedder: Any | None = None) -> dict[str, Any]:
    """Run one golden item through the pipeline and score it."""
    record: dict[str, Any] = {"id": item.id, "kind": item.kind, "text": item.text}
    gold = item.gold_docs
    if item.is_claim:
        result, timings = pipeline.fact_check_timed(item.text)
        record["verdict"] = result.verdict.value
        record["expected_verdict"] = item.verdict
        record["correct"] = result.verdict.value == item.verdict
        retrieved_docs = [s.doc_id for s in result.evidence]
        chunk_texts = _full_chunk_texts(pipeline, item.text)
    else:
        answer, timings = pipeline.ask_timed(item.text)
        retrieved_docs = [s.doc_id for s in answer.sources]
        chunk_texts = _full_chunk_texts(pipeline, item.text)
        record["answer"] = answer.text
        record["abstained"] = abstained(answer)
        record["faithfulness"] = faithfulness(answer)
        if item.kind != "unanswerable":
            record["f1"] = round(best_f1(answer.text, item.references), 4)
            record["contains"] = contains_any(answer.text, item.references)
            if embedder is not None:
                q, a = embedder.embed([item.text, re.sub(r"\[\d+\]", "", answer.text)])
                record["relevancy"] = round(float(np.dot(q, a)), 4)
            if item.yes_no:
                polarity = yes_no_polarity(answer.text)
                record["yes_no_parsed"] = polarity is not None
                record["yes_no_correct"] = polarity == item.yes_no if polarity else None
    if gold:
        record["hit"] = any(d in gold for d in retrieved_docs)
        record["context_precision"] = context_precision(retrieved_docs, gold)
    record["context_recall"] = context_recall(item.evidence, chunk_texts)
    record["timings"] = {k: round(v, 6) for k, v in timings.items()}
    return record


def _full_chunk_texts(pipeline: Any, query: str) -> list[str]:
    """Texts of the chunks the retriever returns for ``query`` (the same call as the pipeline)."""
    return [h.chunk.text for h in pipeline.retriever.retrieve(query, pipeline.top_k)]


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    asks = [r for r in records if r["kind"] != CLAIM_KIND]
    answerable = [r for r in asks if r["kind"] != "unanswerable"]
    unanswerable = [r for r in asks if r["kind"] == "unanswerable"]
    claims = [r for r in records if r["kind"] == CLAIM_KIND]
    with_gold = [r for r in records if "hit" in r]
    yes_no = [r for r in answerable if "yes_no_parsed" in r]
    parsed = [r for r in yes_no if r["yes_no_parsed"]]

    claim_recall: dict[str, float] = {}
    for verdict in Verdict:
        rows = [r for r in claims if r["expected_verdict"] == verdict.value]
        if rows:
            claim_recall[verdict.value] = round(sum(r["correct"] for r in rows) / len(rows), 4)

    stage_times: dict[str, list[float]] = {}
    for r in records:
        for stage, seconds in r["timings"].items():
            stage_times.setdefault(
                f"{'claim' if r['kind'] == CLAIM_KIND else 'ask'}.{stage}", []
            ).append(seconds)

    return {
        "items": len(records),
        "by_kind": dict(Counter(r["kind"] for r in records)),
        "retrieval": {
            "hit": _mean([float(r["hit"]) for r in with_gold]),
            "context_recall": _mean([r.get("context_recall") for r in records]),
            "context_precision": _mean([r.get("context_precision") for r in with_gold]),
        },
        "answer": {
            "f1": _mean([r["f1"] for r in answerable]),
            "contains": _mean([float(r["contains"]) for r in answerable]),
            "relevancy": _mean([r.get("relevancy") for r in answerable]),
            "yes_no_parse_rate": _mean([float(r["yes_no_parsed"]) for r in yes_no]),
            "yes_no_accuracy": _mean([float(r["yes_no_correct"]) for r in parsed]),
        },
        "faithfulness": _mean([r["faithfulness"] for r in asks]),
        "abstention": {
            "rate": _mean([float(r["abstained"]) for r in unanswerable]),
            "false_rate": _mean([float(r["abstained"]) for r in answerable]),
        },
        "claims": {
            "accuracy": _mean([float(r["correct"]) for r in claims]),
            "recall_by_verdict": claim_recall,
        },
        "latency": {name: _quantiles(v) for name, v in sorted(stage_times.items())},
    }


@dataclass
class E2EResult:
    setup: dict[str, Any]
    summary: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"setup": self.setup, "summary": self.summary, "records": self.records}


def run_e2e(
    pipeline: Any,
    items: Sequence[GoldItem],
    embedder: Any | None = None,
    setup: Mapping[str, Any] | None = None,
) -> E2EResult:
    """Score every item. Claims are skipped when the pipeline has no verifier."""
    can_check = pipeline.verifier is not None
    records = [
        evaluate_item(pipeline, item, embedder) for item in items if can_check or not item.is_claim
    ]
    info = dict(setup or {})
    info["items"] = len(records)
    return E2EResult(info, summarize(records), records)


# ------------------------------------------------------------------- regression gates


def gated_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    """Flatten the summary to the metrics that gate regressions."""
    out: dict[str, float | None] = {}
    for name, path in GATED.items():
        node: Any = summary
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        out[name] = node
    return out


def baseline_from(result: E2EResult) -> dict[str, Any]:
    return {"setup": result.setup, "metrics": gated_metrics(result.summary)}


def compare_to_baseline(
    result: E2EResult, baseline: Mapping[str, Any], tolerance: float = 0.05
) -> list[str]:
    """Human-readable regressions: a gated metric below the baseline by more than ``tolerance``.

    A baseline recorded with a different setup (embedder, generator, chunking...) is refused,
    since the numbers would not be comparable.
    """
    problems = []
    for key in SETUP_KEYS:
        if result.setup.get(key) != baseline.get("setup", {}).get(key):
            problems.append(
                f"setup differs on {key!r}: run {result.setup.get(key)!r}, "
                f"baseline {baseline.get('setup', {}).get(key)!r} (record a new baseline)"
            )
    if problems:
        return problems
    current = gated_metrics(result.summary)
    for name, reference in baseline.get("metrics", {}).items():
        if reference is None:
            continue
        value = current.get(name)
        if value is None:
            problems.append(f"{name}: not measured any more (baseline {reference})")
        elif value < reference - tolerance - 1e-9:
            problems.append(
                f"{name}: {value:.4f} is below the baseline {reference:.4f} (tolerance {tolerance})"
            )
    return problems


def format_summary(summary: Mapping[str, Any]) -> str:
    """A compact text table of the summary."""

    def fmt(x: Any) -> str:
        return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    rows = [("items", summary["items"])]
    for name, value in gated_metrics(summary).items():
        rows.append((name, value))
    for extra in (
        ("answer.relevancy", summary["answer"].get("relevancy")),
        ("retrieval.context_precision", summary["retrieval"].get("context_precision")),
        ("abstention.false_rate", summary["abstention"].get("false_rate")),
    ):
        rows.append(extra)
    width = max(len(r[0]) for r in rows)
    lines = [f"{name.ljust(width)}  {fmt(value)}" for name, value in rows]
    for name, q in summary.get("latency", {}).items():
        lines.append(
            f"latency {name}".ljust(width + 8)
            + f" mean {q['mean_ms']:.1f} ms, p95 {q['p95_ms']:.1f} ms"
        )
    return "\n".join(lines)


def nan_free(value: Any) -> Any:
    """JSON-safe copy (NaN and infinity become ``None``)."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: nan_free(v) for k, v in value.items()}
    if isinstance(value, list):
        return [nan_free(v) for v in value]
    return value


# ------------------------------------------------------------------------------ set-up


def prepare_golden_pipeline(
    directory: Path | str, verifier: str = "overlap", base: Any | None = None
) -> tuple[Any, Any, dict[str, Any], list[GoldItem]]:
    """Index the golden corpus in ``directory`` and build the pipeline to evaluate.

    ``verifier``: ``overlap`` (the deterministic double), ``env`` (the verifier configured by
    ``RAGFC_VERIFIER``, normally the NLI model) or ``none``. The embedder and the generator come
    from the environment, so a real-model run only needs the usual ``RAGFC_*`` variables.
    Returns ``(pipeline, embedder, setup, items)``.
    """
    import dataclasses

    from rag_fact_checking.config import Settings
    from rag_fact_checking.corpus import build_index, load_documents, save_documents
    from rag_fact_checking.embeddings import build_embedder
    from rag_fact_checking.generation import build_generator
    from rag_fact_checking.pipeline import RAGPipeline
    from rag_fact_checking.retriever import HybridRetriever
    from rag_fact_checking.verifier import build_verifier

    if verifier not in ("overlap", "env", "none"):
        raise ValueError("verifier must be 'overlap', 'env' or 'none'")
    settings = dataclasses.replace(
        base or Settings.from_env(),
        data_dir=Path(directory),
        chunk_size=GOLDEN_CHUNK_SIZE,
        chunk_overlap=GOLDEN_CHUNK_OVERLAP,
        top_k=GOLDEN_TOP_K,
    )
    golden = golden_dir()
    save_documents(load_documents(golden / "corpus.jsonl"), settings.corpus_path)
    embedder = build_embedder(settings)
    build_index(settings, embedder)
    retriever = HybridRetriever.from_settings(settings, embedder)
    generator = build_generator(settings)
    if verifier == "overlap":
        checker: Any = OverlapVerifier()
        verifier_name = checker.name
    elif verifier == "env":
        checker = build_verifier(settings)
        verifier_name = settings.verifier_model if checker is not None else None
    else:
        checker, verifier_name = None, None
    pipeline = RAGPipeline(
        retriever,
        generator,
        checker,
        top_k=settings.top_k,
        max_new_tokens=settings.max_new_tokens,
        verify_answers=settings.verify_answers,
    )
    setup = {
        "embedder": embedder.name,
        "generator": settings.generator
        + (f":{settings.generator_model}" if settings.generator == "hf" else ""),
        "verifier": verifier_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
    }
    return pipeline, embedder, setup, load_items(golden / "questions.jsonl")
