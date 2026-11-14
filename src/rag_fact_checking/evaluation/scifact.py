"""SciFact benchmark (Wadden et al., 2020): scientific claims labelled against paper abstracts.

Three experiments, each isolating one component:

1. **retrieval**: does the hybrid retriever find the abstracts annotators cited as evidence?
2. **oracle verification**: given the cited abstracts, is the verdict right? (NLI model only)
3. **end to end**: retrieve from the whole corpus, then verify. (what the product does)

Labels are mapped to the project's verdicts: SUPPORT -> SUPPORTED, CONTRADICT -> REFUTED, and a
claim whose cited abstracts hold no evidence -> NOT_ENOUGH_INFO. The public test split has no
labels, so results are reported on the validation split (the original 300-claim dev set).
Data is the Hugging Face parquet conversion of ``allenai/scifact``.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_fact_checking.chunking import chunk_documents
from rag_fact_checking.schemas import Document, Hit, Verdict

HF_BASE = "https://huggingface.co/datasets/allenai/scifact/resolve/refs%2Fconvert%2Fparquet/"
FILES = {
    "corpus": "corpus/train/0000.parquet",
    "train": "claims/train/0000.parquet",
    "validation": "claims/validation/0000.parquet",
    "test": "claims/test/0000.parquet",
}
# Distractor collections (BEIR) used to test how the retriever copes with a larger corpus.
EXTRA_CORPORA = {
    name: f"https://huggingface.co/datasets/BeIR/{name}/resolve/main/corpus/corpus-00000-of-00001.parquet"
    for name in ("nfcorpus", "scidocs")
}
LABEL_MAP = {"SUPPORT": Verdict.SUPPORTED, "CONTRADICT": Verdict.REFUTED}
LABELS = (Verdict.SUPPORTED, Verdict.REFUTED, Verdict.NOT_ENOUGH_INFO)


@dataclass(frozen=True)
class Claim:
    id: int
    text: str
    label: Verdict
    cited: tuple[str, ...]  # abstracts the annotators looked at
    relevant: tuple[str, ...]  # abstracts that actually contain evidence


# --------------------------------------------------------------------------- data


def download(data_dir: Path | str, name: str) -> Path:
    """Fetch one SciFact parquet file into ``data_dir`` (cached)."""
    import requests

    target = Path(data_dir) / name.replace("/", "_")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(HF_BASE + FILES[name], timeout=120)
    resp.raise_for_status()
    tmp = target.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(target)
    return target


def download_extra(data_dir: Path | str, name: str) -> Path:
    """Fetch one distractor collection (cached)."""
    import requests

    target = Path(data_dir) / f"extra_{name}.parquet"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(EXTRA_CORPORA[name], timeout=300)
    resp.raise_for_status()
    tmp = target.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(target)
    return target


def build_extra_documents(rows: Sequence[dict[str, Any]], name: str) -> list[Document]:
    """BEIR rows (``_id``, ``title``, ``text``) as documents; ids are prefixed to never collide."""
    return [
        Document(f"{name}:{r['_id']}", f"{r['title']}. {r['text']}".strip(), r["title"], {})
        for r in rows
        if r.get("text")
    ]


def read_parquet(path: Path | str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("Reading SciFact needs pyarrow: pip install -e '.[bench]'") from exc
    return pq.read_table(str(path)).to_pylist()


def build_documents(rows: Sequence[dict[str, Any]]) -> list[Document]:
    """One document per abstract; sentences are joined, ids are kept as strings."""
    return [Document(str(r["doc_id"]), " ".join(r["abstract"]), r["title"], {}) for r in rows]


def build_claims(rows: Sequence[dict[str, Any]]) -> list[Claim]:
    """Merge the per-evidence rows of each claim into one labelled claim.

    A claim with SUPPORT and CONTRADICT evidence at once would be ambiguous; SciFact does not
    contain such claims (checked by ``mixed_label_claims``), so the first label found is used.
    """
    by_id: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(int(r["id"]), []).append(r)
    claims = []
    for cid, group in by_id.items():
        labels = [LABEL_MAP[r["evidence_label"]] for r in group if r["evidence_label"] in LABEL_MAP]
        relevant = tuple(
            dict.fromkeys(str(r["evidence_doc_id"]) for r in group if r["evidence_label"])
        )
        cited = tuple(dict.fromkeys(str(d) for r in group for d in r["cited_doc_ids"]))
        label = labels[0] if labels else Verdict.NOT_ENOUGH_INFO
        claims.append(Claim(cid, group[0]["claim"], label, cited, relevant))
    return claims


def build_rationales(
    rows: Sequence[dict[str, Any]],
) -> dict[int, list[tuple[str, Verdict, list[int]]]]:
    """Gold evidence per claim: (abstract id, label, indices of the rationale sentences)."""
    out: dict[int, list[tuple[str, Verdict, list[int]]]] = {}
    for r in rows:
        if r["evidence_label"] in LABEL_MAP:
            out.setdefault(int(r["id"]), []).append(
                (
                    str(r["evidence_doc_id"]),
                    LABEL_MAP[r["evidence_label"]],
                    list(r["evidence_sentences"]),
                )
            )
    return out


def mixed_label_claims(rows: Sequence[dict[str, Any]]) -> int:
    seen: dict[int, set[str]] = {}
    for r in rows:
        if r["evidence_label"]:
            seen.setdefault(int(r["id"]), set()).add(r["evidence_label"])
    return sum(len(v) > 1 for v in seen.values())


# ------------------------------------------------------------------------ metrics


def classification_report(gold: Sequence[Verdict], pred: Sequence[Verdict]) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/F1, macro-F1 and the confusion matrix."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    n = len(gold)
    per_class: dict[str, dict[str, float]] = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab.value] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    confusion = {g.value: {p.value: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        confusion[g.value][p.value] += 1
    majority = max(Counter(gold).values()) / n if n else 0.0
    return {
        "n": n,
        "accuracy": round(sum(g == p for g, p in zip(gold, pred, strict=True)) / n, 4)
        if n
        else 0.0,
        "macro_f1": round(sum(c["f1"] for c in per_class.values()) / len(LABELS), 4),
        "majority_baseline": round(majority, 4),
        "per_class": per_class,
        "confusion": confusion,
    }


def retrieval_report(
    ranked: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Document-level hit@k (any relevant abstract in the top k), recall@k and MRR."""
    if len(ranked) != len(relevant):
        raise ValueError("ranked and relevant must have the same length")
    n = len(ranked)
    out: dict[str, Any] = {"n": n}
    for k in ks:
        hits = recalls = 0.0
        for docs, rel in zip(ranked, relevant, strict=True):
            found = set(docs[:k]) & set(rel)
            hits += bool(found)
            recalls += len(found) / len(rel)
        out[f"hit@{k}"] = round(hits / n, 4) if n else 0.0
        out[f"recall@{k}"] = round(recalls / n, 4) if n else 0.0
    rr = 0.0
    for docs, rel in zip(ranked, relevant, strict=True):
        rr += next((1 / i for i, d in enumerate(docs, 1) if d in set(rel)), 0.0)
    out["mrr"] = round(rr / n, 4) if n else 0.0
    return out


def unique_docs(hits: Sequence[Hit]) -> list[str]:
    """Document ids in rank order (a document is ranked where its best chunk appears)."""
    return list(dict.fromkeys(h.chunk.doc_id for h in hits))


def oracle_probs(
    claims: Sequence[Claim],
    documents: Sequence[Document],
    verifier: Any,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> list[list[Any]]:
    """NLI probabilities of each claim's evidence windows, computed once (cited abstracts only).

    Verdicts for any threshold are then a cheap ``aggregate_verdict`` call, which makes the
    threshold sweep free.
    """
    from rag_fact_checking.verifier import select_premises

    by_id = {d.id: d for d in documents}
    out = []
    for c in claims:
        chunks = chunk_documents(
            [by_id[d] for d in c.cited if d in by_id], chunk_size, chunk_overlap, False
        )
        premises = select_premises(c.text, [Hit(ch, 1.0) for ch in chunks])
        out.append(list(verifier.entailment_probs(c.text, premises)) if premises else [])
    return out


def sweep_threshold(
    gold: Sequence[Verdict], probs: Sequence[Sequence[Any]], thresholds: Sequence[float]
) -> dict[float, dict[str, Any]]:
    """Classification report of the oracle verdicts for each decision threshold."""
    from rag_fact_checking.verifier import aggregate_verdict

    return {
        t: classification_report(gold, [aggregate_verdict(p, t)[0] for p in probs])
        for t in thresholds
    }


def abstract_label_report(
    predictions: Sequence[tuple[int, str, Verdict]], claims: Sequence[Claim]
) -> dict[str, Any]:
    """SciFact "Abstract Label-Only" metric (Wadden et al., 2020), the one published systems report.

    A predicted (claim, abstract, label) counts as correct if the abstract is a gold evidence
    abstract of the claim and the label matches. NOT_ENOUGH_INFO predictions are not predictions.
    Micro precision / recall / F1 over all claims.
    """
    gold = {(c.id, d, c.label) for c in claims for d in c.relevant}
    predicted = {p for p in predictions if p[2] != Verdict.NOT_ENOUGH_INFO}
    correct = len(predicted & gold)
    prec = correct / len(predicted) if predicted else 0.0
    rec = correct / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "predicted": len(predicted),
        "gold": len(gold),
        "correct": correct,
    }


def abstract_predictions(
    claims: Sequence[Claim],
    candidates: Sequence[Sequence[str]],
    documents: Sequence[Document],
    verifier: Any,
    *,
    threshold: float = 0.6,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    cache: dict[tuple[int, str], list[Any]] | None = None,
) -> list[tuple[int, str, Verdict]]:
    """Label every candidate abstract of every claim on its own (full abstract as evidence).

    ``candidates[i]`` are the abstract ids to judge for ``claims[i]``: the cited abstracts in
    the "oracle abstract" setting, the retrieved ones in the "open" setting. NLI probabilities
    are kept in ``cache`` so several thresholds or cut-offs reuse them.
    """
    from rag_fact_checking.verifier import aggregate_verdict, select_premises

    by_id = {d.id: d for d in documents}
    cache = {} if cache is None else cache
    out: list[tuple[int, str, Verdict]] = []
    for claim, doc_ids in zip(claims, candidates, strict=True):
        for doc_id in doc_ids:
            if doc_id not in by_id:
                continue
            key = (claim.id, doc_id)
            if key not in cache:
                chunks = chunk_documents([by_id[doc_id]], chunk_size, chunk_overlap, False)
                premises = select_premises(claim.text, [Hit(ch, 1.0) for ch in chunks])
                cache[key] = (
                    list(verifier.entailment_probs(claim.text, premises)) if premises else []
                )
            out.append((claim.id, doc_id, aggregate_verdict(cache[key], threshold)[0]))
    return out


# -------------------------------------------------------------------- experiments


def run_scifact(
    retriever: Any,
    verifier: Any | None,
    claims: Sequence[Claim],
    documents: Sequence[Document],
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    top_k: int = 5,
    threshold: float = 0.6,
    ks: Sequence[int] = (1, 5, 10),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the three experiments; verification ones are skipped without a ``verifier``."""
    from rag_fact_checking.verifier import verify_claim

    say = progress or (lambda _msg: None)
    by_id = {d.id: d for d in documents}
    result: dict[str, Any] = {
        "claims": len(claims),
        "label_counts": {k.value: v for k, v in Counter(c.label for c in claims).items()},
    }

    # 1. retrieval, on claims that have gold evidence
    judged = [c for c in claims if c.relevant]
    ranked = [unique_docs(retriever.retrieve(c.text, max(top_k, 50))) for c in judged]
    result["retrieval"] = retrieval_report(ranked, [c.relevant for c in judged], ks)
    say(f"retrieval done ({len(judged)} claims with evidence)")
    if verifier is None:
        return result

    gold = [c.label for c in claims]

    # 2. oracle verification: only the cited abstracts are available
    start = time.perf_counter()
    oracle = []
    for i, c in enumerate(claims, 1):
        chunks = chunk_documents(
            [by_id[d] for d in c.cited if d in by_id], chunk_size, chunk_overlap, False
        )
        hits = [Hit(ch, 1.0) for ch in chunks]
        oracle.append(verify_claim(c.text, hits, verifier, threshold)[0])
        if i % 50 == 0:
            say(f"oracle {i}/{len(claims)}")
    result["oracle"] = classification_report(gold, oracle)
    result["oracle"]["seconds_per_claim"] = round((time.perf_counter() - start) / len(claims), 3)

    # 3. end to end: retrieve from the whole corpus, then verify
    start = time.perf_counter()
    end_to_end = []
    for i, c in enumerate(claims, 1):
        end_to_end.append(
            verify_claim(c.text, retriever.retrieve(c.text, top_k), verifier, threshold)[0]
        )
        if i % 50 == 0:
            say(f"end-to-end {i}/{len(claims)}")
    result["end_to_end"] = classification_report(gold, end_to_end)
    result["end_to_end"]["seconds_per_claim"] = round(
        (time.perf_counter() - start) / len(claims), 3
    )

    # 4. the published "Abstract Label-Only" metric, per (claim, abstract) pair
    cache: dict[tuple[int, str], list[Any]] = {}
    common = {"threshold": threshold, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    retrieved = [unique_docs(retriever.retrieve(c.text, 50)) for c in claims]
    abstract_level = {
        "oracle_abstract": abstract_label_report(
            abstract_predictions(
                claims, [c.cited for c in claims], documents, verifier, cache=cache, **common
            ),
            claims,
        )
    }
    for k in (3, 10):
        abstract_level[f"open_top{k}"] = abstract_label_report(
            abstract_predictions(
                claims, [r[:k] for r in retrieved], documents, verifier, cache=cache, **common
            ),
            claims,
        )
    result["abstract_level"] = abstract_level
    return result
