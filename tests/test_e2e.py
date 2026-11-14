import json
from pathlib import Path

import pytest

from rag_fact_checking.cli import main
from rag_fact_checking.config import Settings
from rag_fact_checking.evaluation import e2e
from rag_fact_checking.generation import NO_EVIDENCE
from rag_fact_checking.pipeline import RAGPipeline, VerifierNotConfigured
from rag_fact_checking.schemas import Answer, Premise, SentenceCheck, Verdict

BASELINE = Path(__file__).parent / "e2e" / "baseline.json"


# ------------------------------------------------------------------------------ metrics


def test_tokens_drop_articles_and_keep_numbers_whole():
    assert e2e.tokens("The base model has 16.8 million entries, or 32,000 merges.") == [
        "base",
        "model",
        "has",
        "16.8",
        "million",
        "entries",
        "or",
        "32000",
        "merges",
    ]


def test_token_f1_and_containment():
    assert e2e.token_f1("a cat sat", "the cat sat") == 1.0
    assert e2e.token_f1("cat", "dog") == 0.0
    assert e2e.token_f1("cat sat down", "cat sat") == pytest.approx(0.8)
    assert e2e.best_f1("dog", ["cat", "dog"]) == 1.0
    assert e2e.contains("It uses running averages collected during training.", "running averages")
    assert not e2e.contains("averages running", "running averages")  # order matters
    assert e2e.contains_any("the model has 12 layers", ["24", "12"])
    assert not e2e.contains("anything", "")


def test_yes_no_polarity():
    assert e2e.yes_no_polarity("No, it does not.") == "no"
    assert e2e.yes_no_polarity("Yes. Because...") == "yes"
    assert e2e.yes_no_polarity("The model is decoder-only.") is None


def test_context_metrics():
    chunks = ["BERT is an encoder-only Transformer. About 15 percent of tokens are masked."]
    assert e2e.context_recall(["About 15 percent of tokens are masked."], chunks) == 1.0
    assert e2e.context_recall(["Never retrieved sentence."], chunks) == 0.0
    assert e2e.context_recall([], chunks) is None
    assert e2e.context_precision(["g1", "g2", "g1"], ["g1"]) == pytest.approx(2 / 3)
    assert e2e.context_precision(["g2"], []) is None


def check(status):
    return SentenceCheck("a sentence of the answer", status, 0.9)


def test_faithfulness_and_abstention():
    supported = Answer("x", [], [check("supported"), check("unsupported")])
    assert e2e.faithfulness(supported) == 0.5
    assert e2e.faithfulness(Answer("x", [])) is None  # not verified
    assert not e2e.abstained(supported)
    assert e2e.abstained(Answer(NO_EVIDENCE, []))
    assert e2e.abstained(Answer("something", [], [check("unsupported")]))
    assert not e2e.abstained(Answer("something", []))  # unverified answer: counts as an answer


def test_overlap_double_judges_the_best_covering_sentence():
    v = e2e.OverlapVerifier()

    def probe(claim, text):
        (probs,) = v.entailment_probs(claim, [Premise(text, "d", "T", "d#0")])
        return max(probs, key=probs.get)

    window = "Dropout zeroes random units during training. It is usually not applied to the output."
    assert probe("Dropout zeroes random units during training.", window) == "entailment"
    assert probe("Dropout is applied to the output.", window) == "contradiction"  # negation
    assert (
        probe("Quantization shrinks models by 2 times.", "Quantization shrinks models by 4 times.")
        == "contradiction"
    )
    assert (
        probe("BM25 uses a parameter b of 0.75.", "The parameter b is 0.75 in BM25.")
        == "entailment"
    )
    assert probe("Dropout was invented in 2014.", window) == "neutral"


# ------------------------------------------------------------------------- the golden set


def test_golden_set_is_consistent():
    docs = {
        json.loads(line)["id"]: json.loads(line)
        for line in (e2e.golden_dir() / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    }
    items = e2e.load_items(e2e.golden_dir() / "questions.jsonl")
    assert len(docs) == 25 and len(items) == 30
    assert len({i.id for i in items}) == len(items)
    kinds = {i.kind for i in items}
    assert kinds == {*e2e.ASK_KINDS, e2e.CLAIM_KIND}
    for item in items:
        assert all(d in docs for d in item.gold_docs), item.id
        for sentence in item.evidence:  # evidence is quoted word for word from a gold document
            assert any(sentence in docs[d]["text"] for d in item.gold_docs), item.id
        if item.kind == "unanswerable":
            assert not item.gold_docs and not item.references
        elif item.is_claim:
            assert item.verdict in {v.value for v in Verdict}
        else:
            assert item.references, item.id
        assert item.yes_no in (None, "yes", "no")


def test_load_items_reports_the_bad_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "x", "text": "no kind"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"q.jsonl:1.*kind"):
        e2e.load_items(path)


# ------------------------------------------------------------------ summary and baseline


def rec(kind, **kw):
    base = {"id": kind, "kind": kind, "text": "t", "timings": {"retrieve": 0.001}}
    return {**base, **kw}


def test_summarize_separates_answerable_unanswerable_and_claims():
    records = [
        rec(
            "factual",
            f1=1.0,
            contains=True,
            faithfulness=1.0,
            abstained=False,
            hit=True,
            context_precision=0.5,
            context_recall=1.0,
        ),
        rec(
            "factual",
            f1=0.0,
            contains=False,
            faithfulness=0.5,
            abstained=False,
            hit=False,
            context_precision=0.0,
            context_recall=0.0,
        ),
        rec("unanswerable", faithfulness=None, abstained=True),
        rec("unanswerable", faithfulness=None, abstained=False),
        rec(
            "claim",
            correct=True,
            expected_verdict="SUPPORTED",
            verdict="SUPPORTED",
            hit=True,
            context_precision=1.0,
            context_recall=None,
        ),
    ]
    s = e2e.summarize(records)
    assert s["retrieval"]["hit"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["answer"]["f1"] == 0.5 and s["answer"]["contains"] == 0.5
    assert s["abstention"] == {"rate": 0.5, "false_rate": 0.0}
    assert s["claims"]["accuracy"] == 1.0
    assert s["claims"]["recall_by_verdict"] == {"SUPPORTED": 1.0}
    assert s["faithfulness"] == 0.75
    assert "ask.retrieve" in s["latency"] and "claim.retrieve" in s["latency"]


def result_with(**metrics):
    summary = {
        "retrieval": {"hit": metrics.get("hit", 1.0), "context_recall": 1.0},
        "answer": {"f1": 0.5, "contains": 0.5},
        "faithfulness": 1.0,
        "abstention": {"rate": 0.5},
        "claims": {"accuracy": 1.0},
    }
    setup = {k: "x" for k in e2e.SETUP_KEYS}
    return e2e.E2EResult(setup, summary)


def test_baseline_comparison_allows_noise_and_flags_regressions():
    base = e2e.baseline_from(result_with())
    assert e2e.compare_to_baseline(result_with(), base) == []
    assert e2e.compare_to_baseline(result_with(hit=0.96), base) == []  # within 0.05
    problems = e2e.compare_to_baseline(result_with(hit=0.9), base)
    assert len(problems) == 1 and "retrieval.hit" in problems[0]
    assert e2e.compare_to_baseline(result_with(hit=0.9), base, tolerance=0.2) == []


def test_baseline_refuses_a_different_setup_and_a_missing_metric():
    base = e2e.baseline_from(result_with())
    other = result_with()
    other.setup["embedder"] = "something-else"
    assert "embedder" in e2e.compare_to_baseline(other, base)[0]
    lost = result_with()
    lost.summary["faithfulness"] = None
    assert "faithfulness" in e2e.compare_to_baseline(lost, base)[0]


# -------------------------------------------------------- tier A: the real regression gate


def run_tier_a(tmp_path):
    pipeline, embedder, setup, items = e2e.prepare_golden_pipeline(
        tmp_path, verifier="overlap", base=Settings()
    )
    return e2e.run_e2e(pipeline, items, embedder, setup)


def test_tier_a_end_to_end_does_not_regress(tmp_path):
    """The CI-level gate: lightweight components on the golden set against ``baseline.json``.

    If this fails after a deliberate change, look at the numbers first; then record a new
    baseline with ``ragfc e2e --baseline tests/e2e/baseline.json --update-baseline`` and let the
    reviewer see the diff.
    """
    result = run_tier_a(tmp_path)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert e2e.compare_to_baseline(result, baseline) == []
    assert result.summary["items"] == 30 and result.summary["claims"]["accuracy"] is not None


def test_tier_a_run_is_deterministic(tmp_path):
    first = run_tier_a(tmp_path / "a").summary
    second = run_tier_a(tmp_path / "b").summary
    first.pop("latency"), second.pop("latency")
    assert first == second


def test_claims_are_skipped_without_a_verifier(tmp_path):
    pipeline, embedder, setup, items = e2e.prepare_golden_pipeline(
        tmp_path, verifier="none", base=Settings()
    )
    result = e2e.run_e2e(pipeline, items, embedder, setup)
    assert result.summary["items"] == 21 and result.summary["claims"]["accuracy"] is None
    assert result.summary["faithfulness"] is None


def test_unknown_verifier_choice_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="verifier"):
        e2e.prepare_golden_pipeline(tmp_path, verifier="magic", base=Settings())


# ----------------------------------------------------------------------------- the command


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("RAGFC_EMBEDDER", "RAGFC_GENERATOR", "RAGFC_VERIFIER", "RAGFC_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_cli_records_then_checks_a_baseline(tmp_path, clean_env, capsys):
    path = tmp_path / "baseline.json"
    assert main(["e2e", "--baseline", str(path), "--update-baseline"]) == 0
    assert "Baseline written" in capsys.readouterr().out
    assert main(["e2e", "--baseline", str(path)]) == 0
    assert "No regression" in capsys.readouterr().out

    data = json.loads(path.read_text(encoding="utf-8"))
    data["metrics"]["retrieval.hit"] = 1.5  # an impossible reference: the run must "regress"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["e2e", "--baseline", str(path)]) == 1
    assert "retrieval.hit" in capsys.readouterr().err


def test_cli_writes_the_full_result_and_needs_a_baseline_path_to_update(
    tmp_path, clean_env, capsys
):
    out = tmp_path / "result.json"
    assert main(["e2e", "--output", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"setup", "summary", "records"} and len(data["records"]) == 30
    assert main(["e2e", "--update-baseline"]) == 1
    assert "--baseline" in capsys.readouterr().err


# ------------------------------------------------------------------------ timed pipeline


def test_pipeline_timing_methods(pipeline):
    answer, timings = pipeline.ask_timed("How does quantization shrink a neural network?")
    assert answer.text and set(timings) == {"retrieve", "generate", "verify"}
    assert all(t >= 0 for t in timings.values()) and timings["verify"] == 0.0
    with pytest.raises(VerifierNotConfigured):
        pipeline.fact_check_timed("Quantization reduces model size.")


def test_pipeline_timing_with_a_verifier(retriever):
    from rag_fact_checking.generation import ExtractiveGenerator
    from stubs import StubVerifier

    p = RAGPipeline(retriever, ExtractiveGenerator(), StubVerifier(), top_k=3)
    result, timings = p.fact_check_timed("Quantization reduces model size.")
    assert result.verdict in Verdict and set(timings) == {"retrieve", "verify"}


# ------------------------------------------------------- the gate must notice a real break


def test_the_gate_notices_a_broken_generator(tmp_path):
    pipeline, embedder, setup, items = e2e.prepare_golden_pipeline(
        tmp_path, verifier="overlap", base=Settings()
    )
    pipeline.generator.generate = lambda *a, **k: "Lorem ipsum dolor sit amet."
    problems = e2e.compare_to_baseline(
        e2e.run_e2e(pipeline, items, embedder, setup),
        json.loads(BASELINE.read_text(encoding="utf-8")),
    )
    assert any(p.startswith("answer.f1") for p in problems)


def test_the_gate_notices_a_broken_retriever(tmp_path):
    pipeline, embedder, setup, items = e2e.prepare_golden_pipeline(
        tmp_path, verifier="overlap", base=Settings()
    )
    pipeline.retriever.retrieve = lambda query, top_k=5: []
    problems = e2e.compare_to_baseline(
        e2e.run_e2e(pipeline, items, embedder, setup),
        json.loads(BASELINE.read_text(encoding="utf-8")),
    )
    assert any(p.startswith("retrieval.hit") for p in problems)
    assert any(p.startswith("retrieval.context_recall") for p in problems)
