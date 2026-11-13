import pytest

from rag_fact_checking.evaluation import scifact
from rag_fact_checking.schemas import Chunk, Document, Hit, Verdict
from stubs import StubVerifier

S, R, N = Verdict.SUPPORTED, Verdict.REFUTED, Verdict.NOT_ENOUGH_INFO


def row(cid, claim, label="", doc="", cited=()):
    return {
        "id": cid,
        "claim": claim,
        "evidence_doc_id": doc,
        "evidence_label": label,
        "evidence_sentences": [1],
        "cited_doc_ids": list(cited),
    }


ROWS = [
    row(1, "a supported claim", "SUPPORT", "10", [10]),
    row(1, "a supported claim", "SUPPORT", "11", [10, 11]),  # second evidence row, same claim
    row(2, "a refuted claim", "CONTRADICT", "20", [20]),
    row(3, "a claim without evidence", "", "", [30]),
]


def test_claims_are_merged_and_labelled():
    claims = {c.id: c for c in scifact.build_claims(ROWS)}
    assert len(claims) == 3
    assert claims[1].label == S and claims[1].relevant == ("10", "11")
    assert claims[1].cited == ("10", "11")  # de-duplicated, order kept
    assert claims[2].label == R
    assert claims[3].label == N and claims[3].relevant == () and claims[3].cited == ("30",)


def test_mixed_label_claims_are_counted():
    assert scifact.mixed_label_claims(ROWS) == 0
    assert scifact.mixed_label_claims([*ROWS, row(1, "a supported claim", "CONTRADICT", "12")]) == 1


def test_documents_join_sentences_and_use_string_ids():
    docs = scifact.build_documents(
        [{"doc_id": 5, "title": "T", "abstract": ["One.", "Two."], "structured": False}]
    )
    assert docs == [Document("5", "One. Two.", "T", {})]


def test_classification_report_values():
    gold = [S, S, S, R, N, N]
    pred = [S, S, N, R, N, S]
    rep = scifact.classification_report(gold, pred)
    assert rep["n"] == 6 and rep["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert rep["per_class"]["SUPPORTED"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert rep["per_class"]["SUPPORTED"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert rep["per_class"]["REFUTED"]["f1"] == 1.0
    assert rep["per_class"]["NOT_ENOUGH_INFO"]["support"] == 2
    assert rep["confusion"]["SUPPORTED"] == {"SUPPORTED": 2, "REFUTED": 0, "NOT_ENOUGH_INFO": 1}
    assert rep["majority_baseline"] == 0.5
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 1.0 + 0.5) / 3, abs=1e-3)


def test_classification_report_handles_absent_class_and_length_mismatch():
    rep = scifact.classification_report([S, S], [S, S])
    assert rep["per_class"]["REFUTED"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    with pytest.raises(ValueError):
        scifact.classification_report([S], [S, R])


def test_retrieval_report_values():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
    relevant = [["a"], ["z", "y"], ["nope"]]
    rep = scifact.retrieval_report(ranked, relevant, ks=(1, 2))
    assert rep["hit@1"] == pytest.approx(1 / 3, abs=1e-4)  # only claim 1
    assert rep["hit@2"] == pytest.approx(2 / 3, abs=1e-4)  # claim 2 finds y at rank 2
    assert rep["recall@2"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)
    assert rep["mrr"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)


def test_unique_docs_keeps_best_rank_order():
    def hit(doc, pos):
        return Hit(Chunk(f"{doc}#{pos}", doc, pos, "t"), 1.0)

    assert scifact.unique_docs([hit("b", 0), hit("a", 0), hit("b", 1), hit("c", 0)]) == [
        "b",
        "a",
        "c",
    ]


class FakeRetriever:
    """Returns the chunk of the document whose id is named in the query, if any."""

    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=5):
        return [
            Hit(Chunk(f"{d.id}#0", d.id, 0, d.text, d.title), 1.0)
            for d in self.docs
            if f"doc{d.id}" in query
        ][:top_k]


def test_run_scifact_end_to_end_with_stubs():
    docs = [
        Document("10", "Quantization reduces model size a lot.", "A"),
        Document("20", "Other unrelated text about birds.", "B"),
    ]
    claims = [
        scifact.Claim(1, "doc10 quantization reduces model size", S, ("10",), ("10",)),
        scifact.Claim(2, "doc20 quantization never works", R, ("20",), ("20",)),
    ]
    out = scifact.run_scifact(
        FakeRetriever(docs), StubVerifier(), claims, docs, chunk_size=50, chunk_overlap=5
    )
    assert out["claims"] == 2 and out["label_counts"] == {"SUPPORTED": 1, "REFUTED": 1}
    assert out["retrieval"]["hit@1"] == 1.0 and out["retrieval"]["mrr"] == 1.0
    assert out["oracle"]["n"] == 2 and "seconds_per_claim" in out["oracle"]
    assert out["end_to_end"]["accuracy"] >= 0.5


def test_run_scifact_without_verifier_only_measures_retrieval():
    docs = [Document("10", "text", "A")]
    claims = [scifact.Claim(1, "doc10 text", S, ("10",), ("10",))]
    out = scifact.run_scifact(FakeRetriever(docs), None, claims, docs)
    assert "retrieval" in out and "oracle" not in out and "end_to_end" not in out


def test_download_is_cached(tmp_path, monkeypatch):
    calls = []

    class Resp:
        content = b"data"

        def raise_for_status(self):
            pass

    import requests

    monkeypatch.setattr(requests, "get", lambda url, timeout: calls.append(url) or Resp())
    first = scifact.download(tmp_path, "validation")
    second = scifact.download(tmp_path, "validation")
    assert first == second and first.read_bytes() == b"data" and len(calls) == 1
    assert calls[0].endswith("claims/validation/0000.parquet")


def test_parquet_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "x.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    assert scifact.read_parquet(path)[0]["claim"] == "a supported claim"
    assert len(scifact.build_claims(scifact.read_parquet(path))) == 3
