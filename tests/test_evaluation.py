import pytest

from rag_fact_checking.evaluation import evaluate_retrieval, hit_rate, mrr, recall_at_k


def test_metric_functions():
    assert hit_rate(["a", "b"], {"b"}) == 1.0 and hit_rate(["a"], {"z"}) == 0.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}) == 0.5 and recall_at_k(["a"], set()) == 0.0
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3) and mrr(["x"], {"a"}) == 0.0


def test_bundled_testset_reaches_quality_floor(retriever, testset):
    scores = evaluate_retrieval(retriever, testset, k=3)
    assert scores["n"] == len(testset)
    assert scores["hit_rate@3"] >= 0.8 and scores["mrr"] >= 0.6  # regression guard for retrieval


def test_empty_testset_rejected(retriever):
    with pytest.raises(ValueError):
        evaluate_retrieval(retriever, [])
