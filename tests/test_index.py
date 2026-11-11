import numpy as np
import pytest

from rag_fact_checking.index import KnowledgeBase, KnowledgeBaseNotFound
from rag_fact_checking.schemas import Chunk


def make_kb():
    chunks = [Chunk(f"d#{i}", "d", i, f"text {i}", "T", {"a": i}) for i in range(3)]
    vecs = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    return KnowledgeBase(chunks, vecs, "toy")


def test_search_orders_by_cosine():
    res = make_kb().search(np.array([0, 1], dtype=np.float32), 2)
    assert [i for i, _ in res] == [1, 2]
    assert res[0][1] == pytest.approx(1.0)


def test_search_k_larger_than_corpus_and_empty():
    assert len(make_kb().search(np.array([1, 0]), 50)) == 3
    assert KnowledgeBase([], np.zeros((0, 2)), "toy").search(np.array([1, 0]), 3) == []


def test_roundtrip(tmp_path):
    kb = make_kb()
    kb.save(tmp_path / "idx")
    loaded = KnowledgeBase.load(tmp_path / "idx", expected_embedder="toy")
    assert loaded.chunks == kb.chunks and np.array_equal(loaded.vectors, kb.vectors)


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(KnowledgeBaseNotFound, match="ragfc index"):
        KnowledgeBase.load(tmp_path / "nope")


def test_embedder_mismatch_is_detected(tmp_path):
    make_kb().save(tmp_path)
    with pytest.raises(ValueError, match="Rebuild"):
        KnowledgeBase.load(tmp_path, expected_embedder="other")


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([Chunk("a#0", "a", 0, "x")], np.zeros((2, 2)), "toy")
