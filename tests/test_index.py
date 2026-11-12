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


faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")


def random_kb(n, backend, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [Chunk(f"d#{i}", "d", i, f"text {i}") for i in range(n)]
    return KnowledgeBase(chunks, vecs, "toy", backend), vecs


def test_faiss_matches_numpy_on_small_corpus():
    exact, vecs = random_kb(300, "numpy")
    approx, _ = random_kb(300, "faiss")
    for q in vecs[:20]:
        e, a = exact.search(q, 5), approx.search(q, 5)
        assert {i for i, _ in e} == {i for i, _ in a}
        assert [s for _, s in a] == pytest.approx([s for _, s in e], abs=1e-5)


def test_faiss_edge_cases():
    kb, vecs = random_kb(3, "faiss")
    assert len(kb.search(vecs[0], 50)) == 3  # k larger than the corpus
    assert kb.search(vecs[0], 1)[0][0] == 0
    assert KnowledgeBase([], np.zeros((0, 16)), "toy", "faiss").search(vecs[0], 3) == []


def test_faiss_backend_survives_save_and_load(tmp_path):
    kb, vecs = random_kb(50, "numpy")
    kb.save(tmp_path)
    loaded = KnowledgeBase.load(tmp_path, backend="faiss")
    assert loaded.backend == "faiss"
    assert loaded.search(vecs[7], 1)[0][0] == 7


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        KnowledgeBase([], np.zeros((0, 2)), "toy", "annoy")


def test_missing_faiss_gives_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("no faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    kb, vecs = random_kb(5, "faiss")
    with pytest.raises(ImportError, match=r"\[faiss\]"):
        kb.search(vecs[0], 1)
