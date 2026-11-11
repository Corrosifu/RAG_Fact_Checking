from rag_fact_checking.schemas import Hit


def test_finds_relevant_document(retriever):
    hits = retriever.retrieve("How does quantization reduce model size?", top_k=3)
    assert hits[0].chunk.doc_id == "quantization"


def test_top_k_respected_and_scores_descending(retriever):
    hits = retriever.retrieve("neural network training", top_k=4)
    assert len(hits) <= 4
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_no_duplicate_chunks(retriever):
    ids = [h.chunk.id for h in retriever.retrieve("attention transformer gpu", top_k=10)]
    assert len(ids) == len(set(ids))


def test_blank_query_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []


def test_hit_records_both_rankers(retriever):
    top = retriever.retrieve("BM25 inverse document frequency", top_k=1)[0]
    assert top.sparse_rank == 1 and top.dense_rank is not None


def test_reranker_is_applied(retriever):
    class Reverse:
        def rerank(self, query, hits):
            return [Hit(h.chunk, -h.score) for h in reversed(hits)]

    base = [h.chunk.id for h in retriever.retrieve("neural network", top_k=3)]
    assert len(base) == 3
    retriever.reranker = Reverse()
    assert [h.chunk.id for h in retriever.retrieve("neural network", top_k=3)] != base
