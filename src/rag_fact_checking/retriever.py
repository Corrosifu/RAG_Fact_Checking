"""Hybrid retrieval: dense (cosine) + sparse (BM25) fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_fact_checking.bm25 import BM25
from rag_fact_checking.config import Settings
from rag_fact_checking.embeddings import Embedder
from rag_fact_checking.index import KnowledgeBase
from rag_fact_checking.schemas import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class CrossEncoderReranker:
    """Optional precision booster (``ml`` extra). Scores (query, passage) pairs."""

    def __init__(self, model_name: str, token: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install 'rag-fact-checking[ml]'"
            ) from exc
        self._model = CrossEncoder(model_name, device="cpu", token=token)

    def rerank(
        self, query: str, hits: Sequence[Hit]
    ) -> list[Hit]:  # pragma: no cover - needs model
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda p: -float(p[1]))
        return [Hit(h.chunk, float(s), h.dense_rank, h.sparse_rank) for h, s in ranked]


class HybridRetriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.kb = kb
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.bm25 = BM25([c.text for c in kb.chunks])

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder) -> "HybridRetriever":
        kb = KnowledgeBase.load(
            settings.index_dir,
            expected_embedder=embedder.name,
            backend=settings.index_backend,
            ef_search=settings.index_ef_search,
        )
        reranker: Reranker | None = None
        if settings.reranker == "cross-encoder":
            reranker = CrossEncoderReranker(settings.reranker_model, token=settings.hf_token)
        elif settings.reranker != "none":
            raise ValueError(f"Unknown reranker {settings.reranker!r}")
        return cls(kb, embedder, reranker, candidates=settings.candidates)

    def retrieve(self, query: str, top_k: int = 5) -> list[Hit]:
        if not query.strip() or not len(self.kb):
            return []
        n = max(self.candidates, top_k)
        dense = self.kb.search(self.embedder.embed([query])[0], n)
        sparse = self.bm25.top_k(query, n)

        fused: dict[int, float] = {}
        ranks: dict[int, list[int | None]] = {}
        for source, results in enumerate((dense, sparse)):
            for rank, (idx, _) in enumerate(results, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
                ranks.setdefault(idx, [None, None])[source] = rank

        order = sorted(fused, key=lambda i: (-fused[i], i))
        hits = [Hit(self.kb.chunks[i], fused[i], *ranks[i]) for i in order[:n]]
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits)
        return hits[:top_k]
