"""On-disk knowledge base: chunks + dense vectors.

Stored as ``chunks.jsonl`` + ``vectors.npy`` + ``meta.json``. Unlike the previous
FAISS/pickle layout this needs no ``allow_dangerous_deserialization`` and is
diff-able. A flat numpy index is exact and fast for corpora up to ~100k chunks,
which is the target of this project.

For larger corpora an optional FAISS backend (``pip install -e ".[faiss]"``,
``RAGFC_INDEX_BACKEND=faiss``) answers queries with an approximate HNSW graph. It is
built in memory from ``vectors.npy`` when first used, so the on-disk format does not
change and nothing is unpickled.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag_fact_checking.schemas import Chunk


class KnowledgeBaseNotFound(FileNotFoundError):
    """Raised when the index directory has not been built yet."""


BACKENDS = ("numpy", "faiss")


class KnowledgeBase:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder_name: str,
        backend: str = "numpy",
        ef_search: int = 256,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if backend not in BACKENDS:
            raise ValueError(f"Unknown index backend {backend!r}; expected one of {BACKENDS}")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.embedder_name = embedder_name
        self.backend = backend
        self.ef_search = ef_search  # faiss only: higher = better recall, slower queries
        self._faiss_index = None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Cosine search (vectors are normalised). Returns (index, score), best first.

        Exact with the numpy backend, approximate (HNSW) with the faiss backend.
        """
        if not len(self.chunks):
            return []
        if self.backend == "faiss":
            return self._search_faiss(query_vec, k)
        return self._search_numpy(query_vec, k)

    def _search_numpy(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32).reshape(-1)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]

    def _build_faiss(self):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "The faiss backend needs faiss-cpu: pip install -e '.[faiss]' "
                "(or set RAGFC_INDEX_BACKEND=numpy)."
            ) from exc
        index = faiss.IndexHNSWFlat(self.vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(np.ascontiguousarray(self.vectors))
        return index

    def _search_faiss(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._faiss_index is None:
            self._faiss_index = self._build_faiss()
        k = min(k, len(self.chunks))
        self._faiss_index.hnsw.efSearch = max(self.ef_search, k)
        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, ids = self._faiss_index.search(query, k)
        return [(int(i), float(sc)) for i, sc in zip(ids[0], scores[0], strict=True) if i >= 0]

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(
            json.dumps({"embedder": self.embedder_name, "count": len(self.chunks)}),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path | str,
        expected_embedder: str | None = None,
        backend: str = "numpy",
        ef_search: int = 256,
    ) -> "KnowledgeBase":
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise KnowledgeBaseNotFound(
                f"No index found in {d}. Build one with `ragfc index` (after `ragfc ingest` "
                "or `ragfc sample`)."
            )
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if expected_embedder and meta["embedder"] != expected_embedder:
            raise ValueError(
                f"Index was built with embedder {meta['embedder']!r} but {expected_embedder!r} is "
                "configured. Rebuild the index with `ragfc index`."
            )
        with (d / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return cls(chunks, np.load(d / "vectors.npy"), meta["embedder"], backend, ef_search)
