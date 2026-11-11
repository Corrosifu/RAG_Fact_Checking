"""Text embedders.

* :class:`HashingEmbedder` – dependency-free, deterministic feature hashing of
  word uni/bi-grams. Works anywhere, good enough for lexical-ish similarity.
* :class:`SentenceTransformerEmbedder` – real semantic embeddings; requires the
  ``ml`` extra (``pip install .[ml]``) and downloads a model on first use.

Both return L2-normalised ``float32`` matrices so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag_fact_checking.bm25 import tokenize
from rag_fact_checking.config import Settings

# Function words carry no topical signal and, on short passages, dominate cosine similarity.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "without",
        "you",
        "your",
        "i",
        "we",
        "they",
        "than",
        "then",
        "so",
        "such",
        "these",
        "those",
        "there",
        "not",
        "no",
        "but",
        "if",
        "also",
        "more",
        "most",
        "other",
    ]
)


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


class HashingEmbedder:
    def __init__(self, dim: int = 2048) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        toks = [t for t in tokenize(text) if t not in STOPWORDS]
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        # sublinear scaling keeps long repetitive chunks from dominating
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 16, token: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install 'rag-fact-checking[ml]'"
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu", token=token)

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs model
        vecs = self._model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return _normalize(np.asarray(vecs, dtype=np.float32))


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    if settings.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model, token=settings.hf_token)
    raise ValueError(
        f"Unknown embedder {settings.embedder!r} (use 'hashing' or 'sentence-transformers')"
    )
