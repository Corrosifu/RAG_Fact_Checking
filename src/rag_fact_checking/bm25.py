"""Minimal BM25 (Okapi) implementation: pure Python + numpy, no extra dependency."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 and the hashing embedder."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(t)) for t in texts]
        self._lengths = np.array([sum(tf.values()) for tf in self._tfs], dtype=np.float32)
        self._avg_len = float(self._lengths.mean()) if len(texts) else 0.0
        n = len(texts)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self._tfs):
            for t in tf:
                self._postings.setdefault(t, []).append(i)

    def __len__(self) -> int:
        return len(self._tfs)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self._tfs), dtype=np.float32)
        if not self._avg_len:
            return out
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                f = self._tfs[i][term]
                norm = 1 - self.b + self.b * self._lengths[i] / self._avg_len
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` (index, score) pairs with a strictly positive score."""
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]
