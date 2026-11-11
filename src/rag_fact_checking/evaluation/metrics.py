"""Model-free retrieval metrics (no LLM judge needed, runs in milliseconds).

A test set is a JSONL file of ``{"query": ..., "relevant_doc_ids": [...]}``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rag_fact_checking.schemas import Hit


class _Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[Hit]: ...


def _unique_doc_ids(hits: Sequence[Hit]) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.append(h.chunk.doc_id)
    return seen


def hit_rate(retrieved: Sequence[str], relevant: set[str]) -> float:
    return float(any(d in relevant for d in retrieved))


def recall_at_k(retrieved: Sequence[str], relevant: set[str]) -> float:
    return len(relevant.intersection(retrieved)) / len(relevant) if relevant else 0.0


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def load_testset(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_retrieval(
    retriever: _Retriever, testset: Sequence[dict], k: int = 5
) -> dict[str, float]:
    """Average hit-rate@k, recall@k and MRR over the test set."""
    if not testset:
        raise ValueError("empty test set")
    totals = {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}
    for item in testset:
        relevant = set(item["relevant_doc_ids"])
        retrieved = _unique_doc_ids(retriever.retrieve(item["query"], k))
        totals["hit_rate"] += hit_rate(retrieved, relevant)
        totals["recall"] += recall_at_k(retrieved, relevant)
        totals["mrr"] += mrr(retrieved, relevant)
    n = len(testset)
    return {f"{name}@{k}" if name != "mrr" else "mrr": v / n for name, v in totals.items()} | {
        "n": float(n)
    }
