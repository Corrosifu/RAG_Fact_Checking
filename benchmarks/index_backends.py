"""Compare the numpy and faiss index backends: query latency and recall vs exact search.

    pip install -e ".[faiss]"
    python benchmarks/index_backends.py --sizes 10000 50000 200000

Vectors are random unit vectors with cluster structure (a rough stand-in for text
embeddings), so absolute numbers say nothing about retrieval quality on real text:
look at how latency and recall change with the corpus size.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rag_fact_checking.index import KnowledgeBase
from rag_fact_checking.schemas import Chunk


def make_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.normal(size=(max(n // 200, 8), dim))
    x = centers[rng.integers(0, len(centers), n)] + 0.7 * rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def timed(kb: KnowledgeBase, queries: np.ndarray, k: int) -> tuple[list, float]:
    start = time.perf_counter()
    out = [kb.search(q, k) for q in queries]
    return out, (time.perf_counter() - start) / len(queries) * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 200_000])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--ef", type=int, default=256, help="faiss efSearch")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"dim={args.dim} k={args.k} queries={args.queries} efSearch={args.ef}")
    print(f"{'chunks':>9} {'numpy ms':>9} {'faiss ms':>9} {'build s':>8} {'recall@k':>9}")
    for n in args.sizes:
        vecs = make_vectors(n, args.dim, rng)
        chunks = [Chunk(f"d#{i}", "d", i, "") for i in range(n)]
        queries = make_vectors(args.queries, args.dim, rng)
        exact = KnowledgeBase(chunks, vecs, "bench", "numpy")
        approx = KnowledgeBase(chunks, vecs, "bench", "faiss", args.ef)
        truth, t_numpy = timed(exact, queries, args.k)
        start = time.perf_counter()
        approx.search(queries[0], args.k)  # builds the graph
        build = time.perf_counter() - start
        found, t_faiss = timed(approx, queries, args.k)
        recall = np.mean(
            [
                len({i for i, _ in a} & {i for i, _ in b}) / args.k
                for a, b in zip(truth, found, strict=True)
            ]
        )
        print(f"{n:>9} {t_numpy:>9.2f} {t_faiss:>9.2f} {build:>8.1f} {recall:>9.3f}")


if __name__ == "__main__":
    main()
