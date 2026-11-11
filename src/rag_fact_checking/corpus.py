"""Corpus I/O (JSONL) and index building."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from rag_fact_checking.chunking import chunk_documents
from rag_fact_checking.config import Settings
from rag_fact_checking.embeddings import Embedder, build_embedder
from rag_fact_checking.index import KnowledgeBase
from rag_fact_checking.schemas import Document


def save_documents(docs: Iterable[Document], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_documents(path: Path | str) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}. Run `ragfc ingest` (arXiv) or `ragfc sample` first."
        )
    return list(_iter_documents(path))


def _iter_documents(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield Document(**json.loads(line))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid document record ({exc})") from exc


def build_index(
    settings: Settings, embedder: Embedder | None = None, batch_size: int = 64
) -> KnowledgeBase:
    """Chunk the corpus, embed it in batches and persist the knowledge base."""
    embedder = embedder or build_embedder(settings)
    docs = load_documents(settings.corpus_path)
    chunks = chunk_documents(
        docs, settings.chunk_size, settings.chunk_overlap, settings.drop_references
    )
    if not chunks:
        raise ValueError("The corpus is empty: nothing to index.")

    import numpy as np

    batches = [
        embedder.embed([c.text for c in chunks[i : i + batch_size]])
        for i in range(0, len(chunks), batch_size)
    ]
    kb = KnowledgeBase(chunks, np.vstack(batches), embedder.name)
    kb.save(settings.index_dir)
    return kb
