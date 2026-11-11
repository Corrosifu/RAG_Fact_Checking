"""Word-window chunking.

Chunk sizes are expressed in words so that chunking needs no tokenizer download.
The previous implementation used 1000 SciBERT tokens, which exceeds BERT's
512-token limit; 200 words (~270 tokens) fits every common encoder.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_fact_checking.schemas import Chunk, Document
from rag_fact_checking.textutils import strip_references


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of ``size`` words overlapping by ``overlap``."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_document(
    doc: Document, size: int = 200, overlap: int = 40, drop_references: bool = True
) -> list[Chunk]:
    """Chunks of a document; the reference list is left out unless ``drop_references=False``."""
    text = strip_references(doc.text) if drop_references else doc.text
    return [
        Chunk(
            id=f"{doc.id}#{i}",
            doc_id=doc.id,
            position=i,
            text=text,
            title=doc.title,
            metadata=doc.metadata,
        )
        for i, text in enumerate(split_words(text, size, overlap))
    ]


def chunk_documents(
    docs: Iterable[Document], size: int = 200, overlap: int = 40, drop_references: bool = True
) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, size, overlap, drop_references)]
