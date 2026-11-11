"""Lightweight RAG system for scientific fact-checking."""

from rag_fact_checking.config import Settings
from rag_fact_checking.schemas import Answer, Chunk, Document, FactCheckResult, Hit, Verdict

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "FactCheckResult",
    "Hit",
    "Settings",
    "Verdict",
    "__version__",
]
