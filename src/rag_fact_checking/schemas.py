"""Plain data containers shared across the package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document (e.g. one arXiv paper) before chunking."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A contiguous passage of a document."""

    id: str  # "<doc_id>#<position>"
    doc_id: str
    position: int
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its score and provenance."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class Source:
    """Citation-friendly view of a hit, safe to serialise."""

    doc_id: str
    title: str
    chunk_id: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Premise:
    """A short evidence window (a few sentences) extracted from a retrieved chunk."""

    text: str
    doc_id: str
    title: str
    chunk_id: str


@dataclass(frozen=True)
class EvidenceSentence:
    """A premise with the NLI probabilities it received against the claim."""

    text: str
    doc_id: str
    title: str
    chunk_id: str
    entailment: float
    contradiction: float
    neutral: float
    stance: str  # "supports" | "contradicts" | "neutral"


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass(frozen=True)
class FactCheckResult:
    claim: str
    verdict: Verdict
    confidence: float
    evidence: list[Source]
    evidence_sentences: list[EvidenceSentence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out
