"""High-level entry point tying retrieval, generation and verification together."""

from __future__ import annotations

from rag_fact_checking.config import Settings
from rag_fact_checking.embeddings import build_embedder
from rag_fact_checking.generation import Generator, build_generator
from rag_fact_checking.retriever import HybridRetriever
from rag_fact_checking.schemas import Answer, FactCheckResult, Hit, Source
from rag_fact_checking.verifier import (
    Verifier,
    aggregate_verdict,
    build_verifier,
    select_premises,
    to_evidence_sentences,
)


class VerifierNotConfigured(RuntimeError):
    """Fact-checking requested but RAGFC_VERIFIER is 'none'."""


def to_source(hit: Hit, snippet_chars: int = 240) -> Source:
    text = " ".join(hit.chunk.text.split())
    snippet = text if len(text) <= snippet_chars else text[: snippet_chars - 1].rstrip() + "…"
    return Source(hit.chunk.doc_id, hit.chunk.title, hit.chunk.id, round(hit.score, 4), snippet)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        verifier: Verifier | None = None,
        top_k: int = 5,
        max_new_tokens: int = 256,
        nli_threshold: float = 0.6,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.nli_threshold = nli_threshold

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        settings = settings or Settings.from_env()
        retriever = HybridRetriever.from_settings(settings, build_embedder(settings))
        return cls(
            retriever,
            build_generator(settings),
            build_verifier(settings),
            top_k=settings.top_k,
            max_new_tokens=settings.max_new_tokens,
        )

    def ask(self, query: str, top_k: int | None = None) -> Answer:
        hits = self.retriever.retrieve(query, top_k or self.top_k)
        text = self.generator.generate(query, hits, max_new_tokens=self.max_new_tokens)
        return Answer(text, [to_source(h) for h in hits])

    def fact_check(self, claim: str, top_k: int | None = None) -> FactCheckResult:
        if self.verifier is None:
            raise VerifierNotConfigured(
                "Fact-checking needs an NLI model: set RAGFC_VERIFIER=nli "
                "(requires the 'ml' extra)."
            )
        hits = self.retriever.retrieve(claim, top_k or self.top_k)
        premises = select_premises(claim, hits)
        probs = self.verifier.entailment_probs(claim, premises) if premises else []
        verdict, confidence = aggregate_verdict(probs, self.nli_threshold)
        return FactCheckResult(
            claim,
            verdict,
            round(confidence, 4),
            [to_source(h) for h in hits],
            to_evidence_sentences(premises, probs, self.nli_threshold),
        )
