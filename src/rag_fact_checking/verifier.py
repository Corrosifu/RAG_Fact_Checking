"""Claim verification against retrieved evidence using an NLI model.

The decision rule (:func:`aggregate_verdict`) is pure and model-free so it can be
unit-tested; :class:`NLIVerifier` supplies per-passage entailment probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag_fact_checking.config import Settings
from rag_fact_checking.schemas import EvidenceSentence, Hit, Premise, Verdict
from rag_fact_checking.textutils import (
    clean_text,
    content_terms,
    drop_reference_neighbourhood,
    is_prose,
    split_sentences,
    strip_title,
)

Probs = Mapping[str, float]  # keys: "entailment", "contradiction", "neutral"


class Verifier(Protocol):
    def entailment_probs(self, claim: str, premises: Sequence[Premise]) -> list[Probs]: ...


def select_premises(
    claim: str,
    hits: Sequence[Hit],
    max_premises: int = 8,
    window_words: int = 60,
    min_overlap: int = 2,
) -> list[Premise]:
    """Pick short evidence windows (a few sentences) that actually talk about the claim.

    NLI models are trained on short premises: feeding a whole 200-word PDF chunk (with
    references and headings) makes them hallucinate contradictions. Windows that share
    fewer than ``min_overlap`` content terms with the claim are dropped, so unrelated
    passages can neither support nor refute it.
    """
    claim_terms = content_terms(claim)
    need = min(min_overlap, len(claim_terms))
    if need == 0:
        return []
    scored: list[tuple[int, int, int, Premise]] = []
    seen: set[str] = set()
    for h_idx, hit in enumerate(hits):
        text = strip_title(clean_text(hit.chunk.text), hit.chunk.title)
        window: list[str] = []
        sentences = [s for s in drop_reference_neighbourhood(split_sentences(text)) if is_prose(s)]
        sentences.append("")
        for sent in sentences:
            if window and (
                not sent or sum(len(w.split()) for w in window) + len(sent.split()) > window_words
            ):
                premise = " ".join(window)
                overlap = len(claim_terms & content_terms(premise))
                if overlap >= need and premise.lower() not in seen:
                    seen.add(premise.lower())
                    origin = Premise(premise, hit.chunk.doc_id, hit.chunk.title, hit.chunk.id)
                    scored.append((overlap, -h_idx, -len(scored), origin))
                window = []
            if sent:
                window.append(sent)
    scored.sort(reverse=True)
    return [p for *_, p in scored[:max_premises]]


def aggregate_verdict(
    per_passage: Sequence[Probs], threshold: float = 0.6
) -> tuple[Verdict, float]:
    """Combine per-passage NLI probabilities into one verdict.

    Decision rule:

    * only entailment clears ``threshold``  -> SUPPORTED
    * only contradiction clears it          -> REFUTED
    * both clear it (conflicting evidence)  -> NOT_ENOUGH_INFO: a small NLI model often
      attributes a sentence about *another* method to the claim, so refusing to pick a
      side beats refuting a true claim on a 0.01 margin.
    * neither clears it                     -> NOT_ENOUGH_INFO
    """
    if not per_passage:
        return Verdict.NOT_ENOUGH_INFO, 0.0
    ent = max(p.get("entailment", 0.0) for p in per_passage)
    con = max(p.get("contradiction", 0.0) for p in per_passage)
    if ent >= threshold and con >= threshold:
        return Verdict.NOT_ENOUGH_INFO, 1.0 - abs(ent - con)
    if ent >= threshold:
        return Verdict.SUPPORTED, ent
    if con >= threshold:
        return Verdict.REFUTED, con
    return Verdict.NOT_ENOUGH_INFO, 1.0 - max(ent, con)


class NLIVerifier:
    def __init__(self, model_name: str, token: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "transformers/torch are not installed. Run: pip install 'rag-fact-checking[ml]'"
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, token=token
        ).eval()
        self._labels = [str(v).lower() for _, v in sorted(self.model.config.id2label.items())]

    def entailment_probs(  # pragma: no cover - needs model
        self, claim: str, premises: Sequence[Premise]
    ) -> list[Probs]:
        if not premises:
            return []
        enc = self.tokenizer(
            [p.text for p in premises],
            [claim] * len(premises),
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        return [dict(zip(self._labels, row, strict=True)) for row in probs]


def to_evidence_sentences(
    premises: Sequence[Premise], probs: Sequence[Probs], threshold: float = 0.6
) -> list[EvidenceSentence]:
    """Attach NLI probabilities to premises, strongest signal first.

    A sentence "supports" (or "contradicts") only when its own probability clears the
    threshold; everything else is "neutral" and sorted last.
    """
    out = []
    for prem, pr in zip(premises, probs, strict=True):
        ent, con, neu = (pr.get(k, 0.0) for k in ("entailment", "contradiction", "neutral"))
        stance = "neutral"
        if ent >= threshold and ent >= con:
            stance = "supports"
        elif con >= threshold:
            stance = "contradicts"
        out.append(
            EvidenceSentence(
                prem.text,
                prem.doc_id,
                prem.title,
                prem.chunk_id,
                round(ent, 4),
                round(con, 4),
                round(neu, 4),
                stance,
            )
        )
    return sorted(out, key=lambda e: (e.stance == "neutral", -max(e.entailment, e.contradiction)))


def build_verifier(settings: Settings) -> Verifier | None:
    if settings.verifier == "none":
        return None
    if settings.verifier == "nli":
        return NLIVerifier(settings.verifier_model, settings.hf_token)
    raise ValueError(f"Unknown verifier {settings.verifier!r} (use 'none' or 'nli')")
