"""Claim verification against retrieved evidence using an NLI model.

The decision rule (:func:`aggregate_verdict`) is pure and model-free so it can be
unit-tested; :class:`NLIVerifier` supplies per-passage entailment probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag_fact_checking.config import Settings
from rag_fact_checking.schemas import Hit, Verdict

Probs = Mapping[str, float]  # keys: "entailment", "contradiction", "neutral"


class Verifier(Protocol):
    def entailment_probs(self, claim: str, hits: Sequence[Hit]) -> list[Probs]: ...


def aggregate_verdict(
    per_passage: Sequence[Probs], threshold: float = 0.6
) -> tuple[Verdict, float]:
    """Combine per-passage NLI probabilities into one verdict.

    The strongest signal wins: if the best entailment (resp. contradiction) clears
    ``threshold`` and beats the opposite side, the claim is SUPPORTED (REFUTED).
    Otherwise there is not enough information.
    """
    if not per_passage:
        return Verdict.NOT_ENOUGH_INFO, 0.0
    ent = max(p.get("entailment", 0.0) for p in per_passage)
    con = max(p.get("contradiction", 0.0) for p in per_passage)
    if ent >= threshold and ent >= con:
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
        self, claim: str, hits: Sequence[Hit]
    ) -> list[Probs]:
        if not hits:
            return []
        enc = self.tokenizer(
            [h.chunk.text for h in hits],
            [claim] * len(hits),
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        return [dict(zip(self._labels, row, strict=True)) for row in probs]


def build_verifier(settings: Settings) -> Verifier | None:
    if settings.verifier == "none":
        return None
    if settings.verifier == "nli":
        return NLIVerifier(settings.verifier_model, settings.hf_token)
    raise ValueError(f"Unknown verifier {settings.verifier!r} (use 'none' or 'nli')")
