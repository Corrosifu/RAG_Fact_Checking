"""Training pairs for the NLI verifier, built from SciFact with the product's own windows.

At inference the verifier sees ~60-word evidence windows picked by ``select_premises``. The
training premises are produced by the same function, so the model is trained on the
distribution it will meet. A window is labelled from the gold rationale sentences it contains:
SUPPORT -> entailment, CONTRADICT -> contradiction, every other window -> neutral.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_fact_checking.bm25 import tokenize
from rag_fact_checking.chunking import chunk_documents
from rag_fact_checking.evaluation.scifact import Claim
from rag_fact_checking.schemas import Document, Hit, Verdict
from rag_fact_checking.verifier import select_premises

LABEL_OF = {Verdict.SUPPORTED: "entailment", Verdict.REFUTED: "contradiction"}


@dataclass(frozen=True)
class Pair:
    premise: str
    claim: str
    label: str  # "entailment" | "contradiction" | "neutral"


def covers(premise: str, sentence: str, threshold: float = 0.9) -> bool:
    """True if (almost) all words of ``sentence`` appear in ``premise``.

    Word overlap rather than substring: the window text is re-split and cleaned, so a gold
    sentence is not always a literal substring of it.
    """
    words = set(tokenize(sentence))
    if not words:
        return False
    return len(words & set(tokenize(premise))) / len(words) >= threshold


def build_pairs(
    claims: Sequence[Claim],
    rationales: Mapping[int, Sequence[tuple[str, Verdict, Sequence[int]]]],
    documents: Mapping[str, Document],
    sentences: Mapping[str, Sequence[str]],
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    max_neutral_per_claim: int = 3,
    negative_docs: Mapping[int, Sequence[str]] | None = None,
    max_hard_per_claim: int = 3,
    seed: int = 0,
) -> list[Pair]:
    """Labelled (premise, claim) pairs. Neutral windows are capped per claim to limit imbalance;
    gold rationales the window selector missed are added as stand-alone positives.

    ``negative_docs`` maps a claim id to abstracts that are NOT evidence for it (typically the
    retrieved but uncited ones); up to ``max_hard_per_claim`` of their evidence-like windows
    are added as neutral examples, which teaches the verifier to stay silent on look-alikes."""
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for claim in claims:
        gold = list(rationales.get(claim.id, []))
        doc_ids = list(dict.fromkeys([*claim.cited, *[g[0] for g in gold]]))
        chunks = chunk_documents(
            [documents[d] for d in doc_ids if d in documents], chunk_size, chunk_overlap, False
        )
        premises = select_premises(claim.text, [Hit(c, 1.0) for c in chunks])
        found = [False] * sum(len(g[2]) for g in gold)
        positives: list[Pair] = []
        neutrals: list[Pair] = []
        for prem in premises:
            label, k = "neutral", 0
            for doc_id, verdict, idxs in gold:
                for i in idxs:
                    gold_sentence = sentences[doc_id][i]
                    if prem.doc_id == doc_id and covers(prem.text, gold_sentence):
                        label, found[k] = LABEL_OF[verdict], True
                    k += 1
            (neutrals if label == "neutral" else positives).append(
                Pair(prem.text, claim.text, label)
            )
        k = 0
        for doc_id, verdict, idxs in gold:
            for i in idxs:
                if not found[k]:
                    positives.append(Pair(sentences[doc_id][i], claim.text, LABEL_OF[verdict]))
                k += 1
        rng.shuffle(neutrals)
        hard: list[Pair] = []
        if negative_docs:
            neg_chunks = chunk_documents(
                [documents[d] for d in negative_docs.get(claim.id, []) if d in documents],
                chunk_size,
                chunk_overlap,
                False,
            )
            hard = [
                Pair(p.text, claim.text, "neutral")
                for p in select_premises(claim.text, [Hit(c, 1.0) for c in neg_chunks])
            ]
            rng.shuffle(hard)
        pairs += positives + neutrals[:max_neutral_per_claim] + hard[:max_hard_per_claim]
    return pairs


def label_counts(pairs: Sequence[Pair]) -> dict[str, int]:
    return dict(Counter(p.label for p in pairs))


def split_claims(
    claims: Sequence[Claim], holdout: float = 0.15, seed: int = 0
) -> tuple[list[Claim], list[Claim]]:
    """Train / hold-out split by claim, stratified by label so every class is in both."""
    rng = random.Random(seed)
    train: list[Claim] = []
    held: list[Claim] = []
    for label in Verdict:
        group = [c for c in claims if c.label == label]
        rng.shuffle(group)
        n = round(len(group) * holdout)
        held += group[:n]
        train += group[n:]
    return train, held
