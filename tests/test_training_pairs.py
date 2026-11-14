import pytest

from rag_fact_checking.evaluation.scifact import Claim
from rag_fact_checking.schemas import Document, Verdict
from rag_fact_checking.training import pairs as tp

S, N = Verdict.SUPPORTED, Verdict.NOT_ENOUGH_INFO

GOLD_SENTENCES = [
    "Quantization reduces the model size by four times.",
    "The experiments used a small benchmark.",
]
OTHER_SENTENCES = [
    "Quantization of the model size distribution was studied in chemistry.",
    "Results vary widely.",
]


def corpus():
    sentences = {"10": GOLD_SENTENCES, "30": GOLD_SENTENCES[::-1], "40": OTHER_SENTENCES}
    docs = {k: Document(k, " ".join(v), f"T{k}") for k, v in sentences.items()}
    return docs, sentences


def test_covers_uses_word_overlap():
    assert tp.covers(
        "Intro. Quantization reduces the model size by four times. More.", GOLD_SENTENCES[0]
    )
    assert not tp.covers("Birds sing in spring.", GOLD_SENTENCES[0])
    assert not tp.covers("anything", "")


def test_pairs_are_labelled_from_gold_rationales():
    docs, sentences = corpus()
    claims = [
        Claim(1, "quantization reduces model size", S, ("10",), ("10",)),
        Claim(2, "quantization reduces model size", N, ("30",), ()),
    ]
    pairs = tp.build_pairs(claims, {1: [("10", S, [0])]}, docs, sentences, chunk_size=50)
    labels = tp.label_counts(pairs)
    assert labels.get("entailment", 0) >= 1  # the gold window of claim 1
    assert "contradiction" not in labels
    # claim 2 has no gold evidence: everything derived from it is neutral
    assert all(
        p.label == "neutral"
        for p in pairs
        if p.claim == claims[1].text and p.premise.startswith("The exp")
    )


def test_missed_rationale_is_added_as_stand_alone_positive():
    docs, sentences = corpus()
    claim = Claim(1, "something entirely different", S, ("10",), ("10",))
    pairs = tp.build_pairs([claim], {1: [("10", S, [0])]}, docs, sentences, chunk_size=50)
    assert tp.Pair(GOLD_SENTENCES[0], claim.text, "entailment") in pairs


def test_hard_negatives_add_neutral_windows_from_other_abstracts():
    docs, sentences = corpus()
    claim = Claim(1, "quantization reduces model size", S, ("10",), ("10",))
    gold = {1: [("10", S, [0])]}
    plain = tp.build_pairs([claim], gold, docs, sentences, chunk_size=50)
    hard = tp.build_pairs([claim], gold, docs, sentences, chunk_size=50, negative_docs={1: ["40"]})
    extra = [p for p in hard if p not in plain]
    assert extra and all(p.label == "neutral" and "chemistry" in p.premise for p in extra)


def test_split_claims_is_stratified_and_disjoint():
    claims = [Claim(i, f"c{i}", lab, (), ()) for i, lab in enumerate([S] * 20 + [N] * 10)]
    train, held = tp.split_claims(claims, 0.2, seed=1)
    assert len(train) + len(held) == 30 and not {c.id for c in train} & {c.id for c in held}
    assert {c.label for c in held} >= {S, N}
    assert len(held) == pytest.approx(6, abs=1)
