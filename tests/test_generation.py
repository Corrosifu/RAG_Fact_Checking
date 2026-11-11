import pytest

from rag_fact_checking.config import Settings
from rag_fact_checking.generation import (
    NO_EVIDENCE,
    ExtractiveGenerator,
    build_context,
    build_generator,
    build_messages,
)
from rag_fact_checking.schemas import Chunk, Hit


def hit(text, i=0):
    return Hit(Chunk(f"d{i}#0", f"d{i}", 0, text), 1.0)


def test_context_is_numbered_and_truncated():
    ctx = build_context([hit("alpha " * 100), hit("beta")], max_chars=50)
    assert ctx.startswith("[1] alpha") and len(ctx) <= 60 and "[2]" not in ctx


def test_messages_contain_question_and_grounding_rule():
    msgs = build_messages("What is X?", [hit("X is a thing.")])
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    assert (
        "Question: What is X?" in msgs[1]["content"] and "[1] X is a thing." in msgs[1]["content"]
    )


def test_extractive_picks_matching_sentence_with_citation():
    hits = [
        hit("Cats are mammals that purr. Bananas grow in tropical climates."),
        hit("Dogs bark at night.", 1),
    ]
    out = ExtractiveGenerator().generate("Do bananas grow in tropical climates?", hits)
    assert "Bananas grow in tropical climates" in out and out.endswith("[1]")


def test_extractive_without_evidence():
    assert ExtractiveGenerator().generate("anything", []) == NO_EVIDENCE
    assert (
        ExtractiveGenerator().generate("quantum", [hit("Cats purr softly all day long.")])
        == NO_EVIDENCE
    )


def test_build_generator_selection():
    assert isinstance(build_generator(Settings()), ExtractiveGenerator)
    with pytest.raises(ValueError, match="Unknown generator"):
        build_generator(Settings(generator="gpt"))


def test_extractive_drops_weakly_related_sentences(retriever):
    hits = retriever.retrieve("How does quantization reduce model size?", top_k=5)
    out = ExtractiveGenerator().generate("How does quantization reduce model size?", hits)
    assert "uantization" in out
    assert "Retrieval-augmented" not in out and "overfitting" not in out
