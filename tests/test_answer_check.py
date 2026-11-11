from rag_fact_checking.answer_check import check_answer, split_answer_sentences
from rag_fact_checking.schemas import Chunk, Hit
from stubs import StubVerifier

MARKDOWN_ANSWER = """The method works in several steps:

1. **Predicting Depth Maps**: The model predicts one depth map per input view.
2. **Update**: It computes a new set of Gaussian values for the scene.
- Short bit.

This approach is efficient and cheap. Is it good? Yes it is quite good indeed."""


def test_split_drops_markup_leadins_questions_and_fragments():
    sents = split_answer_sentences(MARKDOWN_ANSWER)
    assert "The method works in several steps:" not in sents  # lead-in ending with ':'
    assert "Short bit." not in sents and "Is it good?" not in sents
    assert any(s.startswith("Predicting Depth Maps: The model predicts") for s in sents)
    assert not any("**" in s or s.startswith("1.") for s in sents)
    assert "This approach is efficient and cheap." in sents


def test_split_empty():
    assert split_answer_sentences("") == [] and split_answer_sentences("   \n\n") == []


def hits():
    text = (
        "Quantization reduces the memory footprint of a neural network by using fewer bits. "
        "Bananas are yellow fruits that grow in tropical climates around the world."
    )
    return [Hit(Chunk("q#0", "q", 0, text, "Quantization paper title here"), 1.0)]


def test_statuses_follow_nli_and_order_is_preserved():
    answer = (
        "Quantization reduces the memory footprint of a neural network.\n"
        "Quantization never reduces the memory footprint of a network.\n"
        "Bananas are the main subject of this particular paper here."
    )
    checks = check_answer(answer, hits(), StubVerifier())
    assert [c.status for c in checks] == ["supported", "contradicted", "unsupported"]
    assert checks[0].evidence.stance == "supports" and checks[0].evidence.chunk_id == "q#0"
    assert checks[0].confidence == 0.95


def test_statement_unrelated_to_sources_has_no_evidence():
    checks = check_answer(
        "The Eiffel Tower is located in the city of Berlin.", hits(), StubVerifier()
    )
    assert checks[0].status == "unsupported" and checks[0].evidence is None
    assert checks[0].confidence == 0.0


def test_verifier_is_called_once_per_sentence_with_limited_premises():
    v = StubVerifier()
    check_answer(
        "Quantization reduces the memory footprint of a network.\nBananas are yellow fruits here.",
        hits(),
        v,
        max_premises=1,
    )
    assert len(v.calls) == 2  # one NLI call per answer sentence
    assert all(n <= 1 for _, n in v.calls)


def test_window_size_controls_how_much_context_a_sentence_is_checked_against():
    text = (
        "Quantization reduces memory by using fewer bits for each weight. "
        "It was first studied for neural networks in signal processing venues. "
        "The accuracy loss stays small in most reported experiments overall."
    )
    h = [Hit(Chunk("q#0", "q", 0, text, "Quantization paper title here"), 1.0)]
    claim = "Quantization reduces memory and the accuracy loss stays small."
    narrow, wide = StubVerifier(), StubVerifier()
    check_answer(claim, h, narrow, window_words=12)
    check_answer(claim, h, wide, window_words=100)
    assert max(len(t.split()) for t in narrow.premise_texts) <= 12
    assert any("fewer bits" in t and "accuracy loss" in t for t in wide.premise_texts)
