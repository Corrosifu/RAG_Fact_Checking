import pytest

from rag_fact_checking.config import Settings
from rag_fact_checking.schemas import Chunk, Hit, Verdict
from rag_fact_checking.verifier import aggregate_verdict, build_verifier, select_premises


def p(e=0.0, c=0.0, n=0.0):
    return {"entailment": e, "contradiction": c, "neutral": n}


def test_supported():
    assert aggregate_verdict([p(n=0.9), p(e=0.9)]) == (Verdict.SUPPORTED, 0.9)


def test_refuted():
    assert aggregate_verdict([p(c=0.8), p(n=0.7)]) == (Verdict.REFUTED, 0.8)


def test_not_enough_info_when_weak_or_empty():
    assert aggregate_verdict([p(e=0.3, c=0.3, n=0.4)])[0] == Verdict.NOT_ENOUGH_INFO
    assert aggregate_verdict([]) == (Verdict.NOT_ENOUGH_INFO, 0.0)


def test_conflicting_strong_evidence_is_not_enough_info():
    verdict, conf = aggregate_verdict([p(e=0.96), p(c=0.97)])
    assert verdict == Verdict.NOT_ENOUGH_INFO and conf == pytest.approx(0.99)
    assert aggregate_verdict([p(e=0.7), p(c=0.95)])[0] == Verdict.NOT_ENOUGH_INFO


def test_one_sided_evidence_wins_over_weak_opposition():
    assert aggregate_verdict([p(e=0.9), p(c=0.4)])[0] == Verdict.SUPPORTED
    assert aggregate_verdict([p(e=0.3), p(c=0.9)])[0] == Verdict.REFUTED


def test_threshold_is_configurable():
    assert aggregate_verdict([p(e=0.5)], threshold=0.4)[0] == Verdict.SUPPORTED


def test_build_verifier():
    assert build_verifier(Settings()) is None
    with pytest.raises(ValueError):
        build_verifier(Settings(verifier="x"))


def hit(text, title="Some Document Title Here"):
    return Hit(Chunk("d#0", "d", 0, text, title), 1.0)


def test_select_premises_keeps_only_sentences_about_the_claim():
    h = hit(
        "Kaczmarz is an iterative method for solving large linear systems quickly. "
        "Bananas are yellow fruits that grow in tropical climates around the world."
    )
    prem = select_premises("The Kaczmarz method solves linear systems.", [h], window_words=15)
    assert len(prem) == 1 and "Kaczmarz" in prem[0] and "Bananas" not in prem[0]


def test_unrelated_hits_give_no_premises():
    h = hit("Gradient descent minimizes a loss function by following negative gradients.")
    assert select_premises("The Eiffel Tower is located in Berlin.", [h]) == []


def test_premises_are_short_clean_and_deduplicated():
    sentence = "The learning rate controls the step size taken during gradient descent."
    h1, h2 = hit(f"## {sentence} [3, 4]"), hit(sentence)  # overlapping chunk copy
    prem = select_premises("What does the learning rate control in gradient descent?", [h1, h2])
    assert len(prem) == 1 and "[3" not in prem[0] and "##" not in prem[0]


def test_premises_respect_window_and_max():
    text = " ".join(
        f"Sentence number {i} discusses gradient descent and the learning rate." for i in range(30)
    )
    prem = select_premises(
        "gradient descent learning rate", [hit(text)], max_premises=3, window_words=25
    )
    assert len(prem) == 3 and all(len(x.split()) <= 25 + 12 for x in prem)


def test_empty_claim_terms_give_no_premises():
    assert (
        select_premises("the of and", [hit("Some long enough sentence about things here.")]) == []
    )


def test_short_titles_do_not_erase_words_from_evidence():
    from rag_fact_checking.textutils import strip_title

    assert strip_title("Attention is all you need.", "T") == "Attention is all you need."
    assert strip_title("gradient descent works", "Gradient descent") == "gradient descent works"
    long = strip_title(
        "Where Have All the Iterates Gone? Body text.", "Where Have All the Iterates Gone?"
    )
    assert "Where" not in long and "Body text." in long
