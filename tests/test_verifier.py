import pytest

from rag_fact_checking.config import Settings
from rag_fact_checking.schemas import Verdict
from rag_fact_checking.verifier import aggregate_verdict, build_verifier


def p(e=0.0, c=0.0, n=0.0):
    return {"entailment": e, "contradiction": c, "neutral": n}


def test_supported():
    assert aggregate_verdict([p(n=0.9), p(e=0.9)]) == (Verdict.SUPPORTED, 0.9)


def test_refuted():
    assert aggregate_verdict([p(c=0.8), p(n=0.7)]) == (Verdict.REFUTED, 0.8)


def test_not_enough_info_when_weak_or_empty():
    assert aggregate_verdict([p(e=0.3, c=0.3, n=0.4)])[0] == Verdict.NOT_ENOUGH_INFO
    assert aggregate_verdict([]) == (Verdict.NOT_ENOUGH_INFO, 0.0)


def test_entailment_wins_ties_and_conflicts():
    assert aggregate_verdict([p(e=0.9), p(c=0.9)])[0] == Verdict.SUPPORTED
    assert aggregate_verdict([p(e=0.7), p(c=0.95)])[0] == Verdict.REFUTED


def test_threshold_is_configurable():
    assert aggregate_verdict([p(e=0.5)], threshold=0.4)[0] == Verdict.SUPPORTED


def test_build_verifier():
    assert build_verifier(Settings()) is None
    with pytest.raises(ValueError):
        build_verifier(Settings(verifier="x"))
