"""Check each sentence of a generated answer against the retrieved sources.

Small local LLMs write fluent text that is sometimes not supported by the context. This
module splits the answer into sentences and runs the NLI verifier on each one, so the user
can see which sentences are backed by the sources and which are not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_fact_checking.schemas import Hit, SentenceCheck, Verdict
from rag_fact_checking.textutils import clean_text, split_sentences
from rag_fact_checking.verifier import (
    Verifier,
    aggregate_verdict,
    select_premises,
    to_evidence_sentences,
)

_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([:;,.!?])")
_STATUS = {
    Verdict.SUPPORTED: "supported",
    Verdict.REFUTED: "contradicted",
    Verdict.NOT_ENOUGH_INFO: "unsupported",
}


def split_answer_sentences(text: str, min_words: int = 4) -> list[str]:
    """Split a (possibly markdown) answer into checkable sentences.

    Drops list numbering and markup, lead-ins ending with ":" (they assert nothing) and
    fragments shorter than ``min_words`` words.
    """
    out: list[str] = []
    for line in text.splitlines():
        # removing **bold** markup leaves "Title : text"; restore "Title: text"
        line = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", clean_text(_LIST_PREFIX_RE.sub("", line)))
        if not line:
            continue
        for sent in split_sentences(line):
            sent = sent.strip()
            if len(sent.split()) >= min_words and not sent.endswith(":") and not sent.endswith("?"):
                out.append(sent)
    return out


def check_answer(
    answer: str,
    hits: Sequence[Hit],
    verifier: Verifier,
    threshold: float = 0.6,
    max_premises: int = 4,
    window_words: int = 100,
) -> list[SentenceCheck]:
    """Verify every sentence of ``answer`` against ``hits``; order is preserved.

    ``window_words`` is wider than for claim checking because an answer sentence often
    synthesises facts spread over several source sentences (e.g. a definition). Checked on
    a handful of cases: 60 words missed a faithful definition, 100 recovered it while keeping
    invented sentences unconfirmed, 150 degraded both. Treat it as an empirical setting.
    """
    checks = []
    for sentence in split_answer_sentences(answer):
        premises = select_premises(
            sentence, hits, max_premises=max_premises, window_words=window_words
        )
        probs = verifier.entailment_probs(sentence, premises) if premises else []
        verdict, confidence = aggregate_verdict(probs, threshold)
        evidence = to_evidence_sentences(premises, probs, threshold)
        checks.append(
            SentenceCheck(
                sentence,
                _STATUS[verdict],
                round(confidence, 4),
                evidence[0] if evidence else None,
            )
        )
    return checks
