"""Text helpers shared by the extractive generator and the NLI verifier.

PDF text extracted to markdown is noisy (headings, citation markers, repeated titles);
these helpers turn a chunk into clean sentences.
"""

from __future__ import annotations

import re

from rag_fact_checking.bm25 import tokenize
from rag_fact_checking.embeddings import STOPWORDS

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_MARKUP_RE = re.compile(r"[#*_`>|]+|\[\d+(?:[,\s-]+\d+)*\]")


def stem(token: str) -> str:
    """Crude plural/3rd-person stripping so 'reduces' matches 'reduce'."""
    return (
        token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token
    )


def content_terms(text: str) -> set[str]:
    """Stemmed non-stopword tokens."""
    return {stem(t) for t in tokenize(text) if t not in STOPWORDS}


def clean_text(text: str) -> str:
    """Drop markdown markup and bare citation markers like [11]; normalise whitespace."""
    return " ".join(_MARKUP_RE.sub(" ", text).split())


def strip_title(text: str, title: str, min_words: int = 3) -> str:
    """Remove the document title, which PDF extraction repeats as page headers.

    Short titles (< ``min_words`` words) are left alone: they are too likely to be ordinary
    words that must stay in the evidence (a title like "Attention" or "T"). A title directly
    followed by a lower-case word is the subject of a sentence ("Byte pair encoding builds a
    vocabulary..."), not a page header, and is kept: stripping it would leave "builds a
    vocabulary...".
    """
    title = " ".join(title.split())
    if len(title.split()) < min_words:
        return text
    return re.sub(
        rf"(?<!\w){re.escape(title)}(?!\w)(?!(?-i:\s+[a-z]))", " ", text, flags=re.IGNORECASE
    )


def split_sentences(text: str) -> list[str]:
    return _SENTENCE_RE.split(text)


# --- bibliography handling ---------------------------------------------------------------------
# Extracted PDF text ends with the reference list. It is not evidence: an NLI model reads a
# citation entry as a mismatching claim ("contradiction 0.92"), and extractive answers quote it.
_REFERENCES_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*(?:r[ \t]?eferences|bibliography)"
    r"[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$"
)  # "References", "## References", "**References**", and small caps extracted as "R EFERENCES"
_APPENDIX_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*(?:appendix|appendices|supplementary)\b"
)
_PAGES_RE = re.compile(
    r"\b\d{1,4}\s*(?:\(\d+\))?\s*:\s*\d{1,5}\s*[–-]\s*\d{1,5}\b|\bpp?\.\s*\d|\bvol\.?\s*\d"
)
_VENUE_RE = re.compile(
    r"(?i)\b(?:journal of|proceedings|transactions on|conference on|workshop on|arxiv preprint|"
    r"preprint arxiv|ieee|acm|siam|advances in neural information processing systems|"
    r"international conference|in proc)\b|arxiv:\d{4}\.\d{4,5}"
)
_ENTRY_END_YEAR_RE = re.compile(  # ", 2009." or "(2009)." but not an inline "(Zhang et al., 2024;"
    r"(?<!al\.),\s*(?:19|20)\d{2}[a-z]?\s*\.(?=\s|$)|\(\s*(?:19|20)\d{2}[a-z]?\s*\)\s*\."
)


def strip_references(text: str, min_position: float = 0.4) -> str:
    """Drop the reference list.

    1. With a "References" heading after ``min_position`` of the text (a table of contents or an
       early mention does not count): cut from the last such heading to the end, keeping an appendix
       that follows it.
    2. Without any heading (PDF conversion sometimes loses it): remove dense runs of bibliography
       entries, see :func:`strip_reference_blocks`.
    """
    cut = None
    for m in _REFERENCES_HEADING_RE.finditer(text):
        if m.start() >= min_position * len(text):
            cut = m
    if cut is None:
        return strip_reference_blocks(text)
    appendix = _APPENDIX_HEADING_RE.search(text, cut.end())
    return text[: cut.start()] + (text[appendix.start() :] if appendix else "")


def strip_reference_blocks(
    text: str, window: int = 60, span: int = 9, density: float = 0.4, min_windows: int = 6
) -> str:
    """Remove long runs of bibliography entries, wherever they are in the document.

    The text is cut into ``window``-word windows; a window is a candidate if it carries at least one
    entry marker. Candidates are smoothed over ``span`` windows, and runs of at least
    ``min_windows``
    windows whose smoothed density reaches ``density`` are removed (run edges are trimmed to
    candidates so no ordinary text is lost). Calibrated on 10 arXiv papers: it found the reference
    lists of the two papers that had no "References" heading, and an isolated reference-like
    sentence in the body is never removed.
    """
    words = list(re.finditer(r"\S+", text))
    if len(words) < window * min_windows:
        return text
    spans = [
        (words[i].start(), words[min(i + window, len(words)) - 1].end())
        for i in range(0, len(words), window)
    ]
    flag = [reference_signals(text[a:b]) >= 1 for a, b in spans]
    half = span // 2
    marked = [
        sum(flag[max(0, i - half) : i + half + 1]) / len(flag[max(0, i - half) : i + half + 1])
        >= density
        for i in range(len(flag))
    ]
    pieces, last, i = [], 0, 0
    while i < len(marked):
        if not marked[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(marked) and marked[j + 1]:
            j += 1
        first, end = i, j
        while first <= end and not flag[first]:
            first += 1
        while end >= first and not flag[end]:
            end -= 1
        if end - first + 1 >= min_windows:
            pieces.append(text[last : spans[first][0]])
            last = spans[end][1]
        i = j + 1
    pieces.append(text[last:])
    return "\n".join(p for p in pieces if p.strip()) if len(pieces) > 1 else text


def reference_signals(text: str) -> int:
    """Number of bibliography-entry markers: page ranges, venue names, a year ending an entry."""
    return (
        len(_PAGES_RE.findall(text))
        + len(_VENUE_RE.findall(text))
        + len(_ENTRY_END_YEAR_RE.findall(text))
    )


def looks_like_reference(text: str, min_signals: int = 2) -> bool:
    """True for a bibliography entry. Inline citations such as "(Zhang et al., 2024)" do not count.

    Checked on 778 prose windows and 292 reference windows of 10 arXiv papers: 0 prose windows
    flagged, about half of the reference windows caught (it is a safety net, not a parser).
    """
    return reference_signals(text) >= min_signals


def drop_reference_neighbourhood(sentences: list[str]) -> list[str]:
    """Remove bibliography entries from a list of sentences.

    A sentence-splitter cuts an entry into title / venue / authors; only the venue part carries the
    markers, so its immediate neighbours (the title and the author list) are dropped as well.
    """
    flags = [looks_like_reference(s) for s in sentences]
    return [
        s
        for i, s in enumerate(sentences)
        if not (flags[i] or (i > 0 and flags[i - 1]) or (i + 1 < len(flags) and flags[i + 1]))
    ]


def is_prose(sentence: str) -> bool:
    """Keep prose-like sentences: not shouting caps, not a fragment, not a bibliography entry."""
    words = sentence.split()
    letters = [c for c in sentence if c.isalpha()]
    if len(words) < 5 or len(sentence) > 500 or not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) <= 0.5 and not looks_like_reference(
        sentence
    )
