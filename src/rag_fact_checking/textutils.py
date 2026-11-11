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
    words that must stay in the evidence (a title like "Attention" or "T").
    """
    title = " ".join(title.split())
    if len(title.split()) < min_words:
        return text
    return re.sub(rf"(?<!\w){re.escape(title)}(?!\w)", " ", text, flags=re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    return _SENTENCE_RE.split(text)


def is_prose(sentence: str) -> bool:
    """Keep prose-like sentences: not shouting caps, not a fragment."""
    words = sentence.split()
    letters = [c for c in sentence if c.isalpha()]
    if len(words) < 5 or len(sentence) > 500 or not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) <= 0.5
