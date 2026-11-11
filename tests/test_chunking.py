import pytest

from rag_fact_checking.chunking import chunk_document, split_words
from rag_fact_checking.schemas import Document


def test_windows_and_overlap():
    text = " ".join(str(i) for i in range(10))
    assert split_words(text, size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_short_text_single_chunk():
    assert split_words("a b c", size=10, overlap=2) == ["a b c"]


def test_exact_fit_has_no_trailing_duplicate():
    assert split_words("a b c d", size=4, overlap=2) == ["a b c d"]


def test_empty_and_whitespace():
    assert split_words("", 5, 1) == [] and split_words("  \n ", 5, 1) == []


def test_every_word_is_covered():
    words = [f"w{i}" for i in range(137)]
    chunks = split_words(" ".join(words), size=20, overlap=5)
    assert {w for c in chunks for w in c.split()} == set(words)


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (5, 5), (5, 9), (5, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_words("a b c", size, overlap)  # previously this looped forever for overlap >= size


def test_chunk_document_ids_and_metadata():
    doc = Document(id="p1", title="T", text=" ".join("x" * 1 for _ in range(50)), metadata={"k": 1})
    chunks = chunk_document(doc, size=20, overlap=5)
    assert [c.id for c in chunks] == ["p1#0", "p1#1", "p1#2"]
    assert all(c.doc_id == "p1" and c.title == "T" and c.metadata == {"k": 1} for c in chunks)
