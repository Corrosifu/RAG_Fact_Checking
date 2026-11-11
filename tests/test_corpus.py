import json

import pytest

from rag_fact_checking.corpus import build_index, load_documents, save_documents
from rag_fact_checking.index import KnowledgeBase
from rag_fact_checking.schemas import Document


def test_documents_roundtrip(tmp_path):
    docs = [Document("a", "alpha text", "A", {"x": 1}), Document("b", "βeta ünïcode", "B")]
    assert save_documents(docs, tmp_path / "sub" / "c.jsonl") == 2
    assert load_documents(tmp_path / "sub" / "c.jsonl") == docs


def test_missing_corpus_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="ragfc"):
        load_documents(tmp_path / "x.jsonl")


def test_invalid_line_reports_location(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text(json.dumps({"id": "a", "text": "t"}) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_documents(f)


def test_build_index_persists(settings, sample_corpus):
    kb = build_index(settings, batch_size=3)
    assert len(kb) > 8  # several chunks per document at chunk_size=60
    assert len(KnowledgeBase.load(settings.index_dir)) == len(kb)


def test_empty_corpus_rejected(settings):
    save_documents([], settings.corpus_path)
    with pytest.raises(ValueError, match="empty"):
        build_index(settings)
