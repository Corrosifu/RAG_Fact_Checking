from rag_fact_checking.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_keeps_hyphenated_terms():
    assert tokenize("Fine-Tuning LLMs, fast!") == ["fine-tuning", "llms", "fast"]


def test_ranks_matching_document_first():
    bm = BM25(["cats purr softly", "dogs bark loudly", "stock markets fell"])
    assert bm.top_k("dogs bark", 3)[0][0] == 1


def test_unknown_terms_give_no_results():
    assert BM25(["a b c"]).top_k("zzz", 5) == []


def test_rare_terms_outweigh_common_ones():
    bm = BM25(["the model the model", "the quantization", "the model"])
    assert bm.top_k("the quantization", 1)[0][0] == 1


def test_empty_corpus_and_query():
    assert BM25([]).top_k("x", 3) == []
    assert BM25(["a b"]).top_k("", 3) == []


def test_k_limits_results():
    assert len(BM25(["a x", "b x", "c x"]).top_k("x", 2)) == 2
