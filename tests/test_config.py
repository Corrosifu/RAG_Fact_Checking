import pytest

from rag_fact_checking.config import Settings


def test_defaults_are_lightweight():
    s = Settings()
    assert (s.embedder, s.generator, s.verifier) == ("hashing", "extractive", "none")


def test_from_env_overrides_and_paths(tmp_path):
    s = Settings.from_env(
        {
            "RAGFC_DATA_DIR": str(tmp_path),
            "RAGFC_TOP_K": "3",
            "RAGFC_GENERATOR": "hf",
            "HF_TOKEN": "t",
        }
    )
    assert s.top_k == 3 and s.generator == "hf" and s.hf_token == "t"
    assert s.index_dir == tmp_path / "index" and s.corpus_path == tmp_path / "corpus.jsonl"


def test_from_env_empty_uses_defaults():
    assert Settings.from_env({}) == Settings()


def test_non_integer_env_is_reported():
    with pytest.raises(ValueError, match="RAGFC_TOP_K"):
        Settings.from_env({"RAGFC_TOP_K": "many"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"top_k": 0},
        {"top_k": 9, "candidates": 3},
    ],
)
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_token_not_in_repr():
    assert "secret" not in repr(Settings(hf_token="secret"))


def test_verify_answers_flag():
    assert Settings().verify_answers is True
    assert Settings.from_env({"RAGFC_VERIFY_ANSWERS": "false"}).verify_answers is False
    assert Settings.from_env({"RAGFC_VERIFY_ANSWERS": "1"}).verify_answers is True
    with pytest.raises(ValueError, match="RAGFC_VERIFY_ANSWERS"):
        Settings.from_env({"RAGFC_VERIFY_ANSWERS": "sometimes"})


def test_index_backend_setting():
    assert Settings().index_backend == "numpy"
    assert Settings.from_env({"RAGFC_INDEX_BACKEND": "faiss"}).index_backend == "faiss"
    with pytest.raises(ValueError, match="index_backend"):
        Settings(index_backend="annoy")


def test_index_ef_search_setting():
    assert Settings.from_env({"RAGFC_INDEX_EF_SEARCH": "512"}).index_ef_search == 512
    with pytest.raises(ValueError, match="index_ef_search"):
        Settings(index_ef_search=0)
