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
