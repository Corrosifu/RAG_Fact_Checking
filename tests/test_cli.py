import json

import pytest

from rag_fact_checking.cli import main


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGFC_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def test_sample_index_ask_flow(data_dir, capsys):
    assert main(["sample"]) == 0 and (data_dir / "corpus.jsonl").exists()
    assert main(["index"]) == 0 and (data_dir / "index" / "vectors.npy").exists()
    capsys.readouterr()
    assert main(["ask", "How does LoRA reduce trainable parameters?"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out and "Low-rank adaptation" in out


def test_eval_on_bundled_testset(capsys):
    main(["sample"]), main(["index"])
    capsys.readouterr()
    assert main(["eval", "-k", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 6.0


def test_ask_without_index_fails_cleanly(capsys):
    assert main(["ask", "hello"]) == 1
    assert "ragfc index" in capsys.readouterr().err


def test_check_without_verifier_fails_cleanly(capsys):
    main(["sample"]), main(["index"])
    capsys.readouterr()
    assert main(["check", "Quantization shrinks models."]) == 1
    assert "RAGFC_VERIFIER" in capsys.readouterr().err


def test_index_without_corpus(capsys):
    assert main(["index"]) == 1 and "ragfc" in capsys.readouterr().err


def test_command_required():
    with pytest.raises(SystemExit):
        main([])
