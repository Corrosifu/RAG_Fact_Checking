import numpy as np
import pytest

from rag_fact_checking.config import Settings
from rag_fact_checking.embeddings import HashingEmbedder, build_embedder


def test_shape_and_unit_norm():
    v = HashingEmbedder(128).embed(["hello world", "another text"])
    assert v.shape == (2, 128) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_deterministic_across_instances():
    a = HashingEmbedder().embed(["gradient descent"])
    b = HashingEmbedder().embed(["gradient descent"])
    assert np.array_equal(a, b)


def test_related_texts_are_closer_than_unrelated():
    e = HashingEmbedder()
    q, rel, unrel = e.embed(
        [
            "quantization of neural networks",
            "neural network quantization saves memory",
            "bananas are yellow",
        ]
    )
    assert q @ rel > q @ unrel


def test_empty_text_does_not_produce_nan():
    assert not np.isnan(HashingEmbedder().embed([""])).any()


def test_invalid_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(0)


def test_build_embedder():
    assert build_embedder(Settings()).name.startswith("hashing")
    with pytest.raises(ValueError, match="Unknown embedder"):
        build_embedder(Settings(embedder="magic"))
