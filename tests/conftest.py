import json
from importlib import resources

import pytest

from rag_fact_checking.config import Settings
from rag_fact_checking.corpus import build_index
from rag_fact_checking.embeddings import HashingEmbedder
from rag_fact_checking.generation import ExtractiveGenerator
from rag_fact_checking.pipeline import RAGPipeline
from rag_fact_checking.retriever import HybridRetriever


def sample_path(name: str):
    return resources.files("rag_fact_checking").joinpath("sample", name)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", chunk_size=60, chunk_overlap=10)


@pytest.fixture
def sample_corpus(settings):
    settings.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    settings.corpus_path.write_text(
        sample_path("corpus.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return settings.corpus_path


@pytest.fixture
def retriever(settings, sample_corpus) -> HybridRetriever:
    embedder = HashingEmbedder()
    build_index(settings, embedder)
    return HybridRetriever.from_settings(settings, embedder)


@pytest.fixture
def pipeline(retriever) -> RAGPipeline:
    return RAGPipeline(retriever, ExtractiveGenerator(), top_k=3)


@pytest.fixture
def testset():
    text = sample_path("testset.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
