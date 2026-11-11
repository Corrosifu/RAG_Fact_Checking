import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from rag_fact_checking.api import create_app  # noqa: E402
from rag_fact_checking.index import KnowledgeBaseNotFound  # noqa: E402


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(lambda: pipeline))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_query(client):
    r = client.post("/query", json={"query": "What is batch normalization?"})
    assert r.status_code == 200
    body = r.json()
    assert "normaliz" in body["text"].lower() and body["sources"][0]["doc_id"] == "batchnorm"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "x" * 2001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 99},
    ],
)
def test_invalid_queries_get_422(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_fact_check_without_verifier_is_501(client):
    r = client.post("/fact-check", json={"claim": "Quantization reduces model size."})
    assert r.status_code == 501 and "RAGFC_VERIFIER" in r.json()["detail"]


def test_missing_index_is_503_not_200():
    def factory():
        raise KnowledgeBaseNotFound("No index found")

    r = TestClient(create_app(factory)).post("/query", json={"query": "hello"})
    assert r.status_code == 503 and "No index" in r.json()["detail"]


def test_pipeline_built_once(pipeline):
    calls = []

    def factory():
        calls.append(1)
        return pipeline

    c = TestClient(create_app(factory))
    c.post("/query", json={"query": "a b"})
    c.post("/query", json={"query": "c d"})
    assert len(calls) == 1


def test_query_returns_sentence_checks_and_honours_verify_flag(retriever):
    from rag_fact_checking.pipeline import RAGPipeline
    from stubs import StubVerifier
    from test_pipeline import ANSWER, FixedGenerator

    pipe = RAGPipeline(retriever, FixedGenerator(ANSWER), verifier=StubVerifier())
    c = TestClient(create_app(lambda: pipe))
    body = c.post("/query", json={"query": "What does quantization do?"}).json()
    assert body["checks"][0]["status"] == "supported" and body["checks"][0]["sentence"]
    off = c.post("/query", json={"query": "What does quantization do?", "verify": False}).json()
    assert off["checks"] == []
    assert c.post("/query", json={"query": "x y", "verify": "maybe"}).status_code == 422
