import pytest
import requests

from rag_fact_checking import client


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


def test_post_success(monkeypatch):
    seen = {}

    def fake(url, json, timeout):
        seen.update(url=url, json=json)
        return Resp(body={"ok": 1})

    monkeypatch.setattr(client.requests, "post", fake)
    assert client.post("/query", {"query": "q"}, base_url="http://b:8000/") == {"ok": 1}
    assert seen["url"] == "http://b:8000/query"


def test_backend_url_from_env(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://backend:8000/")
    assert client.backend_url() == "http://backend:8000"


def test_http_error_surfaces_detail(monkeypatch):
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: Resp(503, {"detail": "No index"}))
    with pytest.raises(client.BackendError, match="503.*No index"):
        client.post("/query", {}, base_url="http://b")


def test_non_json_error_body(monkeypatch):
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: Resp(502, None, "Bad Gateway"))
    with pytest.raises(client.BackendError, match="Bad Gateway"):
        client.post("/query", {}, base_url="http://b")


def test_connection_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(client.requests, "post", boom)
    with pytest.raises(client.BackendError, match="Cannot reach"):
        client.post("/query", {}, base_url="http://b")
