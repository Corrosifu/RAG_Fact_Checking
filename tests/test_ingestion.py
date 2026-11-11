import pytest

from rag_fact_checking import ingestion
from rag_fact_checking.ingestion import ArxivPaper, build_query_url, parse_atom, safe_filename

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2510.08562v1</id>
    <title>A  Study
      of Things</title>
    <summary> We study things. </summary>
    <published>2025-10-09T17:59:59Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2510.08562v1" rel="alternate" type="text/html"/>
    <link title="pdf" rel="related" type="application/pdf"
          href="http://arxiv.org/pdf/2510.08562v1"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0101001v2</id>
    <title>No links</title><summary>S</summary><published>2001-01-01T00:00:00Z</published>
  </entry>
</feed>"""


def test_parse_atom():
    a, b = parse_atom(FEED)
    assert (
        a.id == "2510.08562v1"
        and a.title == "A Study of Things"
        and a.summary == "We study things."
    )
    assert a.authors == ["Ada Lovelace", "Alan Turing"] and a.pdf_url.endswith("2510.08562v1")
    assert (
        b.id == "cs/0101001v2"
        and b.pdf_url == "https://arxiv.org/pdf/cs/0101001v2"
        and b.authors == []
    )


def test_query_url_is_encoded():
    url = build_query_url("deep learning & ethics", 5)
    assert (
        "deep+learning+%26+ethics" in url and "max_results=5" in url and url.startswith("https://")
    )
    with pytest.raises(ValueError):
        build_query_url("x", 0)


def test_safe_filename():
    assert safe_filename("cs/0101001v2") == "cs_0101001v2.pdf"
    assert "/" not in safe_filename("../../etc/passwd")


class FakeResponse:
    def __init__(self, text="", content=b"", status=200):
        self.text, self.content, self.status_code = text, content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_ingest_abstract_only(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.requests, "get", lambda url, timeout: FakeResponse(text=FEED))
    docs = ingestion.ingest("things", 2, tmp_path)
    assert [d.id for d in docs] == ["2510.08562v1", "cs/0101001v2"]
    assert (
        docs[0].text.startswith("A Study of Things.")
        and docs[0].metadata["authors"][0] == "Ada Lovelace"
    )
    assert not list(tmp_path.iterdir())  # no PDF downloaded


def test_download_pdf_is_cached(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse(content=b"%PDF-1.4")

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    paper = ArxivPaper("1", "t", [], "s", "d", "http://x/1.pdf")
    p1 = ingestion.download_pdf(paper, tmp_path)
    p2 = ingestion.download_pdf(paper, tmp_path)
    assert p1 == p2 and p1.read_bytes() == b"%PDF-1.4" and len(calls) == 1


def test_full_text_falls_back_to_abstract_on_pdf_error(monkeypatch, tmp_path, capsys):
    def fake_get(url, timeout):
        return FakeResponse(text=FEED) if "api/query" in url else FakeResponse(status=404)

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    docs = ingestion.ingest("things", 2, tmp_path, full_text=True)
    assert len(docs) == 2 and "We study things." in docs[0].text
    assert "PDF unavailable" in capsys.readouterr().out
