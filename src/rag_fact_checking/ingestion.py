"""arXiv ingestion: metadata (Atom API) -> PDF download -> text extraction."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import requests

from rag_fact_checking.schemas import Document

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class ArxivPaper:
    id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    pdf_url: str


def build_query_url(topic: str, max_results: int) -> str:
    if max_results <= 0:
        raise ValueError("max_results must be positive")
    return (
        f"{ARXIV_API_URL}?search_query=all:{quote_plus(topic)}"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_atom(xml_text: str) -> list[ArxivPaper]:
    """Parse an arXiv Atom feed into papers (robust to missing optional fields)."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", _NS):
        raw_id = _clean(entry.findtext("a:id", namespaces=_NS))
        pdf = next(
            (
                link.get("href", "")
                for link in entry.findall("a:link", _NS)
                if link.get("title") == "pdf" or link.get("type") == "application/pdf"
            ),
            "",
        )
        paper_id = raw_id.split("/abs/")[-1]
        papers.append(
            ArxivPaper(
                id=paper_id,
                title=_clean(entry.findtext("a:title", namespaces=_NS)),
                authors=[_clean(n.text) for n in entry.findall("a:author/a:name", _NS)],
                summary=_clean(entry.findtext("a:summary", namespaces=_NS)),
                published=_clean(entry.findtext("a:published", namespaces=_NS)),
                pdf_url=pdf or f"https://arxiv.org/pdf/{paper_id}",
            )
        )
    return papers


def fetch_papers(topic: str, max_results: int = 10, timeout: int = 30) -> list[ArxivPaper]:
    resp = requests.get(build_query_url(topic, max_results), timeout=timeout)
    resp.raise_for_status()
    return parse_atom(resp.text)


def safe_filename(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id) + ".pdf"


def download_pdf(paper: ArxivPaper, directory: Path, timeout: int = 60) -> Path:
    """Download the PDF unless already cached."""
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / safe_filename(paper.id)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    resp = requests.get(paper.pdf_url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "pypdf is not installed. Run: pip install 'rag-fact-checking[pdf]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def ingest(
    topic: str,
    max_results: int,
    pdf_dir: Path,
    full_text: bool = False,
) -> list[Document]:
    """Fetch papers and turn them into documents.

    By default only title + abstract are used (instant, tiny, no PDF parsing). With
    ``full_text=True`` PDFs are downloaded and parsed; a paper whose PDF fails falls
    back to its abstract instead of aborting the whole run.
    """
    docs = []
    for paper in fetch_papers(topic, max_results):
        text = paper.summary
        if full_text:
            try:
                text = extract_pdf_text(download_pdf(paper, pdf_dir)) or paper.summary
            except Exception as exc:  # noqa: BLE001 - keep ingesting the other papers
                print(f"warning: {paper.id}: PDF unavailable ({exc}); using abstract")
        docs.append(
            Document(
                id=paper.id,
                title=paper.title,
                text=f"{paper.title}. {text}",
                metadata={
                    "authors": paper.authors,
                    "published": paper.published,
                    "pdf_url": paper.pdf_url,
                },
            )
        )
    return docs
