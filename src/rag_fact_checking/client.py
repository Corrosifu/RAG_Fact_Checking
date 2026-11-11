"""Tiny HTTP client used by the Streamlit UI (kept separate so it is testable)."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BACKEND_URL = "http://localhost:8000"


class BackendError(RuntimeError):
    pass


def backend_url() -> str:
    return os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def post(
    path: str, payload: dict[str, Any], base_url: str | None = None, timeout: int = 120
) -> dict[str, Any]:
    url = f"{(base_url or backend_url()).rstrip('/')}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise BackendError(f"Cannot reach backend at {url}: {exc}") from exc
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise BackendError(f"Backend returned {resp.status_code}: {detail}")
    return resp.json()
