"""Runtime configuration, read from environment variables (prefix ``RAGFC_``).

The defaults target a small machine: no GPU, no model download, a few hundred MB
of RAM. Heavier backends (sentence-transformers, a HuggingFace LLM, an NLI
verifier) are opt-in through the variables below.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable settings object. Build it with :meth:`from_env` or pass kwargs."""

    data_dir: Path = Path("data")

    # Chunking (measured in words, not model tokens).
    chunk_size: int = 200
    chunk_overlap: int = 40

    # Retrieval.
    top_k: int = 5
    candidates: int = 20
    embedder: str = "hashing"  # "hashing" | "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "none"  # "none" | "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation.
    generator: str = "extractive"  # "extractive" | "hf"
    generator_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 256
    max_context_chars: int = 3000

    # Fact-checking.
    verifier: str = "none"  # "none" | "nli"
    verifier_model: str = "cross-encoder/nli-MiniLM2-L6-H768"

    hf_token: str | None = field(default=None, repr=False)

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if self.top_k <= 0 or self.candidates < self.top_k:
            raise ValueError("need 0 < top_k <= candidates")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        d = cls()

        def get(name: str, default: str) -> str:
            return env.get(f"RAGFC_{name}") or default

        return cls(
            data_dir=Path(get("DATA_DIR", str(d.data_dir))),
            chunk_size=_env_int(env, "RAGFC_CHUNK_SIZE", d.chunk_size),
            chunk_overlap=_env_int(env, "RAGFC_CHUNK_OVERLAP", d.chunk_overlap),
            top_k=_env_int(env, "RAGFC_TOP_K", d.top_k),
            candidates=_env_int(env, "RAGFC_CANDIDATES", d.candidates),
            embedder=get("EMBEDDER", d.embedder),
            embedding_model=get("EMBEDDING_MODEL", d.embedding_model),
            reranker=get("RERANKER", d.reranker),
            reranker_model=get("RERANKER_MODEL", d.reranker_model),
            generator=get("GENERATOR", d.generator),
            generator_model=get("GENERATOR_MODEL", d.generator_model),
            max_new_tokens=_env_int(env, "RAGFC_MAX_NEW_TOKENS", d.max_new_tokens),
            max_context_chars=_env_int(env, "RAGFC_MAX_CONTEXT_CHARS", d.max_context_chars),
            verifier=get("VERIFIER", d.verifier),
            verifier_model=get("VERIFIER_MODEL", d.verifier_model),
            hf_token=env.get("HF_TOKEN") or None,
        )
