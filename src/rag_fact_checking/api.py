"""FastAPI service. The pipeline loads lazily so importing the module is cheap."""

from __future__ import annotations

import threading
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_fact_checking import __version__
from rag_fact_checking.index import KnowledgeBaseNotFound
from rag_fact_checking.pipeline import RAGPipeline, VerifierNotConfigured


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    verify: bool | None = Field(
        default=None,
        description="Check each answer sentence against the sources (needs a verifier).",
    )


class ClaimRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(pipeline_factory: Callable[[], RAGPipeline] = RAGPipeline.from_settings) -> FastAPI:
    app = FastAPI(title="RAG Fact-Checking API", version=__version__)
    state: dict[str, RAGPipeline] = {}
    lock = threading.Lock()

    def get_pipeline() -> RAGPipeline:
        if "p" not in state:
            with lock:
                if "p" not in state:
                    try:
                        state["p"] = pipeline_factory()
                    except KnowledgeBaseNotFound as exc:
                        raise HTTPException(503, detail=str(exc)) from exc
        return state["p"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        return get_pipeline().ask(req.query, req.top_k, req.verify).to_dict()

    @app.post("/fact-check")
    def fact_check(req: ClaimRequest) -> dict:
        try:
            return get_pipeline().fact_check(req.claim, req.top_k).to_dict()
        except VerifierNotConfigured as exc:
            raise HTTPException(501, detail=str(exc)) from exc

    return app


app = create_app()
