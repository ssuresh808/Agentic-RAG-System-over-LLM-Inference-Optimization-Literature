"""FastAPI application exposing health, retrieval, and RAG endpoints.

The retrieval engine and LLM are built lazily (on first request), so ``/health``
boots instantly without loading any model. On first use, the engine indexes the
configured corpus (``CORPUS_DIR``). If nothing is indexed, the query and answer
endpoints degrade gracefully instead of erroring.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.api.rate_limit import enforce_limits, require_api_key
from src.config.settings import get_settings
from src.generation.dispatch import generate_answer
from src.generation.llm import build_llm
from src.ingestion.loader import ingest_directory
from src.retrieval.engine import RetrievalEngine, build_engine

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


class QueryRequest(BaseModel):
    """Request body for a retrieval query."""

    text: str = Field(..., min_length=1, description="Natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50, description="Max chunks to return.")


class QueryResponse(BaseModel):
    """Response body containing the retrieved chunks as plain dicts."""

    results: list[dict]


class AnswerRequest(BaseModel):
    """Request body for a retrieval-augmented generation query."""

    text: str = Field(..., min_length=1, description="Question to answer.")
    top_k: int = Field(default=5, ge=1, le=50, description="Context chunks to use.")


class AnswerResponse(BaseModel):
    """Response body with the generated answer, its sources, and the context."""

    answer: str
    sources: list[str]
    results: list[dict]


@lru_cache
def get_engine() -> RetrievalEngine:
    """Return a cached retrieval engine.

    When connected to a persistent or remote store (``QDRANT_PATH``, or Qdrant
    Cloud via ``QDRANT_CLOUD_URL`` + ``QDRANT_API_KEY``), attach to the pre-seeded
    collection without re-indexing. Otherwise index the local corpus
    (``CORPUS_DIR``) on first use.

    Returns:
        The singleton ``RetrievalEngine``, ready to query.
    """
    settings = get_settings()
    engine = build_engine()
    is_remote = bool(settings.qdrant_cloud_url and settings.qdrant_api_key)
    if settings.qdrant_path or is_remote:
        try:
            engine.connect_existing()
            logger.info("Attached to seeded collection '%s'.", settings.qdrant_collection)
            return engine
        except Exception as exc:  # noqa: BLE001 - fall back to corpus indexing
            logger.warning("No seeded collection to attach (%s); indexing corpus.", exc)
    corpus_dir = settings.corpus_dir
    if Path(corpus_dir).is_dir():
        chunks = ingest_directory(corpus_dir)
        if chunks:
            engine.index(chunks)
    return engine


@lru_cache
def get_llm() -> BaseChatModel:
    """Return a process-wide cached chat model, built on first use.

    Returns:
        The singleton chat model for the configured LLM provider.
    """
    return build_llm()


app = FastAPI(title="RAG Portfolio API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        A small status payload. Does not load any model.
    """
    return {"status": "ok"}


@app.post(
    f"{API_PREFIX}/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_limits)],
)
def query(
    request: QueryRequest,
    engine: Annotated[RetrievalEngine, Depends(get_engine)],
) -> QueryResponse:
    """Retrieve the top-k chunks most similar to the query text.

    Args:
        request: Validated query payload.
        engine: Injected retrieval engine (overridable in tests).

    Returns:
        The matching chunks as ``{"text", "source"}`` dicts, or an empty list if
        nothing has been indexed yet.
    """
    try:
        documents = engine.query(request.text, top_k=request.top_k)
    except RuntimeError:
        logger.warning("Query received but index is empty; returning no results.")
        return QueryResponse(results=[])

    results = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
        }
        for doc in documents
    ]
    return QueryResponse(results=results)


@app.post(
    f"{API_PREFIX}/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_limits)],
)
def answer(
    request: AnswerRequest,
    engine: Annotated[RetrievalEngine, Depends(get_engine)],
    llm: Annotated[BaseChatModel, Depends(get_llm)],
) -> AnswerResponse:
    """Answer a question from retrieved context (retrieval-augmented generation).

    Args:
        request: Validated question payload.
        engine: Injected retrieval engine (overridable in tests).
        llm: Injected chat model (overridable in tests).

    Returns:
        The generated answer with its source filenames and the context chunks,
        or a graceful empty response if nothing has been indexed yet.
    """
    try:
        result = generate_answer(request.text, engine=engine, llm=llm, top_k=request.top_k)
    except RuntimeError:
        logger.warning("Answer requested but index is empty; returning no answer.")
        return AnswerResponse(answer="No corpus has been indexed yet.", sources=[], results=[])

    results = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
        }
        for doc in result.chunks
    ]
    return AnswerResponse(answer=result.answer, sources=result.sources, results=results)
