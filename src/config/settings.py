"""Application configuration.

All settings load from environment variables (and a local ``.env`` file) via
``pydantic-settings``. Secrets must never be hardcoded or committed; see
``.env.example`` for the expected keys.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, validated configuration for the RAG system.

    Attributes:
        embedding_provider: Embedding backend to use. One of ``"huggingface"``
            (local, free, no key - Phase 1 default), ``"nvidia"``, or
            ``"openai"``. Swappable without code changes (see ADR-003).
        embedding_model: Model identifier for the active provider (a
            sentence-transformers repo id by default).
        embedding_device: Torch device for local embeddings: "auto" (prefer
            MPS/CUDA, else CPU) or an explicit "mps"/"cuda"/"cpu".
        embedding_batch_size: Batch size for local embedding encoding.
        openai_api_key: API key for the OpenAI provider. Optional; only needed
            when ``embedding_provider == "openai"``.
        nvidia_api_key: API key for NVIDIA NeMo Retriever (hosted). Optional;
            only needed when ``embedding_provider == "nvidia"``.
        qdrant_url: Qdrant location. Use ``":memory:"`` for the local in-memory
            store (Phase 1 default) or an ``http(s)://host:port`` URL for a
            running Qdrant instance.
        qdrant_path: Local on-disk Qdrant storage path; empty = in-memory.
        qdrant_cloud_url: Qdrant Cloud cluster URL; with ``qdrant_api_key`` set,
            the engine connects to the remote cluster instead of local storage.
        qdrant_api_key: API key for the Qdrant Cloud cluster.
        qdrant_collection: Name of the Qdrant collection holding the corpus.
        llm_provider: LLM backend. One of ``"ollama"`` (local, free - default),
            ``"anthropic"``, or ``"openai"``. Swappable via config (ADR-006).
        llm_model: Model identifier for the active LLM provider.
        ollama_base_url: Base URL of the local Ollama server.
        anthropic_api_key: API key for the Anthropic provider. Optional; only
            needed when ``llm_provider == "anthropic"``.
        corpus_dir: Directory of source documents the API indexes on first use.
        hybrid: Enable hybrid dense+sparse (BM25) retrieval.
        rerank: Enable cross-encoder reranking of retrieved candidates.
        reranker_model: Cross-encoder reranker model id.
        rerank_fetch_k: Number of candidates to fetch before reranking.
        arxiv_dataset: HF dataset id for the on-domain arXiv corpus.
        arxiv_collection: Qdrant collection for the LLM-optimization corpus.
        arxiv_max_docs: Fail-safe cap on documents pulled from the arXiv stream.
        ragas_judge_provider: LLM provider for the RAGAS judge (ollama/openai/anthropic).
        ragas_judge_model: Model id for the RAGAS judge; empty = use llm_model.
        ragas_judge_json_mode: Force JSON output from an Ollama judge so RAGAS's
            structured prompts parse reliably (fewer NaN scores). Ignored for
            hosted judges, which already follow JSON instructions.
        ragas_metrics: Comma-separated RAGAS metrics to score; see
            ``src/evaluation/ragas_eval.py`` for the supported names.
        eval_questions_path: JSON list of ``{id, topic, question}`` items used as
            the generation-eval question set. Kept private (git-ignored) so the
            demo is never tuned to a published question list.
        enable_wandb: When True, log evaluation runs to Weights & Biases.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity (team/user); empty = default account.
        generation_mode: Which generation path the API uses: ``"single"`` (the
            RAGAS-measured single-shot baseline, default) or ``"agent"`` (the
            LangGraph ReAct path, behind this flag - see ADR-018).
        rag_api_key: Shared secret required in the ``X-API-Key`` header on the
            cost-incurring endpoints. Empty (default) disables the check, so
            local/dev runs need no key; set it in production to block anonymous
            ``curl`` against the public backend. See ``src/api/rate_limit.py``.
        api_rate_limit: Sustained per-IP request cap within ``api_rate_window``.
        api_rate_window: Sustained rate-limit window, in seconds.
        api_burst_limit: Burst per-IP request cap within ``api_burst_window``.
        api_burst_window: Burst rate-limit window, in seconds.
        api_daily_budget: Hard ceiling on total cost-incurring calls per UTC day
            across all callers; the API returns 503 once it is spent. This is the
            financial backstop against a draining loop.
        api_paused: Maintenance kill-switch; when True the cost-incurring
            endpoints return 503 without doing any work.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embedding_provider: str = Field(
        default="huggingface",
        description="Embedding backend: 'huggingface' (local/free), 'nvidia', or 'openai'.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Model id for the active provider (HuggingFace repo id by default).",
    )
    embedding_device: str = Field(
        default="auto",
        description="Torch device for local embeddings: 'auto', 'mps', 'cuda', or 'cpu'.",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for local embedding encoding.",
    )
    openai_api_key: str = Field(
        default="", description="API key for the OpenAI provider (optional)."
    )
    nvidia_api_key: str = Field(
        default="", description="API key for NVIDIA NeMo Retriever, hosted (optional)."
    )
    voyage_api_key: str = Field(
        default="", description="API key for Voyage AI hosted embeddings (optional)."
    )
    qdrant_url: str = Field(
        default=":memory:",
        description="Qdrant URL, or ':memory:' for the local in-memory store.",
    )
    qdrant_path: str = Field(
        default="",
        description="Local on-disk Qdrant storage path; empty string = in-memory.",
    )
    qdrant_cloud_url: str = Field(
        default="",
        description="Qdrant Cloud cluster URL; with qdrant_api_key, overrides local storage.",
    )
    qdrant_api_key: str = Field(
        default="", description="API key for the Qdrant Cloud cluster (optional)."
    )
    qdrant_collection: str = Field(
        default="rag_portfolio",
        description="Qdrant collection name for the indexed corpus.",
    )
    llm_provider: str = Field(
        default="ollama",
        description="LLM backend: 'ollama' (local/free), 'anthropic', or 'openai'.",
    )
    llm_model: str = Field(
        default="qwen2.5:14b",
        description="Model id for the active LLM provider.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    anthropic_api_key: str = Field(
        default="", description="API key for the Anthropic provider (optional)."
    )
    corpus_dir: str = Field(
        default="data/sample",
        description="Directory of source documents the API indexes on first use.",
    )
    hybrid: bool = Field(
        default=False,
        description="Enable hybrid dense+sparse (BM25) retrieval.",
    )
    rerank: bool = Field(
        default=False,
        description="Enable cross-encoder reranking of retrieved candidates.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-encoder reranker model id.",
    )
    rerank_fetch_k: int = Field(
        default=20,
        description="Number of candidates to fetch before reranking.",
    )
    arxiv_dataset: str = Field(
        default="librarian-bots/arxiv-metadata-snapshot",
        description="HF dataset id for the on-domain arXiv corpus.",
    )
    arxiv_collection: str = Field(
        default="llm_optimization_domain",
        description="Qdrant collection for the LLM-optimization corpus.",
    )
    arxiv_max_docs: int = Field(
        default=500,
        description="Fail-safe cap on documents pulled from the arXiv stream.",
    )
    ragas_judge_provider: str = Field(
        default="ollama",
        description="LLM provider for the RAGAS judge: ollama, openai, or anthropic.",
    )
    ragas_judge_model: str = Field(
        default="",
        description="Model id for the RAGAS judge; empty = fall back to llm_model.",
    )
    ragas_judge_json_mode: bool = Field(
        default=True,
        description="Force JSON output from an Ollama judge (fewer RAGAS parse failures).",
    )
    ragas_metrics: str = Field(
        default="faithfulness,answer_relevancy,context_precision",
        description="Comma-separated RAGAS metrics to score.",
    )
    eval_questions_path: str = Field(
        default="data/eval/local/generation_questions.json",
        description="Private JSON question set for the generation eval (git-ignored).",
    )
    enable_wandb: bool = Field(
        default=False,
        description="When True, log evaluation runs to Weights & Biases.",
    )
    wandb_project: str = Field(
        default="llm-rag-inference",
        description="Weights & Biases project name.",
    )
    wandb_entity: str = Field(
        default="",
        description="Weights & Biases entity (team/user); empty = default account.",
    )
    generation_mode: Literal["single", "agent"] = Field(
        default="single",
        description="Generation path: 'single' (baseline) or 'agent' (LangGraph ReAct).",
    )
    rag_api_key: str = Field(
        default="",
        description="Shared secret for the X-API-Key header; empty disables the check.",
    )
    api_rate_limit: int = Field(
        default=20,
        description="Sustained per-IP request cap within api_rate_window.",
    )
    api_rate_window: int = Field(
        default=3600,
        description="Sustained rate-limit window, in seconds.",
    )
    api_burst_limit: int = Field(
        default=5,
        description="Burst per-IP request cap within api_burst_window.",
    )
    api_burst_window: int = Field(
        default=60,
        description="Burst rate-limit window, in seconds.",
    )
    api_daily_budget: int = Field(
        default=500,
        description="Hard cap on total cost-incurring calls per UTC day (503 when spent).",
    )
    api_paused: bool = Field(
        default=False,
        description="Maintenance kill-switch: when True the paid endpoints return 503.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance.

    Returns:
        The singleton ``Settings`` object, parsed once and reused. Call
        ``get_settings.cache_clear()`` in tests that mutate the environment.
    """
    return Settings()
