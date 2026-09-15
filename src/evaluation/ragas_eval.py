"""RAGAS generation evaluation (faithfulness, answer_relevancy, context_precision).

Measures generation quality with a **local** judge by default (qwen2.5:14b via
Ollama, forced into JSON mode) and local bge-large embeddings - no paid API.
A hosted judge can be swapped in via ``RAGAS_JUDGE_PROVIDER`` (ADR-016).

The loop is split into two stages so they can be re-run independently:

1. :func:`build_generation_dataset` answers each question through the configured
   generation path (``GENERATION_MODE``: single-shot or agent) and records the
   answer, its context chunks, and its sources.
2. :func:`score_dataset` scores those rows with RAGAS. Because the rows are plain
   dicts, a run can be saved (:func:`save_run`) and re-scored later with a
   different judge without regenerating answers.

Failures from a flaky judge are caught and returned as NaN, never raised.
RAGAS imports are lazy, so this module (and its tests) load without the
``ragas`` package or any network/GPU dependency.
"""

import json
import logging
import math
from collections.abc import Callable, Iterable
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from src.config.settings import Settings, get_settings
from src.generation.dispatch import generate_answer

logger = logging.getLogger(__name__)

#: Metrics scored when ``RAGAS_METRICS`` is empty; all are reference-free.
DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
METRIC_NAMES = DEFAULT_METRICS
#: Columns RAGAS needs; extra row fields (sources, scores) are dropped before scoring.
_RAGAS_COLUMNS = ("question", "answer", "contexts")


def _ensure_ragas_importable() -> None:
    """Shim a removed module that ragas 0.4.x eagerly imports but we never use.

    ragas 0.4.3 imports ``langchain_community.chat_models.vertexai`` at load time;
    that submodule was removed in recent langchain-community. Since we only use
    ragas with our own judge, inject a harmless stand-in so the import succeeds
    (ADR-016).
    """
    import importlib.util
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        shim = types.ModuleType(name)
        shim.ChatVertexAI = object
        sys.modules[name] = shim


def _judge_settings(settings: Settings) -> Settings:
    """Return settings overridden to use the RAGAS judge provider/model.

    Args:
        settings: Base application settings.

    Returns:
        A copy whose ``llm_provider``/``llm_model`` point at the RAGAS judge
        (``ragas_judge_model`` falls back to ``llm_model`` when empty).
    """
    return settings.model_copy(
        update={
            "llm_provider": settings.ragas_judge_provider,
            "llm_model": settings.ragas_judge_model or settings.llm_model,
        }
    )


def build_judge_llm(settings: Settings | None = None) -> BaseChatModel:
    """Build the judge chat model via the provider factory.

    For an Ollama judge, ``RAGAS_JUDGE_JSON_MODE`` (default on) constrains the
    model to emit valid JSON, which is what RAGAS's structured prompts expect;
    local models otherwise drift into prose and the score is lost as NaN.

    Args:
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A LangChain chat model for the configured judge provider.
    """
    settings = settings or get_settings()
    from src.generation.llm import build_llm

    llm = build_llm(_judge_settings(settings))
    if settings.ragas_judge_provider.lower() == "ollama" and settings.ragas_judge_json_mode:
        llm = llm.model_copy(update={"format": "json"})
    return llm


def build_judge(settings: Settings | None = None) -> object:
    """Build the RAGAS judge wrapper around :func:`build_judge_llm`.

    Args:
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A RAGAS-compatible LLM wrapper (no external API unless the configured
        judge provider is hosted).
    """
    _ensure_ragas_importable()
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(build_judge_llm(settings))


def build_judge_embeddings(settings: Settings | None = None) -> object:
    """Build RAGAS embeddings from the configured embedding backend.

    Args:
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A RAGAS-compatible embeddings wrapper.
    """
    settings = settings or get_settings()
    _ensure_ragas_importable()
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from src.retrieval.embeddings import build_embeddings

    return LangchainEmbeddingsWrapper(build_embeddings(settings))


def _faithfulness() -> object:
    from ragas.metrics import faithfulness

    return faithfulness


def _answer_relevancy() -> object:
    from ragas.metrics import answer_relevancy

    return answer_relevancy


def _context_precision() -> object:
    # Reference-free variant: the judge rates each retrieved chunk's usefulness
    # for answering the question, so it needs no gold answers.
    from ragas.metrics import LLMContextPrecisionWithoutReference

    return LLMContextPrecisionWithoutReference(name="context_precision")


_METRIC_FACTORIES: dict[str, Callable[[], object]] = {
    "faithfulness": _faithfulness,
    "answer_relevancy": _answer_relevancy,
    "context_precision": _context_precision,
}


def metric_names(settings: Settings | None = None) -> tuple[str, ...]:
    """Return the metric names configured by ``RAGAS_METRICS``.

    Args:
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        The configured names, or :data:`DEFAULT_METRICS` when unset.
    """
    settings = settings or get_settings()
    names = tuple(name.strip() for name in settings.ragas_metrics.split(",") if name.strip())
    return names or DEFAULT_METRICS


def _validate_metric_names(names: Iterable[str]) -> tuple[str, ...]:
    """Reject unknown metric names without importing ragas."""
    names = tuple(names)
    unknown = [name for name in names if name not in _METRIC_FACTORIES]
    if unknown:
        raise ValueError(
            f"Unknown RAGAS metric(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(_METRIC_FACTORIES)}."
        )
    return names


def ragas_metrics(names: Iterable[str]) -> list[object]:
    """Resolve metric names to RAGAS metric objects (lazy import).

    Args:
        names: Metric names from :data:`_METRIC_FACTORIES`.

    Returns:
        The instantiated RAGAS metrics, in the given order.

    Raises:
        ValueError: If any name is not supported.
    """
    names = _validate_metric_names(names)
    _ensure_ragas_importable()
    return [_METRIC_FACTORIES[name]() for name in names]


def load_questions(path: Path | str | None = None, *, limit: int | None = None) -> list[dict]:
    """Load the question set (a JSON list of ``{id, topic, question}`` items).

    Args:
        path: JSON file. Defaults to ``EVAL_QUESTIONS_PATH``.
        limit: Keep only the first ``limit`` items when set.

    Returns:
        The question items.

    Raises:
        FileNotFoundError: If the file does not exist, with a hint on the format.
    """
    path = Path(path or get_settings().eval_questions_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No question set at {path}. Write a JSON list of "
            '{"id": ..., "topic": ..., "question": ...} items there (it is git-ignored) '
            "or point EVAL_QUESTIONS_PATH / --questions at one."
        )
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit is not None else items


def question_texts(items: Iterable[dict]) -> list[str]:
    """Extract the question strings from loaded question items."""
    return [str(item["question"]) for item in items]


def build_generation_dataset(
    questions: list[str],
    *,
    engine: object | None = None,
    llm: object | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[dict]:
    """Answer each question and record what the generator produced.

    Generation goes through the mode dispatcher, so ``GENERATION_MODE`` selects
    the single-shot or agent path exactly as the API would.

    Args:
        questions: Questions to answer.
        engine: Retrieval engine. Defaults to ``build_engine()``.
        llm: Chat model for the single-shot path. Defaults to the provider factory.
        top_k: Context chunks per question.
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        Rows of ``{"question", "answer", "contexts", "sources"}``.
    """
    if engine is None:
        from src.retrieval.engine import build_engine

        engine = build_engine(settings)

    rows: list[dict] = []
    for question in questions:
        result = generate_answer(question, engine=engine, llm=llm, top_k=top_k, settings=settings)
        rows.append(
            {
                "question": question,
                "answer": result.answer,
                "contexts": [chunk.page_content for chunk in result.chunks],
                "sources": list(result.sources),
            }
        )
    return rows


def _to_dataset(rows: list[dict]) -> object:
    """Build a HuggingFace ``Dataset`` from generation rows."""
    from datasets import Dataset

    return Dataset.from_list(rows)


def _run_ragas(
    dataset: object,
    *,
    judge: object,
    embeddings: object,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> object:
    """Run RAGAS ``evaluate`` over the dataset with the named metrics.

    Maps our ``question``/``answer``/``contexts`` columns to the names ragas
    0.4.x expects (``user_input``/``response``/``retrieved_contexts``) and drops
    any other columns (sources, cached scores) it does not understand.
    """
    _ensure_ragas_importable()
    from ragas import evaluate

    keep = [column for column in dataset.column_names if column in _RAGAS_COLUMNS]
    ragas_dataset = dataset.select_columns(keep).rename_columns(
        {
            "question": "user_input",
            "answer": "response",
            "contexts": "retrieved_contexts",
        }
    )
    return evaluate(
        ragas_dataset,
        metrics=ragas_metrics(metrics),
        llm=judge,
        embeddings=embeddings,
    )


def _nan_scores(error: str, names: Iterable[str]) -> dict:
    """Return a NaN-filled score dict carrying the failure reason."""
    scores: dict = {name: float("nan") for name in names}
    scores["error"] = error
    return scores


def score_dataset(
    rows: list[dict],
    *,
    judge: object | None = None,
    embeddings: object | None = None,
    metrics: Iterable[str] | None = None,
    settings: Settings | None = None,
) -> object:
    """Score generation rows with RAGAS.

    Args:
        rows: Rows from :func:`build_generation_dataset` (or a saved run).
        judge: RAGAS judge LLM. Defaults to :func:`build_judge`.
        embeddings: RAGAS embeddings. Defaults to :func:`build_judge_embeddings`.
        metrics: Metric names. Defaults to ``RAGAS_METRICS``.
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        The RAGAS result (scores), or a NaN-filled dict with an ``error`` key.
    """
    settings = settings or get_settings()
    names = _validate_metric_names(metrics if metrics is not None else metric_names(settings))
    dataset = _to_dataset(rows)

    judge = judge or build_judge(settings)
    embeddings = embeddings or build_judge_embeddings(settings)

    try:
        return _run_ragas(dataset, judge=judge, embeddings=embeddings, metrics=names)
    except Exception as exc:  # noqa: BLE001 - judge output parsing is fragile
        logger.warning("RAGAS evaluation failed (likely judge formatting): %s", exc)
        return _nan_scores(str(exc), names)


def evaluate_generation(
    questions: list[str],
    *,
    engine: object | None = None,
    llm: object | None = None,
    judge: object | None = None,
    embeddings: object | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
    metrics: Iterable[str] | None = None,
) -> object:
    """Generate answers for ``questions`` and score them with RAGAS.

    Convenience composition of :func:`build_generation_dataset` and
    :func:`score_dataset`.

    Args:
        questions: Questions to evaluate.
        engine: Retrieval engine. Defaults to ``build_engine()``.
        llm: Generator LLM for the single-shot path.
        judge: RAGAS judge LLM. Defaults to :func:`build_judge`.
        embeddings: RAGAS embeddings. Defaults to :func:`build_judge_embeddings`.
        top_k: Number of context chunks per question.
        settings: Configuration. Defaults to ``get_settings()``.
        metrics: Metric names. Defaults to ``RAGAS_METRICS``.

    Returns:
        The RAGAS result (scores), or a NaN-filled dict with an ``error`` key.
    """
    rows = build_generation_dataset(
        questions, engine=engine, llm=llm, top_k=top_k, settings=settings
    )
    return score_dataset(
        rows, judge=judge, embeddings=embeddings, metrics=metrics, settings=settings
    )


def attach_scores(rows: list[dict], result: object, names: Iterable[str]) -> list[dict]:
    """Return ``rows`` with a per-question ``scores`` dict merged in.

    Args:
        rows: Generation rows, in the order they were scored.
        result: The RAGAS result (exposes ``to_pandas()``) or the NaN dict.
        names: Metric names to extract.

    Returns:
        New row dicts; scores are NaN where the judge failed.
    """
    names = tuple(names)
    if isinstance(result, dict) or not hasattr(result, "to_pandas"):
        return [{**row, "scores": dict.fromkeys(names, float("nan"))} for row in rows]
    frame = result.to_pandas()
    scored: list[dict] = []
    for index, row in enumerate(rows):
        record = frame.iloc[index] if index < len(frame) else {}
        scored.append(
            {**row, "scores": {name: float(record.get(name, float("nan"))) for name in names}}
        )
    return scored


def _jsonable(value: object) -> object:
    """Convert NaN floats to ``None`` recursively so the JSON stays strict."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_run(
    path: Path | str,
    *,
    config: dict,
    rows: list[dict],
    scores: dict,
    failures: list[dict],
) -> Path:
    """Persist an evaluation run (config, rows, aggregate scores, failures).

    Args:
        path: Destination JSON file; parent directories are created.
        config: Run configuration (mode, top_k, models, ...).
        rows: Generation rows, optionally with per-question ``scores``.
        scores: Aggregate metric scores (NaN allowed; written as ``null``).
        failures: Rows/metrics the judge could not score.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "scores": scores, "failures": failures, "rows": rows}
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def load_run(path: Path | str) -> dict:
    """Load a run saved by :func:`save_run`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
