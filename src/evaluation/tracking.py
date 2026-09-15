"""Weights & Biases experiment tracking for evaluation runs (ADR-017).

Acts strictly as an optional diagnostic wrapper: when ``ENABLE_WANDB`` is False
nothing is imported or initialized. NaN metrics (local-judge parse failures) are
kept out of the performance charts; instead they increment an
``eval/judge_parser_failures`` counter and populate an "Evaluation Failures"
debug table. ``wandb`` is imported lazily via ``_get_wandb`` so it is trivially
mockable and never required offline.
"""

import logging
import math

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy")
FAILURE_TABLE_NAME = "Evaluation Failures"
FAILURE_TABLE_COLUMNS = ("question", "generated_answer", "raw_judge_output", "failed_metric")


def _get_wandb():
    """Import and return the ``wandb`` module (indirection for testability)."""
    import wandb

    return wandb


def _is_nan(value: object) -> bool:
    """Return True if ``value`` is a float NaN or None."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def summarize_result(
    result: object,
    questions: list[str],
    metric_names: tuple[str, ...] = METRIC_NAMES,
) -> tuple[dict, list[dict]]:
    """Reduce an evaluation result to aggregate scores + per-failure rows.

    Args:
        result: Either the NaN-fallback dict from ``evaluate_generation`` or a
            RAGAS result exposing ``to_pandas()``.
        questions: The questions evaluated (used for the dict/total-failure case).
        metric_names: Metrics to summarise (defaults to the two Phase-2c metrics).

    Returns:
        ``(scores, failures)`` where ``scores`` maps each metric to its mean over
        non-NaN rows (NaN if none), and ``failures`` lists the rows/metrics that
        could not be scored.
    """
    if isinstance(result, dict):
        scores = {name: result.get(name, float("nan")) for name in metric_names}
        raw = str(result.get("error", ""))
        failures = [
            {
                "question": question,
                "generated_answer": "",
                "raw_judge_output": raw,
                "failed_metric": "all",
            }
            for question in questions
        ]
        return scores, failures

    frame = result.to_pandas()
    scores = {}
    for name in metric_names:
        column = frame[name].tolist() if name in frame.columns else []
        valid = [value for value in column if not _is_nan(value)]
        scores[name] = float(sum(valid) / len(valid)) if valid else float("nan")

    failures = []
    for record in frame.to_dict(orient="records"):
        for name in metric_names:
            if _is_nan(record.get(name)):
                failures.append(
                    {
                        "question": str(record.get("user_input", "")),
                        "generated_answer": str(record.get("response", "")),
                        "raw_judge_output": "",
                        "failed_metric": name,
                    }
                )
    return scores, failures


def log_evaluation(
    scores: dict,
    failures: list[dict],
    config: dict,
    *,
    settings: Settings | None = None,
) -> None:
    """Log evaluation scores + failure diagnostics to W&B when enabled.

    No-op (and no ``wandb`` import) when ``ENABLE_WANDB`` is False. NaN scores are
    filtered out of the metric log; the number of failures is logged as
    ``eval/judge_parser_failures`` and a debug table captures each failure.

    Args:
        scores: Aggregate metric scores (NaN allowed; filtered before logging).
        failures: Failure rows keyed by ``FAILURE_TABLE_COLUMNS``.
        config: Pipeline hyperparameters logged as the run config.
        settings: Configuration. Defaults to ``get_settings()``.
    """
    settings = settings or get_settings()
    if not settings.enable_wandb:
        logger.debug("W&B disabled (ENABLE_WANDB=False); skipping logging.")
        return

    wandb = _get_wandb()
    wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity or None,
        config=config,
    )

    metrics = {name: float(value) for name, value in scores.items() if not _is_nan(value)}
    metrics["eval/judge_parser_failures"] = len(failures)
    wandb.log(metrics)

    if failures:
        table = wandb.Table(columns=list(FAILURE_TABLE_COLUMNS))
        for failure in failures:
            table.add_data(*(failure.get(column, "") for column in FAILURE_TABLE_COLUMNS))
        wandb.log({FAILURE_TABLE_NAME: table})

    wandb.finish()
