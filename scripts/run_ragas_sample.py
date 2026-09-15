"""Run the RAGAS generation eval over the persistent corpus collection.

Reads the question set from ``EVAL_QUESTIONS_PATH`` (or ``--questions``), answers
each question through the configured generation path (``--mode`` overrides
``GENERATION_MODE``), scores the answers with the configured judge, prints a
summary (overall and per topic), and saves the run as JSON so it can be
re-scored later with a different judge without regenerating answers::

    uv run python -m scripts.run_ragas_sample --run-name single_k5
    uv run python -m scripts.run_ragas_sample --mode agent --limit 10 --run-name agent_k5
    RAGAS_JUDGE_PROVIDER=anthropic uv run python -m scripts.run_ragas_sample
        --rescore data/eval/local/runs/single_k5.json

Seed the collection first with ``scripts/seed_arxiv_db.py``. NaN-tolerant: the
judge's parse failures are reported, never fatal (ADR-016).
"""

import argparse
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from src.config.settings import Settings, get_settings
from src.evaluation.ragas_eval import (
    attach_scores,
    build_generation_dataset,
    load_questions,
    load_run,
    metric_names,
    question_texts,
    save_run,
    score_dataset,
)
from src.evaluation.tracking import log_evaluation, summarize_result
from src.generation.llm import build_llm
from src.generation.rag import DEFAULT_TOP_K
from src.retrieval.engine import RetrievalEngine, build_engine

DEFAULT_RUNS_DIR = Path("data/eval/local/runs")
DEFAULT_QDRANT_PATH = ".qdrant_storage"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="RAGAS generation evaluation.")
    parser.add_argument(
        "--questions", default=None, help="Question set JSON (EVAL_QUESTIONS_PATH)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Context chunks.")
    parser.add_argument("--mode", choices=("single", "agent"), help="Override GENERATION_MODE.")
    parser.add_argument("--metrics", default=None, help="Comma-separated metrics (RAGAS_METRICS).")
    parser.add_argument("--collection", default=None, help="Qdrant collection (ARXIV_COLLECTION).")
    parser.add_argument("--run-name", default=None, help="Output file stem (default: auto).")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Where runs are saved.")
    parser.add_argument(
        "--rescore", default=None, help="Re-score a saved run instead of generating."
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level.")
    return parser.parse_args()


def _connect_engine(settings: Settings, collection: str) -> RetrievalEngine:
    """Attach to the seeded collection with the configured retrieval options."""
    engine = build_engine(
        settings.model_copy(
            update={
                "qdrant_collection": collection,
                "qdrant_path": settings.qdrant_path or DEFAULT_QDRANT_PATH,
            }
        )
    )
    engine.connect_existing()
    return engine


def _fmt(value: object) -> str:
    """Format a score for the console (NaN-safe)."""
    return (
        "  nan"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value:.3f}"
    )


def _print_summary(rows: list[dict], scores: dict, failures: list[dict], names: tuple) -> None:
    """Print overall and per-topic means."""
    print("\nOverall (mean over scored questions):")
    for name in names:
        print(f"  {name:20s} {_fmt(scores.get(name))}")
    print(f"  {'judge failures':20s} {len(failures)}")

    topics = sorted({row.get("topic") or "?" for row in rows})
    if len(topics) > 1:
        print("\nPer topic:")
        header = "  " + "topic".ljust(22) + "".join(name[:18].rjust(20) for name in names) + "   n"
        print(header)
        for topic in topics:
            topic_rows = [row for row in rows if (row.get("topic") or "?") == topic]
            cells = []
            for name in names:
                valid = [
                    row["scores"][name]
                    for row in topic_rows
                    if not math.isnan(row["scores"].get(name, float("nan")))
                ]
                cells.append(_fmt(sum(valid) / len(valid) if valid else float("nan")).rjust(20))
            print("  " + topic.ljust(22) + "".join(cells) + f"{len(topic_rows):4d}")


def main() -> None:
    """Generate (or reload) answers, score them, print a summary, save the run."""
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    if args.mode:
        settings = settings.model_copy(update={"generation_mode": args.mode})
    names = (
        tuple(name.strip() for name in args.metrics.split(",") if name.strip())
        if args.metrics
        else metric_names(settings)
    )
    collection = args.collection or settings.arxiv_collection

    if args.rescore:
        saved = load_run(args.rescore)
        rows = [
            {key: value for key, value in row.items() if key != "scores"} for row in saved["rows"]
        ]
        config = {**saved["config"], "rescored_from": args.rescore}
        print(f"Re-scoring {len(rows)} saved answers from {args.rescore}")
    else:
        items = load_questions(args.questions, limit=args.limit)
        questions = question_texts(items)
        engine = _connect_engine(settings, collection)
        print(
            f"Generating {len(questions)} answers "
            f"(mode={settings.generation_mode}, top_k={args.top_k}, collection={collection})..."
        )
        started = time.perf_counter()
        rows = build_generation_dataset(
            questions, engine=engine, llm=build_llm(settings), top_k=args.top_k, settings=settings
        )
        generation_seconds = time.perf_counter() - started
        engine.close()
        for row, item in zip(rows, items, strict=True):
            row["id"] = item.get("id")
            row["topic"] = item.get("topic")
        config = {
            "generation_mode": settings.generation_mode,
            "retriever_top_k": args.top_k,
            "hybrid": settings.hybrid,
            "rerank": settings.rerank,
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "collection_name": collection,
            "n_questions": len(questions),
            "generation_seconds": round(generation_seconds, 1),
        }

    config.update(
        {
            "judge_provider": settings.ragas_judge_provider,
            "judge_model": settings.ragas_judge_model or settings.llm_model,
            "judge_json_mode": settings.ragas_judge_json_mode,
            "metrics": list(names),
        }
    )
    judge_label = f"{config['judge_provider']}:{config['judge_model']}"
    print(f"Scoring with judge={judge_label} ({', '.join(names)})...")
    started = time.perf_counter()
    result = score_dataset(rows, metrics=names, settings=settings)
    config["scoring_seconds"] = round(time.perf_counter() - started, 1)

    questions = [row["question"] for row in rows]
    scores, failures = summarize_result(result, questions, metric_names=names)
    rows = attach_scores(rows, result, names)
    _print_summary(rows, scores, failures, names)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        args.run_name
        or f"{config.get('generation_mode', 'run')}_k{config.get('retriever_top_k')}_{stamp}"
    )
    out = save_run(
        Path(args.runs_dir) / f"{run_name}.json",
        config=config,
        rows=rows,
        scores=scores,
        failures=failures,
    )
    print(f"\nSaved run to {out}")

    log_evaluation(scores, failures, config, settings=settings)
    if settings.enable_wandb:
        print(f"Logged to W&B project '{settings.wandb_project}'.")


if __name__ == "__main__":
    main()
