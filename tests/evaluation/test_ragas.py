"""Tests for the RAGAS generation-evaluation loop (fully mocked, no network/GPU).

The generator (the mode dispatcher), the retriever (engine), and the RAGAS
scoring call (_run_ragas) are all mocked, so these run instantly and offline.
"""

import json
import math

import pytest
from langchain_core.documents import Document

from src.config.settings import Settings
from src.evaluation import ragas_eval
from src.generation.rag import RagAnswer


def _fake_answer(question, **kwargs):
    return RagAnswer(
        answer=f"answer to {question}",
        sources=["arxiv:1"],
        chunks=[Document(page_content="ctx chunk", metadata={"source": "arxiv:1"})],
    )


def test_evaluate_generation_builds_ragas_dataset_and_returns_scores(monkeypatch):
    monkeypatch.setattr(ragas_eval, "generate_answer", _fake_answer)
    captured = {}

    def fake_run(dataset, *, judge, embeddings, **kwargs):
        captured["dataset"] = dataset
        captured["judge"] = judge
        return {"faithfulness": 0.8, "answer_relevancy": 0.9}

    monkeypatch.setattr(ragas_eval, "_run_ragas", fake_run)

    out = ragas_eval.evaluate_generation(
        ["Q1", "Q2"], engine=object(), judge="JUDGE", embeddings="EMB"
    )

    assert out == {"faithfulness": 0.8, "answer_relevancy": 0.9}
    dataset = captured["dataset"]
    assert {"question", "answer", "contexts"} <= set(dataset.column_names)
    assert len(dataset) == 2
    assert dataset[0]["contexts"] == ["ctx chunk"]
    assert captured["judge"] == "JUDGE"  # injected judge is used (no OpenAI default)


def test_evaluate_generation_is_nan_tolerant(monkeypatch):
    monkeypatch.setattr(ragas_eval, "generate_answer", _fake_answer)

    def boom(dataset, *, judge, embeddings, **kwargs):
        raise ValueError("could not parse JSON from local judge")

    monkeypatch.setattr(ragas_eval, "_run_ragas", boom)

    out = ragas_eval.evaluate_generation(["Q"], engine=object(), judge="J", embeddings="E")

    assert math.isnan(out["faithfulness"])
    assert math.isnan(out["answer_relevancy"])
    assert "could not parse JSON" in out["error"]


def test_evaluate_generation_routes_through_the_mode_dispatcher(monkeypatch):
    # The eval must measure whichever GENERATION_MODE is configured (production
    # runs the agent path), not pin itself to the single-shot function.
    seen = []

    def fake_generate(question, **kwargs):
        seen.append(kwargs.get("settings"))
        return _fake_answer(question)

    monkeypatch.setattr(ragas_eval, "generate_answer", fake_generate)
    monkeypatch.setattr(ragas_eval, "_run_ragas", lambda dataset, **kw: {"faithfulness": 1.0})
    settings = Settings(_env_file=None, generation_mode="agent")

    ragas_eval.evaluate_generation(
        ["Q"], engine=object(), judge="J", embeddings="E", settings=settings
    )

    assert seen and seen[0].generation_mode == "agent"


def test_build_generation_dataset_records_answer_contexts_and_sources(monkeypatch):
    monkeypatch.setattr(ragas_eval, "generate_answer", _fake_answer)

    rows = ragas_eval.build_generation_dataset(["Q1"], engine=object(), top_k=3)

    assert rows == [
        {
            "question": "Q1",
            "answer": "answer to Q1",
            "contexts": ["ctx chunk"],
            "sources": ["arxiv:1"],
        }
    ]


def test_load_questions_reads_json_items(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {"id": "kv-01", "topic": "kv-cache", "question": "How does PagedAttention work?"},
                {"id": "spec-01", "topic": "spec", "question": "What is speculative decoding?"},
            ]
        ),
        encoding="utf-8",
    )

    items = ragas_eval.load_questions(path)

    assert [item["id"] for item in items] == ["kv-01", "spec-01"]
    assert ragas_eval.question_texts(items) == [
        "How does PagedAttention work?",
        "What is speculative decoding?",
    ]


def test_load_questions_honours_limit(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps([{"id": f"q{i}", "question": f"Q{i}"} for i in range(5)]), encoding="utf-8"
    )

    assert len(ragas_eval.load_questions(path, limit=2)) == 2


def test_metric_names_come_from_settings():
    settings = Settings(_env_file=None, ragas_metrics="faithfulness, context_precision")

    assert ragas_eval.metric_names(settings) == ("faithfulness", "context_precision")


def test_default_metric_names_include_context_precision():
    assert "context_precision" in ragas_eval.metric_names(Settings(_env_file=None))


def test_unknown_metric_name_raises_before_importing_ragas():
    with pytest.raises(ValueError, match="bogus"):
        ragas_eval.ragas_metrics(("faithfulness", "bogus"))


def test_judge_llm_uses_json_mode_for_ollama_by_default():
    settings = Settings(_env_file=None, ragas_judge_provider="ollama", llm_model="qwen2.5:14b")

    llm = ragas_eval.build_judge_llm(settings)

    assert llm.format == "json"


def test_judge_llm_json_mode_can_be_disabled():
    settings = Settings(
        _env_file=None,
        ragas_judge_provider="ollama",
        llm_model="qwen2.5:14b",
        ragas_judge_json_mode=False,
    )

    assert ragas_eval.build_judge_llm(settings).format is None


def test_judge_settings_uses_ragas_provider_and_model():
    settings = Settings(
        _env_file=None,
        ragas_judge_provider="openai",
        ragas_judge_model="gpt-4o",
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    )
    judge_settings = ragas_eval._judge_settings(settings)
    assert judge_settings.llm_provider == "openai"
    assert judge_settings.llm_model == "gpt-4o"


def test_judge_settings_falls_back_to_llm_model():
    settings = Settings(
        _env_file=None,
        ragas_judge_provider="ollama",
        ragas_judge_model="",
        llm_model="qwen2.5:14b",
    )
    judge_settings = ragas_eval._judge_settings(settings)
    assert judge_settings.llm_provider == "ollama"
    assert judge_settings.llm_model == "qwen2.5:14b"


def test_save_and_load_run_round_trip(tmp_path):
    path = tmp_path / "results" / "run.json"
    rows = [{"question": "Q", "answer": "A", "contexts": ["c"], "sources": ["s"]}]
    scores = {"faithfulness": 0.5, "answer_relevancy": float("nan")}

    ragas_eval.save_run(
        path,
        config={"mode": "single", "top_k": 5},
        rows=rows,
        scores=scores,
        failures=[{"question": "Q", "failed_metric": "answer_relevancy"}],
    )
    loaded = ragas_eval.load_run(path)

    assert loaded["config"] == {"mode": "single", "top_k": 5}
    assert loaded["rows"] == rows
    assert loaded["scores"]["faithfulness"] == 0.5
    assert loaded["scores"]["answer_relevancy"] is None  # NaN serialised as null
    assert loaded["failures"][0]["failed_metric"] == "answer_relevancy"
