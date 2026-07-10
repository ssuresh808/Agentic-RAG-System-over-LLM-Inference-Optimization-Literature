"""Shared pytest fixtures.

Forces in-memory Qdrant and stubs the API's engine dependency so the suite stays
fast, isolated, and fully offline - it never builds real embeddings (the slim
test env has no torch/bge-large). Tests that need a working engine override the
dependency with their own fake.
"""

import pytest

from src.api import rate_limit
from src.api.main import app, get_engine
from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate_runtime(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_PATH", "")
    # Keep the API guards permissive by default so existing endpoint tests need
    # no key and start from a clean rate-limit/budget slate; tests that exercise
    # the guards override get_settings explicitly (see tests/test_rate_limit.py).
    monkeypatch.setenv("RAG_API_KEY", "")
    monkeypatch.setenv("API_PAUSED", "false")
    rate_limit.reset_state()
    get_settings.cache_clear()
    get_engine.cache_clear()
    # Default the API engine to a stub; tests needing retrieval override it.
    app.dependency_overrides[get_engine] = lambda: object()
    yield
    app.dependency_overrides.pop(get_engine, None)
    rate_limit.reset_state()
    get_settings.cache_clear()
    get_engine.cache_clear()
