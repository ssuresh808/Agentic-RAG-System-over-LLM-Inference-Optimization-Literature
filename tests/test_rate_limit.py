"""Tests for the abuse-protection guards (src/api/rate_limit.py).

Exercised end-to-end through the FastAPI app so the wiring on ``/api/v1/query``
is covered too. ``/query`` is used because it needs only the retrieval engine
(no LLM); a fake engine returns an empty result, so every admitted call is a fast
200 and the assertions are about the guard, not retrieval. Limits are made tiny
by overriding ``get_settings`` so each case needs only a handful of requests.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import rate_limit
from src.api.main import app, get_engine
from src.config.settings import Settings


class _EmptyEngine:
    """Minimal engine stub: every query returns no documents."""

    def query(self, text: str, top_k: int = 5):  # noqa: D102 - test stub
        return []


@pytest.fixture
def client():
    test_client = TestClient(app)
    app.dependency_overrides[get_engine] = lambda: _EmptyEngine()
    yield test_client
    app.dependency_overrides.clear()


def _use_settings(monkeypatch, **overrides) -> Settings:
    """Point both guards at a fixed Settings built from ``overrides``.

    ``require_api_key`` and ``enforce_limits`` read module-level ``get_settings``,
    so patching it here controls the limits without touching the environment.
    """
    settings = Settings(_env_file=None, **overrides)
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    return settings


def _query(client: TestClient, **kwargs):
    return client.post("/api/v1/query", json={"text": "anything"}, **kwargs)


# --- Shared-secret auth -----------------------------------------------------


def test_no_key_configured_allows_anonymous(client, monkeypatch):
    _use_settings(monkeypatch)  # rag_api_key defaults to "" -> auth disabled
    assert _query(client).status_code == 200


def test_missing_key_is_rejected(client, monkeypatch):
    _use_settings(monkeypatch, rag_api_key="s3cret")
    resp = _query(client)
    assert resp.status_code == 401


def test_wrong_key_is_rejected(client, monkeypatch):
    _use_settings(monkeypatch, rag_api_key="s3cret")
    resp = _query(client, headers={"X-API-Key": "nope"})
    assert resp.status_code == 401


def test_correct_key_is_accepted(client, monkeypatch):
    _use_settings(monkeypatch, rag_api_key="s3cret")
    resp = _query(client, headers={"X-API-Key": "s3cret"})
    assert resp.status_code == 200


def test_unauthenticated_request_does_not_spend_budget(client, monkeypatch):
    _use_settings(monkeypatch, rag_api_key="s3cret", api_daily_budget=1)
    _query(client)  # 401, must not count against the budget
    assert rate_limit._daily == {}


# --- Per-IP rate limit ------------------------------------------------------


def test_burst_limit_returns_429_with_retry_after(client, monkeypatch):
    _use_settings(monkeypatch, api_burst_limit=2, api_burst_window=60)
    assert _query(client).status_code == 200
    assert _query(client).status_code == 200
    blocked = _query(client)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_rate_limit_is_per_ip(client, monkeypatch):
    _use_settings(monkeypatch, api_burst_limit=1, api_burst_window=60)
    a = {"X-Forwarded-For": "1.1.1.1"}
    b = {"X-Forwarded-For": "2.2.2.2"}
    assert _query(client, headers=a).status_code == 200
    assert _query(client, headers=a).status_code == 429  # same IP, over limit
    assert _query(client, headers=b).status_code == 200  # different IP, own bucket


def test_rejected_request_does_not_spend_budget(client, monkeypatch):
    _use_settings(monkeypatch, api_burst_limit=1, api_burst_window=60)
    _query(client)  # 200, spends 1
    _query(client)  # 429, must not spend
    today = next(iter(rate_limit._daily))
    assert rate_limit._daily[today] == 1


# --- Global daily budget & pause -------------------------------------------


def test_daily_budget_returns_503_when_spent(client, monkeypatch):
    _use_settings(monkeypatch, api_daily_budget=2)
    assert _query(client).status_code == 200
    assert _query(client).status_code == 200
    blocked = _query(client)
    assert blocked.status_code == 503
    assert int(blocked.headers["Retry-After"]) > 0


def test_pause_switch_returns_503(client, monkeypatch):
    _use_settings(monkeypatch, api_paused=True)
    assert _query(client).status_code == 503


# --- Health stays open ------------------------------------------------------


def test_health_is_not_guarded(client, monkeypatch):
    _use_settings(monkeypatch, rag_api_key="s3cret", api_paused=True)
    # No key, paused: the paid endpoints would 401/503, but /health must answer
    # so Render's health check keeps the service alive.
    assert client.get("/health").status_code == 200


# --- Client-IP resolution ---------------------------------------------------


def test_client_ip_prefers_forwarded_header():
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"9.9.9.9, 10.0.0.1")],
        "client": ("127.0.0.1", 12345),
    }
    assert rate_limit._client_ip(Request(scope)) == "9.9.9.9"


def test_client_ip_falls_back_to_socket_peer():
    from starlette.requests import Request

    scope = {"type": "http", "headers": [], "client": ("203.0.113.5", 12345)}
    assert rate_limit._client_ip(Request(scope)) == "203.0.113.5"
