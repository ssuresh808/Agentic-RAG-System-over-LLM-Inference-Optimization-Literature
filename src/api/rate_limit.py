"""Abuse protection for the cost-incurring API endpoints.

The public ``/api/v1/query`` and ``/api/v1/answer`` endpoints each spend money on
every call (Voyage embeddings, and for answers an Anthropic completion). Left
open, anyone who finds the URL could loop it and run up the bill - a financial
denial-of-service. CORS would not help: it only constrains browsers, not
``curl`` or scripts. This module adds three independent guards, all configured
via :class:`~src.config.settings.Settings` (env vars), with no new dependency:

* **Shared-secret auth** (:func:`require_api_key`): an optional ``X-API-Key``
  header. When ``RAG_API_KEY`` is set, requests without the matching key are
  rejected (401) before any work happens. The Streamlit frontend holds the key
  server-side, so the public demo keeps working while direct ``curl`` is blocked.
* **Per-IP rate limit** (:func:`enforce_limits`, layered burst + sustained
  windows): stops casual hammering from one client (429). Best-effort - behind
  Render's proxy the client IP comes from the spoofable ``X-Forwarded-For``
  header, which is why the daily budget below is the real financial backstop.
* **Global daily budget**: a hard ceiling on total cost-incurring calls per UTC
  day across all callers (503 when spent), plus an ``API_PAUSED`` kill-switch.

State is in-memory, which is correct for the single-worker Render deployment
(``--workers 1`` in ``render.yaml``): the counters are authoritative for that
process. They reset on restart and are not shared across instances, so a
horizontally scaled deployment would need a shared store (e.g. Redis).
"""

import math
import secrets
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.config.settings import Settings, get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Per-IP request timestamps and the per-UTC-day call counter. In-memory only:
# authoritative under a single worker, reset on restart (see module docstring).
_hits: dict[str, list[float]] = defaultdict(list)
_daily: dict[str, int] = defaultdict(int)


def reset_state() -> None:
    """Clear all in-memory counters. Test helper; not used in production."""
    _hits.clear()
    _daily.clear()


def _client_ip(request: Request) -> str:
    """Return a best-effort client IP, honoring ``X-Forwarded-For``.

    Render terminates TLS and forwards the real client IP in the left-most
    ``X-Forwarded-For`` entry; without it (e.g. local runs) we fall back to the
    socket peer. The header is spoofable, so per-IP limiting is a speed bump, not
    a guarantee - the daily budget is the hard cap.

    Args:
        request: The incoming request.

    Returns:
        The resolved client IP, or ``"unknown"`` if none is available.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_retry_after(ip: str, settings: Settings) -> int:
    """Return seconds the caller must wait, or 0 if the request is allowed.

    Two windows apply: a short burst window catches rapid-fire abuse and a long
    window caps sustained use; whichever trips first wins. The hit is recorded
    only when the request is allowed, so a rejected caller's own retries do not
    push them further past the limit.

    Args:
        ip: The caller's resolved IP, used as the bucket key.
        settings: Active configuration supplying the limits and windows.

    Returns:
        Seconds until the caller may retry, or 0 if allowed now.
    """
    now = time.time()
    window = settings.api_rate_window
    hits = _hits[ip] = [t for t in _hits[ip] if now - t < window]
    if len(hits) >= settings.api_rate_limit:
        return math.ceil(window - (now - hits[0]))
    burst_window = settings.api_burst_window
    recent = [t for t in hits if now - t < burst_window]
    if len(recent) >= settings.api_burst_limit:
        return math.ceil(burst_window - (now - recent[0]))
    hits.append(now)
    return 0


def _budget_retry_after(settings: Settings) -> int:
    """Return seconds until the daily budget resets if it is spent, else 0.

    Args:
        settings: Active configuration supplying ``api_daily_budget``.

    Returns:
        Seconds until 00:00 UTC if today's budget is exhausted, otherwise 0.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    for day in [d for d in _daily if d != today]:
        del _daily[day]  # keep only today's counter
    if _daily[today] >= settings.api_daily_budget:
        now = datetime.now(UTC)
        reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return math.ceil((reset - now).total_seconds())
    return 0


def _spend_budget() -> None:
    """Record one cost-incurring call against today's budget."""
    _daily[datetime.now(UTC).strftime("%Y-%m-%d")] += 1


def require_api_key(key: str | None = Security(_API_KEY_HEADER)) -> None:
    """Reject the request (401) unless the shared secret matches.

    A no-op when ``RAG_API_KEY`` is unset, so local/dev runs need no key. Uses a
    constant-time comparison to avoid leaking the key through response timing.

    Args:
        key: The ``X-API-Key`` header value, if present.

    Raises:
        HTTPException: 401 when a key is configured but the header is missing or
            wrong.
    """
    expected = get_settings().rag_api_key
    if not expected:
        return
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def enforce_limits(request: Request) -> None:
    """Apply the pause switch, daily budget, then per-IP rate limit, in order.

    Runs after :func:`require_api_key` so unauthenticated callers never consume
    budget. On success, records one unit against today's budget and returns.

    Args:
        request: The incoming request, used to resolve the caller's IP.

    Raises:
        HTTPException: 503 when paused or the daily budget is spent; 429 when the
            caller is rate-limited. Both carry a ``Retry-After`` header where a
            wait time is known.
    """
    settings = get_settings()
    if settings.api_paused:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API is paused for maintenance. Please check back soon.",
        )
    wait = _budget_retry_after(settings)
    if wait:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API has reached its daily capacity. Please try again tomorrow.",
            headers={"Retry-After": str(wait)},
        )
    wait = _rate_retry_after(_client_ip(request), settings)
    if wait:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit reached. Please slow down.",
            headers={"Retry-After": str(wait)},
        )
    _spend_budget()  # an admitted request is the unit of cost we cap
