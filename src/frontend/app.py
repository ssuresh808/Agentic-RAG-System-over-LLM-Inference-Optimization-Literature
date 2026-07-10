"""Streamlit frontend for the RAG API: a clean, technical retrieval tool.

Connects to the FastAPI ``/api/v1/answer`` endpoint and renders the grounded
answer (native markdown) in the main canvas, with source citations and the
retrieval setting in the sidebar, so provenance is the visible proof of work.
Styling lives in ``style.css``. Run::

    streamlit run src/frontend/app.py

Set ``RAG_API_URL`` to point at the backend (default ``http://localhost:8000``;
``http://backend:8000`` under docker-compose).
"""

import html
import os
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
# Shared secret for the backend's X-API-Key guard. Held server-side here (never
# sent to the browser), so the public demo works while direct curl is blocked.
# Empty = the backend has auth disabled (local/dev); no header is sent.
API_KEY = os.environ.get("RAG_API_KEY", "")
REQUEST_TIMEOUT = 180.0
ANALYSIS_HEIGHT = 800  # fixed height (px) of the scrollable analysis container
_CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

_SUBTITLE = (
    "Grounded answers over a curated arXiv corpus on vLLM, KV-cache, "
    "quantization, speculative decoding, and continuous batching. Every claim "
    "links back to the paper it came from."
)


def _arxiv_url(source: str | None) -> str:
    """Build an arXiv abstract URL from an ``arxiv:<id>`` source string.

    Args:
        source: A source id such as ``"arxiv:2605.22850"``.

    Returns:
        The ``https://arxiv.org/abs/<id>`` URL (best-effort if unprefixed).
    """
    if not source:
        return "https://arxiv.org"
    arxiv_id = source.split(":", 1)[1] if source.startswith("arxiv:") else source
    return f"https://arxiv.org/abs/{arxiv_id}"


def _dedupe_sources(results: list[dict]) -> list[dict]:
    """Drop duplicate sources (multiple chunks per paper), preserving order."""
    seen: set[str] = set()
    unique: list[dict] = []
    for record in results:
        source = record.get("source") or ""
        if source in seen:
            continue
        seen.add(source)
        unique.append(record)
    return unique


def _request_answer(question: str, top_k: int) -> dict:
    """POST a question to the RAG API and return the parsed JSON response.

    Args:
        question: The user's question.
        top_k: Number of context passages to retrieve.

    Returns:
        The decoded ``/api/v1/answer`` response.

    Raises:
        httpx.HTTPError: On connection failure or a non-2xx response.
    """
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    response = httpx.post(
        f"{API_URL}/api/v1/answer",
        json={"text": question, "top_k": top_k},
        headers=headers,
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
    )
    response.raise_for_status()
    return response.json()


def _render_header() -> None:
    """Render the page header (single eyebrow, title, subtitle)."""
    st.markdown(
        '<div class="eyebrow">arXiv &middot; LLM inference optimization</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="app-title">Ask the inference-optimization literature.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="app-sub">{_SUBTITLE}</p>', unsafe_allow_html=True)


def _render_settings() -> int:
    """Render the retrieval setting in the sidebar; return the clamped top_k."""
    with st.sidebar:
        st.markdown('<div class="side-label">Retrieval</div>', unsafe_allow_html=True)
        st.number_input(
            "Sources to retrieve",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="k_sources",
        )
    return max(1, min(int(st.session_state.get("k_sources", 5)), 20))


def _render_search() -> tuple[bool, str]:
    """Render the main query input and Search button; return (submitted, text)."""
    question = st.text_input(
        "Question",
        placeholder="How does continuous batching improve GPU throughput?",
        key="question",
        label_visibility="collapsed",
    )
    submitted = st.button("Search")
    return submitted, question


def _render_citations(results: list[dict]) -> None:
    """Render retrieved sources as dense, rule-separated rows in the sidebar."""
    st.markdown('<div class="side-label">Citations</div>', unsafe_allow_html=True)
    sources = _dedupe_sources(results)
    if not sources:
        st.markdown('<p class="empty">No sources retrieved.</p>', unsafe_allow_html=True)
        return
    rows = [
        f'<a class="src-row" href="{_arxiv_url(record.get("source"))}" '
        f'target="_blank" rel="noopener">'
        f'<span class="src-rank">{index:02d}</span>'
        f'<span class="src-main">'
        f'<span class="src-title">{html.escape(record.get("title") or "Untitled")}</span>'
        f'<span class="src-id">{html.escape(record.get("source") or "n/a")}</span>'
        f"</span></a>"
        for index, record in enumerate(sources, start=1)
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_analysis(answer: str) -> None:
    """Render the synthesized answer as native markdown in a scroll container."""
    st.markdown('<div class="col-label">Synthesis</div>', unsafe_allow_html=True)
    with st.container(height=ANALYSIS_HEIGHT, border=True):
        st.markdown(answer or "_No answer was generated._")


def _render_output() -> None:
    """Render the current result, error, or the initial invitation."""
    error = st.session_state.get("error")
    result = st.session_state.get("result")

    if error:
        st.markdown(
            f'<div class="notice">Can\'t reach the backend at {html.escape(API_URL)}. '
            f"Start it, then search again.<br>{html.escape(error)}</div>",
            unsafe_allow_html=True,
        )
        return
    if not result:
        st.markdown(
            '<p class="empty">Ask a question to query the corpus.</p>',
            unsafe_allow_html=True,
        )
        return

    with st.sidebar:
        _render_citations(result.get("results", []))
    _render_analysis(result.get("answer", ""))


def main() -> None:
    """Compose and render the single-page Streamlit application."""
    st.set_page_config(
        page_title="Inference Atlas",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

    top_k = _render_settings()
    _render_header()
    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)
    submitted, question = _render_search()

    if submitted and question.strip():
        with st.spinner("Retrieving and synthesizing"):
            try:
                st.session_state["result"] = _request_answer(question.strip(), top_k)
                st.session_state["error"] = None
            except httpx.HTTPError as exc:
                st.session_state["result"] = None
                st.session_state["error"] = str(exc)

    _render_output()


main()
