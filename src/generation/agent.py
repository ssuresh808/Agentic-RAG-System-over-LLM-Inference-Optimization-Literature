"""Agentic generation via a LangGraph ReAct agent (ADR-018, behind a flag).

Wraps the Qdrant retrieval engine as a ``search_arxiv_literature`` tool and lets
a ReAct agent plan + actively search to synthesize an answer. This path is
config-gated by ``GENERATION_MODE``; the single-shot :func:`answer_question`
remains the default, RAGAS-measured baseline.

``langgraph`` is imported lazily inside :func:`build_react_agent`, so this module
and its tests load without constructing an agent and make no network/GPU calls.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from src.config.settings import Settings, get_settings
from src.generation.rag import RagAnswer
from src.retrieval.engine import RetrievalEngine, build_engine

logger = logging.getLogger(__name__)

#: Directory of Markdown analysis "skills" the agent can load on demand.
SKILLS_DIR = (Path(__file__).resolve().parent.parent / "skills").resolve()

#: Stable default checkpointer thread id (gives the demo a single memory thread).
DEFAULT_THREAD_ID = "portfolio-default-thread"


@tool
def read_agent_skill(skill_name: str) -> str:
    """Load a named analysis skill (a Markdown playbook) to guide reasoning.

    Use for evaluative/comparison questions to load reusable analysis guidance
    before answering (e.g. ``"critical_analysis"``).

    Args:
        skill_name: Bare skill name, no path or extension (e.g. ``critical_analysis``).

    Returns:
        The skill's Markdown content, or a message listing available skills if
        the name is unknown. Reads are sandboxed to the skills directory.
    """
    target = (SKILLS_DIR / f"{Path(skill_name).stem}.md").resolve()
    if SKILLS_DIR not in target.parents or not target.is_file():
        available = sorted(path.stem for path in SKILLS_DIR.glob("*.md"))
        return f"Unknown skill '{skill_name}'. Available skills: {', '.join(available) or 'none'}."
    return target.read_text(encoding="utf-8")


AGENT_INSTRUCTIONS = (
    "You are an expert assistant on LLM inference optimization (vLLM, "
    "TensorRT-LLM, quantization, KV-cache, continuous batching, speculative "
    "decoding, GPU serving cost/latency).\n\n"
    "You have NO prior knowledge of this corpus. You MUST call the "
    "`search_arxiv_literature` tool at least once to gather evidence BEFORE you "
    "write any answer. Never answer from memory. If the question has multiple "
    "parts, search for each part. Base your synthesis ONLY on the passages the "
    "tool returns, and cite the source ids you used. If the tool returns nothing "
    "relevant, say so plainly rather than guessing.\n\n"
    "For evaluative or comparison questions, you may first call "
    "`read_agent_skill('critical_analysis')` to load an analysis playbook."
)


def _make_search_tool(engine: RetrievalEngine, top_k: int, collected: list[Document]) -> BaseTool:
    """Build a retrieval tool bound to ``engine`` that records what it returns.

    Args:
        engine: The retrieval engine to search.
        top_k: Passages to return per call.
        collected: List mutated in place with every retrieved document, so the
            caller can assemble citations after the agent finishes.

    Returns:
        A LangChain tool the agent can call with a ``query`` string.
    """

    @tool
    def search_arxiv_literature(query: str) -> str:
        """Search the arXiv LLM-inference-optimization corpus for passages.

        Args:
            query: A focused natural-language search query.

        Returns:
            Retrieved passages, each prefixed with its source id.
        """
        documents = engine.query(query, top_k=top_k)
        collected.extend(documents)
        if not documents:
            return "No relevant passages found."
        return "\n\n".join(
            f"[{doc.metadata.get('source', '?')}] {doc.page_content}" for doc in documents
        )

    return search_arxiv_literature


def build_react_agent(tools: list[BaseTool], settings: Settings | None = None) -> object:
    """Create a LangGraph ReAct agent driven by the configured local/hosted LLM.

    A lightweight tool-calling harness (chosen over deepagents for reliable
    tool use on local ~14B models - see ADR-018).

    Args:
        tools: Tools the agent may call (e.g. the retrieval tool).
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A compiled LangGraph agent, invoked with ``{"messages": [...]}`` and a
        ``configurable.thread_id``. A ``MemorySaver`` checkpointer makes the run
        checkpoint-capable (per-thread state); cross-turn memory requires a
        stable ``thread_id``, which the single-shot API does not yet expose.
    """
    settings = settings or get_settings()
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    from src.generation.llm import build_llm

    return create_react_agent(
        build_llm(settings),
        tools,
        prompt=AGENT_INSTRUCTIONS,
        checkpointer=MemorySaver(),
    )


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    """Drop repeated chunks (same source and content), preserving order.

    Args:
        documents: Retrieved chunks, possibly overlapping across searches.

    Returns:
        The unique chunks in first-seen order.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        key = (str(doc.metadata.get("source")), doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _cited_documents(answer: str, documents: list[Document]) -> list[Document]:
    """Keep only the chunks whose source id the answer actually cites.

    A source counts as cited when its full id (``arxiv:2601.12345``) or bare id
    (``2601.12345``) appears in the answer text. If no source matches (e.g. the
    model paraphrased without ids), all documents are kept so citations never
    silently vanish.

    Args:
        answer: The agent's final answer text.
        documents: Deduplicated retrieved chunks.

    Returns:
        The chunks backing the answer, or ``documents`` unchanged if the answer
        cites none of them.
    """
    cited: list[Document] = []
    for doc in documents:
        source = str(doc.metadata.get("source") or "")
        bare_id = source.split(":", 1)[-1]
        if source and (source in answer or (bare_id and bare_id in answer)):
            cited.append(doc)
    return cited or documents


def _cap_to_top_k(documents: list[Document], top_k: int) -> list[Document]:
    """Keep at most ``top_k`` chunks, taking one chunk per source before extras.

    A multi-search run collects up to ``searches x top_k`` chunks, so the
    citations must be re-bounded by the caller's ``top_k`` to honour the same
    contract the single-shot path meets. Ranking one chunk per distinct source
    first guarantees every source survives the cap whenever the answer cites at
    most ``top_k`` of them; remaining slots are filled with further chunks in
    retrieval order.

    Args:
        documents: Deduplicated, citation-filtered chunks in retrieval order.
        top_k: Maximum number of chunks (and therefore sources) to return.

    Returns:
        At most ``top_k`` chunks, distinct sources first.
    """
    first_per_source: list[Document] = []
    extras: list[Document] = []
    seen: set[str] = set()
    for doc in documents:
        source = str(doc.metadata.get("source"))
        if source in seen:
            extras.append(doc)
            continue
        seen.add(source)
        first_per_source.append(doc)
    return (first_per_source + extras)[:top_k]


def _extract_final_text(result: object) -> str:
    """Pull the final assistant message text out of an agent result state."""
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
    return content if isinstance(content, str) else str(content)


def agentic_answer_question(
    question: str,
    *,
    engine: RetrievalEngine | None = None,
    agent: object | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
    thread_id: str | None = None,
) -> RagAnswer:
    """Answer ``question`` with the ReAct agent, returning the baseline schema.

    The agent decides when/what to search; retrieved documents are captured,
    deduplicated, narrowed to the sources the final answer actually cites
    (falling back to everything retrieved if the answer cites none), and then
    capped at ``top_k`` chunks, so a multi-search run never returns more
    citations than the caller asked for. The output matches :class:`RagAnswer`
    exactly (answer + sources + chunks), keeping the API response schema
    identical across modes.

    Args:
        question: The user's question.
        engine: Retrieval engine. Defaults to ``build_engine()``.
        agent: Pre-built agent (injected in tests). Defaults to a fresh agent.
        top_k: Passages per tool call, and the ceiling on returned chunks and
            sources regardless of how many searches the agent runs.
        settings: Configuration. Defaults to ``get_settings()``.
        thread_id: Checkpointer thread id. Defaults to ``DEFAULT_THREAD_ID``;
            pass a per-conversation id to isolate agent memory across callers.

    Returns:
        A ``RagAnswer`` with the agent's synthesis, cited sources, and chunks.
    """
    settings = settings or get_settings()
    engine = engine or build_engine()
    collected: list[Document] = []
    search_tool = _make_search_tool(engine, top_k, collected)
    if agent is None:
        agent = build_react_agent([search_tool, read_agent_skill], settings)

    config = {"configurable": {"thread_id": thread_id or DEFAULT_THREAD_ID}}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    answer_text = _extract_final_text(result)
    chunks = _cap_to_top_k(_cited_documents(answer_text, _dedupe_documents(collected)), top_k)
    sources = list(
        dict.fromkeys(doc.metadata.get("source") for doc in chunks if doc.metadata.get("source"))
    )
    return RagAnswer(answer=answer_text, sources=sources, chunks=chunks)
