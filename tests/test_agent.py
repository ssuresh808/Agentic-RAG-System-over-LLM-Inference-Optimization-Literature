"""Tests for the agentic generation path (offline; langgraph + LLM not loaded)."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from src.generation.agent import (
    DEFAULT_THREAD_ID,
    _make_search_tool,
    agentic_answer_question,
    read_agent_skill,
)
from src.retrieval.engine import RetrievalEngine


def _engine() -> RetrievalEngine:
    engine = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="agent_test")
    engine.index(
        [
            Document(
                page_content="PagedAttention manages the KV cache in vLLM.",
                metadata={"source": "paged.md"},
            ),
            Document(
                page_content="Speculative decoding reduces decode latency.",
                metadata={"source": "spec.md"},
            ),
        ]
    )
    return engine


def test_search_tool_returns_passages_and_records_sources():
    collected: list[Document] = []
    search = _make_search_tool(_engine(), top_k=2, collected=collected)

    out = search.invoke({"query": "paged attention kv cache"})

    assert isinstance(out, str) and out
    assert len(collected) == 2  # tool records what it retrieved for citations


class _FakeAgent:
    """Stand-in deep agent that returns a fixed final message (no LLM/tool)."""

    def __init__(self, messages):
        self.messages = messages

    def invoke(self, state, config=None):
        self._state = state
        self._config = config
        return {"messages": self.messages}


def test_agentic_answer_extracts_final_dict_message():
    agent = _FakeAgent(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Agentic synthesis."}]
    )
    out = agentic_answer_question(
        "How does paged attention work?", engine=_engine(), agent=agent, top_k=2
    )
    assert out.answer == "Agentic synthesis."


def test_agentic_answer_extracts_final_aimessage_object():
    agent = _FakeAgent([AIMessage(content="From an AIMessage.")])
    out = agentic_answer_question("q", engine=_engine(), agent=agent, top_k=1)
    assert out.answer == "From an AIMessage."


def test_agentic_answer_passes_question_to_agent():
    agent = _FakeAgent([AIMessage(content="ok")])
    agentic_answer_question("explain KV cache", engine=_engine(), agent=agent, top_k=1)
    assert agent._state["messages"][0]["content"] == "explain KV cache"


def test_agentic_answer_uses_default_thread_id():
    agent = _FakeAgent([AIMessage(content="ok")])
    agentic_answer_question("q", engine=_engine(), agent=agent, top_k=1)
    assert agent._config["configurable"]["thread_id"] == DEFAULT_THREAD_ID


class _SearchingFakeAgent:
    """Fake agent that actually calls the search tool, then answers.

    Mirrors a real ReAct run: each ``queries`` entry triggers one search-tool
    call (populating the collected-documents list), then the fixed ``answer``
    is returned as the final message.
    """

    def __init__(self, tools, queries, answer):
        self._search = tools[0]
        self._queries = queries
        self._answer = answer

    def invoke(self, state, config=None):
        for query in self._queries:
            self._search.invoke({"query": query})
        return {"messages": [AIMessage(content=self._answer)]}


def _patch_agent(monkeypatch, queries, answer):
    monkeypatch.setattr(
        "src.generation.agent.build_react_agent",
        lambda tools, settings: _SearchingFakeAgent(tools, queries, answer),
    )


def test_agentic_citations_only_include_sources_cited_in_answer(monkeypatch):
    # Two searches retrieve both corpus docs, but the answer only uses one.
    _patch_agent(
        monkeypatch,
        queries=["paged attention", "kv cache management"],
        answer="PagedAttention blocks the KV cache [paged.md].",
    )
    out = agentic_answer_question("How does vLLM manage KV cache?", engine=_engine(), top_k=2)

    assert out.sources == ["paged.md"]
    assert {doc.metadata["source"] for doc in out.chunks} == {"paged.md"}


def test_agentic_citations_dedupe_repeated_chunks(monkeypatch):
    # The same search twice must not duplicate chunks in the response.
    _patch_agent(
        monkeypatch,
        queries=["paged attention", "paged attention"],
        answer="Cites both [paged.md] and [spec.md].",
    )
    out = agentic_answer_question("q", engine=_engine(), top_k=2)

    keys = [(doc.metadata["source"], doc.page_content) for doc in out.chunks]
    assert len(keys) == len(set(keys))
    assert sorted(out.sources) == ["paged.md", "spec.md"]


def test_agentic_citations_fall_back_to_all_retrieved_when_none_cited(monkeypatch):
    # An answer with no recognizable source ids keeps all (deduped) retrievals.
    _patch_agent(
        monkeypatch,
        queries=["paged attention"],
        answer="The corpus does not cover this topic.",
    )
    out = agentic_answer_question("q", engine=_engine(), top_k=2)

    assert sorted(out.sources) == ["paged.md", "spec.md"]
    assert len(out.chunks) == 2


def test_agentic_citations_match_bare_arxiv_ids(monkeypatch):
    # Models often cite "2601.12345" without the "arxiv:" prefix.
    engine = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="agent_arxiv")
    engine.index(
        [
            Document(
                page_content="Continuous batching raises GPU utilization.",
                metadata={"source": "arxiv:2601.12345"},
            ),
            Document(
                page_content="Quantization shrinks weight memory.",
                metadata={"source": "arxiv:2602.67890"},
            ),
        ]
    )
    _patch_agent(
        monkeypatch,
        queries=["continuous batching", "quantization"],
        answer="Continuous batching keeps the GPU saturated (2601.12345).",
    )
    out = agentic_answer_question("q", engine=engine, top_k=2)

    assert out.sources == ["arxiv:2601.12345"]


def test_read_agent_skill_loads_known_skill():
    out = read_agent_skill.invoke({"skill_name": "critical_analysis"})
    assert "critical analysis" in out.lower()
    assert len(out) > 100


def test_read_agent_skill_rejects_unknown_and_path_traversal():
    assert "Unknown skill" in read_agent_skill.invoke({"skill_name": "does_not_exist"})
    # path traversal is neutralized (stem-only + sandboxed to the skills dir)
    assert "Unknown skill" in read_agent_skill.invoke({"skill_name": "../../etc/passwd"})
