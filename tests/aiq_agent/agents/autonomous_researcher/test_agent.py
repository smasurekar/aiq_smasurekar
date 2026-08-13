# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent lifecycle: extraction order, the two finalize exits, and the termination fallbacks.

Everything asserted here is the invariant part of the change — the API and report contract the
eval harnesses and the UI depend on. It is deliberately identical in shape to the adaptive
agent's contract; only the routing above it differs.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from aiq_agent.agents.autonomous_researcher.agent import AutonomousResearcherAgent
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchAgentState
from aiq_agent.agents.autonomous_researcher.tools.finalize import FINAL_REPORT_META_PATH
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import SourceEntry

GRAPH_TARGET = "aiq_agent.agents.autonomous_researcher.factory.create_deep_agent"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


def output_markdown_file(markdown: str) -> dict:
    """The writer exit."""
    return {"/shared/output.md": {"content": markdown, "encoding": "utf-8"}}


def final_report_files(markdown: str = "# Answer\n\nBody [1].", researched: bool = True) -> dict:
    """The inline exit."""
    return {
        "/shared/final_report.md": {"content": markdown, "encoding": "utf-8"},
        FINAL_REPORT_META_PATH: {"content": json.dumps({"researched": researched}), "encoding": "utf-8"},
    }


@pytest.fixture(autouse=True)
def mock_research_summarization_middleware():
    """The researcher runnable is built via deep_researcher.factory; avoid a concrete model."""

    class FakeSummarizationMiddleware(AgentMiddleware):
        pass

    researcher_runnable = MagicMock(name="researcher_runnable")
    researcher_runnable.ainvoke = AsyncMock()
    with (
        patch(
            "aiq_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=FakeSummarizationMiddleware(),
        ),
        patch("aiq_agent.agents.deep_researcher.factory.create_agent", return_value=researcher_runnable),
    ):
        yield


@pytest.fixture
def mock_llm_provider():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    provider = LLMProvider()
    provider.set_default(llm)
    for role in (LLMRole.ORCHESTRATOR, LLMRole.PLANNER, LLMRole.RESEARCHER, LLMRole.REPORT_WRITER):
        provider.configure(role, llm)
    return provider


@pytest.fixture
def agent(mock_llm_provider):
    return AutonomousResearcherAgent(llm_provider=mock_llm_provider, tools=[web_search_tool])


def _mock_graph(result: dict):
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    graph.ainvoke = AsyncMock(return_value=result)
    return graph


def _raising_graph(exc: BaseException):
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    graph.ainvoke = AsyncMock(side_effect=exc)
    return graph


class TestPromptLoading:
    def test_loads_exactly_four_prompts(self, agent):
        assert set(agent._prompts) == {"planner", "researcher", "orchestrator", "writer"}

    def test_no_source_router_prompt(self, agent):
        assert "source_router" not in agent._prompts


class TestExtractFinalMarkdown:
    """Extraction order is ported verbatim: writer exit, then inline exit, then salvage."""

    def test_prefers_output_md_over_final_report(self, agent):
        result = {
            "messages": [AIMessage(content="Wrote /shared/output.md")],
            "files": {**output_markdown_file("# Writer"), **final_report_files("# Inline")},
        }
        assert agent._extract_final_markdown(result) == "# Writer"

    def test_uses_final_report_when_no_output_md(self, agent):
        result = {"messages": [AIMessage(content="done")], "files": final_report_files("# Inline")}
        assert agent._extract_final_markdown(result) == "# Inline"

    def test_salvage_is_the_last_resort(self, agent):
        result = {"messages": [AIMessage(content="Hello! I can research things for you.")], "files": {}}
        assert agent._extract_final_markdown(result) == "Hello! I can research things for you."

    def test_salvage_rejects_only_empty_and_the_writer_marker(self, agent):
        for content in ("", "   ", "Wrote /shared/output.md"):
            assert agent._salvage_inline_report({"messages": [AIMessage(content=content)], "files": {}}) is None


class TestResearchedFlag:
    def test_false_round_trips(self):
        result = {"files": final_report_files(researched=False)}
        assert AutonomousResearcherAgent._read_researched_flag(result) is False

    def test_true_round_trips(self):
        result = {"files": final_report_files(researched=True)}
        assert AutonomousResearcherAgent._read_researched_flag(result) is True

    def test_defaults_true_when_absent(self):
        assert AutonomousResearcherAgent._read_researched_flag({"files": {}}) is True

    def test_defaults_true_on_bad_json(self):
        result = {"files": {FINAL_REPORT_META_PATH: {"content": "{not json"}}}
        assert AutonomousResearcherAgent._read_researched_flag(result) is True


class TestBothExitsReachTheSameBoundary:
    """Neither exit may bypass citation verification or sanitization."""

    @pytest.mark.asyncio
    async def test_inline_exit_verifies_citations(self, agent):
        result = {
            "messages": [AIMessage(content="marker")],
            "files": final_report_files(
                "# Answer\n\nFact [1].\n\n## Sources\n[1] Example: https://example.com", researched=True
            ),
        }
        with patch(GRAPH_TARGET, return_value=_mock_graph(result)):
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="what is X?")]))
        assert "Fact [1]" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_writer_exit_verifies_citations(self, agent):
        result = {
            "messages": [AIMessage(content="Wrote /shared/output.md")],
            "files": output_markdown_file("# Report\n\nFact [1].\n\n## Sources\n[1] Example: https://example.com"),
        }
        with patch(GRAPH_TARGET, return_value=_mock_graph(result)):
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="report on X")]))
        assert "Fact [1]" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_writer_exit_strips_an_unverifiable_citation(self, agent):
        result = {
            "messages": [AIMessage(content="Wrote /shared/output.md")],
            "files": output_markdown_file(
                "# Report\n\nFact [1].\n\n## Sources\n[1] Fabricated: https://not-in-registry.example"
            ),
        }
        with patch(GRAPH_TARGET, return_value=_mock_graph(result)):
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="report on X")]))
        assert "not-in-registry.example" not in out.messages[-1].content


class TestNoResearchSafeguard:
    @pytest.mark.asyncio
    async def test_researched_false_skips_the_empty_registry_raise(self, agent):
        result = {
            "messages": [AIMessage(content="marker")],
            "files": final_report_files("# Answer\n\nHello, I'm the AI-Q assistant.", researched=False),
        }
        with patch(GRAPH_TARGET, return_value=_mock_graph(result)):
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="who are you?")]))
        assert "AI-Q assistant" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_researched_true_with_empty_registry_still_raises(self, agent):
        """An honest 'could not verify' keeps research-failure semantics rather than downgrading."""
        result = {
            "messages": [AIMessage(content="marker")],
            "files": final_report_files("# Answer\n\nResearched body [1].", researched=True),
        }
        with patch(GRAPH_TARGET, return_value=_mock_graph(result)):
            with pytest.raises(EmptySourceRegistryError):
                await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="latest news on X")]))

    @pytest.mark.asyncio
    async def test_salvaged_greeting_with_empty_registry_returns(self, agent):
        greeting = "Hello! I'm the AI-Q research assistant. Ask me a question and I'll research it."
        with patch(GRAPH_TARGET, return_value=_mock_graph({"messages": [AIMessage(content=greeting)], "files": {}})):
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="Hi")]))
        assert "AI-Q research assistant" in out.messages[-1].content


class TestForcedTermination:
    """A deadline or recursion abort must yield a citation-safe partial, never an opaque 500."""

    @pytest.mark.asyncio
    async def test_timeout_returns_a_bounded_failure_when_no_evidence(self, agent):
        with patch(GRAPH_TARGET, return_value=_raising_graph(TimeoutError())):
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="q")]))
        assert "Research could not be completed" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_recursion_limit_returns_a_partial_with_gathered_sources(self, agent):
        agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
        with patch(GRAPH_TARGET, return_value=_raising_graph(GraphRecursionError())):
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content="q")]))
        content = out.messages[-1].content
        assert "Partial research result" in content
        assert "https://example.com" in content

    @pytest.mark.asyncio
    async def test_task_delegated_notes_feed_the_partial(self, agent):
        """The partial harvests /shared/research_note_* from BOTH research paths."""
        state = AutonomousResearchAgentState(
            messages=[HumanMessage(content="q")],
            files={
                "/shared/research_note_task_01_topic_abc.json": {
                    "content": json.dumps({"summary": "delegated finding", "gaps": []}),
                    "encoding": "utf-8",
                }
            },
        )
        with patch(GRAPH_TARGET, return_value=_raising_graph(TimeoutError())):
            out = await agent.run(state)
        assert "delegated finding" in out.messages[-1].content
