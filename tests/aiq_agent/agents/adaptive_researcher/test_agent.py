# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for AdaptiveResearcherAgent report extraction and the no-research safeguard."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.adaptive_researcher.agent import _WRITER_COMPLETION_MARKER
from aiq_agent.agents.adaptive_researcher.agent import AdaptiveResearcherAgent
from aiq_agent.agents.adaptive_researcher.models import AdaptiveResearchAgentState
from aiq_agent.agents.adaptive_researcher.tools.finalize import FINAL_REPORT_META_PATH
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import SourceEntry


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


def output_markdown_file(markdown: str) -> dict:
    return {"/shared/output.md": {"content": markdown, "encoding": "utf-8"}}


def final_report_files(markdown: str = "# Answer\n\nBody [1].", researched: bool = True) -> dict:
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
        patch(
            "aiq_agent.agents.deep_researcher.factory.create_agent",
            return_value=researcher_runnable,
        ),
    ):
        yield


@pytest.fixture
def mock_llm_provider():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    provider = LLMProvider()
    provider.set_default(llm)
    for role in (
        LLMRole.ORCHESTRATOR,
        LLMRole.ROUTER,
        LLMRole.PLANNER,
        LLMRole.RESEARCHER,
        LLMRole.REPORT_WRITER,
    ):
        provider.configure(role, llm)
    return provider


@pytest.fixture
def agent(mock_llm_provider):
    """A constructed agent (no graph build needed for extraction-level tests)."""
    return AdaptiveResearcherAgent(llm_provider=mock_llm_provider, tools=[web_search_tool])


def _mock_graph(result: dict):
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    graph.ainvoke = AsyncMock(return_value=result)
    return graph


class TestExtractFinalMarkdown:
    def test_prefers_output_md_over_final_report(self, agent):
        result = {
            "messages": [AIMessage(content="marker")],
            "files": {**output_markdown_file("OUTPUT MD"), **final_report_files("FINAL MD")},
        }
        assert agent._extract_final_markdown(result, {}) == "OUTPUT MD"

    def test_uses_final_report_when_no_output_md(self, agent):
        result = {"messages": [AIMessage(content="marker")], "files": final_report_files("# Final\n\nInline body.")}
        assert agent._extract_final_markdown(result, {}) == "# Final\n\nInline body."

    def test_salvage_is_last_resort_for_substantive_inline(self, agent):
        long_md = "# Heading\n\n" + ("word " * 120)
        result = {"messages": [AIMessage(content=long_md)], "files": {}}
        assert agent._extract_final_markdown(result, {}) == long_md.strip()

    def test_salvage_accepts_headingless_inline(self, agent):
        """A short, heading-less conversational reply (e.g. a greeting) is now salvaged."""
        result = {"messages": [AIMessage(content="Hello! I'm the AI-Q assistant.")], "files": {}}
        assert agent._extract_final_markdown(result, {}) == "Hello! I'm the AI-Q assistant."

    def test_salvage_rejects_only_empty_and_writer_marker(self, agent):
        """Salvage rejects an empty message and the writer completion marker; nothing else."""
        empty = {"messages": [AIMessage(content="   ")], "files": {}}
        assert agent._extract_final_markdown(empty, {}) is None
        marker = {"messages": [AIMessage(content=_WRITER_COMPLETION_MARKER)], "files": {}}
        assert agent._extract_final_markdown(marker, {}) is None


class TestReadResearchedFlag:
    def test_false_round_trips(self):
        assert AdaptiveResearcherAgent._read_researched_flag({"files": final_report_files(researched=False)}) is False

    def test_true_round_trips(self):
        assert AdaptiveResearcherAgent._read_researched_flag({"files": final_report_files(researched=True)}) is True

    def test_defaults_true_when_absent(self):
        assert AdaptiveResearcherAgent._read_researched_flag({"files": {}}) is True
        assert AdaptiveResearcherAgent._read_researched_flag({"messages": []}) is True

    def test_defaults_true_on_bad_json(self):
        result = {"files": {FINAL_REPORT_META_PATH: {"content": "not-json", "encoding": "utf-8"}}}
        assert AdaptiveResearcherAgent._read_researched_flag(result) is True


class TestNoResearchSafeguard:
    @pytest.mark.asyncio
    async def test_direct_answer_skips_empty_registry_raise(self, agent):
        """researched=False (a deliberate no-research answer) must NOT raise on an empty registry."""
        result = {
            "messages": [AIMessage(content="marker")],
            "files": final_report_files("# Answer\n\nHello, I'm the AI-Q assistant.", researched=False),
        }
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_mock_graph(result),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="hi, who are you?")])
            # registry intentionally left empty
            out = await agent.run(state)
        assert out is not None
        assert "AI-Q assistant" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_researched_but_empty_registry_still_raises(self, agent):
        """researched=True with an empty registry is a real failure and must still raise."""
        result = {
            "messages": [AIMessage(content="marker")],
            "files": final_report_files("# Answer\n\nResearched body [1].", researched=True),
        }
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_mock_graph(result),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="latest news on X")])
            with pytest.raises(EmptySourceRegistryError):
                await agent.run(state)

    @pytest.mark.asyncio
    async def test_inline_report_with_sources_verifies_and_returns(self, agent):
        """A researched inline report with a populated registry flows through verification."""
        result = {
            "messages": [AIMessage(content="marker")],
            "files": final_report_files(
                "# Answer\n\nFact [1].\n\n## Sources\n[1] Example: https://example.com", researched=True
            ),
        }
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_mock_graph(result),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="what is X?")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
            out = await agent.run(state)
        assert out is not None
        assert "Fact [1]" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_salvaged_greeting_empty_registry_returns(self, agent):
        """A greeting answered inline (no submit_final_report, no output file, empty registry)
        must be salvaged and returned, not raise ValueError or EmptySourceRegistryError."""
        greeting = "Hello! I'm the AI-Q research assistant. Ask me a question and I'll research it."
        result = {"messages": [AIMessage(content=greeting)], "files": {}}
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_mock_graph(result),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="Hi")])
            # registry intentionally left empty; no submit_final_report was called
            out = await agent.run(state)
        assert out is not None
        assert "AI-Q research assistant" in out.messages[-1].content
