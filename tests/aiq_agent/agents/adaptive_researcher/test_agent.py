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

import asyncio
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

from aiq_agent.agents.adaptive_researcher.agent import _WRITER_COMPLETION_MARKER
from aiq_agent.agents.adaptive_researcher.agent import AdaptiveResearcherAgent
from aiq_agent.agents.adaptive_researcher.models import AdaptiveRequestTerminationConfig
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


def _raising_graph(exc: BaseException):
    """A mock graph whose ainvoke raises the given exception."""
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    graph.ainvoke = AsyncMock(side_effect=exc)
    return graph


def research_note_file(path: str, summary: str, gap_descriptions: list[str]) -> dict:
    gaps = [{"description": d, "impact": "matters", "suggested_follow_up_queries": []} for d in gap_descriptions]
    payload = {"summary": summary, "gaps": gaps}
    return {path: {"content": json.dumps(payload), "encoding": "utf-8"}}


class TestPersistedNotesAndGaps:
    def test_parses_summaries_and_gaps(self):
        files = {
            **research_note_file(
                "/shared/research_note_01_apple_ab12cd34.json", "Apple 2022/2023 covered", ["FY2024 10-K missing"]
            ),
            "/shared/plan.json": {"content": "{}", "encoding": "utf-8"},
        }
        summaries, gaps = AdaptiveResearcherAgent._persisted_notes_and_gaps(files)
        assert summaries == ["Apple 2022/2023 covered"]
        assert gaps == ["FY2024 10-K missing"]

    def test_ignores_non_note_files_and_bad_json(self):
        files = {
            "/shared/research_note_02_x.json": {"content": "not-json", "encoding": "utf-8"},
            "/shared/output.md": {"content": "# hi", "encoding": "utf-8"},
        }
        assert AdaptiveResearcherAgent._persisted_notes_and_gaps(files) == ([], [])

    def test_empty_or_non_dict(self):
        assert AdaptiveResearcherAgent._persisted_notes_and_gaps({}) == ([], [])
        assert AdaptiveResearcherAgent._persisted_notes_and_gaps(None) == ([], [])


class TestDeterministicPartial:
    def test_bounded_failure_when_no_evidence(self, agent):
        state = AdaptiveResearchAgentState(messages=[HumanMessage(content="q")])
        md = agent._render_deterministic_partial(state, "the time limit was reached")
        assert "could not be completed" in md.lower()
        assert "the time limit was reached" in md

    def test_partial_lists_gaps_and_sources(self, agent):
        agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Apple 10-Q 2023"))
        files = research_note_file(
            "/shared/research_note_01_x.json", "Found 2023 data", ["FY2024 unavailable", "FY2025 unavailable"]
        )
        state = AdaptiveResearchAgentState(messages=[HumanMessage(content="q")], files=files)
        md = agent._render_deterministic_partial(state, "the research budget was reached")
        assert "# Partial research result" in md
        assert "Found 2023 data" in md
        assert "## Evidence gaps" in md
        assert "FY2024 unavailable" in md
        assert "FY2025 unavailable" in md
        assert "## Sources" in md
        assert "Apple 10-Q 2023" in md


class TestForcedTerminationPaths:
    @pytest.mark.asyncio
    async def test_timeout_returns_bounded_failure_when_no_evidence(self, agent):
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_raising_graph(TimeoutError()),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="apple fy2024 inventory")])
            out = await agent.run(state)
        assert out is not None
        assert "could not be completed" in out.messages[-1].content.lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_with_gathered_sources(self, agent):
        agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Apple 2023 10-K"))
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_raising_graph(TimeoutError()),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="apple fy2024 inventory")])
            out = await agent.run(state)
        content = out.messages[-1].content
        assert "Partial research result" in content
        assert "Apple 2023 10-K" in content

    @pytest.mark.asyncio
    async def test_recursion_limit_routes_to_partial(self, agent):
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_raising_graph(GraphRecursionError("recursion limit reached")),
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="q")])
            out = await agent.run(state)
        assert out is not None
        assert out.messages[-1].content  # a terminal answer, not a raised error

    @pytest.mark.asyncio
    async def test_reuses_completed_output_on_termination(self, agent):
        agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
        completed = "# Answer\n\nComplete body [1].\n\n## Sources\n[1] Example: https://example.com"
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=_raising_graph(TimeoutError()),
        ):
            state = AdaptiveResearchAgentState(
                messages=[HumanMessage(content="q")],
                files=output_markdown_file(completed),
            )
            out = await agent.run(state)
        # The already-completed report is reused rather than a partial being synthesized.
        assert "Complete body [1]" in out.messages[-1].content
        assert "Partial research result" not in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_real_deadline_cancels_hung_graph(self, mock_llm_provider):
        """asyncio.timeout actually fires on a graph whose ainvoke never returns."""
        agent = AdaptiveResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            request_termination=AdaptiveRequestTerminationConfig(
                workflow_timeout_seconds=2,
                fallback_finalizer_timeout_seconds=1,
            ),
        )

        async def _never_returns(*_args, **_kwargs):
            await asyncio.sleep(30)

        graph = MagicMock()
        graph.with_config = MagicMock(return_value=graph)
        graph.ainvoke = _never_returns
        with patch(
            "aiq_agent.agents.adaptive_researcher.factory.create_deep_agent",
            return_value=graph,
        ):
            state = AdaptiveResearchAgentState(messages=[HumanMessage(content="q")])
            out = await asyncio.wait_for(agent.run(state), timeout=10)
        assert out is not None
        assert "could not be completed" in out.messages[-1].content.lower()


# ---------------------------------------------------------------------------
# single_shot shallow sub-agent: capture recovery and cancellation ownership
# ---------------------------------------------------------------------------

SHALLOW_RECOVERED_REPORT = "# Shallow answer\n\nThe shallow researcher finished before the deadline."


def _capture(status="completed", *, tier="single_shot", markdown=SHALLOW_RECOVERED_REPORT):
    from aiq_agent.agents.adaptive_researcher.subagents import ShallowSubagentCapture

    capture = ShallowSubagentCapture()
    capture.status = status
    capture.declared_tier = tier
    capture.markdown = markdown if status == "completed" else None
    capture.attempts = 1
    return capture


def _bundle(capture, *, raises=None, result=None):
    """An AdaptiveResearchGraphRun whose runnable raises (or returns) on ainvoke."""
    from aiq_agent.agents.adaptive_researcher.factory import AdaptiveResearchGraphRun

    runnable = MagicMock()
    if raises is not None:
        runnable.ainvoke = AsyncMock(side_effect=raises)
    else:
        runnable.ainvoke = AsyncMock(return_value=result)
    return AdaptiveResearchGraphRun(runnable=runnable, shallow_capture=capture)


class TestShallowCaptureRecovery:
    """After a forced exit the graph returns no state, so the run-scoped capture is the only seam."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [TimeoutError(), GraphRecursionError("recursion limit reached")])
    async def test_completed_single_shot_capture_is_recovered(self, agent, error):
        capture = _capture()
        with patch.object(agent, "_build_orchestrator_agent", return_value=_bundle(capture, raises=error)):
            out = await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        assert "The shallow researcher finished before the deadline." in out.messages[-1].content
        assert "Partial research result" not in out.messages[-1].content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "capture",
        [
            _capture(status="running"),
            _capture(status="failed"),
            _capture(status="completed", markdown=""),
            _capture(status="completed", tier="deep"),
        ],
        ids=["running", "failed", "empty", "escalated"],
    )
    async def test_unusable_captures_fall_back_to_the_deterministic_partial(self, agent, capture):
        with patch.object(agent, "_build_orchestrator_agent", return_value=_bundle(capture, raises=TimeoutError())):
            out = await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        content = out.messages[-1].content
        assert "The shallow researcher finished before the deadline." not in content
        assert "could not be completed" in content.lower()

    @pytest.mark.asyncio
    async def test_recovery_is_inert_when_the_subagent_is_disabled(self, agent):
        with patch.object(agent, "_build_orchestrator_agent", return_value=_bundle(None, raises=TimeoutError())):
            out = await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        assert "could not be completed" in out.messages[-1].content.lower()


class TestShallowCaptureCancellation:
    """`asyncio.create_task` detaches the shallow run, so `run()` must always cancel it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [TimeoutError(), GraphRecursionError("recursion limit reached")],
        ids=["timeout", "recursion"],
    )
    async def test_cancelled_on_forced_exit_paths(self, agent, error):
        capture = _capture(status="running")
        with (
            patch.object(agent, "_build_orchestrator_agent", return_value=_bundle(capture, raises=error)),
            patch.object(capture, "cancel") as cancel,
        ):
            await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancelled_on_normal_completion(self, agent):
        capture = _capture()
        agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
        result = {
            "messages": [AIMessage(content="done")],
            "files": {"/shared/final_report.md": {"content": "# Answer\n\nBody [1].", "encoding": "utf-8"}},
        }
        with (
            patch.object(agent, "_build_orchestrator_agent", return_value=_bundle(capture, result=result)),
            patch.object(capture, "cancel") as cancel,
        ):
            await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancelled_when_the_graph_raises_an_unexpected_error(self, agent):
        capture = _capture(status="running")
        with (
            patch.object(
                agent, "_build_orchestrator_agent", return_value=_bundle(capture, raises=RuntimeError("boom"))
            ),
            patch.object(capture, "cancel") as cancel,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancelled_when_the_request_is_cancelled(self, agent):
        """A client disconnect raises CancelledError, which run() deliberately does not catch."""
        capture = _capture(status="running")
        with (
            patch.object(
                agent, "_build_orchestrator_agent", return_value=_bundle(capture, raises=asyncio.CancelledError())
            ),
            patch.object(capture, "cancel") as cancel,
            pytest.raises(asyncio.CancelledError),
        ):
            await agent.run(AdaptiveResearchAgentState(messages=[HumanMessage(content="q")]))
        cancel.assert_called_once()
