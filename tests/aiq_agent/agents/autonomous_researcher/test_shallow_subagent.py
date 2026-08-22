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

"""Tests for the autonomous ``shallow-researcher`` sub-agent and its auto-finalization seam.

Two things are under test and they fail in different ways:

* the **adapter** (``subagents/shallow.py``) — turns ``ShallowResearcherAgent`` into a
  ``CompiledSubAgent``, and must never raise, never run twice concurrently, and never leave an
  orphaned task behind;
* the **finalization middleware** — commits a successful report and arms the end-jump, and must
  do neither of those on failure, because a failure notice is this design's only escalation path.

The shallow agent itself is stubbed throughout: what it produces is covered by
``tests/aiq_agent/agents/shallow_researcher/``.
"""

import asyncio
import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.autonomous_researcher.agent import AutonomousResearcherAgent
from aiq_agent.agents.autonomous_researcher.custom_middleware import SHALLOW_SUBAGENT
from aiq_agent.agents.autonomous_researcher.custom_middleware import TASK_TOOL
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalReportCommitTracker
from aiq_agent.agents.autonomous_researcher.custom_middleware import ShallowFinalizationMiddleware
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchAgentState
from aiq_agent.agents.autonomous_researcher.subagents import MAX_SHALLOW_ATTEMPTS
from aiq_agent.agents.autonomous_researcher.subagents import SHALLOW_RESEARCHER_SUBAGENT
from aiq_agent.agents.autonomous_researcher.subagents import ShallowSubagentCapture
from aiq_agent.agents.autonomous_researcher.subagents import build_shallow_researcher_subagent
from aiq_agent.agents.autonomous_researcher.tools.finalize import FINAL_REPORT_META_PATH
from aiq_agent.agents.autonomous_researcher.tools.finalize import FINAL_REPORT_PATH
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import get_session_registry

ORIGINAL_QUERY = "who is the current ceo of intel?"
SHALLOW_REPORT = "Lip-Bu Tan [1].\n\n**References:**\n- [1] Example - https://example.com"
DESCRIPTION = "shallow description (routing text is asserted in test_factory)"


# =================================================================================================
# Helpers
# =================================================================================================


class _StubShallowResult:
    """Minimal stand-in for ``ShallowResearchAgentState`` as ``run()`` returns it."""

    def __init__(self, content: str):
        self.messages = [AIMessage(content=content)]


def _build_subagent(
    *,
    run_side_effect=None,
    content: str = SHALLOW_REPORT,
    capture: ShallowSubagentCapture | None = None,
    original_query: str | None = ORIGINAL_QUERY,
    registry_middleware: SourceRegistryMiddleware | None = None,
    escalation_route: str = "run_research_batch",
):
    """Build the sub-agent spec with a stubbed ``ShallowResearcherAgent``.

    Returns ``(spec, capture, shallow_stub)`` so tests can assert how many times the shallow agent
    actually ran — the signal that neither retry multiplication nor duplicate dispatch happened.
    """
    capture = capture or ShallowSubagentCapture()
    registry_middleware = registry_middleware or SourceRegistryMiddleware(source_tool_names={"web_search_tool"})
    shallow_stub = MagicMock()
    shallow_stub.run_call_count = 0

    async def _run(_state):
        shallow_stub.run_call_count += 1
        if run_side_effect is not None:
            outcome = run_side_effect(shallow_stub.run_call_count)
            if isinstance(outcome, BaseException):
                raise outcome
        return _StubShallowResult(content)

    shallow_stub.run = _run
    with patch(
        "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent",
        return_value=shallow_stub,
    ):
        spec = build_shallow_researcher_subagent(
            llm_provider=MagicMock(),
            tools=[],
            callbacks=[],
            capture=capture,
            source_registry_middleware=registry_middleware,
            original_query=original_query,
            description=DESCRIPTION,
            escalation_route=escalation_route,
            max_llm_turns=10,
            max_tool_iterations=5,
        )
    return spec, capture, shallow_stub


def _subagent_state(description: str = "a paraphrased task description") -> dict:
    """The state shape DeepAgents hands a CompiledSubAgent runnable."""
    return {"messages": [HumanMessage(content=description)], "data_sources": ["web_search"], "user_info": None}


class _FakeToolCallRequest:
    """Minimal ``ToolCallRequest`` stand-in."""

    def __init__(self, tool_call: dict, state: dict | None = None):
        self.tool_call = tool_call
        self.state = state if state is not None else {"messages": []}


def _task_request(subagent_type: str = SHALLOW_SUBAGENT) -> _FakeToolCallRequest:
    return _FakeToolCallRequest({"name": TASK_TOOL, "args": {"subagent_type": subagent_type}, "id": "call-x"})


class _RecordingBackend:
    """Captures the files ``commit_final_report`` uploads."""

    def __init__(self, error: str | None = None):
        self.uploaded: dict[str, bytes] = {}
        self._error = error

    def upload_files(self, files):
        responses = []
        for path, content in files:
            if self._error is None:
                self.uploaded[path] = content
            responses.append(MagicMock(path=path, error=self._error))
        return responses


def _completed_capture() -> ShallowSubagentCapture:
    capture = ShallowSubagentCapture()
    capture.markdown = SHALLOW_REPORT
    capture.status = "completed"
    capture.attempts = 1
    return capture


def _middleware(
    capture: ShallowSubagentCapture,
    *,
    backend=None,
    tracker=None,
    registry_middleware=None,
) -> ShallowFinalizationMiddleware:
    return ShallowFinalizationMiddleware(
        capture=capture,
        backend=backend if backend is not None else _RecordingBackend(),
        tracker=tracker if tracker is not None else AutonomousFinalReportCommitTracker(),
        source_registry_middleware=registry_middleware,
    )


async def _passthrough(request):
    """Handler standing in for the executed tool."""
    return request.tool_call


# =================================================================================================
# Sub-agent spec
# =================================================================================================


class TestSubagentSpec:
    def test_builder_returns_compiled_subagent_spec(self):
        spec, _, _ = _build_subagent()
        assert spec["name"] == SHALLOW_RESEARCHER_SUBAGENT == "shallow-researcher"
        assert spec["description"] == DESCRIPTION
        assert "runnable" in spec
        # A CompiledSubAgent must NOT carry tools/model/middleware: deepagents runs the runnable
        # as given, and the shallow agent owns its own graph.
        assert "tools" not in spec and "model" not in spec

    def test_sync_invocation_is_rejected_with_an_actionable_error(self):
        spec, _, _ = _build_subagent()
        with pytest.raises(RuntimeError, match="ainvoke"):
            spec["runnable"].invoke(_subagent_state())


# =================================================================================================
# Adapter: success path
# =================================================================================================


class TestAdapterSuccess:
    async def test_original_query_wins_over_the_orchestrator_description(self):
        """The build-time snapshot of the user's own words is authoritative."""
        seen = {}
        spec, _, _ = _build_subagent()
        with patch(
            "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearchAgentState",
            side_effect=lambda **kwargs: seen.update(kwargs) or MagicMock(),
        ):
            spec, _, _ = _build_subagent()
            await spec["runnable"].ainvoke(_subagent_state("a badly paraphrased ask"))
        assert seen["messages"][0].content == ORIGINAL_QUERY

    async def test_falls_back_to_the_task_description_without_an_original_query(self):
        seen = {}
        with patch(
            "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearchAgentState",
            side_effect=lambda **kwargs: seen.update(kwargs) or MagicMock(),
        ):
            spec, _, _ = _build_subagent(original_query=None)
            await spec["runnable"].ainvoke(_subagent_state("the fallback ask"))
        assert seen["messages"][0].content == "the fallback ask"

    async def test_returns_messages_and_writes_the_final_report_files(self):
        spec, capture, _ = _build_subagent()
        result = await spec["runnable"].ainvoke(_subagent_state())

        assert result["messages"][-1].content == SHALLOW_REPORT
        assert capture.has_report and capture.markdown == SHALLOW_REPORT
        files = result["files"]
        assert FINAL_REPORT_PATH in files and FINAL_REPORT_META_PATH in files

    async def test_metadata_carries_no_tier_vocabulary(self):
        """This agent is tier-free; the adaptive copy writes `tier: single_shot` and must not here."""
        spec, _, _ = _build_subagent()
        result = await spec["runnable"].ainvoke(_subagent_state())
        raw = result["files"][FINAL_REPORT_META_PATH]
        content = raw["content"] if isinstance(raw, dict) else raw
        meta = json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
        assert meta == {"researched": True, "source": SHALLOW_RESEARCHER_SUBAGENT}

    async def test_sources_captured_by_the_shallow_run_reach_the_parent_registry(self):
        """Without the shared session registry the parent verifies against an empty one."""
        registry_middleware = SourceRegistryMiddleware(source_tool_names={"web_search_tool"})

        def _capture_registry(_call_count):
            get_session_registry().add(SourceEntry(url="https://example.com", title="Example"))

        spec, _, _ = _build_subagent(run_side_effect=_capture_registry, registry_middleware=registry_middleware)
        await spec["runnable"].ainvoke(_subagent_state())

        assert [entry.url for entry in registry_middleware.registry.all_sources()] == ["https://example.com"]


# =================================================================================================
# Adapter: concurrency and cancellation
# =================================================================================================


class TestAdapterConcurrencyAndCancellation:
    async def test_parallel_task_calls_share_one_shallow_run(self):
        """ToolNode dispatches a turn's calls together; only one full run may result."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow(_state):
            started.set()
            await release.wait()
            return _StubShallowResult(SHALLOW_REPORT)

        capture = ShallowSubagentCapture()
        stub = MagicMock()
        stub.run = _slow
        with patch(
            "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=capture,
                source_registry_middleware=SourceRegistryMiddleware(source_tool_names=set()),
                original_query=ORIGINAL_QUERY,
                description=DESCRIPTION,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        first = asyncio.create_task(spec["runnable"].ainvoke(_subagent_state()))
        await started.wait()
        second = asyncio.create_task(spec["runnable"].ainvoke(_subagent_state()))
        release.set()
        results = await asyncio.gather(first, second)

        assert capture.attempts == 1
        assert all(r["messages"][-1].content == SHALLOW_REPORT for r in results)

    async def test_cancel_stops_an_in_flight_run_and_is_a_noop_once_finished(self):
        release = asyncio.Event()
        started = asyncio.Event()

        async def _slow(_state):
            started.set()
            await release.wait()
            return _StubShallowResult(SHALLOW_REPORT)

        capture = ShallowSubagentCapture()
        stub = MagicMock()
        stub.run = _slow
        with patch(
            "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=capture,
                source_registry_middleware=SourceRegistryMiddleware(source_tool_names=set()),
                original_query=ORIGINAL_QUERY,
                description=DESCRIPTION,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        task = asyncio.create_task(spec["runnable"].ainvoke(_subagent_state()))
        await started.wait()
        capture.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        capture.cancel()  # idempotent

    async def test_cancellation_inside_the_shallow_run_is_never_swallowed(self):
        """Cancellation means teardown; converting it to a failure notice would hide that."""
        spec, capture, _ = _build_subagent(run_side_effect=lambda _n: asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await spec["runnable"].ainvoke(_subagent_state())
        assert capture.attempts == 0 and not capture.has_report


# =================================================================================================
# Adapter: failure contract — the only escalation path
# =================================================================================================


class TestAdapterFailureContract:
    async def test_failure_becomes_a_notice_and_costs_exactly_one_run(self):
        """Raising would be retried 3x by ToolRetryMiddleware — four full shallow runs."""
        spec, capture, stub = _build_subagent(run_side_effect=lambda _n: EmptySourceRegistryError())
        result = await spec["runnable"].ainvoke(_subagent_state())

        assert stub.run_call_count == 1
        assert capture.attempts == 1 and capture.status == "failed"
        assert not capture.has_report
        assert "did not complete" in result["messages"][-1].content
        assert "files" not in result

    async def test_failure_notice_names_the_exception_type_but_never_its_message(self):
        # Fake credential-shaped text, present precisely so the assertion below can prove it does
        # not reach the model. Not a real secret.
        secret = "api-key-abc123 leaked from a source"  # pragma: allowlist secret
        spec, capture, _ = _build_subagent(run_side_effect=lambda _n: RuntimeError(secret))
        result = await spec["runnable"].ainvoke(_subagent_state())

        content = result["messages"][-1].content
        assert "RuntimeError" in content and secret not in content
        assert capture.error_type == "RuntimeError"

    async def test_failure_notice_tells_the_orchestrator_to_research_it_itself(self):
        """The notice IS the escalation instruction; on success the model gets no turn at all."""
        spec, _, _ = _build_subagent(run_side_effect=lambda _n: RuntimeError("boom"))
        result = await spec["runnable"].ainvoke(_subagent_state())
        assert "run_research_batch" in result["messages"][-1].content

    async def test_empty_report_is_rejected_rather_than_captured(self):
        spec, capture, _ = _build_subagent(content="   ")
        result = await spec["runnable"].ainvoke(_subagent_state())
        assert not capture.has_report
        assert "did not complete" in result["messages"][-1].content

    async def test_a_failed_attempt_leaves_the_slot_retryable(self):
        spec, capture, stub = _build_subagent(
            run_side_effect=lambda n: RuntimeError("transient") if n == 1 else None,
        )
        await spec["runnable"].ainvoke(_subagent_state())
        result = await spec["runnable"].ainvoke(_subagent_state())

        assert stub.run_call_count == 2
        assert capture.has_report and result["messages"][-1].content == SHALLOW_REPORT

    async def test_exhausted_budget_returns_the_notice_without_executing(self):
        """Nothing hides this sub-agent after a failure, so refusing to run is the only backstop."""
        spec, capture, stub = _build_subagent(run_side_effect=lambda _n: RuntimeError("systematic"))
        for _ in range(MAX_SHALLOW_ATTEMPTS):
            await spec["runnable"].ainvoke(_subagent_state())
        assert capture.exhausted and stub.run_call_count == MAX_SHALLOW_ATTEMPTS

        result = await spec["runnable"].ainvoke(_subagent_state())
        assert stub.run_call_count == MAX_SHALLOW_ATTEMPTS, "an exhausted budget must not run again"
        assert "No further shallow-researcher attempts" in result["messages"][-1].content


# =================================================================================================
# Auto-finalization middleware
# =================================================================================================


class TestShallowFinalization:
    async def test_successful_report_is_committed_and_arms_the_end_jump(self):
        backend, tracker = _RecordingBackend(), AutonomousFinalReportCommitTracker()
        middleware = _middleware(_completed_capture(), backend=backend, tracker=tracker)

        await middleware.awrap_tool_call(_task_request(), _passthrough)

        assert middleware.finalized
        assert backend.uploaded[FINAL_REPORT_PATH].decode("utf-8") == SHALLOW_REPORT
        assert json.loads(backend.uploaded[FINAL_REPORT_META_PATH]) == {"researched": True}
        assert tracker.inline_digest is not None, "the inline exit must be recorded"
        assert middleware.before_model({}, None) == {"jump_to": "end"}

    async def test_no_jump_before_a_shallow_report_exists(self):
        middleware = _middleware(ShallowSubagentCapture())
        assert middleware.before_model({}, None) is None

    async def test_failure_commits_nothing_and_leaves_the_run_to_the_orchestrator(self):
        """This is the escalation path: no commit, no jump, orchestrator keeps its full menu."""
        capture = ShallowSubagentCapture()
        capture.status = "failed"
        capture.attempts = 1
        backend, tracker = _RecordingBackend(), AutonomousFinalReportCommitTracker()
        middleware = _middleware(capture, backend=backend, tracker=tracker)

        await middleware.awrap_tool_call(_task_request(), _passthrough)

        assert not middleware.finalized
        assert backend.uploaded == {}
        assert tracker.inline_digest is None
        assert middleware.before_model({}, None) is None

    async def test_delegations_to_other_subagents_pass_straight_through(self):
        backend = _RecordingBackend()
        middleware = _middleware(_completed_capture(), backend=backend)

        for other in ("researcher-agent", "planner-agent", "writer-agent"):
            await middleware.awrap_tool_call(_task_request(other), _passthrough)

        assert not middleware.finalized and backend.uploaded == {}

    async def test_non_task_tool_calls_pass_straight_through(self):
        backend = _RecordingBackend()
        middleware = _middleware(_completed_capture(), backend=backend)

        request = _FakeToolCallRequest({"name": "run_research_batch", "args": {}, "id": "c"})
        await middleware.awrap_tool_call(request, _passthrough)

        assert not middleware.finalized and backend.uploaded == {}

    async def test_the_tool_result_is_returned_untouched(self):
        """It is a Command carrying the sub-agent's `files` write; replacing it would lose it."""
        middleware = _middleware(_completed_capture())
        sentinel = _task_request()
        result = await middleware.awrap_tool_call(sentinel, _passthrough)
        assert result is sentinel.tool_call

    async def test_a_failed_commit_does_not_arm_the_jump(self):
        """Better one wasted orchestrator turn than a run that ends with nothing recorded."""
        middleware = _middleware(_completed_capture(), backend=_RecordingBackend(error="disk full"))

        await middleware.awrap_tool_call(_task_request(), _passthrough)

        assert not middleware.finalized
        assert middleware.before_model({}, None) is None

    async def test_commit_happens_only_once(self):
        backend = _RecordingBackend()
        middleware = _middleware(_completed_capture(), backend=backend)

        await middleware.awrap_tool_call(_task_request(), _passthrough)
        backend.uploaded.clear()
        await middleware.awrap_tool_call(_task_request(), _passthrough)

        assert backend.uploaded == {}, "a second delegation must not re-commit"

    async def test_shallow_sources_are_promoted_into_the_compact_set(self):
        """On the escalation path a later batch registers compact keys; unpromoted sources vanish."""
        registry_middleware = SourceRegistryMiddleware(source_tool_names={"web_search_tool"})
        registry_middleware.registry.add(SourceEntry(url="https://example.com", title="Example"))
        middleware = _middleware(_completed_capture(), registry_middleware=registry_middleware)

        # A research note registered later is what makes the compact filter selective. Simulating
        # one here is the whole point: without promotion the shallow source would drop out of the
        # very list the already-published answer cites.
        await middleware.awrap_tool_call(_task_request(), _passthrough)
        registry_middleware.registry.add(SourceEntry(url="https://later-note.example", title="Later"))
        registry_middleware.register_compact_sources([SourceEntry(url="https://later-note.example")])

        compact = {entry.url for entry in registry_middleware.get_source_entries(mode="compact")}
        assert compact == {"https://example.com", "https://later-note.example"}


# =================================================================================================
# Graph level: the zero-turn exit, end to end
# =================================================================================================


class _ToolBindingFakeChatModel(FakeMessagesListChatModel):
    """Scripted chat model that accepts the tools bound by ``create_agent``."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _delegate_message(subagent_type: str = SHALLOW_SUBAGENT) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": TASK_TOOL,
                "args": {"description": ORIGINAL_QUERY, "subagent_type": subagent_type},
                "id": "c1",
            }
        ],
    )


@tool
def _probe_web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


def _provider(*responses: AIMessage) -> tuple[LLMProvider, list[int]]:
    """A provider whose model records every call, so a second turn is observable."""
    calls: list[int] = []

    class _Counting(_ToolBindingFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls.append(1)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    llm = _Counting(responses=list(responses))
    provider = LLMProvider()
    provider.set_default(llm)
    for role in (LLMRole.ORCHESTRATOR, LLMRole.PLANNER, LLMRole.RESEARCHER, LLMRole.REPORT_WRITER):
        provider.configure(role, llm)
    return provider, calls


async def _invoke_graph(provider, *, shallow_run):
    """Build and run the REAL compiled graph with only the shallow agent stubbed."""
    stub = MagicMock()
    stub.run = shallow_run
    with patch(
        "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent",
        return_value=stub,
    ):
        agent = AutonomousResearcherAgent(
            llm_provider=provider,
            tools=[_probe_web_search],
            enable_citation_verification=False,
        )
        state = AutonomousResearchAgentState(messages=[HumanMessage(content=ORIGINAL_QUERY)])
        tracker = AutonomousFinalReportCommitTracker()
        built = agent._build_orchestrator_agent(state, tracker)
        result = await built.runnable.ainvoke(state)
    return result, tracker, built


class TestZeroTurnExit:
    """The claim this whole design rests on, asserted against a real compiled graph.

    A unit test of ``before_model`` proves the hook returns the right dict; only running the graph
    proves LangChain honours it from a ``before_model`` node and that the run actually stops. The
    scripted model is given extra responses on purpose — consuming one of them IS the failure.
    """

    async def test_successful_shallow_run_costs_exactly_one_model_call(self):
        provider, calls = _provider(
            _delegate_message(),
            AIMessage(content="a second orchestrator turn means the end-jump did not fire"),
            AIMessage(content="nor a third"),
        )

        async def _run(_state):
            return _StubShallowResult(SHALLOW_REPORT)

        result, tracker, built = await _invoke_graph(provider, shallow_run=_run)

        assert len(calls) == 1, f"expected only the delegation turn, got {len(calls)} model calls"
        assert built.shallow_capture.has_report
        assert tracker.inline_digest is not None, "the inline exit must be committed"
        report = result["files"][FINAL_REPORT_PATH]
        content = report["content"] if isinstance(report, dict) else report
        assert SHALLOW_REPORT in (content.decode("utf-8") if isinstance(content, bytes) else content)

    async def test_full_run_verifies_the_shallow_report_and_renders_its_sources(self):
        """The whole contract through ``run()``, with citation verification ON.

        This is where a shallow-only run could silently break: the parent re-verifies against its
        own registry, so if the sub-run's sources did not reach it the answer would be stripped of
        citations or the run would raise EmptySourceRegistryError on a perfectly good report.
        """
        provider, calls = _provider(_delegate_message(), AIMessage(content="a second turn"))

        async def _run(_state):
            # What a real shallow run leaves behind: sources in the shared session registry.
            get_session_registry().add(SourceEntry(url="https://example.com", title="Example"))
            return _StubShallowResult(SHALLOW_REPORT)

        stub = MagicMock()
        stub.run = _run
        with patch(
            "aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=stub,
        ):
            agent = AutonomousResearcherAgent(
                llm_provider=provider,
                tools=[_probe_web_search],
                enable_citation_verification=True,
            )
            out = await agent.run(AutonomousResearchAgentState(messages=[HumanMessage(content=ORIGINAL_QUERY)]))

        answer = str(out.messages[-1].content)
        assert len(calls) == 1
        assert "Lip-Bu Tan [1]" in answer
        assert "https://example.com" in answer, "the citation survived verification with a target"

    async def test_a_failed_shallow_run_returns_the_turn_to_the_orchestrator(self):
        """The escalation path must remain open — the graph may NOT end on a failure notice."""
        provider, calls = _provider(
            _delegate_message(),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_final_report",
                        "args": {"markdown": "Recovered inline.", "researched": True},
                        "id": "c2",
                    }
                ],
            ),
        )

        async def _run(_state):
            raise RuntimeError("systematic failure")

        _result, tracker, built = await _invoke_graph(provider, shallow_run=_run)

        assert len(calls) == 2, "the orchestrator must get a turn to escalate"
        assert not built.shallow_capture.has_report
        assert tracker.inline_digest is not None, "the orchestrator finished the run itself"


class TestFailureNoticeEscalationRoute:
    """The notice is the only escalation path, so it must name a door this build actually holds."""

    @pytest.mark.parametrize(
        "route",
        ["run_research_batch", 'task(subagent_type="researcher-agent", ...)'],
    )
    async def test_the_configured_route_is_what_the_notice_names(self, route):
        spec, _, _ = _build_subagent(
            run_side_effect=lambda _n: RuntimeError("boom"),
            escalation_route=route,
        )
        result = await spec["runnable"].ainvoke(_subagent_state())
        assert route in result["messages"][-1].content

    async def test_the_exhausted_attempt_notice_also_carries_the_route(self):
        """Both branches of the notice escalate; only one of them was ever exercised before."""
        capture = ShallowSubagentCapture()
        capture.attempts = MAX_SHALLOW_ATTEMPTS
        spec, _, _ = _build_subagent(
            run_side_effect=lambda _n: RuntimeError("boom"),
            capture=capture,
            escalation_route="researcher-agent",
        )
        result = await spec["runnable"].ainvoke(_subagent_state())
        content = result["messages"][-1].content
        assert "researcher-agent" in content
        assert "run_research_batch" not in content
