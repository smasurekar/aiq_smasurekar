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

"""Tests for the ``single_shot`` shallow-researcher compiled sub-agent and its routing guard."""

import asyncio
import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.adaptive_researcher.custom_middleware import _DECLARE_EFFORT_TIER_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _FINALIZE_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _TASK_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import SingleShotShallowDelegationMiddleware
from aiq_agent.agents.adaptive_researcher.subagents import MAX_SHALLOW_ATTEMPTS
from aiq_agent.agents.adaptive_researcher.subagents import SHALLOW_RESEARCHER_SUBAGENT
from aiq_agent.agents.adaptive_researcher.subagents import ShallowSubagentCapture
from aiq_agent.agents.adaptive_researcher.subagents import build_shallow_researcher_subagent
from aiq_agent.agents.adaptive_researcher.tools.finalize import FINAL_REPORT_META_PATH
from aiq_agent.agents.adaptive_researcher.tools.finalize import FINAL_REPORT_PATH
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import get_session_registry

ORIGINAL_QUERY = "who won the 2026 world cup?"
SHALLOW_REPORT = "The answer is 42 [1].\n\n**References:**\n- [1] Example - https://example.com"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
):
    """Build the sub-agent spec with a stubbed ``ShallowResearcherAgent``.

    Returns ``(spec, capture, shallow_stub)`` so tests can assert on how many times the shallow
    agent actually ran — the signal that retry multiplication is not happening.
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
        "aiq_agent.agents.adaptive_researcher.subagents.shallow.ShallowResearcherAgent",
        return_value=shallow_stub,
    ):
        spec = build_shallow_researcher_subagent(
            llm_provider=MagicMock(),
            tools=[],
            callbacks=[],
            capture=capture,
            source_registry_middleware=registry_middleware,
            original_query=original_query,
            max_llm_turns=10,
            max_tool_iterations=5,
        )
    return spec, capture, shallow_stub


def _subagent_state(description: str = "a paraphrased task description") -> dict:
    """The state shape DeepAgents hands a CompiledSubAgent runnable."""
    return {"messages": [HumanMessage(content=description)], "data_sources": ["web_search"], "user_info": None}


class _FakeToolCallRequest:
    """Minimal ``ToolCallRequest`` stand-in supporting the immutable ``override`` API."""

    def __init__(self, tool_call: dict, state: dict | None = None):
        self.tool_call = tool_call
        self.state = state if state is not None else {"messages": []}

    def override(self, **overrides):
        return _FakeToolCallRequest(overrides.get("tool_call", self.tool_call), self.state)


def _ai_message_with_calls(*calls) -> AIMessage:
    """Build the AIMessage whose tool-call batch ToolNode is currently executing."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call-{i}"} for i, (name, args) in enumerate(calls)],
    )


def _request(name: str, args: dict, *, batch=None) -> _FakeToolCallRequest:
    """Build a tool-call request, optionally with sibling calls in the same batch."""
    state = {"messages": [_ai_message_with_calls(*batch)]} if batch else {"messages": []}
    return _FakeToolCallRequest({"name": name, "args": args, "id": "call-x"}, state)


async def _passthrough(request):
    """Handler that records the (possibly rewritten) tool call it was given."""
    return request.tool_call


def _middleware(capture: ShallowSubagentCapture) -> SingleShotShallowDelegationMiddleware:
    return SingleShotShallowDelegationMiddleware(capture=capture, original_query=ORIGINAL_QUERY)


async def _declare(middleware, tier: str) -> None:
    await middleware.awrap_tool_call(_request(_DECLARE_EFFORT_TIER_TOOL, {"tier": tier}), _passthrough)


def _completed_capture(tier: str = "single_shot") -> ShallowSubagentCapture:
    capture = ShallowSubagentCapture()
    capture.markdown = SHALLOW_REPORT
    capture.status = "completed"
    capture.attempts = 1
    capture.declared_tier = tier
    return capture


# ---------------------------------------------------------------------------
# Sub-agent spec and adapter
# ---------------------------------------------------------------------------


class TestSubagentSpec:
    def test_builder_returns_compiled_subagent_spec(self):
        spec, _, _ = _build_subagent()
        assert spec["name"] == SHALLOW_RESEARCHER_SUBAGENT
        assert spec["description"]
        assert hasattr(spec["runnable"], "ainvoke")

    @pytest.mark.asyncio
    async def test_sync_invocation_is_rejected_with_a_clear_error(self):
        spec, _, _ = _build_subagent()
        with pytest.raises(RuntimeError, match="requires the async path"):
            spec["runnable"].invoke(_subagent_state())


class TestAdapterSuccess:
    @pytest.mark.asyncio
    async def test_uses_the_original_query_not_the_task_description(self):
        captured_queries = []
        capture = ShallowSubagentCapture()
        registry = SourceRegistryMiddleware(source_tool_names=set())
        shallow_stub = MagicMock()

        async def _run(state):
            captured_queries.append(state.messages[-1].content)
            return _StubShallowResult(SHALLOW_REPORT)

        shallow_stub.run = _run
        with patch(
            "aiq_agent.agents.adaptive_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=shallow_stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=capture,
                source_registry_middleware=registry,
                original_query=ORIGINAL_QUERY,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        await spec["runnable"].ainvoke(_subagent_state("a paraphrase the model invented"))
        assert captured_queries == [ORIGINAL_QUERY]

    @pytest.mark.asyncio
    async def test_falls_back_to_the_task_description_when_no_original_query(self):
        captured_queries = []
        shallow_stub = MagicMock()

        async def _run(state):
            captured_queries.append(state.messages[-1].content)
            return _StubShallowResult(SHALLOW_REPORT)

        shallow_stub.run = _run
        with patch(
            "aiq_agent.agents.adaptive_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=shallow_stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=ShallowSubagentCapture(),
                source_registry_middleware=SourceRegistryMiddleware(source_tool_names=set()),
                original_query=None,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        await spec["runnable"].ainvoke(_subagent_state("the only text available"))
        assert captured_queries == ["the only text available"]

    @pytest.mark.asyncio
    async def test_returns_messages_and_final_report_files(self):
        spec, capture, _ = _build_subagent()
        result = await spec["runnable"].ainvoke(_subagent_state())

        assert result["messages"][-1].content == SHALLOW_REPORT
        assert result["files"][FINAL_REPORT_PATH]["content"] == SHALLOW_REPORT
        meta = json.loads(result["files"][FINAL_REPORT_META_PATH]["content"])
        assert meta == {"researched": True, "tier": "single_shot", "source": SHALLOW_RESEARCHER_SUBAGENT}
        assert capture.status == "completed"
        assert capture.markdown == SHALLOW_REPORT
        assert capture.has_report is True
        assert capture.attempts == 1

    @pytest.mark.asyncio
    async def test_sources_captured_by_the_shallow_run_reach_the_parent_registry(self):
        """The registry bridge is what stops a good answer failing the adaptive citation gate."""
        registry_middleware = SourceRegistryMiddleware(source_tool_names={"web_search_tool"})
        shallow_stub = MagicMock()

        async def _run(_state):
            # Stand in for the shallow agent's tool_node_with_source_capture, which writes to
            # `get_session_registry() or self.source_registry`.
            session_registry = get_session_registry()
            assert session_registry is not None, "adapter must bind a session registry for the sub-run"
            session_registry.add(SourceEntry(url="https://example.com", title="Example", tool_name="web_search_tool"))
            return _StubShallowResult(SHALLOW_REPORT)

        shallow_stub.run = _run
        with patch(
            "aiq_agent.agents.adaptive_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=shallow_stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=ShallowSubagentCapture(),
                source_registry_middleware=registry_middleware,
                original_query=ORIGINAL_QUERY,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        assert registry_middleware.has_sources() is False
        await spec["runnable"].ainvoke(_subagent_state())
        assert registry_middleware.has_sources() is True
        # The contextvar must be reset so it cannot leak into the rest of the request.
        assert get_session_registry() is None


class TestAdapterConcurrencyAndCancellation:
    @pytest.mark.asyncio
    async def test_parallel_task_calls_share_one_shallow_run(self):
        started = asyncio.Event()
        release = asyncio.Event()
        shallow_stub = MagicMock()
        shallow_stub.calls = 0

        async def _run(_state):
            shallow_stub.calls += 1
            started.set()
            await release.wait()
            return _StubShallowResult(SHALLOW_REPORT)

        shallow_stub.run = _run
        with patch(
            "aiq_agent.agents.adaptive_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=shallow_stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=ShallowSubagentCapture(),
                source_registry_middleware=SourceRegistryMiddleware(source_tool_names=set()),
                original_query=ORIGINAL_QUERY,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        first = asyncio.create_task(spec["runnable"].ainvoke(_subagent_state()))
        await started.wait()
        second = asyncio.create_task(spec["runnable"].ainvoke(_subagent_state()))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second)

        assert shallow_stub.calls == 1, "concurrent task calls must coalesce onto one shallow run"
        assert results[0]["messages"][-1].content == results[1]["messages"][-1].content

    @pytest.mark.asyncio
    async def test_cancel_stops_an_in_flight_run_and_is_a_noop_when_finished(self):
        started = asyncio.Event()
        capture = ShallowSubagentCapture()
        shallow_stub = MagicMock()

        async def _run(_state):
            started.set()
            await asyncio.sleep(30)
            return _StubShallowResult(SHALLOW_REPORT)

        shallow_stub.run = _run
        with patch(
            "aiq_agent.agents.adaptive_researcher.subagents.shallow.ShallowResearcherAgent",
            return_value=shallow_stub,
        ):
            spec = build_shallow_researcher_subagent(
                llm_provider=MagicMock(),
                tools=[],
                callbacks=[],
                capture=capture,
                source_registry_middleware=SourceRegistryMiddleware(source_tool_names=set()),
                original_query=ORIGINAL_QUERY,
                max_llm_turns=10,
                max_tool_iterations=5,
            )
        waiter = asyncio.create_task(spec["runnable"].ainvoke(_subagent_state()))
        await started.wait()
        detached = capture._task  # the task run() must own; the awaiter alone cannot stop it

        capture.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert detached.cancelled() is True

        capture.cancel()  # idempotent, and safe once the task has finished

    @pytest.mark.asyncio
    async def test_cancellation_inside_the_shallow_run_is_never_swallowed(self):
        spec, capture, _ = _build_subagent(run_side_effect=lambda _n: asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await spec["runnable"].ainvoke(_subagent_state())
        # No attempt is spent and nothing partial is recoverable.
        assert capture.attempts == 0
        assert capture.status == "running"
        assert capture.has_report is False


class TestAdapterFailureContract:
    @pytest.mark.asyncio
    async def test_empty_source_registry_error_does_not_escape_and_costs_one_run(self):
        """Raising would be retried 3x by ToolRetryMiddleware — three more full shallow runs."""
        spec, capture, stub = _build_subagent(
            run_side_effect=lambda _n: EmptySourceRegistryError("shallow research"),
        )
        result = await spec["runnable"].ainvoke(_subagent_state())

        assert stub.run_call_count == 1, "one delegation must cost exactly one shallow run"
        assert "files" not in result, "a failed attempt must not look like a report"
        assert "EmptySourceRegistryError" in result["messages"][-1].content
        assert capture.status == "failed"
        assert capture.error_type == "EmptySourceRegistryError"
        assert capture.attempts == 1
        assert capture.has_report is False
        # researched is never flipped to False: that value is reserved for a deliberate
        # no-research answer, not a research attempt that failed.
        assert capture.researched is True

    @pytest.mark.asyncio
    async def test_empty_report_is_rejected_rather_than_captured(self):
        spec, capture, _ = _build_subagent(content="   ")
        result = await spec["runnable"].ainvoke(_subagent_state())
        assert capture.status == "failed"
        assert capture.error_type == "ValueError"
        assert "files" not in result

    @pytest.mark.asyncio
    async def test_attempt_budget_is_spent_after_max_attempts(self):
        spec, capture, stub = _build_subagent(run_side_effect=lambda _n: RuntimeError("source down"))
        for _ in range(MAX_SHALLOW_ATTEMPTS):
            await spec["runnable"].ainvoke(_subagent_state())

        assert stub.run_call_count == MAX_SHALLOW_ATTEMPTS
        assert capture.attempts == MAX_SHALLOW_ATTEMPTS
        assert capture.exhausted is True

    @pytest.mark.asyncio
    async def test_a_failed_attempt_leaves_the_slot_retryable(self):
        spec, capture, stub = _build_subagent(
            run_side_effect=lambda n: RuntimeError("transient") if n == 1 else None,
        )
        await spec["runnable"].ainvoke(_subagent_state())
        assert capture.status == "failed"

        result = await spec["runnable"].ainvoke(_subagent_state())
        assert stub.run_call_count == 2, "the second delegation must be a real new attempt"
        assert capture.status == "completed"
        assert result["files"][FINAL_REPORT_PATH]["content"] == SHALLOW_REPORT


# ---------------------------------------------------------------------------
# SingleShotShallowDelegationMiddleware
# ---------------------------------------------------------------------------


class TestDelegationRouting:
    @pytest.mark.asyncio
    async def test_declaration_is_mirrored_onto_the_capture(self):
        capture = ShallowSubagentCapture()
        middleware = _middleware(capture)
        await _declare(middleware, "single_shot")
        assert capture.declared_tier == "single_shot"
        # Escalation must be tracked too, so the finalizer override switches itself off.
        await _declare(middleware, "deep")
        assert capture.declared_tier == "deep"

    @pytest.mark.asyncio
    async def test_task_without_any_declaration_is_rejected(self):
        middleware = _middleware(ShallowSubagentCapture())
        result = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": "planner-agent", "description": "plan"}), _passthrough
        )
        assert result.status == "error"
        assert "Declare the effort tier" in result.content

    @pytest.mark.asyncio
    async def test_single_shot_task_is_forced_to_the_shallow_subagent_and_original_query(self):
        middleware = _middleware(ShallowSubagentCapture())
        await _declare(middleware, "single_shot")
        forwarded = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": "planner-agent", "description": "a paraphrase"}), _passthrough
        )
        assert forwarded["args"]["subagent_type"] == SHALLOW_RESEARCHER_SUBAGENT
        assert forwarded["args"]["description"] == ORIGINAL_QUERY

    @pytest.mark.asyncio
    @pytest.mark.parametrize("order", [("declare_first"), ("task_first")])
    async def test_same_turn_declaration_routes_identically_in_either_wrapper_order(self, order):
        """ToolNode may run sibling wrappers in any order; routing must not depend on it."""
        middleware = _middleware(ShallowSubagentCapture())
        batch = [
            (_DECLARE_EFFORT_TIER_TOOL, {"tier": "single_shot"}),
            (_TASK_TOOL, {"subagent_type": "planner-agent", "description": "paraphrase"}),
        ]
        if order == "declare_first":
            await middleware.awrap_tool_call(_request(_DECLARE_EFFORT_TIER_TOOL, {"tier": "single_shot"}), _passthrough)
        forwarded = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": "planner-agent", "description": "paraphrase"}, batch=batch),
            _passthrough,
        )
        assert forwarded["args"]["subagent_type"] == SHALLOW_RESEARCHER_SUBAGENT
        assert forwarded["args"]["description"] == ORIGINAL_QUERY

    @pytest.mark.asyncio
    async def test_conflicting_same_turn_declarations_are_rejected(self):
        middleware = _middleware(ShallowSubagentCapture())
        batch = [
            (_DECLARE_EFFORT_TIER_TOOL, {"tier": "single_shot"}),
            (_DECLARE_EFFORT_TIER_TOOL, {"tier": "deep"}),
            (_TASK_TOOL, {"subagent_type": "planner-agent", "description": "x"}),
        ]
        result = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": "planner-agent", "description": "x"}, batch=batch), _passthrough
        )
        assert result.status == "error"
        assert "Conflicting effort tiers" in result.content

    @pytest.mark.asyncio
    async def test_shallow_subagent_is_rejected_on_other_tiers(self):
        middleware = _middleware(ShallowSubagentCapture())
        await _declare(middleware, "deep")
        result = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": SHALLOW_RESEARCHER_SUBAGENT, "description": "x"}), _passthrough
        )
        assert result.status == "error"
        assert "only on the single_shot tier" in result.content

    @pytest.mark.asyncio
    async def test_planner_and_writer_delegation_is_untouched_on_other_tiers(self):
        middleware = _middleware(ShallowSubagentCapture())
        await _declare(middleware, "deep")
        for subagent in ("planner-agent", "writer-agent", "source-router-agent"):
            forwarded = await middleware.awrap_tool_call(
                _request(_TASK_TOOL, {"subagent_type": subagent, "description": "do it"}), _passthrough
            )
            assert forwarded["args"] == {"subagent_type": subagent, "description": "do it"}

    @pytest.mark.asyncio
    async def test_task_is_rejected_once_the_attempt_budget_is_spent(self):
        capture = ShallowSubagentCapture()
        capture.status = "failed"
        capture.attempts = MAX_SHALLOW_ATTEMPTS
        middleware = _middleware(capture)
        await _declare(middleware, "single_shot")
        result = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": SHALLOW_RESEARCHER_SUBAGENT, "description": "x"}), _passthrough
        )
        assert result.status == "error"
        assert "Do not delegate again" in result.content


class TestFinalizeOverride:
    @pytest.mark.asyncio
    async def test_finalize_is_rejected_while_an_attempt_is_still_viable(self):
        middleware = _middleware(ShallowSubagentCapture())
        await _declare(middleware, "single_shot")
        result = await middleware.awrap_tool_call(
            _request(_FINALIZE_TOOL, {"markdown": "my own answer", "researched": True}), _passthrough
        )
        assert result.status == "error"
        assert "has not completed" in result.content

    @pytest.mark.asyncio
    async def test_paraphrased_report_is_replaced_by_the_captured_one(self):
        middleware = _middleware(_completed_capture())
        await _declare(middleware, "single_shot")
        forwarded = await middleware.awrap_tool_call(
            _request(_FINALIZE_TOOL, {"markdown": "a shortened paraphrase", "researched": False, "tier": "direct"}),
            _passthrough,
        )
        assert forwarded["args"]["markdown"] == SHALLOW_REPORT
        assert forwarded["args"]["researched"] is True
        assert forwarded["args"]["tier"] == "single_shot"

    @pytest.mark.asyncio
    async def test_escalation_disables_the_override(self):
        capture = _completed_capture()
        middleware = _middleware(capture)
        await _declare(middleware, "single_shot")
        await _declare(middleware, "deep")
        forwarded = await middleware.awrap_tool_call(
            _request(_FINALIZE_TOOL, {"markdown": "a deep-tier answer", "researched": True}), _passthrough
        )
        assert forwarded["args"]["markdown"] == "a deep-tier answer"
        assert capture.declared_tier == "deep"

    @pytest.mark.asyncio
    async def test_same_turn_escalation_disables_the_override_regardless_of_order(self):
        middleware = _middleware(_completed_capture())
        await _declare(middleware, "single_shot")
        batch = [
            (_DECLARE_EFFORT_TIER_TOOL, {"tier": "standard"}),
            (_FINALIZE_TOOL, {"markdown": "a standard-tier answer"}),
        ]
        forwarded = await middleware.awrap_tool_call(
            _request(_FINALIZE_TOOL, {"markdown": "a standard-tier answer"}, batch=batch), _passthrough
        )
        assert forwarded["args"]["markdown"] == "a standard-tier answer"

    @pytest.mark.asyncio
    async def test_exhausted_budget_opens_the_escape_hatch_with_researched_forced_true(self):
        """Rejecting forever would livelock: `task` is closed and finalize is the only exit."""
        capture = ShallowSubagentCapture()
        capture.status = "failed"
        capture.attempts = MAX_SHALLOW_ATTEMPTS
        capture.error_type = "EmptySourceRegistryError"
        middleware = _middleware(capture)
        await _declare(middleware, "single_shot")

        rejected_task = await middleware.awrap_tool_call(
            _request(_TASK_TOOL, {"subagent_type": SHALLOW_RESEARCHER_SUBAGENT, "description": "x"}), _passthrough
        )
        forwarded = await middleware.awrap_tool_call(
            _request(_FINALIZE_TOOL, {"markdown": "what little I have", "researched": False}), _passthrough
        )

        assert rejected_task.status == "error"
        # The finalizer runs, keeps the model's text (there is no captured report to enforce),
        # but researched=True so the existing empty-registry gate still decides the outcome.
        assert forwarded["args"]["markdown"] == "what little I have"
        assert forwarded["args"]["researched"] is True
        assert forwarded["args"]["tier"] == "single_shot"

    @pytest.mark.asyncio
    async def test_other_tiers_finalize_untouched(self):
        middleware = _middleware(_completed_capture(tier="deep"))
        await _declare(middleware, "deep")
        forwarded = await middleware.awrap_tool_call(
            _request(_FINALIZE_TOOL, {"markdown": "deep answer", "researched": True, "tier": "deep"}), _passthrough
        )
        assert forwarded["args"] == {"markdown": "deep answer", "researched": True, "tier": "deep"}

    @pytest.mark.asyncio
    async def test_unrelated_tools_pass_through_untouched(self):
        middleware = _middleware(_completed_capture())
        await _declare(middleware, "single_shot")
        forwarded = await middleware.awrap_tool_call(_request("think", {"thought": "hmm"}), _passthrough)
        assert forwarded["args"] == {"thought": "hmm"}
