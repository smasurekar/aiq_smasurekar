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

"""The autonomous-only evidence and integrity seams.

Each of these replaces something the tier machinery used to provide for free, so they are tested
against the behaviour that machinery guaranteed: evidence reaches the compact source set no
matter which of the three research paths produced it, the writer never runs without a plan, and a
run cannot end without committing one of the two valid exits.
"""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalizationMiddleware
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalReportCommitTracker
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousOrchestratorLoopGuardMiddleware
from aiq_agent.agents.autonomous_researcher.custom_middleware import DirectSourcePromotionMiddleware
from aiq_agent.agents.autonomous_researcher.custom_middleware import PlanBeforeWriterMiddleware
from aiq_agent.agents.autonomous_researcher.custom_middleware import ResearcherTaskPersistenceMiddleware
from aiq_agent.agents.autonomous_researcher.models import AutonomousRequestTerminationConfig
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.resource_limits import DeepResearchResourceLimits
from aiq_agent.agents.deep_researcher.resource_limits import StateBudgetLedger
from aiq_agent.common.citation_verification import SourceEntry

INLINE_PATH = "/shared/final_report.md"
WRITER_PATH = "/shared/output.md"
PLAN_PATH = "/shared/plan.json"


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _make_tool(name: str):
    tool = MagicMock()
    tool.name = name
    return tool


def _request(tool_call: dict, *, state: dict | None = None):
    request = MagicMock()
    request.tool_call = tool_call
    request.state = state if state is not None else {}
    return request


def _notes_json(topic: str = "gpu launches", locator: str = "https://example.com/a") -> str:
    """A minimal valid ResearchNotes payload, as the task tool would serialize it."""
    return json.dumps(
        {
            "query_topic": topic,
            "target_components": ["c1"],
            "summary": "s",
            "findings": [{"claim": "c", "evidence": "e", "source_ids": [1], "confidence": "high", "caveats": []}],
            "gaps": [],
            "sources": [{"id": 1, "title": "A", "source_type": "url", "locator": locator}],
            "narrative_notes": "n",
            "language": "en",
        }
    )


def _query(query: str = "apple inventory", **over) -> dict:
    payload = {
        "query": query,
        "subqueries": [],
        "preferred_tools": ["web_search_tool"],
        "target_components": ["c1"],
        "depth": "medium",
        "rationale": "why",
        "fallback_tools": [],
    }
    payload.update(over)
    return payload


def _batch_request(queries: list[dict]):
    """A run_research_batch call whose `override` records what the guard forwards downstream.

    The depth clamp rewrites the tool call rather than rejecting it, so asserting on the clamp
    means asserting on the args the tool would actually receive — which arrive through
    `request.override(tool_call=...)`, not through the original request.
    """
    request = _request({"name": "run_research_batch", "args": {"queries": queries}, "id": "call_1"})

    def _override(**kwargs):
        forwarded = MagicMock()
        forwarded.tool_call = kwargs.get("tool_call", request.tool_call)
        forwarded.state = request.state
        request.forwarded = forwarded
        return forwarded

    request.forwarded = None
    request.override = MagicMock(side_effect=_override)
    return request


def _forwarded_queries(handler) -> list[dict]:
    """The `queries` list the tool handler was actually invoked with."""
    return handler.await_args.args[0].tool_call["args"]["queries"]


def _source_request(query: str = "apple inventory", *, tool: str = "web_search_tool"):
    return _request({"name": tool, "args": {"query": query}, "id": "src_1"})


def _task_request(subagent: str, *, description: str = "research the thing"):
    return _request({"name": "task", "args": {"subagent_type": subagent, "description": description}, "id": "task_1"})


# ---------------------------------------------------------------------------------------------
# Dual-exit commit tracking
# ---------------------------------------------------------------------------------------------


class TestAutonomousFinalReportCommitTracker:
    def test_neither_exit_committed_by_default(self):
        assert not AutonomousFinalReportCommitTracker().any_exit_committed({})

    def test_inline_exit_alone_satisfies_the_contract(self):
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record_inline("# answer")
        assert tracker.any_exit_committed({INLINE_PATH: {"content": "# answer"}})
        assert tracker.inline_committed_text({INLINE_PATH: {"content": "# answer"}}) == "# answer"

    def test_writer_exit_alone_satisfies_the_contract(self):
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record("# report")
        assert tracker.any_exit_committed({WRITER_PATH: {"content": "# report"}})

    def test_writer_path_behaviour_is_unchanged(self):
        """RequiredOutputFileMiddleware keeps using committed_text; the inline digest must not leak."""
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record_inline("# inline only")
        assert tracker.committed_text({INLINE_PATH: {"content": "# inline only"}}) is None

    def test_tampered_inline_file_does_not_match(self):
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record_inline("# answer")
        assert tracker.inline_committed_text({INLINE_PATH: {"content": "# something else"}}) is None

    def test_digest_alone_counts_after_the_return_direct_exit(self):
        """submit_final_report is return_direct, so the run can end before files are visible."""
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record_inline("# answer")
        assert tracker.any_exit_committed({})


# ---------------------------------------------------------------------------------------------
# Path-independent evidence state
# ---------------------------------------------------------------------------------------------


class TestDirectSourcePromotion:
    @pytest.fixture
    def registry_middleware(self):
        return SourceRegistryMiddleware(source_tool_names={"web_search_tool"})

    @pytest.mark.asyncio
    async def test_direct_source_is_promoted_into_the_compact_set(self, registry_middleware):
        middleware = DirectSourcePromotionMiddleware(
            source_registry_middleware=registry_middleware,
            source_tool_names={"web_search_tool"},
        )

        async def handler(_request):
            registry_middleware.active_registry().add(SourceEntry(url="https://direct.example/a", title="Direct"))
            return ToolMessage(content="ok", tool_call_id="1", name="web_search_tool")

        await middleware.awrap_tool_call(_request({"name": "web_search_tool", "args": {}, "id": "1"}), handler)
        compact = registry_middleware.get_source_entries(mode="compact")
        assert [entry.url for entry in compact] == ["https://direct.example/a"]

    @pytest.mark.asyncio
    async def test_direct_evidence_survives_a_later_batch(self, registry_middleware):
        """Without promotion, a batch's compact subset would hide the direct source entirely."""
        middleware = DirectSourcePromotionMiddleware(
            source_registry_middleware=registry_middleware,
            source_tool_names={"web_search_tool"},
        )

        async def handler(_request):
            registry_middleware.active_registry().add(SourceEntry(url="https://direct.example/a", title="Direct"))
            return ToolMessage(content="ok", tool_call_id="1", name="web_search_tool")

        await middleware.awrap_tool_call(_request({"name": "web_search_tool", "args": {}, "id": "1"}), handler)

        # A batch then registers its own note sources, establishing a compact subset.
        registry_middleware.active_registry().add(SourceEntry(url="https://batch.example/b", title="Batch"))
        note = MagicMock()
        note.sources = [MagicMock(locator="https://batch.example/b")]
        registry_middleware.register_research_note_sources([note])

        urls = {entry.url for entry in registry_middleware.get_source_entries(mode="compact")}
        assert urls == {"https://direct.example/a", "https://batch.example/b"}

    @pytest.mark.asyncio
    async def test_non_source_tools_are_ignored(self, registry_middleware):
        middleware = DirectSourcePromotionMiddleware(
            source_registry_middleware=registry_middleware,
            source_tool_names={"web_search_tool"},
        )
        handler = AsyncMock(return_value=ToolMessage(content="thought", tool_call_id="1", name="think"))
        await middleware.awrap_tool_call(_request({"name": "think", "args": {}, "id": "1"}), handler)
        assert registry_middleware.get_source_entries(mode="compact") == []


class TestResearcherTaskPersistence:
    @pytest.fixture
    def middleware_and_backend(self):
        backend = MagicMock()
        backend.upload_files = MagicMock(return_value=[MagicMock(path="p", error=None)])
        limits = DeepResearchResourceLimits()
        registry = SourceRegistryMiddleware(source_tool_names={"web_search_tool"})
        middleware = ResearcherTaskPersistenceMiddleware(
            backend=backend,
            state_budget=StateBudgetLedger(limits=limits, files={}, sandbox_enabled=False),
            resource_limits=limits,
            source_registry_middleware=registry,
        )
        return middleware, backend, registry

    @staticmethod
    def _task_request(subagent: str = "researcher-agent"):
        return _request(
            {"name": "task", "args": {"subagent_type": subagent, "description": "investigate X"}, "id": "1"}
        )

    @staticmethod
    def _command_result(content: str) -> Command:
        return Command(update={"messages": [ToolMessage(content=content, tool_call_id="1")]})

    @pytest.mark.asyncio
    async def test_persists_one_collision_safe_note_and_registers_its_sources(self, middleware_and_backend):
        middleware, backend, registry = middleware_and_backend
        registry.active_registry().add(SourceEntry(url="https://example.com/a", title="A"))
        handler = AsyncMock(return_value=self._command_result(_notes_json()))

        await middleware.awrap_tool_call(self._task_request(), handler)

        (note_files,) = backend.upload_files.call_args.args
        assert len(note_files) == 1
        path, payload = note_files[0]
        assert path.startswith("/shared/research_note_task_01_")
        assert path.endswith(".json")
        assert b"gpu launches" in payload
        assert [e.url for e in registry.get_source_entries(mode="compact")] == ["https://example.com/a"]

    @pytest.mark.asyncio
    async def test_repeated_delegations_do_not_collide(self, middleware_and_backend):
        middleware, backend, _ = middleware_and_backend
        # A fresh Command per call: the real task tool builds a new ToolMessage each time, and
        # reusing one would let the first call's appended persistence note corrupt the second
        # parse.
        handler = AsyncMock(side_effect=lambda _r: self._command_result(_notes_json()))
        await middleware.awrap_tool_call(self._task_request(), handler)
        await middleware.awrap_tool_call(self._task_request(), handler)
        paths = [call.args[0][0][0] for call in backend.upload_files.call_args_list]
        assert len(set(paths)) == 2, "identical delegations must not overwrite each other's note"

    @pytest.mark.asyncio
    async def test_other_subagents_are_untouched(self, middleware_and_backend):
        middleware, backend, _ = middleware_and_backend
        handler = AsyncMock(return_value=self._command_result("some plan text"))
        await middleware.awrap_tool_call(self._task_request("planner-agent"), handler)
        backend.upload_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_non_structured_answer_is_still_returned(self, middleware_and_backend):
        """Losing a bookkeeping side effect must never lose the research itself."""
        middleware, backend, _ = middleware_and_backend
        handler = AsyncMock(return_value=self._command_result("I could not produce structured notes."))
        result = await middleware.awrap_tool_call(self._task_request(), handler)
        backend.upload_files.assert_not_called()
        assert "could not produce" in result.update["messages"][0].content

    @pytest.mark.asyncio
    async def test_a_persistence_failure_is_not_fatal(self, middleware_and_backend):
        middleware, backend, registry = middleware_and_backend
        backend.upload_files.side_effect = RuntimeError("backend down")
        handler = AsyncMock(return_value=self._command_result(_notes_json()))
        result = await middleware.awrap_tool_call(self._task_request(), handler)
        assert result is not None
        # Sources are still registered even though the file could not be written.
        registry.active_registry().add(SourceEntry(url="https://example.com/a", title="A"))
        assert [e.url for e in registry.get_source_entries(mode="compact")] == ["https://example.com/a"]


class TestPlanBeforeWriter:
    @pytest.mark.asyncio
    async def test_writer_is_rejected_before_a_plan_exists(self):
        middleware = PlanBeforeWriterMiddleware()
        handler = AsyncMock()
        result = await middleware.awrap_tool_call(
            _request({"name": "task", "args": {"subagent_type": "writer-agent"}, "id": "1"}),
            handler,
        )
        handler.assert_not_called()
        assert result.status == "error"
        assert "planner-agent" in result.content

    @pytest.mark.asyncio
    async def test_writer_is_accepted_after_a_planner_run(self):
        middleware = PlanBeforeWriterMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="planned", tool_call_id="1"))
        await middleware.awrap_tool_call(
            _request({"name": "task", "args": {"subagent_type": "planner-agent"}, "id": "1"}), handler
        )
        result = await middleware.awrap_tool_call(
            _request({"name": "task", "args": {"subagent_type": "writer-agent"}, "id": "2"}), handler
        )
        assert getattr(result, "status", None) != "error"

    @pytest.mark.asyncio
    async def test_writer_is_accepted_when_the_plan_is_already_in_state(self):
        middleware = PlanBeforeWriterMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="wrote", tool_call_id="1"))
        result = await middleware.awrap_tool_call(
            _request(
                {"name": "task", "args": {"subagent_type": "writer-agent"}, "id": "1"},
                state={"files": {PLAN_PATH: {"content": "{}"}}},
            ),
            handler,
        )
        handler.assert_awaited_once()
        assert getattr(result, "status", None) != "error"

    @pytest.mark.asyncio
    async def test_a_failed_planner_run_does_not_unlock_the_writer(self):
        middleware = PlanBeforeWriterMiddleware()
        failed = AsyncMock(return_value=ToolMessage(content="boom", tool_call_id="1", status="error"))
        await middleware.awrap_tool_call(
            _request({"name": "task", "args": {"subagent_type": "planner-agent"}, "id": "1"}), failed
        )
        result = await middleware.awrap_tool_call(
            _request({"name": "task", "args": {"subagent_type": "writer-agent"}, "id": "2"}), AsyncMock()
        )
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_researcher_delegation_is_never_gated(self):
        middleware = PlanBeforeWriterMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="notes", tool_call_id="1"))
        await middleware.awrap_tool_call(
            _request({"name": "task", "args": {"subagent_type": "researcher-agent"}, "id": "1"}), handler
        )
        handler.assert_awaited_once()


class TestAutonomousFinalization:
    @staticmethod
    def _state(messages, files=None):
        return {"messages": messages, "files": files or {}}

    def test_inline_exit_satisfies_the_guard_without_writer_delegation(self):
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record_inline("# answer")
        middleware = AutonomousFinalizationMiddleware(tracker=tracker)
        state = self._state([AIMessage(content="done")], {INLINE_PATH: {"content": "# answer"}})
        assert middleware._check_after_model(state) is None

    def test_writer_exit_also_satisfies_the_guard(self):
        tracker = AutonomousFinalReportCommitTracker()
        tracker.record("# report")
        middleware = AutonomousFinalizationMiddleware(tracker=tracker)
        state = self._state([AIMessage(content="Wrote /shared/output.md")], {WRITER_PATH: {"content": "# report"}})
        assert middleware._check_after_model(state) is None

    def test_neither_exit_gets_one_bounded_corrective_turn(self):
        middleware = AutonomousFinalizationMiddleware(tracker=AutonomousFinalReportCommitTracker())
        result = middleware._check_after_model(self._state([AIMessage(content="here you go")]))
        assert result is not None
        assert result["jump_to"] == "model"
        retry = result["messages"][0]
        assert "submit_final_report" in retry.content

    def test_the_corrective_turn_does_not_push_toward_the_writer(self):
        """Forcing writer delegation is exactly what RequiredWriterDelegationMiddleware got wrong."""
        middleware = AutonomousFinalizationMiddleware(tracker=AutonomousFinalReportCommitTracker())
        retry = middleware._check_after_model(self._state([AIMessage(content="x")]))["messages"][0]
        assert "whichever exit matches" in retry.content

    def test_the_retry_budget_is_spent_only_once_then_falls_through(self):
        """Raising here would turn a recoverable conversational answer into a request failure."""
        middleware = AutonomousFinalizationMiddleware(tracker=AutonomousFinalReportCommitTracker())
        first = middleware._check_after_model(self._state([AIMessage(content="x")]))
        messages = [AIMessage(content="x"), first["messages"][0], AIMessage(content="still text")]
        assert middleware._check_after_model(self._state(messages)) is None

    def test_a_turn_with_tool_calls_is_not_a_run_ending(self):
        middleware = AutonomousFinalizationMiddleware(tracker=AutonomousFinalReportCommitTracker())
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {"query": "x"}, "id": "1"}],
        )
        assert middleware._check_after_model(self._state([ai])) is None


# ---------------------------------------------------------------------------------------------
# Request-wide termination
# ---------------------------------------------------------------------------------------------


def _config(**over) -> AutonomousRequestTerminationConfig:
    base = dict(
        max_batch_calls=2,
        max_total_research_queries=3,
        max_orchestrator_turns=4,
        max_identical_research_queries=1,
        workflow_timeout_seconds=100,
        fallback_finalizer_timeout_seconds=10,
        recursion_limit=50,
    )
    base.update(over)
    return AutonomousRequestTerminationConfig(**base)


class TestAutonomousOrchestratorLoopGuard:
    @staticmethod
    def _guard(**over):
        return AutonomousOrchestratorLoopGuardMiddleware(
            config=_config(**over),
            source_tool_names=frozenset({"web_search_tool"}),
        )

    @pytest.mark.asyncio
    async def test_budgets_apply_with_no_tier_declaration(self):
        """The adaptive guard was inert until a tier was declared; this one never is."""
        guard = self._guard(max_batch_calls=1)
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("q1")]), handler)
        blocked = await guard.awrap_tool_call(_batch_request([_query("q2")]), handler)
        assert blocked.status == "error"
        assert "batch budget" in blocked.content
        assert guard.phase == "finalizing"

    @pytest.mark.asyncio
    async def test_total_query_budget_blocks_the_whole_batch(self):
        guard = self._guard(max_total_research_queries=3)
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("a"), _query("b")]), handler)
        blocked = await guard.awrap_tool_call(_batch_request([_query("c"), _query("d")]), handler)
        assert blocked.status == "error"
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_queries_are_blocked(self):
        guard = self._guard()
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("same")]), handler)
        blocked = await guard.awrap_tool_call(_batch_request([_query("same")]), handler)
        assert blocked.status == "error"
        assert "Duplicate research query" in blocked.content

    @pytest.mark.asyncio
    async def test_turn_budget_forces_finalizing(self):
        guard = self._guard(max_orchestrator_turns=2)
        request = MagicMock()
        request.tools = []
        request.override = MagicMock(return_value=request)
        handler = MagicMock(return_value="response")
        for _ in range(3):
            guard.wrap_model_call(request, handler)
        assert guard.phase == "finalizing"

    def test_finalizing_withdraws_every_research_affordance(self):
        """Including the direct source tools — the adaptive guard never had to hide those."""
        guard = self._guard()
        guard._mark_finalizing("test")
        tools = [_make_tool(n) for n in ("run_research_batch", "think", "web_search_tool", "submit_final_report")]
        remaining = {t.name for t in guard._filter_tools(tools)}
        assert remaining == {"submit_final_report"}

    def test_active_phase_hides_nothing(self):
        guard = self._guard()
        tools = [_make_tool(n) for n in ("run_research_batch", "think", "web_search_tool")]
        assert len(guard._filter_tools(tools)) == 3

    @pytest.mark.asyncio
    async def test_disabled_guard_is_inert(self):
        guard = AutonomousOrchestratorLoopGuardMiddleware(
            config=_config(enabled=False, max_batch_calls=1),
            source_tool_names=frozenset({"web_search_tool"}),
        )
        handler = AsyncMock(return_value="notes")
        for _ in range(5):
            await guard.awrap_tool_call(_batch_request([_query("q")]), handler)
        assert handler.await_count == 5

    @pytest.mark.asyncio
    async def test_guards_are_isolated_per_request(self):
        first, second = self._guard(max_batch_calls=1), self._guard(max_batch_calls=1)
        handler = AsyncMock(return_value="notes")
        await first.awrap_tool_call(_batch_request([_query("q")]), handler)
        result = await second.awrap_tool_call(_batch_request([_query("q")]), handler)
        assert getattr(result, "status", None) != "error"

    # --- the orchestrator's own direct source calls ------------------------------------------

    @pytest.mark.asyncio
    async def test_direct_source_budget_blocks_the_call_past_the_ceiling(self):
        guard = self._guard(max_direct_source_calls=2)
        handler = AsyncMock(return_value="results")
        await guard.awrap_tool_call(_source_request("a"), handler)
        await guard.awrap_tool_call(_source_request("b"), handler)
        blocked = await guard.awrap_tool_call(_source_request("c"), handler)
        assert blocked.status == "error"
        assert "Direct-search budget reached" in blocked.content
        assert handler.await_count == 2

    def test_spent_direct_budget_withdraws_sources_but_keeps_the_delegation_paths(self):
        """The point of the budget is to redirect research, not to end it."""
        guard = self._guard(max_direct_source_calls=1)
        guard._direct_source_call_count = 1
        tools = [_make_tool(n) for n in ("run_research_batch", "think", "web_search_tool", "task")]
        remaining = {t.name for t in guard._filter_tools(tools)}
        assert remaining == {"run_research_batch", "think", "task"}

    @pytest.mark.asyncio
    async def test_spending_the_direct_budget_does_not_finalize_the_request(self):
        guard = self._guard(max_direct_source_calls=1)
        handler = AsyncMock(return_value="results")
        await guard.awrap_tool_call(_source_request("a"), handler)
        assert guard.phase == "active"
        # ...and delegated research still runs afterwards.
        assert getattr(await guard.awrap_tool_call(_batch_request([_query("q")]), handler), "status", None) != "error"

    @pytest.mark.asyncio
    async def test_parallel_direct_calls_share_one_hard_ceiling(self):
        """Counting after the await would let a whole turn of parallel searches through."""
        guard = self._guard(max_direct_source_calls=2)
        handler = AsyncMock(return_value="results")
        results = await asyncio.gather(*(guard.awrap_tool_call(_source_request(f"q{i}"), handler) for i in range(5)))
        assert handler.await_count == 2
        assert sum(getattr(r, "status", None) == "error" for r in results) == 3

    @pytest.mark.asyncio
    async def test_repeated_direct_call_is_blocked_while_a_different_one_runs(self):
        guard = self._guard(max_direct_source_calls=5, max_identical_direct_source_calls=1)
        handler = AsyncMock(return_value="results")
        await guard.awrap_tool_call(_source_request("same"), handler)
        blocked = await guard.awrap_tool_call(_source_request("same"), handler)
        assert blocked.status == "error"
        assert "Duplicate direct search" in blocked.content
        assert getattr(await guard.awrap_tool_call(_source_request("other"), handler), "status", None) != "error"
        assert handler.await_count == 2

    @pytest.mark.asyncio
    async def test_a_direct_call_may_repeat_a_question_a_worker_already_researched(self):
        """Verification is the documented purpose of a direct call, so dedup must not cross paths."""
        guard = self._guard(max_direct_source_calls=2)
        handler = AsyncMock(return_value="results")
        await guard.awrap_tool_call(_batch_request([_query("apple inventory")]), handler)
        verification = await guard.awrap_tool_call(_source_request("apple inventory"), handler)
        assert getattr(verification, "status", None) != "error"

    @pytest.mark.asyncio
    async def test_the_last_allowed_direct_call_carries_the_delegation_nudge(self):
        guard = self._guard(max_direct_source_calls=1)
        handler = AsyncMock(return_value=ToolMessage(content="findings", tool_call_id="src_1", name="web_search_tool"))
        result = await guard.awrap_tool_call(_source_request("a"), handler)
        assert result.content.startswith("findings")
        assert "direct-search budget reached" in result.content
        assert "researcher-agent" in result.content

    @pytest.mark.asyncio
    async def test_a_non_pydantic_result_still_enforces_the_direct_budget(self):
        """A missing nudge is a soft degradation; the ceiling itself must not depend on it."""
        guard = self._guard(max_direct_source_calls=1)
        handler = AsyncMock(return_value="a plain string")
        assert await guard.awrap_tool_call(_source_request("a"), handler) == "a plain string"
        assert (await guard.awrap_tool_call(_source_request("b"), handler)).status == "error"

    @pytest.mark.asyncio
    async def test_helper_tools_are_never_budgeted(self):
        guard = self._guard(max_direct_source_calls=1)
        handler = AsyncMock(return_value="ok")
        for name in ("think", "get_verified_sources", "write_todos", "read_file", "submit_final_report"):
            await guard.awrap_tool_call(_request({"name": name, "args": {}, "id": "h"}), handler)
        assert handler.await_count == 5
        assert guard._direct_source_call_count == 0

    # --- depth clamping -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_high_depth_queries_past_the_allowance_are_clamped_not_rejected(self):
        guard = self._guard(max_high_depth_queries=1)
        handler = AsyncMock(return_value="notes")
        request = _batch_request([_query("a", depth="high"), _query("b", depth="high"), _query("c", depth="low")])
        result = await guard.awrap_tool_call(request, handler)
        assert getattr(result, "status", None) != "error"
        assert [q["depth"] for q in _forwarded_queries(handler)] == ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_the_high_depth_allowance_spans_the_whole_request(self):
        guard = self._guard(max_high_depth_queries=1)
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("a", depth="high")]), handler)
        await guard.awrap_tool_call(_batch_request([_query("b", depth="high")]), handler)
        assert [q["depth"] for q in _forwarded_queries(handler)] == ["medium"]

    @pytest.mark.asyncio
    async def test_a_batch_within_the_allowance_is_forwarded_untouched(self):
        """No clamp means no rewrite, so the original request object reaches the handler."""
        guard = self._guard(max_high_depth_queries=2)
        handler = AsyncMock(return_value="notes")
        request = _batch_request([_query("a", depth="high"), _query("b", depth="low")])
        await guard.awrap_tool_call(request, handler)
        assert handler.await_args.args[0] is request
        assert request.forwarded is None

    @pytest.mark.asyncio
    async def test_a_rejected_batch_does_not_consume_the_high_depth_allowance(self):
        guard = self._guard(max_total_research_queries=1, max_high_depth_queries=1)
        handler = AsyncMock(return_value="notes")
        blocked = await guard.awrap_tool_call(
            _batch_request([_query("a", depth="high"), _query("b", depth="high")]), handler
        )
        assert blocked.status == "error"
        assert guard._high_depth_query_count == 0

    # --- researcher-agent delegation shares the research budget -------------------------------

    @pytest.mark.asyncio
    async def test_researcher_delegation_spends_the_same_budget_as_a_batch(self):
        """One capability behind two doors; budgeting one leaves the other a free escape hatch."""
        guard = self._guard(max_batch_calls=2)
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("q")]), handler)
        await guard.awrap_tool_call(_task_request("researcher-agent"), handler)
        blocked = await guard.awrap_tool_call(_task_request("researcher-agent"), handler)
        assert blocked.status == "error"
        assert "Research budget reached" in blocked.content
        assert guard.phase == "finalizing"

    @pytest.mark.asyncio
    async def test_researcher_delegation_is_refused_once_finalizing(self):
        guard = self._guard()
        guard._mark_finalizing("test")
        handler = AsyncMock(return_value="notes")
        blocked = await guard.awrap_tool_call(_task_request("researcher-agent"), handler)
        assert blocked.status == "error"
        assert handler.await_count == 0

    @pytest.mark.asyncio
    async def test_planner_and_writer_delegation_are_never_budgeted_or_blocked(self):
        """`task(writer-agent)` is one of the two ways a run legitimately ends."""
        guard = self._guard(max_batch_calls=1)
        guard._mark_finalizing("test")
        handler = AsyncMock(return_value="done")
        for subagent in ("planner-agent", "writer-agent"):
            result = await guard.awrap_tool_call(_task_request(subagent), handler)
            assert getattr(result, "status", None) != "error"
        assert guard._batch_call_count == 0

    def test_the_task_tool_is_never_withdrawn(self):
        guard = self._guard()
        guard._mark_finalizing("test")
        tools = [_make_tool(n) for n in ("task", "run_research_batch", "submit_final_report")]
        assert "task" in {t.name for t in guard._filter_tools(tools)}

    # --- duplicate detection: four defects found in review, each pinned here ------------------

    @pytest.mark.asyncio
    async def test_clamping_cannot_launder_a_repeat_into_a_new_query(self):
        """Regression: `depth` in the signature made the clamp itself a duplicate bypass.

        The same `high` query in a second batch was clamped to `medium`, hashed differently from
        its own earlier run, and executed again. The request-wide signature now excludes `depth`.
        """
        guard = self._guard(max_high_depth_queries=1)
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("X", depth="high")]), handler)
        blocked = await guard.awrap_tool_call(_batch_request([_query("X", depth="high")]), handler)
        assert blocked.status == "error"
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_asking_the_same_question_at_a_greater_depth_is_still_a_repeat(self):
        guard = self._guard()
        handler = AsyncMock(return_value="notes")
        await guard.awrap_tool_call(_batch_request([_query("X", depth="low")]), handler)
        blocked = await guard.awrap_tool_call(_batch_request([_query("X", depth="high")]), handler)
        assert blocked.status == "error"
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_a_batch_repeating_a_query_inside_itself_is_blocked(self):
        """Regression: signatures were all checked against the ledger before any were recorded,
        so two copies in one batch each saw a count of zero and both went through."""
        guard = self._guard(max_identical_research_queries=1)
        handler = AsyncMock(return_value="notes")
        blocked = await guard.awrap_tool_call(_batch_request([_query("Y"), _query("Y")]), handler)
        assert blocked.status == "error"
        assert handler.await_count == 0
        assert not guard._query_signature_counts

    @pytest.mark.asyncio
    async def test_distinct_queries_in_one_batch_are_unaffected(self):
        guard = self._guard(max_identical_research_queries=1)
        handler = AsyncMock(return_value="notes")
        result = await guard.awrap_tool_call(_batch_request([_query("a"), _query("b")]), handler)
        assert getattr(result, "status", None) != "error"
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_direct_call_duplicates_are_matched_on_normalized_text(self):
        """Regression: the signature sorted JSON keys but not string values, so `"same"` and
        `"  SAME  "` hashed differently and both ran."""
        guard = self._guard(max_direct_source_calls=5, max_identical_direct_source_calls=1)
        handler = AsyncMock(return_value="results")
        await guard.awrap_tool_call(_source_request("same"), handler)
        for variant in ("  SAME  ", "Same", "same\n"):
            blocked = await guard.awrap_tool_call(_source_request(variant), handler)
            assert blocked.status == "error", variant
        assert handler.await_count == 1


class TestRequestTerminationConfig:
    def test_finalizer_deadline_must_be_below_the_workflow_deadline(self):
        with pytest.raises(ValueError, match="strictly less than"):
            AutonomousRequestTerminationConfig(workflow_timeout_seconds=60, fallback_finalizer_timeout_seconds=60)

    def test_there_is_no_per_tier_budget_lookup(self):
        config = AutonomousRequestTerminationConfig()
        assert not hasattr(config, "budgets_for_tier")
        assert not hasattr(config, "standard")
        assert not hasattr(config, "deep")

    def test_defaults_encode_the_intended_contract(self):
        config = AutonomousRequestTerminationConfig()
        # max_batch_calls stays a runaway backstop (it never bound in the 90-task job); the two
        # budgets that actually shape a run are the direct-search cap and the `high` allowance.
        assert config.max_batch_calls == 6
        assert config.max_direct_source_calls == 2
        assert config.max_identical_direct_source_calls == 1
        assert config.max_high_depth_queries == 1

    @pytest.mark.parametrize(
        "field",
        ["max_batch_calls", "max_direct_source_calls", "max_identical_direct_source_calls", "max_high_depth_queries"],
    )
    def test_budgets_must_be_positive(self, field):
        with pytest.raises(ValueError):
            AutonomousRequestTerminationConfig(**{field: 0})

    def test_unknown_budget_fields_are_rejected(self):
        with pytest.raises(ValueError):
            AutonomousRequestTerminationConfig(max_direct_searches=2)

    def test_config_is_frozen(self):
        config = AutonomousRequestTerminationConfig()
        with pytest.raises(ValueError):
            config.max_direct_source_calls = 99


def test_finalization_retry_marker_matches_the_upstream_convention():
    """A retry injected by this agent must not be mistaken for user input by other middleware."""
    middleware = AutonomousFinalizationMiddleware(tracker=AutonomousFinalReportCommitTracker())
    result = middleware._check_after_model({"messages": [AIMessage(content="x")], "files": {}})
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].additional_kwargs["aiq_generated_retry"] == "autonomous_finalization"
