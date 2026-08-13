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
    return _request({"name": "run_research_batch", "args": {"queries": queries}, "id": "call_1"})


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


class TestRequestTerminationConfig:
    def test_finalizer_deadline_must_be_below_the_workflow_deadline(self):
        with pytest.raises(ValueError, match="strictly less than"):
            AutonomousRequestTerminationConfig(workflow_timeout_seconds=60, fallback_finalizer_timeout_seconds=60)

    def test_there_is_no_per_tier_budget_lookup(self):
        config = AutonomousRequestTerminationConfig()
        assert not hasattr(config, "budgets_for_tier")
        assert not hasattr(config, "standard")
        assert not hasattr(config, "deep")


def test_finalization_retry_marker_matches_the_upstream_convention():
    """A retry injected by this agent must not be mistaken for user input by other middleware."""
    middleware = AutonomousFinalizationMiddleware(tracker=AutonomousFinalReportCommitTracker())
    result = middleware._check_after_model({"messages": [AIMessage(content="x")], "files": {}})
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].additional_kwargs["aiq_generated_retry"] == "autonomous_finalization"
