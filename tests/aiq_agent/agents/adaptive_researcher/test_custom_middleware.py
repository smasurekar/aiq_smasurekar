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

"""Tests for adaptive researcher middleware and loop guards."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from aiq_agent.agents.adaptive_researcher.custom_middleware import _DECLARE_EFFORT_TIER_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _DEFAULT_SINGLE_SHOT_SEARCH_BUDGET
from aiq_agent.agents.adaptive_researcher.custom_middleware import _MAX_FORCED_RETURN_MODEL_CALLS
from aiq_agent.agents.adaptive_researcher.custom_middleware import _RESEARCHER_BUDGET_NUDGE
from aiq_agent.agents.adaptive_researcher.custom_middleware import _RUN_RESEARCH_BATCH_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _SINGLE_SHOT_BUDGET_NUDGE
from aiq_agent.agents.adaptive_researcher.custom_middleware import _THINK_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import ComplexityRouterMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import ConsecutiveThinkGuardMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import ResearcherForcedReturnExhausted
from aiq_agent.agents.adaptive_researcher.custom_middleware import ResearcherLoopGuardMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import _canonical_source_signature
from aiq_agent.agents.adaptive_researcher.models import ResearcherLoopGuardConfig
from aiq_agent.agents.adaptive_researcher.models import ResearcherSourceCallBudgets
from aiq_agent.agents.deep_researcher.researcher_context import CURRENT_RESEARCHER_GUARD_STATE
from aiq_agent.agents.deep_researcher.researcher_context import ResearcherRunGuardState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str):
    """Return a minimal mock tool with the given name."""
    t = MagicMock()
    t.name = name
    return t


def _declare_tier_request(tier: str):
    """Build a mock tool-call request for declare_effort_tier."""
    req = MagicMock()
    req.tool_call = {"name": _DECLARE_EFFORT_TIER_TOOL, "args": {"tier": tier}}
    return req


def _other_tool_request(tool_name: str = "some_tool"):
    """Build a mock tool-call request for a non-declare tool."""
    req = MagicMock()
    req.tool_call = {"name": tool_name, "args": {}}
    return req


def _think_request():
    """Build a mock tool-call request for the think tool."""
    req = MagicMock()
    req.tool_call = {"name": _THINK_TOOL, "args": {"thought": "some thought"}}
    return req


class _FakeModelRequest:
    """Minimal stand-in for LangChain's ``ModelRequest`` with immutable ``override`` semantics.

    Only the attributes the loop guard reads are modelled. ``override`` returns a *new* instance,
    matching ``dataclasses.replace``, so a test can assert the original request was not mutated.
    """

    def __init__(self, *, tools=None, messages=None, response_format=object(), tool_choice=None):
        self.tools = list(tools or [])
        self.messages = list(messages or [])
        self.response_format = response_format
        self.tool_choice = tool_choice

    def override(self, **overrides):
        replacement = _FakeModelRequest(
            tools=self.tools,
            messages=self.messages,
            response_format=self.response_format,
            tool_choice=self.tool_choice,
        )
        for key, value in overrides.items():
            setattr(replacement, key, value)
        return replacement


def _model_response(structured_response=None):
    """Return a stand-in ``ModelResponse`` carrying only the field the guard inspects."""
    response = MagicMock()
    response.structured_response = structured_response
    return response


def _tool_message(content: str = "Thought recorded."):
    """Return a mock ToolMessage whose model_copy produces a new mock with updated content."""
    msg = MagicMock()
    msg.content = content

    def _model_copy(*, update=None):
        new_msg = MagicMock()
        new_msg.content = (update or {}).get("content", content)
        return new_msg

    msg.model_copy = _model_copy
    return msg


# ---------------------------------------------------------------------------
# awrap_tool_call — tier caching
# ---------------------------------------------------------------------------


class TestAwrapToolCallTierCaching:
    def setup_method(self):
        self.mw = ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            direct_source_tools=[_make_tool("web_search_tool")],
            single_loop_single_shot=True,
        )

    @pytest.mark.asyncio
    async def test_caches_single_shot_tier(self):
        handler = AsyncMock(return_value="result")
        await self.mw.awrap_tool_call(_declare_tier_request("single_shot"), handler)
        assert self.mw._declared_tier == "single_shot"

    @pytest.mark.asyncio
    async def test_caches_standard_tier(self):
        handler = AsyncMock(return_value="result")
        await self.mw.awrap_tool_call(_declare_tier_request("standard"), handler)
        assert self.mw._declared_tier == "standard"

    @pytest.mark.asyncio
    async def test_ignores_non_declare_tool_calls(self):
        handler = AsyncMock(return_value="result")
        await self.mw.awrap_tool_call(_other_tool_request("run_research_batch"), handler)
        assert self.mw._declared_tier is None

    @pytest.mark.asyncio
    async def test_still_calls_handler(self):
        handler = AsyncMock(return_value="tool_result")
        result = await self.mw.awrap_tool_call(_declare_tier_request("single_shot"), handler)
        handler.assert_awaited_once()
        assert result == "tool_result"

    def test_declared_tier_starts_none(self):
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            single_loop_single_shot=True,
        )
        assert mw._declared_tier is None


# ---------------------------------------------------------------------------
# _filter_tools — static ceiling behavior (single_loop_single_shot=False)
# ---------------------------------------------------------------------------


class TestFilterToolsStaticCeiling:
    def setup_method(self):
        self.tools = [
            _make_tool("run_research_batch"),
            _make_tool("advanced_web_search_tool"),
            _make_tool("task"),
            _make_tool("write_todos"),
            _make_tool("get_verified_sources"),
        ]

    def test_deep_ceiling_hides_nothing(self):
        mw = ComplexityRouterMiddleware(enabled_tiers=["direct", "single_shot", "standard", "deep"])
        result_names = {t.name for t in mw._filter_tools(self.tools)}
        assert result_names == {t.name for t in self.tools}

    def test_standard_ceiling_hides_advanced_web_search(self):
        mw = ComplexityRouterMiddleware(enabled_tiers=["direct", "single_shot", "standard"])
        result_names = {t.name for t in mw._filter_tools(self.tools)}
        assert "advanced_web_search_tool" not in result_names
        assert "run_research_batch" in result_names
        assert "task" in result_names

    def test_single_shot_ceiling_hides_delegation_tools(self):
        mw = ComplexityRouterMiddleware(enabled_tiers=["direct", "single_shot"])
        result_names = {t.name for t in mw._filter_tools(self.tools)}
        assert "advanced_web_search_tool" not in result_names
        assert "task" not in result_names
        assert "write_todos" not in result_names
        assert "run_research_batch" in result_names

    def test_allow_delegation_preserves_task_tools(self):
        mw = ComplexityRouterMiddleware(enabled_tiers=["single_shot"], allow_delegation=True)
        result_names = {t.name for t in mw._filter_tools(self.tools)}
        assert "task" in result_names
        assert "write_todos" in result_names


# ---------------------------------------------------------------------------
# _filter_tools — single-shot swap (single_loop_single_shot=True)
# ---------------------------------------------------------------------------


class TestFilterToolsSingleLoopSwap:
    def setup_method(self):
        self.direct_source_tools = [
            _make_tool("web_search_tool"),
            _make_tool("knowledge_search_tool"),
        ]
        self.orchestrator_tools = [
            _make_tool("run_research_batch"),
            _make_tool("get_verified_sources"),
            _make_tool("declare_effort_tier"),
            _make_tool("submit_final_report"),
            _make_tool("web_search_tool"),
            _make_tool("knowledge_search_tool"),
        ]

    def _mw(self, declared_tier: str | None = None) -> ComplexityRouterMiddleware:
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            direct_source_tools=self.direct_source_tools,
            single_loop_single_shot=True,
        )
        mw._declared_tier = declared_tier
        return mw

    def test_undeclared_hides_source_tools_keeps_run_research_batch(self):
        mw = self._mw(declared_tier=None)
        result_names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "web_search_tool" not in result_names
        assert "knowledge_search_tool" not in result_names
        assert _RUN_RESEARCH_BATCH_TOOL in result_names

    def test_single_shot_removes_run_research_batch(self):
        mw = self._mw(declared_tier="single_shot")
        result_names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert _RUN_RESEARCH_BATCH_TOOL not in result_names

    def test_single_shot_keeps_source_tools(self):
        mw = self._mw(declared_tier="single_shot")
        result_names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "web_search_tool" in result_names
        assert "knowledge_search_tool" in result_names

    def test_single_shot_keeps_helper_tools(self):
        mw = self._mw(declared_tier="single_shot")
        result_names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "get_verified_sources" in result_names
        assert "submit_final_report" in result_names

    def test_standard_tier_hides_source_tools(self):
        mw = self._mw(declared_tier="standard")
        result_names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "web_search_tool" not in result_names
        assert "knowledge_search_tool" not in result_names
        assert _RUN_RESEARCH_BATCH_TOOL in result_names

    def test_deep_tier_hides_source_tools(self):
        mw = self._mw(declared_tier="deep")
        result_names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "web_search_tool" not in result_names
        assert _RUN_RESEARCH_BATCH_TOOL in result_names


# ---------------------------------------------------------------------------
# Integration: awrap_tool_call → awrap_model_call sequence
# ---------------------------------------------------------------------------


class TestTierCachingIntegration:
    """Verify the full sequence: capture tier from tool call, then apply to model call."""

    @pytest.mark.asyncio
    async def test_full_sequence_single_shot(self):
        """After declare_effort_tier(single_shot) fires, the next model call removes run_research_batch."""
        direct_tools = [_make_tool("web_search_tool")]
        all_tools = [
            _make_tool("run_research_batch"),
            _make_tool("web_search_tool"),
            _make_tool("get_verified_sources"),
        ]

        mw = ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            direct_source_tools=direct_tools,
            single_loop_single_shot=True,
        )

        # Simulate declare_effort_tier tool call firing
        tool_handler = AsyncMock(return_value="Tier recorded.")
        await mw.awrap_tool_call(_declare_tier_request("single_shot"), tool_handler)
        assert mw._declared_tier == "single_shot"

        # Simulate next model call — run_research_batch must be absent
        model_request = MagicMock()
        model_request.tools = all_tools
        model_request.override = MagicMock(return_value=model_request)
        model_handler = AsyncMock(return_value=MagicMock())

        await mw.awrap_model_call(model_request, model_handler)

        called_tools = model_request.override.call_args.kwargs["tools"]
        result_names = {t.name for t in called_tools}
        assert "run_research_batch" not in result_names
        assert "web_search_tool" in result_names

    @pytest.mark.asyncio
    async def test_full_sequence_standard_keeps_run_research_batch(self):
        """After declare_effort_tier(standard) fires, the next model call keeps run_research_batch."""
        direct_tools = [_make_tool("web_search_tool")]
        all_tools = [_make_tool("run_research_batch"), _make_tool("web_search_tool")]

        mw = ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            direct_source_tools=direct_tools,
            single_loop_single_shot=True,
        )

        tool_handler = AsyncMock(return_value="Tier recorded.")
        await mw.awrap_tool_call(_declare_tier_request("standard"), tool_handler)

        model_request = MagicMock()
        model_request.tools = all_tools
        model_request.override = MagicMock(return_value=model_request)
        model_handler = AsyncMock(return_value=MagicMock())

        await mw.awrap_model_call(model_request, model_handler)

        called_tools = model_request.override.call_args.kwargs["tools"]
        result_names = {t.name for t in called_tools}
        assert "run_research_batch" in result_names
        assert "web_search_tool" not in result_names


# ---------------------------------------------------------------------------
# Dynamic per-tier prompt swap (prompt_renderer)
# ---------------------------------------------------------------------------


class TestDynamicPromptSwap:
    """When a prompt_renderer is supplied, the middleware swaps the system message to the
    declared tier's prompt on model calls; without one it only ever overrides tools.
    """

    def _model_request(self):
        req = MagicMock()
        req.tools = [_make_tool("run_research_batch")]
        req.override = MagicMock(return_value=req)
        return req

    @pytest.mark.asyncio
    async def test_no_swap_before_tier_declared(self):
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["single_shot", "deep"],
            prompt_renderer=lambda tier: f"PROMPT[{tier}]",
        )
        req = self._model_request()
        await mw.awrap_model_call(req, AsyncMock(return_value=MagicMock()))
        # tier unknown on turn 1 -> tools filtered but system prompt left intact
        assert "system_message" not in req.override.call_args.kwargs

    @pytest.mark.asyncio
    async def test_swaps_system_message_after_declaration(self):
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["single_shot", "deep"],
            prompt_renderer=lambda tier: f"PROMPT[{tier}]",
        )
        await mw.awrap_tool_call(_declare_tier_request("single_shot"), AsyncMock(return_value="ok"))
        req = self._model_request()
        await mw.awrap_model_call(req, AsyncMock(return_value=MagicMock()))
        swapped = req.override.call_args.kwargs["system_message"]
        assert swapped.content == "PROMPT[single_shot]"

    @pytest.mark.asyncio
    async def test_render_is_memoized_per_tier(self):
        calls = []

        def renderer(tier):
            calls.append(tier)
            return f"PROMPT[{tier}]"

        mw = ComplexityRouterMiddleware(enabled_tiers=["single_shot", "deep"], prompt_renderer=renderer)
        await mw.awrap_tool_call(_declare_tier_request("single_shot"), AsyncMock(return_value="ok"))
        for _ in range(3):
            await mw.awrap_model_call(self._model_request(), AsyncMock(return_value=MagicMock()))
        assert calls == ["single_shot"]  # rendered once, reused across model calls

    @pytest.mark.asyncio
    async def test_escalation_rerenders_for_new_tier(self):
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["single_shot", "deep"],
            prompt_renderer=lambda tier: f"PROMPT[{tier}]",
        )
        await mw.awrap_tool_call(_declare_tier_request("single_shot"), AsyncMock(return_value="ok"))
        req1 = self._model_request()
        await mw.awrap_model_call(req1, AsyncMock(return_value=MagicMock()))
        assert req1.override.call_args.kwargs["system_message"].content == "PROMPT[single_shot]"
        # model steps up and re-declares -> next model call renders the deeper prompt
        await mw.awrap_tool_call(_declare_tier_request("deep"), AsyncMock(return_value="ok"))
        req2 = self._model_request()
        await mw.awrap_model_call(req2, AsyncMock(return_value=MagicMock()))
        assert req2.override.call_args.kwargs["system_message"].content == "PROMPT[deep]"

    @pytest.mark.asyncio
    async def test_no_renderer_never_swaps_prompt(self):
        # Default (flag off / other wiring reasons): tools-only override, exactly as before.
        mw = ComplexityRouterMiddleware(enabled_tiers=["single_shot", "deep"])
        await mw.awrap_tool_call(_declare_tier_request("single_shot"), AsyncMock(return_value="ok"))
        req = self._model_request()
        await mw.awrap_model_call(req, AsyncMock(return_value=MagicMock()))
        assert "system_message" not in req.override.call_args.kwargs


# ---------------------------------------------------------------------------
# Single-shot search budget (hard cap on direct source-tool calls)
# ---------------------------------------------------------------------------


class TestSingleShotSearchBudget:
    """The single_loop_single_shot single_shot path caps direct source-tool calls.

    Once the budget is spent the middleware (a) appends a finalize nudge to the search result
    that spent it and (b) withdraws the source tools from later model calls. standard / deep,
    and any run before single_shot is declared, are never counted or capped.
    """

    def setup_method(self):
        # Two configured source tools; the orchestrator also holds helper / finalize tools.
        self.direct_source_tools = [
            _make_tool("knowledge_search"),
            _make_tool("web_search_tool"),
        ]
        self.orchestrator_tools = [
            _make_tool("run_research_batch"),
            _make_tool("get_verified_sources"),
            _make_tool("declare_effort_tier"),
            _make_tool("submit_final_report"),
            _make_tool("knowledge_search"),
            _make_tool("web_search_tool"),
        ]

    def _mw(self, *, budget: int = 2, declared_tier: str | None = "single_shot") -> ComplexityRouterMiddleware:
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            direct_source_tools=self.direct_source_tools,
            single_loop_single_shot=True,
            single_shot_search_budget=budget,
        )
        mw._declared_tier = declared_tier
        return mw

    # --- counting -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_counter_increments_on_source_tool_call(self):
        mw = self._mw(budget=5)
        handler = AsyncMock(return_value=_tool_message("search results"))
        await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
        assert mw._source_call_count == 1

    @pytest.mark.asyncio
    async def test_counter_ignores_helper_tools(self):
        mw = self._mw(budget=5)
        handler = AsyncMock(return_value=_tool_message("sources"))
        await mw.awrap_tool_call(_other_tool_request("get_verified_sources"), handler)
        assert mw._source_call_count == 0

    @pytest.mark.asyncio
    async def test_counter_ignores_before_single_shot_declared(self):
        mw = self._mw(budget=5, declared_tier=None)
        handler = AsyncMock(return_value=_tool_message("search results"))
        await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
        assert mw._source_call_count == 0

    @pytest.mark.asyncio
    async def test_counter_ignores_on_standard_tier(self):
        mw = self._mw(budget=5, declared_tier="standard")
        handler = AsyncMock(return_value=_tool_message("search results"))
        await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
        assert mw._source_call_count == 0

    @pytest.mark.asyncio
    async def test_handler_always_called_for_source_call(self):
        mw = self._mw(budget=2)
        handler = AsyncMock(return_value=_tool_message("search results"))
        await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
        handler.assert_awaited_once()

    # --- nudge --------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_nudge_appended_at_threshold_preserves_content(self):
        mw = self._mw(budget=2)
        handler = AsyncMock(return_value=_tool_message("original retrieval"))
        await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)  # 1/2
        result = await mw.awrap_tool_call(_other_tool_request("web_search_tool"), handler)  # 2/2
        # Original evidence preserved (appended, not overwritten) + nudge present.
        assert result.content.startswith("original retrieval")
        assert _SINGLE_SHOT_BUDGET_NUDGE in result.content

    @pytest.mark.asyncio
    async def test_no_nudge_below_threshold(self):
        mw = self._mw(budget=2)
        handler = AsyncMock(return_value=_tool_message("original retrieval"))
        result = await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)  # 1/2
        assert result.content == "original retrieval"

    @pytest.mark.asyncio
    async def test_graceful_on_immutable_result_at_threshold(self):
        """If model_copy raises, the budget is still enforced via _filter_tools; return original."""
        mw = self._mw(budget=1)
        msg = MagicMock()
        msg.content = "original retrieval"
        msg.model_copy = MagicMock(side_effect=AttributeError("immutable"))
        handler = AsyncMock(return_value=msg)
        result = await mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
        assert result is msg

    # --- tool hiding --------------------------------------------------------

    def test_filter_keeps_source_tools_within_budget(self):
        mw = self._mw(budget=2)
        mw._source_call_count = 1
        names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "knowledge_search" in names
        assert "web_search_tool" in names
        assert _RUN_RESEARCH_BATCH_TOOL not in names

    def test_filter_hides_source_tools_after_budget(self):
        mw = self._mw(budget=2)
        mw._source_call_count = 2
        names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "knowledge_search" not in names
        assert "web_search_tool" not in names
        assert _RUN_RESEARCH_BATCH_TOOL not in names

    def test_filter_keeps_finalize_tools_after_budget(self):
        mw = self._mw(budget=2)
        mw._source_call_count = 2
        names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "get_verified_sources" in names
        assert "submit_final_report" in names

    def test_custom_budget_of_one_hides_after_single_call(self):
        mw = self._mw(budget=1)
        mw._source_call_count = 1
        names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert "knowledge_search" not in names

    def test_standard_tier_not_capped_by_budget(self):
        """standard keeps run_research_batch regardless of any source-call count."""
        mw = self._mw(budget=2, declared_tier="standard")
        mw._source_call_count = 99
        names = {t.name for t in mw._filter_tools(self.orchestrator_tools)}
        assert _RUN_RESEARCH_BATCH_TOOL in names
        assert "knowledge_search" not in names

    def test_source_call_count_starts_at_zero(self):
        mw = self._mw()
        assert mw._source_call_count == 0

    def test_default_budget_constant(self):
        # Sanity: the wired default matches the documented constant (2).
        mw = ComplexityRouterMiddleware(
            enabled_tiers=["single_shot"],
            direct_source_tools=self.direct_source_tools,
            single_loop_single_shot=True,
        )
        assert mw._search_budget == _DEFAULT_SINGLE_SHOT_SEARCH_BUDGET


# ---------------------------------------------------------------------------
# ResearcherLoopGuardMiddleware
# ---------------------------------------------------------------------------


class TestResearcherLoopGuardMiddleware:
    def setup_method(self):
        self.config = ResearcherLoopGuardConfig(
            source_call_budgets=ResearcherSourceCallBudgets(low=1, medium=3, high=6),
            max_identical_source_calls=2,
            max_consecutive_thinks=2,
            max_consecutive_blocked_source_calls=3,
        )
        self.mw = ResearcherLoopGuardMiddleware(
            source_tool_names={"knowledge_search", "web_search_tool"},
            config=self.config,
        )
        self.state = ResearcherRunGuardState(invocation_id="test-run", depth="medium")
        self.token = CURRENT_RESEARCHER_GUARD_STATE.set(self.state)

    def teardown_method(self):
        CURRENT_RESEARCHER_GUARD_STATE.reset(self.token)

    @pytest.mark.asyncio
    async def test_depth_budget_exhausts_and_withdraws_source_and_think_tools(self):
        self.state.depth = "low"
        handler = AsyncMock(return_value=_tool_message("source evidence"))

        result = await self.mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)

        assert result.content == f"source evidence{_RESEARCHER_BUDGET_NUDGE}"
        assert self.state.source_call_count == 1
        assert self.state.exhausted is True
        names = {
            tool.name
            for tool in self.mw._filter_tools(
                [_make_tool("knowledge_search"), _make_tool(_THINK_TOOL), _make_tool("get_verified_sources")]
            )
        }
        assert names == {"get_verified_sources"}

    @pytest.mark.parametrize(("depth", "budget"), [("low", 1), ("medium", 3), ("high", 6)])
    @pytest.mark.asyncio
    async def test_each_depth_enforces_its_configured_budget(self, depth, budget):
        self.state.depth = depth
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        result = None
        for index in range(budget):
            request = _other_tool_request("knowledge_search")
            request.tool_call["args"] = {"query": f"distinct-{index}"}
            result = await self.mw.awrap_tool_call(request, handler)

        assert handler.await_count == budget
        assert self.state.source_call_count == budget
        assert self.state.exhausted is True
        assert _RESEARCHER_BUDGET_NUDGE in result.content

        blocked = _other_tool_request("knowledge_search")
        blocked.tool_call["args"] = {"query": "one-too-many"}
        blocked_result = await self.mw.awrap_tool_call(blocked, handler)
        assert handler.await_count == budget
        assert "not executed" in blocked_result.content

    @pytest.mark.asyncio
    async def test_immutable_final_result_still_exhausts_budget(self):
        self.state.depth = "low"
        result = MagicMock()
        result.content = "source evidence"
        result.model_copy.side_effect = AttributeError("immutable")
        handler = AsyncMock(return_value=result)

        returned = await self.mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)

        assert returned is result
        assert self.state.exhausted is True
        assert self.mw._filter_tools([_make_tool("knowledge_search")]) == []

    @pytest.mark.asyncio
    async def test_third_identical_source_call_is_rejected_without_exhausting_the_worker(self):
        """R6: a repeat costs one rejected signature, not the whole worker."""
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        request = _other_tool_request("knowledge_search")
        request.tool_call["args"] = {"query": "same query", "filters": {"b": 2, "a": 1}}
        await self.mw.awrap_tool_call(request, handler)
        await self.mw.awrap_tool_call(request, handler)
        result = await self.mw.awrap_tool_call(request, handler)

        assert handler.await_count == 2
        assert self.state.source_call_count == 2
        assert "repeated source-call limit" in result.content
        # The rejection tells the model to vary the query, not to stop researching.
        assert "materially different query" in result.content
        assert self.state.exhausted is False
        assert self.state.exhaustion_reason is None
        assert self.state.blocked_source_calls == 1
        assert self.state.consecutive_blocked_source_calls == 1
        # Source tools stay visible: the worker still has budget for a different query.
        names = {tool.name for tool in self.mw._filter_tools([_make_tool("knowledge_search")])}
        assert names == {"knowledge_search"}

    @pytest.mark.asyncio
    async def test_repeated_signature_does_not_consume_remaining_budget(self):
        """After a rejected repeat, a materially different query still executes."""
        # `high` (budget 6) leaves headroom, so nothing here can trip the budget rule instead.
        self.state.depth = "high"
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        repeated = _other_tool_request("knowledge_search")
        repeated.tool_call["args"] = {"query": "same query"}
        for _ in range(3):
            await self.mw.awrap_tool_call(repeated, handler)

        distinct = _other_tool_request("knowledge_search")
        distinct.tool_call["args"] = {"query": "a different query"}
        result = await self.mw.awrap_tool_call(distinct, handler)

        assert handler.await_count == 3
        assert result.content == "source evidence"
        assert self.state.source_call_count == 3
        assert self.state.exhausted is False

    @pytest.mark.asyncio
    async def test_executed_source_call_resets_the_blocked_run(self):
        """Only real progress clears the run, so block/execute/block never reaches the ceiling."""
        # `high` (budget 6) leaves headroom, so nothing here can trip the budget rule instead.
        self.state.depth = "high"
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        repeated = _other_tool_request("knowledge_search")
        repeated.tool_call["args"] = {"query": "same query"}
        for _ in range(2):
            await self.mw.awrap_tool_call(repeated, handler)
        await self.mw.awrap_tool_call(repeated, handler)
        await self.mw.awrap_tool_call(repeated, handler)
        assert self.state.consecutive_blocked_source_calls == 2

        distinct = _other_tool_request("knowledge_search")
        distinct.tool_call["args"] = {"query": "a different query"}
        await self.mw.awrap_tool_call(distinct, handler)
        assert self.state.consecutive_blocked_source_calls == 0

        await self.mw.awrap_tool_call(repeated, handler)
        await self.mw.awrap_tool_call(repeated, handler)
        assert self.state.consecutive_blocked_source_calls == 2
        assert self.state.force_structured_return is False
        assert self.state.exhausted is False

    @pytest.mark.asyncio
    async def test_alternating_think_and_same_search_trips_the_blocked_ceiling(self):
        """Interleaved ``think`` must not launder the blocked run; the ceiling still fires."""
        think_guard = ConsecutiveThinkGuardMiddleware(max_consecutive_thinks=3)
        handler = AsyncMock(return_value=_tool_message("tool result"))

        async def guarded_call(request):
            async def invoke_think_guard(inner_request):
                return await think_guard.awrap_tool_call(inner_request, handler)

            return await self.mw.awrap_tool_call(request, invoke_think_guard)

        search = _other_tool_request("knowledge_search")
        search.tool_call["args"] = {"query": "same query"}
        for _ in range(2):
            await guarded_call(_think_request())
            await guarded_call(search)
            assert self.state.consecutive_think_count == 0

        result = None
        for _ in range(self.config.max_consecutive_blocked_source_calls):
            await guarded_call(_think_request())
            result = await guarded_call(search)

        assert self.state.source_call_count == 2
        assert self.state.consecutive_blocked_source_calls == 3
        assert self.state.exhausted is True
        assert self.state.exhaustion_reason == "consecutive blocked source calls"
        assert self.state.force_structured_return is True
        assert "not executed" in result.content

    @pytest.mark.asyncio
    async def test_ceiling_reports_its_own_reason_not_the_budget(self, caplog):
        """The regression itself: a non-budget trip must never be logged as ``reason=total_budget``."""
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        repeated = _other_tool_request("knowledge_search")
        repeated.tool_call["args"] = {"query": "same query"}
        with caplog.at_level("WARNING", logger="aiq_agent.agents.adaptive_researcher.custom_middleware"):
            for _ in range(5):
                await self.mw.awrap_tool_call(repeated, handler)

        assert self.state.source_call_count < self.config.source_call_budgets.for_depth(self.state.depth)
        assert "reason=total_budget" not in caplog.text
        assert "reason=repeated_signature" in caplog.text
        assert "forcing structured return" in caplog.text

    @pytest.mark.asyncio
    async def test_budget_exhaustion_still_reports_total_budget(self, caplog):
        """The other direction: a genuine budget trip keeps its own reason."""
        self.state.depth = "low"
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        with caplog.at_level("WARNING", logger="aiq_agent.agents.adaptive_researcher.custom_middleware"):
            first = _other_tool_request("knowledge_search")
            first.tool_call["args"] = {"query": "first"}
            await self.mw.awrap_tool_call(first, handler)
            second = _other_tool_request("knowledge_search")
            second.tool_call["args"] = {"query": "second"}
            await self.mw.awrap_tool_call(second, handler)

        assert self.state.exhaustion_reason == "total source-call budget"
        assert "reason=total_budget" in caplog.text

    def test_signature_is_stable_across_mapping_key_order(self):
        first = _canonical_source_signature("knowledge_search", {"query": "x", "filters": {"a": 1, "b": 2}})
        second = _canonical_source_signature("knowledge_search", {"filters": {"b": 2, "a": 1}, "query": "x"})
        assert first == second

    @pytest.mark.asyncio
    async def test_distinct_source_arguments_do_not_collide(self):
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        first = _other_tool_request("knowledge_search")
        first.tool_call["args"] = {"query": "first"}
        second = _other_tool_request("knowledge_search")
        second.tool_call["args"] = {"query": "second"}
        await self.mw.awrap_tool_call(first, handler)
        await self.mw.awrap_tool_call(second, handler)
        assert handler.await_count == 2
        assert len(self.state.source_signature_counts) == 2
        assert self.state.exhausted is False

    @pytest.mark.asyncio
    async def test_non_source_tools_are_not_counted(self):
        handler = AsyncMock(return_value=_tool_message("helper result"))
        result = await self.mw.awrap_tool_call(_other_tool_request("get_verified_sources"), handler)
        assert result.content == "helper result"
        assert self.state.source_call_count == 0

    @pytest.mark.asyncio
    async def test_parallel_calls_share_one_hard_budget(self):
        self.state.depth = "low"
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        first = _other_tool_request("knowledge_search")
        first.tool_call["args"] = {"query": "first"}
        second = _other_tool_request("web_search_tool")
        second.tool_call["args"] = {"query": "second"}
        results = await asyncio.gather(
            self.mw.awrap_tool_call(first, handler),
            self.mw.awrap_tool_call(second, handler),
        )
        assert handler.await_count == 1
        assert self.state.source_call_count == 1
        assert any("not executed" in result.content for result in results)

    @pytest.mark.asyncio
    async def test_context_state_is_isolated_between_concurrent_invocations(self):
        async def run(invocation_id: str):
            state = ResearcherRunGuardState(invocation_id=invocation_id, depth="low")
            token = CURRENT_RESEARCHER_GUARD_STATE.set(state)
            try:
                await asyncio.sleep(0)
                handler = AsyncMock(return_value=_tool_message("source evidence"))
                await self.mw.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
                return state
            finally:
                CURRENT_RESEARCHER_GUARD_STATE.reset(token)

        first, second = await asyncio.gather(run("first"), run("second"))
        assert first.invocation_id != second.invocation_id
        assert first.source_call_count == second.source_call_count == 1
        assert self.state.source_call_count == 0

    @pytest.mark.asyncio
    async def test_consecutive_think_limit_withdraws_think_for_researcher(self):
        think_guard = ConsecutiveThinkGuardMiddleware(max_consecutive_thinks=2)
        handler = AsyncMock(return_value=_tool_message())
        await think_guard.awrap_tool_call(_think_request(), handler)
        result = await think_guard.awrap_tool_call(_think_request(), handler)
        assert "WARNING" in result.content
        assert self.state.think_blocked is True
        names = {tool.name for tool in self.mw._filter_tools([_make_tool(_THINK_TOOL), _make_tool("knowledge_search")])}
        assert names == {"knowledge_search"}

    @pytest.mark.asyncio
    async def test_forced_return_binds_only_the_structured_output_tool(self):
        """The hard stop: an empty tool list leaves LangChain's ToolStrategy tool as the only option."""
        self.state.force_structured_return = True
        self.state.consecutive_blocked_source_calls = 3
        original_tools = [_make_tool("knowledge_search"), _make_tool(_THINK_TOOL)]
        original_messages = [HumanMessage(content="research this")]
        request = _FakeModelRequest(tools=original_tools, messages=original_messages)
        seen = {}

        async def handler(sent):
            seen["request"] = sent
            return _model_response(structured_response={"findings": []})

        await self.mw.awrap_model_call(request, handler)
        sent = seen["request"]

        assert sent.tools == []
        # response_format must pass through by identity so LangChain reuses the ToolStrategy built
        # at construction time; rebuilding it would rename the output tool the graph routes on.
        assert sent.response_format is request.response_format
        assert sent.tool_choice is request.tool_choice
        assert len(sent.messages) == len(original_messages) + 1
        assert isinstance(sent.messages[-1], HumanMessage)
        assert "source tools are now closed" in sent.messages[-1].content
        # The instruction rides on the request only; graph state must be untouched.
        assert request.messages == original_messages
        assert request.tools == original_tools
        assert self.state.forced_return_model_calls == 1

    @pytest.mark.asyncio
    async def test_forced_return_is_skipped_when_response_format_is_none(self):
        """StructuredResponseTextFallbackMiddleware owns its own tools-disabled corrective call."""
        # The ceiling sets both flags together, so reproduce that state exactly.
        self.state.force_structured_return = True
        self.state.exhausted = True
        request = _FakeModelRequest(tools=[_make_tool("knowledge_search")], response_format=None)
        seen = {}

        async def handler(sent):
            seen["request"] = sent
            return _model_response()

        await self.mw.awrap_model_call(request, handler)

        # Falls back to plain withdrawal; no instruction appended, no forced-return recorded.
        assert seen["request"].tools == []
        assert seen["request"].messages == request.messages
        assert self.state.forced_return_model_calls == 0

    @pytest.mark.asyncio
    async def test_forced_return_raises_only_after_max_attempts(self):
        """A model that ignores the forced return is stopped here, not by the recursion limit."""
        self.state.force_structured_return = True

        async def handler(_sent):
            return _model_response()

        for _ in range(_MAX_FORCED_RETURN_MODEL_CALLS - 1):
            await self.mw.awrap_model_call(_FakeModelRequest(), handler)
        assert self.state.forced_return_model_calls == _MAX_FORCED_RETURN_MODEL_CALLS - 1

        with pytest.raises(ResearcherForcedReturnExhausted):
            await self.mw.awrap_model_call(_FakeModelRequest(), handler)

    @pytest.mark.asyncio
    async def test_model_call_is_untouched_before_the_ceiling(self):
        """Nothing is forced while the worker is healthy."""
        request = _FakeModelRequest(tools=[_make_tool("knowledge_search")])
        seen = {}

        async def handler(sent):
            seen["request"] = sent
            return _model_response()

        await self.mw.awrap_model_call(request, handler)

        assert [tool.name for tool in seen["request"].tools] == ["knowledge_search"]
        assert seen["request"].messages == request.messages

    def test_filter_tools_counts_only_model_calls_it_reached(self, caplog):
        """The withdrawal counter is the evidence that the model was told; it must not overcount."""
        self.state.exhausted = True
        with caplog.at_level("INFO", logger="aiq_agent.agents.adaptive_researcher.custom_middleware"):
            self.mw._filter_tools([_make_tool("get_verified_sources")])
            assert self.state.tools_withdrawn_model_calls == 0

            self.mw._filter_tools([_make_tool("knowledge_search"), _make_tool("get_verified_sources")])
            assert self.state.tools_withdrawn_model_calls == 1

        assert "withdrew tools" in caplog.text
        assert "hidden=knowledge_search" in caplog.text

    @pytest.mark.asyncio
    async def test_disabled_guard_passes_through_without_counting(self):
        middleware = ResearcherLoopGuardMiddleware(
            source_tool_names={"knowledge_search"},
            config=ResearcherLoopGuardConfig(enabled=False),
        )
        handler = AsyncMock(return_value=_tool_message("source evidence"))
        result = await middleware.awrap_tool_call(_other_tool_request("knowledge_search"), handler)
        assert result.content == "source evidence"
        assert self.state.source_call_count == 0


# ---------------------------------------------------------------------------
# ConsecutiveThinkGuardMiddleware
# ---------------------------------------------------------------------------


class TestConsecutiveThinkGuardMiddleware:
    def setup_method(self):
        self.mw = ConsecutiveThinkGuardMiddleware(max_consecutive_thinks=3)

    @pytest.mark.asyncio
    async def test_counter_increments_on_think(self):
        handler = AsyncMock(return_value=_tool_message())
        await self.mw.awrap_tool_call(_think_request(), handler)
        assert self.mw._consecutive_think_count == 1

    @pytest.mark.asyncio
    async def test_counter_resets_on_non_think(self):
        handler = AsyncMock(return_value=_tool_message())
        await self.mw.awrap_tool_call(_think_request(), handler)
        await self.mw.awrap_tool_call(_think_request(), handler)
        await self.mw.awrap_tool_call(_other_tool_request("run_research_batch"), handler)
        assert self.mw._consecutive_think_count == 0

    @pytest.mark.asyncio
    async def test_no_modification_below_threshold(self):
        msg = _tool_message("Thought recorded.")
        handler = AsyncMock(return_value=msg)
        result = await self.mw.awrap_tool_call(_think_request(), handler)
        assert result.content == "Thought recorded."

    @pytest.mark.asyncio
    async def test_injects_warning_at_threshold(self):
        handler = AsyncMock(return_value=_tool_message())
        # Call think 3 times — threshold is 3
        for _ in range(2):
            await self.mw.awrap_tool_call(_think_request(), handler)
        result = await self.mw.awrap_tool_call(_think_request(), handler)
        assert "WARNING" in result.content
        assert "think" in result.content
        assert "real tool" in result.content

    @pytest.mark.asyncio
    async def test_injects_warning_beyond_threshold(self):
        handler = AsyncMock(return_value=_tool_message())
        for _ in range(5):
            result = await self.mw.awrap_tool_call(_think_request(), handler)
        assert "WARNING" in result.content

    @pytest.mark.asyncio
    async def test_handler_always_called(self):
        handler = AsyncMock(return_value=_tool_message())
        for _ in range(4):
            await self.mw.awrap_tool_call(_think_request(), handler)
        assert handler.await_count == 4

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        mw = ConsecutiveThinkGuardMiddleware(max_consecutive_thinks=2)
        msg = _tool_message("Thought recorded.")
        handler = AsyncMock(return_value=msg)
        # First call — below threshold
        result = await mw.awrap_tool_call(_think_request(), handler)
        assert "WARNING" not in result.content
        # Second call — at threshold
        result = await mw.awrap_tool_call(_think_request(), handler)
        assert "WARNING" in result.content

    @pytest.mark.asyncio
    async def test_counter_starts_at_zero(self):
        mw = ConsecutiveThinkGuardMiddleware()
        assert mw._consecutive_think_count == 0

    @pytest.mark.asyncio
    async def test_non_think_calls_not_modified(self):
        msg = MagicMock()
        msg.content = "search results"
        handler = AsyncMock(return_value=msg)
        result = await self.mw.awrap_tool_call(_other_tool_request("run_research_batch"), handler)
        assert result.content == "search results"

    @pytest.mark.asyncio
    async def test_graceful_on_immutable_result(self):
        """If model_copy raises, awrap_tool_call returns the original result without crashing."""
        msg = MagicMock()
        msg.content = "Thought recorded."
        msg.model_copy = MagicMock(side_effect=AttributeError("immutable"))
        handler = AsyncMock(return_value=msg)
        for _ in range(3):
            result = await self.mw.awrap_tool_call(_think_request(), handler)
        # Should not raise; returns original msg
        assert result is msg


# ---------------------------------------------------------------------------
# Shallow sub-agent mode — tool exposure
# ---------------------------------------------------------------------------


class TestShallowSubagentToolExposure:
    """`task` must stay reachable, and retrieval must move entirely to the sub-agent."""

    @staticmethod
    def _router(capture):
        return ComplexityRouterMiddleware(
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            shallow_subagent_capture=capture,
        )

    @staticmethod
    def _tools():
        return [
            _make_tool("task"),
            _make_tool("submit_final_report"),
            _make_tool("get_verified_sources"),
            _make_tool(_RUN_RESEARCH_BATCH_TOOL),
            _make_tool("web_search_tool"),
        ]

    def test_ceiling_keeps_task_but_still_hides_write_todos(self):
        from aiq_agent.agents.adaptive_researcher.custom_middleware import hidden_tools_for_ceiling

        hidden = hidden_tools_for_ceiling("single_shot", allow_shallow_subagent=True)
        assert "task" not in hidden
        assert "write_todos" in hidden
        # Unchanged without the flag, and unchanged for a deep ceiling.
        assert "task" in hidden_tools_for_ceiling("single_shot")
        assert hidden_tools_for_ceiling("deep", allow_shallow_subagent=True) == set()

    def test_research_batch_hidden_and_task_exposed_on_single_shot(self):
        capture = MagicMock(invoked=False)
        router = self._router(capture)
        router._declared_tier = "single_shot"
        names = [t.name for t in router._filter_tools(self._tools())]
        assert "task" in names
        assert _RUN_RESEARCH_BATCH_TOOL not in names
        assert "submit_final_report" in names

    def test_task_hidden_after_the_subagent_has_been_invoked(self):
        capture = MagicMock(invoked=True)
        router = self._router(capture)
        router._declared_tier = "single_shot"
        names = [t.name for t in router._filter_tools(self._tools())]
        assert "task" not in names
        assert "submit_final_report" in names

    def test_other_tiers_keep_the_research_batch_tool(self):
        capture = MagicMock(invoked=False)
        router = self._router(capture)
        router._declared_tier = "deep"
        names = [t.name for t in router._filter_tools(self._tools())]
        assert _RUN_RESEARCH_BATCH_TOOL in names
        assert "task" in names

    def test_before_any_declaration_the_research_batch_tool_remains(self):
        capture = MagicMock(invoked=False)
        router = self._router(capture)
        names = [t.name for t in router._filter_tools(self._tools())]
        assert _RUN_RESEARCH_BATCH_TOOL in names
