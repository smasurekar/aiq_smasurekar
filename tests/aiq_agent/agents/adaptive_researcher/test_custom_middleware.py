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

"""Tests for ComplexityRouterMiddleware — static ceiling hiding and single-shot dynamic swap."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_agent.agents.adaptive_researcher.custom_middleware import _DECLARE_EFFORT_TIER_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _RUN_RESEARCH_BATCH_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _THINK_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import ComplexityRouterMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import ConsecutiveThinkGuardMiddleware

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
