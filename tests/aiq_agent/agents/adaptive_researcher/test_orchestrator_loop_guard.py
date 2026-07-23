# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the request-wide OrchestratorLoopGuardMiddleware."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_agent.agents.adaptive_researcher.custom_middleware import _DECLARE_EFFORT_TIER_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _RUN_RESEARCH_BATCH_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import _THINK_TOOL
from aiq_agent.agents.adaptive_researcher.custom_middleware import OrchestratorLoopGuardMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import _canonical_research_query_signature
from aiq_agent.agents.adaptive_researcher.models import AdaptiveRequestTerminationConfig
from aiq_agent.agents.adaptive_researcher.models import AdaptiveTierBudgets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str):
    t = MagicMock()
    t.name = name
    return t


def _cfg(**over) -> AdaptiveRequestTerminationConfig:
    """Small, crisp budgets for tests (deep >= standard as the model requires)."""
    base = dict(
        standard=AdaptiveTierBudgets(max_batch_calls=2, max_total_research_queries=3, max_orchestrator_turns=4),
        deep=AdaptiveTierBudgets(max_batch_calls=4, max_total_research_queries=16, max_orchestrator_turns=60),
        max_identical_research_queries=1,
        workflow_timeout_seconds=100,
        fallback_finalizer_timeout_seconds=10,
        recursion_limit=50,
    )
    base.update(over)
    return AdaptiveRequestTerminationConfig(**base)


def _q(query="apple inventory", **over) -> dict:
    d = {
        "query": query,
        "subqueries": [],
        "preferred_tools": ["knowledge_search"],
        "target_components": ["c1"],
        "depth": "medium",
        "rationale": "why",
        "fallback_tools": [],
    }
    d.update(over)
    return d


def _batch_request(queries):
    req = MagicMock()
    req.tool_call = {"name": _RUN_RESEARCH_BATCH_TOOL, "args": {"queries": queries}, "id": "call_1"}
    return req


def _declare_request(tier: str):
    req = MagicMock()
    req.tool_call = {"name": _DECLARE_EFFORT_TIER_TOOL, "args": {"tier": tier}}
    return req


def _model_request(tool_names):
    req = MagicMock()
    req.tools = [_make_tool(n) for n in tool_names]
    req.override = MagicMock(return_value=req)
    return req


def _standard_guard(**over):
    mw = OrchestratorLoopGuardMiddleware(config=_cfg(**over))
    mw._declared_tier = "standard"
    return mw


def _is_blocked(result) -> bool:
    return getattr(result, "status", None) == "error"


# ---------------------------------------------------------------------------
# Signature normalization
# ---------------------------------------------------------------------------


class TestQuerySignature:
    def test_rationale_and_fallback_ignored(self):
        a = _q(rationale="one", fallback_tools=["web"])
        b = _q(rationale="a completely different rationale", fallback_tools=[])
        assert _canonical_research_query_signature(a) == _canonical_research_query_signature(b)

    def test_whitespace_and_case_normalized(self):
        a = _q(query="Apple FY2024 10-K inventory")
        b = _q(query="  apple   fy2024   10-k    INVENTORY ")
        assert _canonical_research_query_signature(a) == _canonical_research_query_signature(b)

    def test_preferred_tool_order_ignored(self):
        a = _q(preferred_tools=["knowledge_search", "web"])
        b = _q(preferred_tools=["web", "knowledge_search"])
        assert _canonical_research_query_signature(a) == _canonical_research_query_signature(b)

    def test_subquery_order_matters(self):
        a = _q(subqueries=["x", "y"])
        b = _q(subqueries=["y", "x"])
        assert _canonical_research_query_signature(a) != _canonical_research_query_signature(b)

    def test_different_target_component_differs(self):
        a = _q(target_components=["c1"])
        b = _q(target_components=["c2"])
        assert _canonical_research_query_signature(a) != _canonical_research_query_signature(b)


# ---------------------------------------------------------------------------
# Batch / query / duplicate enforcement
# ---------------------------------------------------------------------------


class TestBatchEnforcement:
    @pytest.mark.asyncio
    async def test_first_batch_executes(self):
        mw = _standard_guard()
        handler = AsyncMock(return_value="notes")
        result = await mw.awrap_tool_call(_batch_request([_q()]), handler)
        handler.assert_awaited_once()
        assert result == "notes"
        assert mw._batch_call_count == 1
        assert mw._total_query_count == 1
        assert mw.phase == "active"

    @pytest.mark.asyncio
    async def test_batch_over_call_budget_is_blocked(self):
        # standard max_batch_calls=2 → third batch blocked. Use distinct queries so the duplicate
        # guard does not fire first, and keep each batch within the total-query budget.
        mw = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=2, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await mw.awrap_tool_call(_batch_request([_q("q1")]), handler)
        await mw.awrap_tool_call(_batch_request([_q("q2")]), handler)
        blocked = await mw.awrap_tool_call(_batch_request([_q("q3")]), handler)
        assert _is_blocked(blocked)
        assert handler.await_count == 2  # third batch never executed
        assert mw.phase == "finalizing"
        assert mw.exhaustion_reason is not None

    @pytest.mark.asyncio
    async def test_total_query_budget_enforced_across_batches(self):
        # standard max_total_research_queries=3. First batch of 2 ok; second batch of 2 would make 4.
        mw = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=9, max_total_research_queries=3, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await mw.awrap_tool_call(_batch_request([_q("a"), _q("b")]), handler)
        assert mw._total_query_count == 2
        blocked = await mw.awrap_tool_call(_batch_request([_q("c"), _q("d")]), handler)
        assert _is_blocked(blocked)
        assert handler.await_count == 1
        assert mw.phase == "finalizing"

    @pytest.mark.asyncio
    async def test_identical_query_blocked_across_batches(self):
        mw = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=9, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await mw.awrap_tool_call(_batch_request([_q("same query")]), handler)
        blocked = await mw.awrap_tool_call(_batch_request([_q("same query")]), handler)
        assert _is_blocked(blocked)
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_rationale_only_change_does_not_bypass_dedup(self):
        mw = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=9, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await mw.awrap_tool_call(_batch_request([_q("same", rationale="first")]), handler)
        blocked = await mw.awrap_tool_call(_batch_request([_q("same", rationale="reworded")]), handler)
        assert _is_blocked(blocked)

    @pytest.mark.asyncio
    async def test_materially_different_query_is_allowed(self):
        mw = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=9, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await mw.awrap_tool_call(_batch_request([_q("first", target_components=["c1"])]), handler)
        result = await mw.awrap_tool_call(_batch_request([_q("first", target_components=["c2"])]), handler)
        assert not _is_blocked(result)
        assert handler.await_count == 2

    @pytest.mark.asyncio
    async def test_sequential_calls_cannot_overshoot_batch_budget(self):
        # Counting happens before the handler is awaited, so once the budget is spent the next
        # call is blocked deterministically (the property parallel calls in one turn rely on).
        mw = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=1, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await mw.awrap_tool_call(_batch_request([_q("q1")]), handler)
        blocked = await mw.awrap_tool_call(_batch_request([_q("q2")]), handler)
        assert _is_blocked(blocked)
        assert mw._batch_call_count == 1


class TestInertTiers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier", ["single_shot", "direct"])
    async def test_inert_tier_never_blocks_batches(self, tier):
        mw = OrchestratorLoopGuardMiddleware(config=_cfg())
        mw._declared_tier = tier
        handler = AsyncMock(return_value="notes")
        for _ in range(5):
            result = await mw.awrap_tool_call(_batch_request([_q("q")]), handler)
            assert not _is_blocked(result)
        assert handler.await_count == 5
        assert mw._batch_call_count == 0  # not counted for inert tiers

    @pytest.mark.asyncio
    async def test_disabled_guard_passes_through(self):
        mw = OrchestratorLoopGuardMiddleware(config=_cfg(enabled=False))
        mw._declared_tier = "standard"
        handler = AsyncMock(return_value="notes")
        for _ in range(5):
            await mw.awrap_tool_call(_batch_request([_q("q")]), handler)
        assert handler.await_count == 5


class TestTierCapture:
    @pytest.mark.asyncio
    async def test_declare_effort_tier_is_captured(self):
        mw = OrchestratorLoopGuardMiddleware(config=_cfg())
        handler = AsyncMock(return_value="ok")
        await mw.awrap_tool_call(_declare_request("deep"), handler)
        assert mw._declared_tier == "deep"
        handler.assert_awaited_once()


# ---------------------------------------------------------------------------
# Model-turn budget and tool withdrawal
# ---------------------------------------------------------------------------


class TestFinalizationToolWithdrawal:
    @pytest.mark.asyncio
    async def test_research_and_think_withdrawn_when_finalizing(self):
        mw = _standard_guard()
        mw._phase = "finalizing"
        req = _model_request([_RUN_RESEARCH_BATCH_TOOL, _THINK_TOOL, "get_verified_sources", "submit_final_report"])
        await mw.awrap_model_call(req, AsyncMock(return_value=MagicMock()))
        passed = {t.name for t in req.override.call_args.kwargs["tools"]}
        assert _RUN_RESEARCH_BATCH_TOOL not in passed
        assert _THINK_TOOL not in passed
        assert {"get_verified_sources", "submit_final_report"} <= passed

    @pytest.mark.asyncio
    async def test_tools_intact_while_active(self):
        mw = _standard_guard()
        req = _model_request([_RUN_RESEARCH_BATCH_TOOL, _THINK_TOOL, "submit_final_report"])
        await mw.awrap_model_call(req, AsyncMock(return_value=MagicMock()))
        passed = {t.name for t in req.override.call_args.kwargs["tools"]}
        assert _RUN_RESEARCH_BATCH_TOOL in passed

    @pytest.mark.asyncio
    async def test_turn_budget_forces_finalization(self):
        # standard max_orchestrator_turns=4 → the 5th model call flips to finalizing.
        mw = _standard_guard()
        handler = AsyncMock(return_value=MagicMock())
        for _ in range(4):
            await mw.awrap_model_call(_model_request(["submit_final_report"]), handler)
        assert mw.phase == "active"
        await mw.awrap_model_call(_model_request(["submit_final_report"]), handler)
        assert mw.phase == "finalizing"

    @pytest.mark.asyncio
    async def test_batch_blocked_after_finalizing_via_turns(self):
        mw = _standard_guard()
        handler = AsyncMock(return_value=MagicMock())
        for _ in range(5):
            await mw.awrap_model_call(_model_request(["submit_final_report"]), handler)
        assert mw.phase == "finalizing"
        blocked = await mw.awrap_tool_call(_batch_request([_q("late")]), AsyncMock(return_value="notes"))
        assert _is_blocked(blocked)


class TestRequestIsolation:
    @pytest.mark.asyncio
    async def test_independent_instances_do_not_share_counts(self):
        a = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=1, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        b = _standard_guard(
            standard=AdaptiveTierBudgets(max_batch_calls=1, max_total_research_queries=99, max_orchestrator_turns=99),
        )
        handler = AsyncMock(return_value="notes")
        await a.awrap_tool_call(_batch_request([_q("q")]), handler)
        # b has its own fresh budget — its first batch still executes.
        result = await b.awrap_tool_call(_batch_request([_q("q")]), handler)
        assert not _is_blocked(result)
        assert b._batch_call_count == 1
