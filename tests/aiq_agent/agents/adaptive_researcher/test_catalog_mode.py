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

"""Tests for catalog mode: batch tier resolution, compatibility, and turn-1 tool exposure.

Catalog mode removes the dedicated tier-selection LLM call: the orchestrator declares its tier
*alongside* the first action of that tier. That makes the tier a property of a whole tool-call
batch rather than of one earlier turn, so these tests focus on the three things that can go
wrong as a result — a batch resolved differently depending on wrapper scheduling order, an
action executing under a tier its configured path cannot perform, and a first-turn budget that
a parallel batch can overshoot.
"""

import itertools
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_agent.agents.adaptive_researcher.custom_middleware import ComplexityRouterMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import TierResolver

ALL_TIERS = ["direct", "single_shot", "standard", "deep"]
SOURCE_TOOLS = frozenset({"web_search_tool", "advanced_web_search_tool"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BATCH_SEQ = itertools.count()


def _batch(*calls):
    """Build a mock state whose last message carries ``calls`` as one tool-call batch.

    Each call gets a fresh message id: the resolver memoizes per batch, so reusing an id would
    silently return the previous turn's decision instead of resolving the new one.
    """
    message = MagicMock()
    message.id = f"ai-{next(_BATCH_SEQ)}"
    message.tool_calls = [dict(c) for c in calls]
    return {"messages": [MagicMock(), message]}


def _declare(tier):
    return {"name": "declare_effort_tier", "args": {"tier": tier}, "id": "c-declare"}


def _research_batch(n=1):
    return {"name": "run_research_batch", "args": {"queries": [{"query": f"q{i}"} for i in range(n)]}, "id": "c-rb"}


def _task(subagent):
    return {"name": "task", "args": {"subagent_type": subagent, "description": "d"}, "id": f"c-{subagent}"}


def _source(name="web_search_tool"):
    return {"name": name, "args": {"query": "q"}, "id": f"c-{name}"}


def _finalize(researched=True, tier=None):
    return {
        "name": "submit_final_report",
        "args": {"markdown": "m", "researched": researched, "tier": tier},
        "id": "c-fin",
    }


def _resolver(**over):
    kwargs = dict(
        enabled_tiers=ALL_TIERS,
        single_loop_single_shot=False,
        shallow_mode=False,
        direct_source_tool_names=frozenset(),
        backend=None,
    )
    kwargs.update(over)
    return TierResolver(**kwargs)


def _make_tool(name):
    tool = MagicMock()
    tool.name = name
    return tool


def _request(tool_call, state):
    req = MagicMock()
    req.tool_call = tool_call
    req.state = state
    return req


# ---------------------------------------------------------------------------
# Resolution precedence and determinism
# ---------------------------------------------------------------------------


class TestBatchResolutionPrecedence:
    def test_declaration_in_the_same_batch_wins(self):
        r = _resolver()
        decision = r.decide(_batch(_declare("deep"), _task("planner-agent")))
        assert decision.tier == "deep"
        assert decision.source == "declared"
        assert decision.error is None

    def test_order_within_the_batch_is_irrelevant(self):
        # ToolNode may schedule the sibling wrappers in any order; the decision must not depend
        # on which call happens to be inspected first.
        forward = _resolver().decide(_batch(_declare("standard"), _research_batch()))
        reversed_ = _resolver().decide(_batch(_research_batch(), _declare("standard")))
        assert forward.tier == reversed_.tier == "standard"

    def test_decision_is_memoized_per_batch(self):
        r = _resolver()
        state = _batch(_declare("standard"), _research_batch())
        first = r.decide(state)
        assert r.decide(state) is first

    def test_cached_tier_used_when_a_later_batch_omits_the_declaration(self):
        r = _resolver()
        r.commit(r.decide(_batch(_declare("standard"), _research_batch())))
        later = r.decide(_batch(_research_batch()))
        assert later.tier == "standard"
        assert later.source == "cached"

    def test_helper_only_batch_resolves_nothing(self):
        # think / file reads must never establish a tier by themselves.
        for call in (
            {"name": "think", "args": {"thought": "t"}, "id": "c"},
            {"name": "read_file", "args": {"path": "/shared/x"}, "id": "c"},
            {"name": "get_verified_sources", "args": {}, "id": "c"},
        ):
            decision = _resolver().decide(_batch(call))
            assert decision.tier is None
            assert decision.error is None

    def test_bare_declaration_still_resolves(self):
        # A model that ignores the co-declaration contract degrades to the old behaviour rather
        # than erroring: one wasted turn, not a failed run.
        decision = _resolver().decide(_batch(_declare("single_shot")))
        assert decision.tier == "single_shot"


class TestDeclarationValidation:
    def test_conflicting_declarations_rejected(self):
        decision = _resolver().decide(_batch(_declare("single_shot"), _declare("deep"), _research_batch()))
        assert decision.tier is None
        assert "Conflicting" in decision.error

    def test_disabled_tier_rejected(self):
        r = _resolver(enabled_tiers=["single_shot", "standard"])
        decision = r.decide(_batch(_declare("deep"), _task("planner-agent")))
        assert decision.tier is None
        assert "not enabled" in decision.error

    def test_unknown_tier_rejected(self):
        decision = _resolver().decide(_batch(_declare("turbo"), _research_batch()))
        assert decision.tier is None
        assert "Unknown effort tier" in decision.error

    def test_downgrade_rejected(self):
        r = _resolver()
        r.commit(r.decide(_batch(_declare("deep"), _task("planner-agent"))))
        decision = r.decide(_batch(_declare("single_shot"), _research_batch()))
        assert decision.tier is None
        assert "downgrade" in decision.error

    def test_escalation_accepted(self):
        r = _resolver()
        r.commit(r.decide(_batch(_declare("standard"), _research_batch())))
        decision = r.decide(_batch(_declare("deep"), _task("planner-agent")))
        assert decision.tier == "deep"

    def test_meta_requires_the_terminal_finalizer(self):
        ok = _resolver().decide(_batch(_declare("meta"), _finalize(researched=False, tier="meta")))
        assert ok.tier == "meta"
        bad = _resolver().decide(_batch(_declare("meta"), _research_batch()))
        assert bad.tier is None
        assert "No-Research Meta" in bad.error


# ---------------------------------------------------------------------------
# Inference fallback (the model skipped its declaration)
# ---------------------------------------------------------------------------


class TestInference:
    def test_lone_direct_finalizer(self):
        decision = _resolver().decide(_batch(_finalize(researched=False, tier="direct")))
        assert decision.tier == "direct"
        assert decision.source == "inferred"

    def test_lone_meta_finalizer(self):
        assert _resolver().decide(_batch(_finalize(researched=False, tier="meta"))).tier == "meta"

    def test_direct_finalizer_rejected_when_direct_disabled(self):
        r = _resolver(enabled_tiers=["single_shot", "standard", "deep"])
        decision = r.decide(_batch(_finalize(researched=False, tier="direct")))
        assert decision.tier is None
        assert "not enabled" in decision.error

    def test_source_tool_implies_single_shot_only_on_the_fast_lane(self):
        fast = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        assert fast.decide(_batch(_source())).tier == "single_shot"

    def test_shallow_task_implies_single_shot(self):
        r = _resolver(shallow_mode=True)
        assert r.decide(_batch(_task("shallow-researcher"))).tier == "single_shot"

    def test_shallow_task_rejected_when_not_configured(self):
        decision = _resolver().decide(_batch(_task("shallow-researcher")))
        assert decision.tier is None
        assert "not available" in decision.error

    def test_planner_task_maps_to_the_enabled_ceiling(self):
        assert _resolver().decide(_batch(_task("planner-agent"))).tier == "deep"
        capped = _resolver(enabled_tiers=["direct", "single_shot", "standard"])
        assert capped.decide(_batch(_task("planner-agent"))).tier == "standard"

    def test_writer_task_can_never_establish_a_tier(self):
        decision = _resolver().decide(_batch(_task("writer-agent")))
        assert decision.tier is None
        assert "writer-agent cannot be the first action" in decision.error

    def test_todos_alone_is_not_enough(self):
        decision = _resolver().decide(_batch({"name": "write_todos", "args": {"todos": []}, "id": "c"}))
        assert decision.tier is None
        assert decision.error is None

    def test_todos_plus_planner_resolves_from_the_planner(self):
        decision = _resolver().decide(
            _batch({"name": "write_todos", "args": {"todos": []}, "id": "c"}, _task("planner-agent"))
        )
        assert decision.tier == "deep"

    def test_incompatible_mixed_actions_rejected(self):
        r = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        decision = r.decide(_batch(_source(), _research_batch()))
        assert decision.tier is None
        assert "different effort levels" in decision.error

    def test_inference_is_logged_and_persisted(self):
        backend = MagicMock()
        r = _resolver(backend=backend)
        r.commit(r.decide(_batch(_task("planner-agent"))))
        # declare_effort_tier never ran, so nothing else would write /shared/effort_tier.json
        backend.upload_files.assert_called_once()
        assert b"deep" in backend.upload_files.call_args[0][0][0][1]

    def test_declared_tier_is_not_double_persisted(self):
        # The declaration tool already writes the file; the resolver must not race a second write.
        backend = MagicMock()
        r = _resolver(backend=backend)
        r.commit(r.decide(_batch(_declare("standard"), _research_batch())))
        backend.upload_files.assert_not_called()


class TestResearchBatchInferenceNeverDisablesTheGuard:
    """`run_research_batch` must not infer `single_shot` while a budgeted tier is enabled.

    ``budgets_for_tier`` returns None for single_shot, which makes OrchestratorLoopGuardMiddleware
    fully inert — no batch cap, no query cap, and no turn cap. Guessing low would therefore strip
    every request-wide research bound in exactly the situation where the model is already
    misbehaving by skipping its declaration. See plan doc §9.1.
    """

    def test_prefers_standard_over_single_shot(self):
        assert _resolver().decide(_batch(_research_batch())).tier == "standard"

    def test_prefers_deep_when_standard_disabled(self):
        r = _resolver(enabled_tiers=["direct", "single_shot", "deep"])
        assert r.decide(_batch(_research_batch())).tier == "deep"

    def test_falls_back_to_single_shot_only_when_it_is_the_sole_research_tier(self):
        # Nothing better exists here: the guard would have been inert either way.
        r = _resolver(enabled_tiers=["direct", "single_shot"])
        assert r.decide(_batch(_research_batch())).tier == "single_shot"

    def test_rejected_when_no_research_tier_is_enabled(self):
        decision = _resolver(enabled_tiers=["direct"]).decide(_batch(_research_batch()))
        assert decision.tier is None
        assert "No research-capable tier" in decision.error

    def test_never_single_shot_under_a_fast_lane(self):
        for kwargs in (
            {"single_loop_single_shot": True, "direct_source_tool_names": SOURCE_TOOLS},
            {"shallow_mode": True},
        ):
            assert _resolver(**kwargs).decide(_batch(_research_batch())).tier == "standard"


# ---------------------------------------------------------------------------
# Compatibility matrix: declared tier vs the action actually taken
# ---------------------------------------------------------------------------


class TestCompatibilityMatrix:
    def test_single_shot_plus_research_batch_blocked_on_the_fast_lane(self):
        # Promoting would silently turn a cheap lookup into a standard run; block so the model
        # corrects itself onto the fast lane instead.
        r = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        decision = r.decide(_batch(_declare("single_shot"), _research_batch()))
        assert decision.tier is None
        assert "not available on the single_shot path" in decision.error

    def test_single_shot_plus_research_batch_allowed_in_ordinary_mode(self):
        # Without a fast lane, run_research_batch *is* the single_shot procedure.
        decision = _resolver().decide(_batch(_declare("single_shot"), _research_batch()))
        assert decision.tier == "single_shot"

    def test_standard_plus_source_tool_blocked(self):
        r = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        decision = r.decide(_batch(_declare("standard"), _source()))
        assert decision.tier is None
        assert "cannot be called directly" in decision.error

    def test_deep_plus_shallow_subagent_blocked(self):
        r = _resolver(shallow_mode=True)
        decision = r.decide(_batch(_declare("deep"), _task("shallow-researcher")))
        assert decision.tier is None
        assert "only on the single_shot tier" in decision.error

    def test_direct_plus_research_promotes_upward(self):
        decision = _resolver().decide(_batch(_declare("direct"), _research_batch()))
        assert decision.tier == "standard"
        assert decision.source == "inferred"

    def test_promotion_never_lands_on_a_disabled_tier(self):
        r = _resolver(
            enabled_tiers=["direct", "single_shot"], single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS
        )
        decision = r.decide(_batch(_declare("direct"), _research_batch()))
        # standard/deep are disabled, so the only compatible research tier is single_shot — and
        # single_shot cannot run run_research_batch on this fast lane.
        assert decision.tier is None
        assert decision.error

    def test_finalizing_is_always_structurally_allowed(self):
        r = _resolver()
        r.commit(r.decide(_batch(_declare("deep"), _task("planner-agent"))))
        assert r.decide(_batch(_finalize(researched=True))).tier == "deep"


# ---------------------------------------------------------------------------
# ComplexityRouterMiddleware: enforcement, turn-1 exposure, hard budget
# ---------------------------------------------------------------------------


class TestMiddlewareEnforcement:
    def _mw(self, resolver, **over):
        kwargs = dict(
            enabled_tiers=ALL_TIERS,
            direct_source_tools=[_make_tool(n) for n in sorted(SOURCE_TOOLS)],
            single_loop_single_shot=True,
            single_shot_search_budget=2,
            tier_resolver=resolver,
        )
        kwargs.update(over)
        return ComplexityRouterMiddleware(**kwargs)

    @pytest.mark.asyncio
    async def test_rejected_batch_blocks_the_substantive_call(self):
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        mw = self._mw(resolver)
        state = _batch(_declare("single_shot"), _research_batch())
        handler = AsyncMock(return_value="ran")
        result = await mw.awrap_tool_call(_request(_research_batch(), state), handler)
        handler.assert_not_awaited()
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_rejected_batch_still_allows_helpers(self):
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        mw = self._mw(resolver)
        state = _batch(_declare("single_shot"), _research_batch())
        handler = AsyncMock(return_value="thought")
        think = {"name": "think", "args": {"thought": "t"}, "id": "c"}
        assert await mw.awrap_tool_call(_request(think, state), handler) == "thought"

    @pytest.mark.asyncio
    async def test_valid_batch_commits_the_tier(self):
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        mw = self._mw(resolver)
        state = _batch(_declare("standard"), _research_batch())
        await mw.awrap_tool_call(_request(_research_batch(), state), AsyncMock(return_value="ok"))
        assert resolver.tier == "standard"
        assert mw._tier == "standard"

    def test_turn_one_exposes_the_union(self):
        # The tier is unknown on turn 1, so both research paths must be reachable; the
        # compatibility matrix — not tool hiding — keeps the model on one of them.
        mw = self._mw(_resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS))
        tools = [_make_tool(n) for n in ("run_research_batch", "web_search_tool", "submit_final_report")]
        names = {t.name for t in mw._filter_tools(tools)}
        assert names == {"run_research_batch", "web_search_tool", "submit_final_report"}

    def test_source_tools_hidden_once_a_deep_tier_is_resolved(self):
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        resolver.commit(resolver.decide(_batch(_declare("standard"), _research_batch())))
        mw = self._mw(resolver)
        tools = [_make_tool(n) for n in ("run_research_batch", "web_search_tool")]
        assert {t.name for t in mw._filter_tools(tools)} == {"run_research_batch"}

    def test_research_batch_hidden_once_single_shot_is_resolved(self):
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        resolver.commit(resolver.decide(_batch(_declare("single_shot"), _source())))
        mw = self._mw(resolver)
        tools = [_make_tool(n) for n in ("run_research_batch", "web_search_tool")]
        assert {t.name for t in mw._filter_tools(tools)} == {"web_search_tool"}

    @pytest.mark.asyncio
    async def test_parallel_first_turn_searches_cannot_overshoot_the_budget(self):
        # The whole point of reserving before awaiting: a catalog first turn can contain several
        # parallel source calls, and counting after the fact would let all of them run.
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        mw = self._mw(resolver, single_shot_search_budget=2)
        state = _batch(_declare("single_shot"), _source("web_search_tool"))
        handler = AsyncMock(return_value=MagicMock(content="hit"))
        executed = 0
        for _ in range(5):
            result = await mw.awrap_tool_call(_request(_source("web_search_tool"), state), handler)
            if getattr(result, "status", None) != "error":
                executed += 1
        assert executed == 2
        assert mw._source_call_count == 2

    @pytest.mark.asyncio
    async def test_budget_is_counted_from_the_first_turn(self):
        # In catalog mode the very first turn can already search, so the first call must consume
        # budget rather than being treated as pre-declaration.
        resolver = _resolver(single_loop_single_shot=True, direct_source_tool_names=SOURCE_TOOLS)
        mw = self._mw(resolver, single_shot_search_budget=1)
        state = _batch(_declare("single_shot"), _source())
        await mw.awrap_tool_call(_request(_source(), state), AsyncMock(return_value=MagicMock(content="hit")))
        assert mw._source_call_count == 1


class TestLegacyModeUnchanged:
    """With no resolver (``dynamic_orchestrator_sections: false``) nothing may change."""

    def _mw(self):
        return ComplexityRouterMiddleware(
            enabled_tiers=ALL_TIERS,
            direct_source_tools=[_make_tool("web_search_tool")],
            single_loop_single_shot=True,
        )

    @pytest.mark.asyncio
    async def test_declaration_still_cached_locally(self):
        mw = self._mw()
        req = MagicMock()
        req.tool_call = {"name": "declare_effort_tier", "args": {"tier": "single_shot"}}
        await mw.awrap_tool_call(req, AsyncMock(return_value="ok"))
        assert mw._declared_tier == "single_shot"
        assert mw._tier == "single_shot"

    def test_source_tools_hidden_before_declaration(self):
        # Legacy turn-1 behaviour: no union exposure, source tools stay hidden until declared.
        mw = self._mw()
        tools = [_make_tool(n) for n in ("run_research_batch", "web_search_tool")]
        assert {t.name for t in mw._filter_tools(tools)} == {"run_research_batch"}
