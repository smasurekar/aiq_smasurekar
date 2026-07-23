# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the request-wide termination configuration model."""

import pytest
from pydantic import ValidationError

from aiq_agent.agents.adaptive_researcher.models import AdaptiveRequestTerminationConfig
from aiq_agent.agents.adaptive_researcher.models import AdaptiveTierBudgets


class TestDefaults:
    def test_defaults_are_enabled_and_finite(self):
        cfg = AdaptiveRequestTerminationConfig()
        assert cfg.enabled is True
        assert cfg.standard.max_batch_calls == 3
        assert cfg.standard.max_total_research_queries == 9
        assert cfg.standard.max_orchestrator_turns == 24
        assert cfg.deep.max_batch_calls == 6
        assert cfg.deep.max_total_research_queries == 24
        assert cfg.deep.max_orchestrator_turns == 100
        assert cfg.max_identical_research_queries == 1
        assert cfg.workflow_timeout_seconds == 1200
        assert cfg.fallback_finalizer_timeout_seconds == 60
        assert cfg.recursion_limit == 250

    def test_frozen(self):
        cfg = AdaptiveRequestTerminationConfig()
        with pytest.raises(ValidationError):
            cfg.enabled = False


class TestValidation:
    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            AdaptiveRequestTerminationConfig(unexpected=1)

    def test_rejects_unknown_tier_budget_field(self):
        with pytest.raises(ValidationError):
            AdaptiveTierBudgets(max_batch_calls=2, oops=1)

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_budgets(self, value):
        with pytest.raises(ValidationError):
            AdaptiveTierBudgets(max_batch_calls=value)

    @pytest.mark.parametrize("field", ["workflow_timeout_seconds", "recursion_limit", "max_identical_research_queries"])
    def test_rejects_non_positive_top_level(self, field):
        with pytest.raises(ValidationError):
            AdaptiveRequestTerminationConfig(**{field: 0})

    def test_standard_may_not_exceed_deep_on_any_budget(self):
        deep = AdaptiveTierBudgets(max_batch_calls=4, max_total_research_queries=16, max_orchestrator_turns=60)
        # batch calls
        too_many_batches = AdaptiveTierBudgets(
            max_batch_calls=5, max_total_research_queries=6, max_orchestrator_turns=18
        )
        with pytest.raises(ValidationError):
            AdaptiveRequestTerminationConfig(standard=too_many_batches, deep=deep)
        # total queries
        too_many_queries = AdaptiveTierBudgets(
            max_batch_calls=2, max_total_research_queries=99, max_orchestrator_turns=1
        )
        with pytest.raises(ValidationError):
            AdaptiveRequestTerminationConfig(standard=too_many_queries, deep=deep)

    def test_equal_standard_and_deep_is_allowed(self):
        budgets = AdaptiveTierBudgets(max_batch_calls=3, max_total_research_queries=8, max_orchestrator_turns=20)
        cfg = AdaptiveRequestTerminationConfig(standard=budgets, deep=budgets)
        assert cfg.standard == cfg.deep

    def test_fallback_must_be_below_workflow_timeout(self):
        with pytest.raises(ValidationError):
            AdaptiveRequestTerminationConfig(workflow_timeout_seconds=60, fallback_finalizer_timeout_seconds=60)
        with pytest.raises(ValidationError):
            AdaptiveRequestTerminationConfig(workflow_timeout_seconds=60, fallback_finalizer_timeout_seconds=61)


class TestBudgetsForTier:
    def test_standard_maps_to_standard_budgets(self):
        cfg = AdaptiveRequestTerminationConfig()
        assert cfg.budgets_for_tier("standard") is cfg.standard

    @pytest.mark.parametrize("tier", ["deep", "delta"])
    def test_deep_and_delta_map_to_deep_budgets(self, tier):
        cfg = AdaptiveRequestTerminationConfig()
        assert cfg.budgets_for_tier(tier) is cfg.deep

    @pytest.mark.parametrize("tier", ["single_shot", "direct", "meta", None, "unknown"])
    def test_inert_tiers_return_none(self, tier):
        cfg = AdaptiveRequestTerminationConfig()
        assert cfg.budgets_for_tier(tier) is None
