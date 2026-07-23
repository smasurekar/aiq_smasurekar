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

"""Tests for the adaptive researcher effort tiers and the enabled_tiers config field."""

import pytest
from pydantic import ValidationError

from aiq_agent.agents.adaptive_researcher.custom_middleware import hidden_tools_for_ceiling
from aiq_agent.agents.adaptive_researcher.register import AdaptiveResearchAgentConfig
from aiq_agent.agents.adaptive_researcher.tiers import _TIER_ORDER
from aiq_agent.agents.adaptive_researcher.tiers import SECTION_FLAGS
from aiq_agent.agents.adaptive_researcher.tiers import SECTION_PRESETS
from aiq_agent.agents.adaptive_researcher.tiers import clamp_to_enabled_tiers
from aiq_agent.agents.adaptive_researcher.tiers import enabled_tier_profiles
from aiq_agent.agents.adaptive_researcher.tiers import escalation_possible
from aiq_agent.agents.adaptive_researcher.tiers import normalize_enabled_tiers
from aiq_agent.agents.adaptive_researcher.tiers import sections_for_tier
from aiq_agent.agents.adaptive_researcher.tiers import tier_ceiling


class TestClampToEnabledTiers:
    """clamp_to_enabled_tiers snaps a resolved tier into the enabled allow-list."""

    def test_enabled_tier_passes_through(self):
        assert clamp_to_enabled_tiers("deep", ["single_shot", "deep"]) == "deep"
        assert clamp_to_enabled_tiers("single_shot", ["single_shot", "deep"]) == "single_shot"

    def test_disabled_tier_snaps_to_nearest_by_rank(self):
        # direct disabled, nearest enabled is single_shot (rank distance 1)
        assert clamp_to_enabled_tiers("direct", ["single_shot", "standard"]) == "single_shot"
        # deep disabled, nearest enabled is standard
        assert clamp_to_enabled_tiers("deep", ["direct", "standard"]) == "standard"

    def test_tie_breaks_to_deeper_tier(self):
        # standard (rank 2) is equidistant from single_shot (rank 1) and deep (rank 3);
        # ties resolve to the deeper tier for safer quality.
        assert clamp_to_enabled_tiers("standard", ["single_shot", "deep"]) == "deep"

    def test_single_enabled_tier_floors_and_ceilings(self):
        assert clamp_to_enabled_tiers("direct", ["deep"]) == "deep"
        assert clamp_to_enabled_tiers("deep", ["single_shot"]) == "single_shot"

    def test_empty_enabled_falls_back_to_all(self):
        assert clamp_to_enabled_tiers("deep", []) == "deep"


class TestNormalizeAndProfiles:
    def test_normalize_orders_and_dedupes(self):
        assert normalize_enabled_tiers(["deep", "single_shot", "deep"]) == ["single_shot", "deep"]

    def test_normalize_empty_returns_all(self):
        assert normalize_enabled_tiers([]) == list(_TIER_ORDER)
        assert normalize_enabled_tiers(None) == list(_TIER_ORDER)

    def test_tier_ceiling_is_highest_enabled(self):
        assert tier_ceiling(["single_shot", "standard"]) == "standard"
        assert tier_ceiling(["direct", "deep"]) == "deep"
        assert tier_ceiling(None) == "deep"

    def test_enabled_tier_profiles_are_filtered_and_ordered(self):
        profiles = enabled_tier_profiles(["deep", "single_shot"])
        assert [p.name for p in profiles] == ["single_shot", "deep"]
        # every profile exposes the prompt-facing fields, including the finalize mechanism
        for p in profiles:
            assert p.when and p.planner and p.writer and p.width and p.tools and p.finalize


class TestHiddenToolsForCeiling:
    """Layer-B static tool-hiding derived from the enabled-tiers ceiling."""

    def test_deep_ceiling_hides_nothing(self):
        assert hidden_tools_for_ceiling("deep") == set()

    def test_standard_ceiling_hides_advanced_web_only(self):
        assert hidden_tools_for_ceiling("standard") == {"advanced_web_search_tool"}

    def test_shallow_ceiling_also_hides_delegation_tools(self):
        assert hidden_tools_for_ceiling("single_shot") == {"advanced_web_search_tool", "task", "write_todos"}
        assert hidden_tools_for_ceiling("direct") == {"advanced_web_search_tool", "task", "write_todos"}

    def test_delta_override_preserves_delegation_tools(self):
        assert hidden_tools_for_ceiling("single_shot", allow_delegation=True) == {"advanced_web_search_tool"}


class TestEnabledTiersConfig:
    """The enabled_tiers / enforce_tier_tools config surface."""

    def test_defaults(self):
        config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
        assert config.enabled_tiers == ["direct", "single_shot", "standard", "deep"]
        assert config.enforce_tier_tools is False
        assert config.enable_source_router is False

    def test_custom_enabled_tiers(self):
        config = AdaptiveResearchAgentConfig(orchestrator_llm="llm", enabled_tiers=["single_shot", "deep"])
        assert config.enabled_tiers == ["single_shot", "deep"]

    def test_empty_enabled_tiers_rejected(self):
        with pytest.raises(ValidationError):
            AdaptiveResearchAgentConfig(orchestrator_llm="llm", enabled_tiers=[])

    def test_unknown_tier_rejected(self):
        with pytest.raises(ValidationError):
            AdaptiveResearchAgentConfig(orchestrator_llm="llm", enabled_tiers=["turbo"])

    def test_dynamic_orchestrator_sections_default_off(self):
        config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
        assert config.dynamic_orchestrator_sections is False


class TestSectionPresets:
    """Dynamic per-tier prompt sections: SECTION_PRESETS + sections_for_tier()."""

    ALL_MODES = ("router", "direct", "single_shot", "standard", "deep", "delta")

    def test_every_mode_expands_to_full_flag_set(self):
        # sections_for_tier must return every flag (in SECTION_FLAGS), so rendering is
        # deterministic and no template flag is left undefined.
        for mode in self.ALL_MODES:
            resolved = sections_for_tier(mode, enabled=["direct", "single_shot", "standard", "deep"])
            assert set(resolved) == set(SECTION_FLAGS)
            assert all(isinstance(v, bool) for v in resolved.values())

    def test_preset_flags_are_known(self):
        # No preset may reference a flag the template doesn't understand.
        for mode, on_flags in SECTION_PRESETS.items():
            assert on_flags <= set(SECTION_FLAGS), mode

    def test_router_selects_but_does_not_execute(self):
        r = sections_for_tier("router", enabled=["direct", "single_shot", "standard", "deep"])
        assert r["effort_catalog"] and r["effort_selection"]
        assert not r["workflow"] and not r["research_loop"] and not r["subagents"]

    def test_cheap_tiers_drop_selection_and_subagents(self):
        for mode in ("direct", "single_shot"):
            s = sections_for_tier(mode, enabled=["direct", "single_shot", "standard", "deep"])
            assert not s["effort_catalog"] and not s["effort_selection"]
            assert not s["subagents"]

    def test_deep_and_delta_carry_subagents(self):
        for mode in ("deep", "delta"):
            s = sections_for_tier(mode, enabled=["direct", "single_shot", "standard", "deep"])
            assert s["subagents"] and s["workflow"]
        assert sections_for_tier("delta", enabled=["deep"])["delta_rule"]
        assert not sections_for_tier("deep", enabled=["deep"])["delta_rule"]

    def test_escalation_resolves_against_enabled_tiers(self):
        # a higher tier exists -> escalation section kept
        assert sections_for_tier("single_shot", enabled=["single_shot", "deep"])["escalation"]
        # single_shot is the ceiling -> nothing to escalate to
        assert not sections_for_tier("single_shot", enabled=["single_shot"])["escalation"]
        # deep never advertises escalation regardless of enabled set
        assert not sections_for_tier("deep", enabled=["deep"])["escalation"]


class TestEscalationPossible:
    def test_true_when_higher_tier_enabled(self):
        assert escalation_possible("single_shot", ["single_shot", "standard", "deep"]) is True

    def test_false_at_ceiling(self):
        assert escalation_possible("deep", ["single_shot", "deep"]) is False
        assert escalation_possible("single_shot", ["single_shot"]) is False
