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

import itertools

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
from aiq_agent.agents.adaptive_researcher.tiers import sections_for_catalog
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

    ALL_MODES = ("direct", "single_shot", "standard", "deep", "delta")

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

    def test_router_preset_is_gone(self):
        # Catalog mode replaced the two-turn router flow outright; a leftover preset would be a
        # silently reachable path with no code building it.
        assert "router" not in SECTION_PRESETS
        with pytest.raises(KeyError):
            sections_for_tier("router", enabled=["direct", "deep"])

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


class TestSectionsForCatalog:
    """The turn-1 catalog map: union of the enabled tiers, derived (not a stored preset)."""

    ALL = ["direct", "single_shot", "standard", "deep"]

    @staticmethod
    def _subsets():
        for n in range(1, 5):
            for combo in itertools.combinations(TestSectionsForCatalog.ALL, n):
                yield list(combo)

    def test_every_subset_expands_to_the_full_ordered_flag_set(self):
        # Byte-stable rendering requires a fully populated map in SECTION_FLAGS order: a missing
        # flag would render as "default on" and silently change the prompt.
        for enabled in self._subsets():
            resolved = sections_for_catalog(enabled)
            assert list(resolved) == list(SECTION_FLAGS), enabled
            assert all(isinstance(v, bool) for v in resolved.values())

    def test_catalog_is_the_union_of_the_enabled_tiers(self):
        for enabled in self._subsets():
            resolved = sections_for_catalog(enabled)
            for tier in enabled:
                for flag in SECTION_PRESETS[tier]:
                    # delta_rule and escalation are resolved against the enabled set rather than
                    # unioned: delta is never catalog-routed, and escalation is meaningless when
                    # only one tier is enabled. Both have dedicated tests below.
                    if flag in ("delta_rule", "escalation"):
                        continue
                    assert resolved[flag], (enabled, tier, flag)

    def test_selection_blocks_always_on(self):
        # The model is choosing a tier on turn 1, so it needs the level descriptions and the
        # selection contract even for a single-tier config.
        for enabled in self._subsets():
            resolved = sections_for_catalog(enabled)
            assert resolved["effort_catalog"] and resolved["effort_selection"], enabled

    def test_delta_rule_never_in_a_normal_catalog(self):
        # Parent-report requests are detected at build time and get the forced delta prompt, so
        # a normal catalog must never carry the delta machinery.
        for enabled in self._subsets():
            assert sections_for_catalog(enabled)["delta_rule"] is False, enabled

    def test_escalation_only_with_more_than_one_tier(self):
        assert sections_for_catalog(["single_shot", "deep"])["escalation"] is True
        assert sections_for_catalog(["single_shot"])["escalation"] is False
        assert sections_for_catalog(["deep"])["escalation"] is False

    def test_shallow_only_catalog_omits_writer_machinery(self):
        # The whole point of deriving the catalog: a config that can never reach the planned
        # writer pipeline must not pay for its prompt sections.
        shallow = sections_for_catalog(["direct", "single_shot"])
        assert not shallow["subagents"]
        assert not shallow["sequential_handoffs"]
        assert not shallow["filesystem"]

    def test_empty_falls_back_to_all_tiers(self):
        assert sections_for_catalog([]) == sections_for_catalog(self.ALL)
        assert sections_for_catalog(None) == sections_for_catalog(self.ALL)

    def test_deterministic_for_a_given_enabled_set(self):
        # Prompt KV-cache stability: two calls must produce an identical map, and enabled-tier
        # ordering must not matter.
        assert sections_for_catalog(["deep", "direct"]) == sections_for_catalog(["direct", "deep"])


class TestEscalationPossible:
    def test_true_when_higher_tier_enabled(self):
        assert escalation_possible("single_shot", ["single_shot", "standard", "deep"]) is True

    def test_false_at_ceiling(self):
        assert escalation_possible("deep", ["single_shot", "deep"]) is False
        assert escalation_possible("single_shot", ["single_shot"]) is False
