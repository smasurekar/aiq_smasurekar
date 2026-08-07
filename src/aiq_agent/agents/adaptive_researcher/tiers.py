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

"""Effort tiers for the adaptive research orchestrator.

Adaptivity is soft and prompt-driven: the orchestrator self-selects an effort level per
request and self-limits its planning, fan-out, and tool use. These tiers are prompt-facing
*descriptions* (not code presets) plus a small clamp helper. ``enabled_tiers`` (config)
controls which tiers are described to the model — the primary (Layer-A) enforcement — and,
when ``enforce_tier_tools`` is on, which heavier tools the Layer-B middleware hides.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered from least to most effort. The index is the tier's rank, used by the clamp and by
# the Layer-B tool-exposure ceiling.
_TIER_ORDER: list[str] = ["direct", "single_shot", "standard", "deep"]

Tier = str


@dataclass(frozen=True)
class TierProfile:
    """A prompt-facing description of one effort tier (behavioral guidance, not a code preset)."""

    name: str
    when: str
    planner: str
    writer: str
    width: str
    tools: str
    finalize: str


# The behavioral guidance rendered into orchestrator.j2. Mirrors the effort-profile table in
# the POC plan (§4.3.5). All four normal tiers exist in the one graph; ``enabled_tiers`` controls
# which are described to the model. Parent-report delta work is a separate mandatory writer
# safety workflow. ``finalize`` states how each normal level ends its run (the mechanism-based
# rule: inline -> submit_final_report; writer-agent -> return the marker).
TIER_PROFILES: dict[str, TierProfile] = {
    "direct": TierProfile(
        name="direct",
        when="a trivially known, time-insensitive fact where research adds nothing",
        planner="skip",
        writer="answer inline",
        width="0 queries — perform NO research",
        tools="none (answer directly)",
        finalize="submit_final_report(researched=false)",
    ),
    "single_shot": TierProfile(
        name="single_shot",
        when="one bounded, factual question",
        planner="skip",
        writer="write inline",
        width="1–3 queries",
        tools="basic web / knowledge retrieval",
        finalize="submit_final_report(researched=true)",
    ),
    "standard": TierProfile(
        name="standard",
        when="a lightly multi-part question, or a comparison across a bounded set of entities",
        planner="skip for inline; required for writer-agent",
        writer="inline or Planned Writer Pipeline",
        width="~3–5 queries",
        tools="basic web + advanced web if needed",
        finalize="inline → submit_final_report(researched=true), or writer-agent → return marker",
    ),
    "deep": TierProfile(
        name="deep",
        when="trend / multi-hop / 'comprehensive report'",
        planner="use planner-agent",
        writer="delegate to writer-agent",
        width="up to the width cap",
        tools="full set incl. advanced web search",
        finalize="writer-agent → return /shared/output.md marker",
    ),
}


def _rank(tier: str) -> int:
    return _TIER_ORDER.index(tier)


def normalize_enabled_tiers(enabled: list[str] | None) -> list[str]:
    """Return the enabled tiers in canonical rank order, de-duplicated.

    Falls back to all four tiers when ``enabled`` is empty/None so callers never operate on an
    empty allow-list (the config validator also enforces a non-empty list).
    """
    if not enabled:
        return list(_TIER_ORDER)
    seen = set()
    ordered: list[str] = []
    for tier in _TIER_ORDER:
        if tier in enabled and tier not in seen:
            seen.add(tier)
            ordered.append(tier)
    return ordered or list(_TIER_ORDER)


def clamp_to_enabled_tiers(tier: str, enabled: list[str]) -> str:
    """Snap ``tier`` to the nearest enabled tier by rank; ties resolve to the deeper tier.

    Deterministic and LLM-free. Used both to floor/ceiling the effort the run maps to and to
    compute the Layer-B tool-exposure ceiling. Ties break toward the deeper tier because that
    is the safer choice for answer quality.
    """
    allowed = normalize_enabled_tiers(enabled)
    if tier in allowed:
        return tier
    idx = _rank(tier)
    return min(allowed, key=lambda t: (abs(_rank(t) - idx), -_rank(t)))


def enabled_tier_profiles(enabled: list[str] | None) -> list[TierProfile]:
    """Return the ``TierProfile`` objects for the enabled tiers, in rank order (for the prompt)."""
    return [TIER_PROFILES[t] for t in normalize_enabled_tiers(enabled)]


def tier_ceiling(enabled: list[str] | None) -> str:
    """Return the highest-effort enabled tier (the deep-most tier the agent may reach)."""
    return normalize_enabled_tiers(enabled)[-1]


# ---------------------------------------------------------------------------
# Dynamic prompt sections (opt-in via ``dynamic_orchestrator_sections``)
# ---------------------------------------------------------------------------
# The orchestrator prompt (orchestrator.j2) wraps every logical block in
# ``{% if S.get('<flag>', True) %}`` so it can be trimmed per effort tier. To avoid
# re-sending the full deep/writer/delta machinery on cheap ``direct``/``single_shot``
# runs, we render only the sections a given mode needs.
#
# Flow (see misc/adaptive-orchestrator-skip-tier-turn-plan.md):
#   * Turn 1 uses the "catalog" mode — the union of every enabled tier's sections, so the
#     model can select a tier AND take that tier's first action in the same turn.
#   * Once the tier is resolved (declared in that same batch, or inferred from the action),
#     ComplexityRouterMiddleware swaps in the trimmed prompt for that tier (or "delta" for
#     parent-report rewrites) for the rest of the run.
#
# The catalog is computed from the enabled subset rather than stored as a preset: a
# shallow-only config must not pay for the deep/writer machinery it can never reach. That is
# why ``SECTION_PRESETS`` holds only the per-tier modes and ``sections_for_catalog`` derives
# the turn-1 map dynamically.

# Every section flag the template understands, in template render order. Listing them here
# (a) makes a preset that forgets a flag obvious, and (b) lets ``sections_for_tier`` build a
# deterministic, fully-populated map so the same mode always renders a byte-identical prefix
# (required for prompt KV-cache stability across a run's model calls).
SECTION_FLAGS: tuple[str, ...] = (
    "intro",
    "effort_catalog",
    "effort_selection",
    "research_depth",
    "delta_rule",
    "subagents",
    "research_routing",
    "filesystem",
    "sequential_handoffs",
    "workflow",
    "research_loop",
    "escalation",
    "stopping",
    "finalize",
    "citation_contract",
    "important",
)

# One entry per prompt "mode": the set of sections that are ON for that mode. Every flag not
# listed is OFF. Encodes the tier→sections table in the design doc (§3). Modeling presets as
# ON-sets (rather than full boolean dicts) keeps them short and hard to misread.
#
# There is deliberately no "catalog" entry: the turn-1 catalog depends on which tiers are
# enabled, while these values are static frozensets. ``sections_for_catalog`` unions the
# relevant entries below instead.
#
#   direct      — no research; answer inline via the ### direct workflow block, then finalize.
#   single_shot — one inline retrieval batch; needs the research loop + inline citation rules.
#   standard    — union of the inline and writer branches (so either path is fully specified):
#                 inline needs citation rules; the writer branch needs subagents/filesystem/handoffs.
#   deep        — full plan → research → writer pipeline (no inline citation rules; writer owns those).
#   delta       — parent-report rewrite: the deep machinery PLUS the delta rule. Never inline.
#
# ``escalation`` is listed for tiers that *could* step up; sections_for_tier() drops it when no
# higher tier is enabled (nothing to escalate to).
SECTION_PRESETS: dict[str, frozenset[str]] = {
    "direct": frozenset({"intro", "workflow", "finalize", "important"}),
    "single_shot": frozenset(
        {
            "intro",
            "research_depth",
            "research_routing",
            "workflow",
            "research_loop",
            "escalation",
            "stopping",
            "finalize",
            "citation_contract",
            "important",
        }
    ),
    "standard": frozenset(
        {
            "intro",
            "research_depth",
            "subagents",
            "research_routing",
            "filesystem",
            "sequential_handoffs",
            "workflow",
            "research_loop",
            "escalation",
            "stopping",
            "finalize",
            "citation_contract",
            "important",
        }
    ),
    "deep": frozenset(
        {
            "intro",
            "research_depth",
            "subagents",
            "research_routing",
            "filesystem",
            "sequential_handoffs",
            "workflow",
            "research_loop",
            "stopping",
            "finalize",
            "important",
        }
    ),
    "delta": frozenset(
        {
            "intro",
            "research_depth",
            "delta_rule",
            "subagents",
            "research_routing",
            "filesystem",
            "sequential_handoffs",
            "workflow",
            "research_loop",
            "stopping",
            "finalize",
            "important",
        }
    ),
}


def escalation_possible(tier: str, enabled: list[str] | None) -> bool:
    """Return True when a higher enabled tier exists to step up to from ``tier``.

    Used to decide whether the ``escalation`` prompt section is worth including: if the run's
    effort maps to the deep-most enabled tier already, there is nothing to escalate to, so the
    section is dropped. ``tier`` is snapped into the enabled set first (via
    ``clamp_to_enabled_tiers``) so a disabled tier name still compares sensibly.
    """
    return _rank(tier_ceiling(enabled)) > _rank(clamp_to_enabled_tiers(tier, enabled))


def sections_for_tier(mode: str, *, enabled: list[str] | None) -> dict[str, bool]:
    """Expand a preset into a full ``{flag: bool}`` map for rendering orchestrator.j2.

    ``mode`` is a key of ``SECTION_PRESETS`` (a tier name or "delta"). The result always
    contains every flag in ``SECTION_FLAGS`` (in order) so rendering is deterministic and
    byte-stable per mode. The ``escalation`` section is included only when the preset turns it
    on *and* a higher enabled tier actually exists (``escalation_possible``).
    """
    on_flags = SECTION_PRESETS[mode]
    resolved = {flag: (flag in on_flags) for flag in SECTION_FLAGS}
    if resolved["escalation"]:
        resolved["escalation"] = escalation_possible(mode, enabled)
    return resolved


def sections_for_catalog(enabled: list[str] | None) -> dict[str, bool]:
    """Build the turn-1 "catalog" section map: the union of every enabled tier's sections.

    Catalog mode replaces the old two-turn router flow. The orchestrator must be able to pick a
    tier *and* run that tier's first step in one turn, so the turn-1 prompt has to carry every
    enabled tier's procedure — but only the enabled ones, which is why this is derived rather
    than stored as a static preset (a shallow-only config must not pay for the deep/writer
    machinery it can never reach).

    Rules:

    - union of ``SECTION_PRESETS`` over the enabled tiers, so every reachable procedure block is
      described;
    - ``effort_catalog`` / ``effort_selection`` forced on — the model is choosing, so it needs
      the level descriptions and the selection contract;
    - ``delta_rule`` forced OFF: a parent-report request is detected at graph build time and gets
      the dedicated "delta" prompt instead, so a normal catalog never needs the delta machinery;
    - ``escalation`` kept only when more than one tier is enabled (nothing to step up to
      otherwise).

    Returns a fully populated map in ``SECTION_FLAGS`` order so the rendered prefix is
    byte-identical for a given enabled set (prompt KV-cache stability).
    """
    allowed = normalize_enabled_tiers(enabled)
    on_flags: set[str] = set()
    for tier in allowed:
        on_flags |= SECTION_PRESETS[tier]
    on_flags |= {"effort_catalog", "effort_selection"}
    on_flags.discard("delta_rule")
    resolved = {flag: (flag in on_flags) for flag in SECTION_FLAGS}
    # More than one enabled tier means some tier can step up; with exactly one there is nowhere
    # to escalate to, so the section would be dead text.
    resolved["escalation"] = len(allowed) > 1
    return resolved
