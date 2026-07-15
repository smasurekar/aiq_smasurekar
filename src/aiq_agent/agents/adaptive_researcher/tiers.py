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
        when="a lightly multi-part question",
        planner="skip for inline; required for writer-agent",
        writer="inline or Planned Writer Pipeline",
        width="~3–5 queries",
        tools="basic web + advanced web if needed",
        finalize="inline → submit_final_report(researched=true), or writer-agent → return marker",
    ),
    "deep": TierProfile(
        name="deep",
        when="comparison / trend / multi-hop / 'comprehensive report'",
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
