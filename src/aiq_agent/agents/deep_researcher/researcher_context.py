# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-invocation context for reusable researcher workers."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

ResearchDepth = Literal["low", "medium", "high"]


@dataclass
class ResearcherRunGuardState:
    """Mutable loop-guard state isolated to one researcher invocation."""

    invocation_id: str
    depth: ResearchDepth
    source_call_count: int = 0
    source_signature_counts: dict[str, int] = field(default_factory=dict)
    exhausted: bool = False
    exhaustion_reason: str | None = None
    consecutive_think_count: int = 0
    think_blocked: bool = False


CURRENT_RESEARCHER_GUARD_STATE: ContextVar[ResearcherRunGuardState | None] = ContextVar(
    "current_researcher_guard_state",
    default=None,
)


def normalize_research_depth(value: object) -> ResearchDepth:
    """Return a supported research depth, defaulting shared deep queries to medium."""
    if value in ("low", "medium", "high"):
        return value
    return "medium"
