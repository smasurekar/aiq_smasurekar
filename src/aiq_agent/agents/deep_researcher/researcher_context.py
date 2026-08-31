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
    # --- Blocked source-call accounting ---------------------------------------------------
    # A rejected source call is not free: the model still spends a turn on it and re-sends the
    # whole transcript on the next one. ``blocked_source_calls`` is the per-worker total, reported
    # once at worker completion so a job log can be graded without replaying it.
    # ``consecutive_blocked_source_calls`` is the run of rejections since the last source call that
    # actually executed, and is what trips the forced-return ceiling. ``think`` neither increments
    # nor resets that run - alternating think / repeat-search is the exact loop it exists to break.
    blocked_source_calls: int = 0
    consecutive_blocked_source_calls: int = 0
    # Latched once the ceiling trips. Tool withdrawal is advisory, never enforcement: LangChain
    # routes a tool call to the tools node by *registered* name (``_make_model_to_tools_edge``),
    # not by the middleware-narrowed list the model was shown, so a model replaying a withdrawn
    # name out of its own history still gets it executed. This flag drives the deterministic
    # escape hatch instead - a model call bound to the structured-output tool alone.
    force_structured_return: bool = False
    # Diagnostics only. Together these answer "did the withdrawal reach the model?": a worker with
    # ``tools_withdrawn_model_calls`` > 0 and a still-rising ``blocked_source_calls`` ignored it.
    tools_withdrawn_model_calls: int = 0
    forced_return_model_calls: int = 0


CURRENT_RESEARCHER_GUARD_STATE: ContextVar[ResearcherRunGuardState | None] = ContextVar(
    "current_researcher_guard_state",
    default=None,
)


def normalize_research_depth(value: object) -> ResearchDepth:
    """Return a supported research depth, defaulting shared deep queries to medium."""
    if value in ("low", "medium", "high"):
        return value
    return "medium"
