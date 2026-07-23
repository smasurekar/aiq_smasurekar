# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated configuration for request-wide adaptive-researcher termination.

The per-researcher loop guard (``ResearcherLoopGuardConfig``) bounds *one* delegated
researcher invocation. It cannot bound the *top-level request*: each ``run_research_batch``
call spawns fresh researcher invocations with fresh budgets, so an orchestrator that keeps
authoring new batches can run indefinitely even while every per-researcher guard fires
correctly. This config bounds the whole request instead — the number of research batches, the
total delegated queries, repeated identical queries, orchestrator model turns, the graph
recursion ceiling, and a hard wall-clock deadline for the entire workflow.

Kept deliberately separate from ``ResearcherLoopGuardConfig`` because the two protect different
lifetimes: the loop guard protects one researcher; this protects the request. Defaults are
enabled and finite so production is bounded without operators having to opt in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

# Tiers that delegate research through ``run_research_batch`` and are therefore bounded by this
# guard. ``single_shot`` keeps its own direct-search budget (``single_shot_search_budget``) and
# ``direct`` / ``meta`` perform no delegated research, so all three are left inert here.
_STANDARD_TIER = "standard"
_DEEP_TIERS = frozenset({"deep", "delta"})


class AdaptiveTierBudgets(BaseModel):
    """Per-tier request-wide budgets for orchestrator research delegation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_batch_calls: int = Field(
        default=3,
        ge=1,
        description="Maximum run_research_batch calls the orchestrator may make in one request.",
    )
    max_total_research_queries: int = Field(
        default=9,
        ge=1,
        description="Maximum delegated ResearchQuery items summed across every batch in one request.",
    )
    max_orchestrator_turns: int = Field(
        default=24,
        ge=1,
        description="Maximum orchestrator model turns before finalization is forced for this request.",
    )

    def _at_least(self, other: AdaptiveTierBudgets) -> bool:
        """Return True when every budget here is >= the corresponding budget in ``other``."""
        return (
            self.max_batch_calls >= other.max_batch_calls
            and self.max_total_research_queries >= other.max_total_research_queries
            and self.max_orchestrator_turns >= other.max_orchestrator_turns
        )


def _default_standard_budgets() -> AdaptiveTierBudgets:
    return AdaptiveTierBudgets(max_batch_calls=3, max_total_research_queries=9, max_orchestrator_turns=24)


def _default_deep_budgets() -> AdaptiveTierBudgets:
    return AdaptiveTierBudgets(max_batch_calls=6, max_total_research_queries=24, max_orchestrator_turns=100)


class AdaptiveRequestTerminationConfig(BaseModel):
    """Request-wide budgets, deadlines, and the recursion ceiling for one adaptive request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(
        default=True,
        description="Enable request-wide batch/query/turn enforcement, the workflow deadline, and the fallback.",
    )
    standard: AdaptiveTierBudgets = Field(
        default_factory=_default_standard_budgets,
        description="Request-wide budgets applied when the declared effort tier is 'standard'.",
    )
    deep: AdaptiveTierBudgets = Field(
        default_factory=_default_deep_budgets,
        description="Request-wide budgets applied when the declared effort tier is 'deep' or a parent-report delta.",
    )
    max_identical_research_queries: int = Field(
        default=1,
        ge=1,
        description="Maximum executions of one normalized ResearchQuery signature across all batches in a request.",
    )
    workflow_timeout_seconds: int = Field(
        default=1200,
        ge=1,
        description="Hard wall-clock deadline around the entire agent.ainvoke workflow. Exceeding it forces fallback.",
    )
    fallback_finalizer_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description=(
            "Independent deadline reserved for a bounded tool-free finalizer. Unused by the "
            "deterministic first-slice fallback; must stay below workflow_timeout_seconds."
        ),
    )
    recursion_limit: int = Field(
        default=250,
        ge=1,
        description="LangGraph recursion ceiling — a last-resort safety net below the batch/query/turn budgets.",
    )

    @model_validator(mode="after")
    def _validate_relationships(self) -> AdaptiveRequestTerminationConfig:
        # deep must be at least as permissive as standard on every budget so escalating effort
        # never tightens limits.
        if not self.deep._at_least(self.standard):
            raise ValueError(
                "request_termination 'deep' budgets must be >= the corresponding 'standard' budgets "
                "(escalating effort must not reduce any limit)"
            )
        # A finalizer that could outlive the whole request would defeat the workflow deadline.
        if self.fallback_finalizer_timeout_seconds >= self.workflow_timeout_seconds:
            raise ValueError(
                "request_termination fallback_finalizer_timeout_seconds must be strictly less than "
                "workflow_timeout_seconds"
            )
        return self

    def budgets_for_tier(self, tier: str | None) -> AdaptiveTierBudgets | None:
        """Return the request-wide budgets for a declared tier, or ``None`` when the guard is inert.

        ``standard`` maps to the standard budgets; ``deep`` and a parent-report ``delta`` rewrite
        map to the deep budgets. ``single_shot``, ``direct``, ``meta``, and the pre-declaration
        state (``None``) return ``None`` — those paths either self-limit (single_shot's own search
        budget) or perform no delegated research, so this guard does not bound them.
        """
        if tier == _STANDARD_TIER:
            return self.standard
        if tier in _DEEP_TIERS:
            return self.deep
        return None


RequestPhase = Literal["active", "finalizing", "terminal"]
