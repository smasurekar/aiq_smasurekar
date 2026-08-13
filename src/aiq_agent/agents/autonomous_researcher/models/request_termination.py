# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated configuration for request-wide autonomous-researcher termination.

Two independent limit layers already exist and are *not* duplicated here:

1. ``ResearcherLoopGuardConfig`` bounds **one** delegated researcher invocation (source calls,
   repeated calls, think loops). Its budget resets for every new invocation, so it cannot bound
   the top-level request.
2. ``deep_researcher.resource_limits.DeepResearchResourceLimits`` is upstream's tier-independent
   hard-limits layer: per-job query counts, note/plan/report byte ceilings, state-file budgets,
   todo quotas, and the execution-seconds ceiling. It is already "one flat set of caps" and is
   wired into the graph context unchanged.

This config holds **only** what neither of those covers: how many ``run_research_batch`` calls
one request may make, how many orchestrator model turns it may spend, how often one normalized
query may repeat, the hard wall-clock deadline around ``ainvoke``, and the LangGraph recursion
ceiling.

Unlike the adaptive agent's ``AdaptiveRequestTerminationConfig`` there is no per-tier lookup and
no ``budgets_for_tier()``: one flat budget set always applies. That is a strict improvement —
under the tier design ``single_shot`` and ``direct`` resolved to ``None``, i.e. no request-wide
guard at all.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from aiq_agent.agents.deep_researcher.resource_limits import DEFAULT_MAX_RESEARCH_QUERIES


class AutonomousRequestTerminationConfig(BaseModel):
    """Request-wide budgets, deadlines, and the recursion ceiling for one autonomous request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(
        default=True,
        description="Enable request-wide batch/query/turn enforcement and the workflow deadline.",
    )
    max_batch_calls: int = Field(
        default=6,
        ge=1,
        description="Maximum run_research_batch calls the orchestrator may make in one request.",
    )
    max_total_research_queries: int = Field(
        default=DEFAULT_MAX_RESEARCH_QUERIES,
        ge=1,
        description=(
            "Maximum delegated ResearchQuery items summed across every batch in one request. "
            "DeepResearchResourceLimits.max_research_queries is the hard per-job ceiling above "
            "this; setting a higher value here has no effect."
        ),
    )
    max_orchestrator_turns: int = Field(
        default=100,
        ge=1,
        description="Maximum orchestrator model turns before finalization is forced for this request.",
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
            "deterministic partial-result fallback; must stay below workflow_timeout_seconds."
        ),
    )
    recursion_limit: int = Field(
        default=250,
        ge=1,
        description="LangGraph recursion ceiling — a last-resort safety net below the batch/query/turn budgets.",
    )

    @model_validator(mode="after")
    def _validate_relationships(self) -> AutonomousRequestTerminationConfig:
        # A finalizer that could outlive the whole request would defeat the workflow deadline.
        if self.fallback_finalizer_timeout_seconds >= self.workflow_timeout_seconds:
            raise ValueError(
                "request_termination fallback_finalizer_timeout_seconds must be strictly less than "
                "workflow_timeout_seconds"
            )
        return self


RequestPhase = Literal["active", "finalizing", "terminal"]
