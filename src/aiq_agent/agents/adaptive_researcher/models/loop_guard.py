# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated configuration for adaptive researcher loop guards."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from aiq_agent.agents.deep_researcher.researcher_context import ResearchDepth


class ResearcherSourceCallBudgets(BaseModel):
    """Maximum source-tool calls for each adaptive research depth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    low: int = Field(default=1, ge=1)
    medium: int = Field(default=3, ge=1)
    high: int = Field(default=6, ge=1)

    @model_validator(mode="after")
    def _validate_order(self) -> ResearcherSourceCallBudgets:
        if not self.low <= self.medium <= self.high:
            raise ValueError("researcher source-call budgets must satisfy low <= medium <= high")
        return self

    def for_depth(self, depth: ResearchDepth) -> int:
        """Return the configured source-call budget for one query depth."""
        return getattr(self, depth)


class ResearcherLoopGuardConfig(BaseModel):
    """Hard limits for one adaptive researcher sub-agent invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    source_call_budgets: ResearcherSourceCallBudgets = Field(default_factory=ResearcherSourceCallBudgets)
    max_identical_source_calls: int = Field(default=2, ge=1)
    max_consecutive_thinks: int = Field(default=3, ge=1)
    # The hard backstop behind tool withdrawal, which only changes what the model is *told* about:
    # a model that re-emits a withdrawn-but-registered tool name from its own history still gets it
    # executed, so a rejected source call can repeat indefinitely (one eval worker did it 116 times
    # for 6.1M input tokens). After this many consecutive rejections the guard stops asking and
    # forces the structured return. Three is ample: two informative rejections, then out.
    max_consecutive_blocked_source_calls: int = Field(default=3, ge=1)
