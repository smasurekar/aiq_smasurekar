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

"""Adaptive-only subagent contracts.

These subclass the shared ``deep_researcher`` schemas to add a per-query ``depth`` hint
without touching ``deep_researcher``. ``depth`` is orthogonal to breadth (the number of
queries the orchestrator fans out): the orchestrator and planner set it per query, and the
adaptive researcher prompt scales its search/iteration budget to it.
"""

from typing import Literal

from pydantic import Field

from aiq_agent.agents.deep_researcher.models import ResearchPlan
from aiq_agent.agents.deep_researcher.models import ResearchQuery


class AdaptiveResearchQuery(ResearchQuery):
    """A ResearchQuery carrying a per-query research ``depth`` hint for the researcher worker."""

    depth: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Research depth for this query, independent of how many queries are fanned out: "
            "'low' = one quick source-tool lookup, return immediately; "
            "'medium' = a few corroborating searches; "
            "'high' = iterative multi-hop, chaining several sequential searches where each "
            "result informs the next. Use 'high' for multi-hop questions and 'low' for simple "
            "self-contained lookups."
        ),
    )


class AdaptiveResearchPlan(ResearchPlan):
    """A ResearchPlan whose queries carry the per-query ``depth`` hint."""

    queries: list[AdaptiveResearchQuery] = Field(
        description="Queries for researcher workers to execute, each with a per-query depth hint."
    )
