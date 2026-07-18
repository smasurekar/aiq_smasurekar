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

"""Models for the adaptive research agent.

Subagent contracts (ResearchPlan / ResearchQuery / ResearchNotes / ...) are reused verbatim
from ``deep_researcher`` to avoid drift; only the state model is subclassed locally.
"""

from aiq_agent.agents.deep_researcher.models import AnswerComponent
from aiq_agent.agents.deep_researcher.models import AnswerStrategy
from aiq_agent.agents.deep_researcher.models import Constraint
from aiq_agent.agents.deep_researcher.models import EvidenceJudgment
from aiq_agent.agents.deep_researcher.models import ResearchFinding
from aiq_agent.agents.deep_researcher.models import ResearchGap
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.deep_researcher.models import ResearchPlan
from aiq_agent.agents.deep_researcher.models import ResearchQuery
from aiq_agent.agents.deep_researcher.models import ResearchSource
from aiq_agent.agents.deep_researcher.models import SourceRecommendation
from aiq_agent.agents.deep_researcher.models import SourceRoutingPlan
from aiq_agent.agents.deep_researcher.models import TaskAnalysis

from .state import AdaptiveResearchAgentState
from .subagent_contracts import AdaptiveResearchPlan
from .subagent_contracts import AdaptiveResearchQuery

__all__ = [
    "AdaptiveResearchAgentState",
    "AdaptiveResearchPlan",
    "AdaptiveResearchQuery",
    "AnswerComponent",
    "AnswerStrategy",
    "Constraint",
    "EvidenceJudgment",
    "ResearchFinding",
    "ResearchGap",
    "ResearchNotes",
    "ResearchPlan",
    "ResearchQuery",
    "ResearchSource",
    "SourceRecommendation",
    "SourceRoutingPlan",
    "TaskAnalysis",
]
