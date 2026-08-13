# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Models for the autonomous research agent.

Nothing here is copied. The subagent contracts (``ResearchPlan`` / ``ResearchQuery`` /
``ResearchNotes`` / ...) come from ``deep_researcher``; the depth-carrying query/plan subclasses
and the per-researcher loop-guard config come from ``adaptive_researcher``. None of those three
are tier-coupled — ``depth`` is a per-query knob, not an effort level — so they are *imported*
rather than forked, which keeps a single source of truth across all three research agents.

``AutonomousResearchAgentState`` is an alias for the adaptive state model (which is itself
``DeepResearchAgentState`` with a distinct name). The autonomous agent adds no state fields, and
sharing the class means the reused deep-researcher runtime, tools, and middleware operate on it
unchanged.

Only ``AutonomousRequestTerminationConfig`` is new: it replaces the tier-keyed
``AdaptiveRequestTerminationConfig`` + ``budgets_for_tier()`` with one flat budget set.
"""

from aiq_agent.agents.adaptive_researcher.models.loop_guard import ResearcherLoopGuardConfig
from aiq_agent.agents.adaptive_researcher.models.loop_guard import ResearcherSourceCallBudgets
from aiq_agent.agents.adaptive_researcher.models.state import AdaptiveResearchAgentState
from aiq_agent.agents.adaptive_researcher.models.subagent_contracts import AdaptiveResearchPlan
from aiq_agent.agents.adaptive_researcher.models.subagent_contracts import AdaptiveResearchQuery
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
from aiq_agent.agents.deep_researcher.models import TaskAnalysis

from .request_termination import AutonomousRequestTerminationConfig

# Autonomous-facing names for the shared models. Aliases, not subclasses: a subclass would be a
# second identity for the same shape and would have to be threaded through every reused
# deep/adaptive helper that is typed to the original.
AutonomousResearchAgentState = AdaptiveResearchAgentState
AutonomousResearchQuery = AdaptiveResearchQuery
AutonomousResearchPlan = AdaptiveResearchPlan

__all__ = [
    "AnswerComponent",
    "AnswerStrategy",
    "AutonomousRequestTerminationConfig",
    "AutonomousResearchAgentState",
    "AutonomousResearchPlan",
    "AutonomousResearchQuery",
    "Constraint",
    "EvidenceJudgment",
    "ResearchFinding",
    "ResearchGap",
    "ResearcherLoopGuardConfig",
    "ResearcherSourceCallBudgets",
    "ResearchNotes",
    "ResearchPlan",
    "ResearchQuery",
    "ResearchSource",
    "TaskAnalysis",
]
