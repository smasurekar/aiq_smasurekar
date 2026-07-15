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

"""State model for the adaptive research agent.

The adaptive researcher shares the deep researcher's state shape exactly for the first
iteration; it is subclassed here only to give the adaptive agent a distinct type name and a
future extension point (e.g. an effort/tier trace field). No new required fields are added,
so the deep-research runtime, tools, and middleware all operate on it unchanged.
"""

from aiq_agent.agents.deep_researcher.models.state import DeepResearchAgentState


class AdaptiveResearchAgentState(DeepResearchAgentState):
    """State for the adaptive research agent (identical shape to deep research for iteration 1)."""
