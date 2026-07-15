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

"""Tests for adaptive research registration (config markers + workflow wrapper wiring)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_agent.agents.adaptive_researcher.register import AdaptiveResearchAgentConfig
from aiq_agent.agents.adaptive_researcher.register import AdaptiveResearchWorkflowConfig
from aiq_agent.agents.adaptive_researcher.register import adaptive_research_workflow


def test_config_type_markers():
    assert AdaptiveResearchAgentConfig.static_type() == "adaptive_research_agent"
    assert AdaptiveResearchWorkflowConfig.static_type() == "adaptive_research_workflow"


def test_config_reuses_deep_runtime_models():
    from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
    from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSkillsConfig

    config = AdaptiveResearchAgentConfig(
        orchestrator_llm="llm",
        skills=DeepResearchSkillsConfig(agents={"writer-agent": ("synthesis",)}),
        sandbox=DeepResearchSandboxConfig(app_name="custom-aiq"),
        enabled_tiers=["single_shot", "deep"],
    )
    assert config.skills.agents == {"writer-agent": ("synthesis",)}
    assert config.sandbox.app_name == "custom-aiq"
    assert config.enabled_tiers == ["single_shot", "deep"]


@pytest.mark.asyncio
async def test_workflow_wrapper_invokes_adaptive_agent_by_name():
    """The workflow wrapper looks up adaptive_research_agent by fixed name and wraps a string query."""
    agent_fn = MagicMock()
    agent_fn.ainvoke = AsyncMock(return_value=MagicMock(messages=[MagicMock(content="final answer")]))
    builder = MagicMock()
    builder.get_function = AsyncMock(return_value=agent_fn)

    config = AdaptiveResearchWorkflowConfig()
    registration = adaptive_research_workflow.__wrapped__(config, builder)
    function_info = await anext(registration)
    try:
        response = await function_info.single_fn("what is X?")
    finally:
        await registration.aclose()

    builder.get_function.assert_awaited_once_with("adaptive_research_agent")
    agent_fn.ainvoke.assert_awaited_once()
    # the string query is wrapped into a HumanMessage state
    state_arg = agent_fn.ainvoke.call_args.args[0]
    assert state_arg.messages[0].content == "what is X?"
    assert "final answer" in str(response)
