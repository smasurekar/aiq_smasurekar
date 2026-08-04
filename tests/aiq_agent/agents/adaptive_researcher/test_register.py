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


def test_single_loop_single_shot_defaults_false():
    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
    assert config.single_loop_single_shot is False


def test_single_shot_researcher_llm_defaults_none():
    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
    assert config.single_shot_researcher_llm is None


def test_single_loop_single_shot_can_be_enabled():
    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm", single_loop_single_shot=True)
    assert config.single_loop_single_shot is True


def test_single_shot_search_budget_defaults_to_two():
    from aiq_agent.agents.adaptive_researcher.agent import DEFAULT_SINGLE_SHOT_SEARCH_BUDGET

    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
    assert config.single_shot_search_budget == DEFAULT_SINGLE_SHOT_SEARCH_BUDGET == 2


def test_single_shot_search_budget_can_be_overridden():
    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm", single_shot_search_budget=4)
    assert config.single_shot_search_budget == 4


def test_single_shot_search_budget_rejects_below_one():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AdaptiveResearchAgentConfig(orchestrator_llm="llm", single_shot_search_budget=0)


def test_researcher_loop_guard_defaults_match_prompt_budgets():
    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
    guard = config.researcher_loop_guard
    assert guard.enabled is True
    assert guard.source_call_budgets.model_dump() == {"low": 1, "medium": 3, "high": 6}
    assert guard.max_identical_source_calls == 2
    assert guard.max_consecutive_thinks == 3


def test_researcher_loop_guard_accepts_nested_overrides():
    config = AdaptiveResearchAgentConfig(
        orchestrator_llm="llm",
        researcher_loop_guard={
            "source_call_budgets": {"low": 2, "medium": 4, "high": 8},
            "max_identical_source_calls": 1,
            "max_consecutive_thinks": 2,
        },
    )
    assert config.researcher_loop_guard.source_call_budgets.high == 8
    assert config.researcher_loop_guard.max_identical_source_calls == 1


@pytest.mark.parametrize(
    "guard",
    [
        {"source_call_budgets": {"low": 0, "medium": 3, "high": 6}},
        {"source_call_budgets": {"low": 4, "medium": 3, "high": 6}},
        {"max_identical_source_calls": 0},
        {"unknown_option": True},
    ],
)
def test_researcher_loop_guard_rejects_invalid_configuration(guard):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AdaptiveResearchAgentConfig(orchestrator_llm="llm", researcher_loop_guard=guard)


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


# ---------------------------------------------------------------------------
# single_shot shallow sub-agent config
# ---------------------------------------------------------------------------


def test_single_shot_shallow_subagent_defaults_false():
    config = AdaptiveResearchAgentConfig(orchestrator_llm="llm")
    assert config.single_shot_shallow_subagent is False
    assert config.shallow_subagent_max_llm_turns == 10
    assert config.shallow_subagent_max_tool_iterations == 5


def test_single_shot_shallow_subagent_can_be_enabled():
    config = AdaptiveResearchAgentConfig(
        orchestrator_llm="llm",
        single_shot_shallow_subagent=True,
        shallow_subagent_max_tool_iterations=3,
    )
    assert config.single_shot_shallow_subagent is True
    assert config.shallow_subagent_max_tool_iterations == 3


def test_shallow_subagent_bounds_reject_below_one():
    with pytest.raises(ValueError):
        AdaptiveResearchAgentConfig(orchestrator_llm="llm", shallow_subagent_max_tool_iterations=0)
    with pytest.raises(ValueError):
        AdaptiveResearchAgentConfig(orchestrator_llm="llm", shallow_subagent_max_llm_turns=0)


def test_the_two_single_shot_modes_are_mutually_exclusive():
    """Both own the single_shot execution path, so this is a config error, not a precedence rule."""
    with pytest.raises(ValueError, match="enable at most one"):
        AdaptiveResearchAgentConfig(
            orchestrator_llm="llm",
            single_shot_shallow_subagent=True,
            single_loop_single_shot=True,
        )


def test_either_single_shot_mode_alone_is_accepted():
    assert AdaptiveResearchAgentConfig(orchestrator_llm="llm", single_loop_single_shot=True)
    assert AdaptiveResearchAgentConfig(orchestrator_llm="llm", single_shot_shallow_subagent=True)
    assert AdaptiveResearchAgentConfig(orchestrator_llm="llm")


def test_agent_construction_forwards_every_new_field():
    """Both AdaptiveResearcherAgent construction sites in register.py must stay in sync."""
    import inspect

    from aiq_agent.agents.adaptive_researcher import register as register_module

    source = inspect.getsource(register_module)
    for field in (
        "single_shot_shallow_subagent=config.single_shot_shallow_subagent",
        "shallow_subagent_max_llm_turns=config.shallow_subagent_max_llm_turns",
        "shallow_subagent_max_tool_iterations=config.shallow_subagent_max_tool_iterations",
    ):
        assert source.count(field) == 2, f"{field} must be forwarded at both construction sites"
