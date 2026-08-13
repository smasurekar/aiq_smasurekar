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

"""Graph, prompt, and subagent wiring for the autonomous researcher.

The assertions here are the architectural contract, not incidental detail: the orchestrator holds
the full retrieval menu, exactly three subagents can act, deepagents' default general-purpose
subagent is never built, and no tier artifact survives anywhere in the rendered prompt.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.autonomous_researcher.agent import AutonomousResearcherAgent
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalReportCommitTracker
from aiq_agent.agents.autonomous_researcher.factory import GENERAL_PURPOSE_SUBAGENT_NAME
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchAgentState
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchPlan
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole

# Strings that would prove some part of the tier machinery leaked into this agent.
TIER_ARTIFACTS = (
    "declare_effort_tier",
    "effort tier",
    "Effort Levels",
    "Choosing Effort",
    "single_shot",
    "shallow-researcher",
    "source-router-agent",
    "enabled_tiers",
)


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool
def knowledge_search(query: str) -> str:
    """Search uploaded documents."""
    return f"Docs for: {query}"


@pytest.fixture
def mock_llm_provider():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    provider = LLMProvider()
    provider.set_default(llm)
    for role in (LLMRole.ORCHESTRATOR, LLMRole.PLANNER, LLMRole.RESEARCHER, LLMRole.REPORT_WRITER):
        provider.configure(role, llm)
    return provider


class _FakeSummarizationMiddleware(AgentMiddleware):
    pass


def _build_and_capture(mock_llm_provider, *, state=None, tools=None, **agent_kwargs) -> dict:
    """Build the orchestrator graph and return the ``create_deep_agent`` kwargs."""
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    with (
        patch("aiq_agent.agents.autonomous_researcher.factory.create_deep_agent", return_value=graph) as create,
        patch("aiq_agent.agents.deep_researcher.factory.create_agent", return_value=graph),
        patch(
            "aiq_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=_FakeSummarizationMiddleware(),
        ),
    ):
        agent = AutonomousResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=tools if tools is not None else [web_search_tool, knowledge_search],
            **agent_kwargs,
        )
        state = state or AutonomousResearchAgentState(messages=[HumanMessage(content="q")])
        agent._build_orchestrator_agent(state, AutonomousFinalReportCommitTracker())
    return dict(create.call_args.kwargs)


def _middleware_names(captured: dict) -> list[str]:
    return [type(m).__name__ for m in captured["middleware"]]


class TestOrchestratorTools:
    """The orchestrator holds the full menu, unconditionally."""

    def test_holds_source_tools_directly_alongside_batch_and_finalize(self, mock_llm_provider):
        names = [t.name for t in _build_and_capture(mock_llm_provider)["tools"]]
        assert {"think", "get_verified_sources", "run_research_batch", "submit_final_report"} <= set(names)
        assert {"web_search_tool", "knowledge_search"} <= set(names), "source tools must be held directly"

    def test_no_tier_declaration_tool(self, mock_llm_provider):
        names = [t.name for t in _build_and_capture(mock_llm_provider)["tools"]]
        assert "declare_effort_tier" not in names

    def test_source_tool_names_join_the_sanitizer_allowlist(self, mock_llm_provider):
        """Upstream excludes source tools there; here a source-tool name is a legitimate call."""
        captured = _build_and_capture(mock_llm_provider)
        sanitizer = next(m for m in captured["middleware"] if type(m).__name__ == "ToolNameSanitizationMiddleware")
        allowlist = set(sanitizer.valid_tool_names)
        assert {"web_search_tool", "knowledge_search", "submit_final_report", "run_research_batch"} <= allowlist


class TestSubagents:
    """`task` must advertise exactly three usable delegation routes."""

    def test_exactly_researcher_planner_writer_plus_inert_stub(self, mock_llm_provider):
        names = [s["name"] for s in _build_and_capture(mock_llm_provider)["subagents"]]
        assert names == ["researcher-agent", "planner-agent", "writer-agent", GENERAL_PURPOSE_SUBAGENT_NAME]

    def test_source_router_subagent_is_never_built(self, mock_llm_provider):
        names = [s["name"] for s in _build_and_capture(mock_llm_provider)["subagents"]]
        assert "source-router-agent" not in names

    def test_general_purpose_stub_is_inert(self, mock_llm_provider):
        """The default GP subagent inherits the parent's ENTIRE tool list; this one holds none."""
        specs = _build_and_capture(mock_llm_provider)["subagents"]
        gp = next(s for s in specs if s["name"] == GENERAL_PURPOSE_SUBAGENT_NAME)
        assert gp["tools"] == []
        assert "tools" in gp, "an omitted 'tools' key makes deepagents inherit the parent's tools"
        assert "submit_final_report" not in str(gp["tools"])
        assert "run_research_batch" not in str(gp["tools"])

    def test_general_purpose_description_points_back_at_researcher(self, mock_llm_provider):
        """Its description must not compete with researcher-agent for research delegation."""
        specs = _build_and_capture(mock_llm_provider)["subagents"]
        gp = next(s for s in specs if s["name"] == GENERAL_PURPOSE_SUBAGENT_NAME)
        assert "researcher-agent" in gp["description"]
        assert "researching complex questions" not in gp["description"], "deepagents' default description leaked"

    def test_researcher_subagent_returns_structured_notes(self, mock_llm_provider):
        specs = _build_and_capture(mock_llm_provider)["subagents"]
        researcher = next(s for s in specs if s["name"] == "researcher-agent")
        assert researcher["response_format"].__name__ == "ResearchNotes"
        assert [t.name for t in researcher["tools"]], "researcher must hold source tools"

    def test_planner_returns_depth_carrying_plan(self, mock_llm_provider):
        specs = _build_and_capture(mock_llm_provider)["subagents"]
        planner = next(s for s in specs if s["name"] == "planner-agent")
        assert planner["response_format"] is AutonomousResearchPlan

    def test_subagent_descriptions_route_without_effort_vocabulary(self, mock_llm_provider):
        """Descriptions are the routing mechanism; none may mention an effort level."""
        for spec in _build_and_capture(mock_llm_provider)["subagents"]:
            description = spec["description"]
            assert not any(artifact in description for artifact in TIER_ARTIFACTS), description


class TestOrchestratorPrompt:
    def test_carries_no_tier_artifacts(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert not [artifact for artifact in TIER_ARTIFACTS if artifact in prompt]

    def test_states_the_anti_memory_rule(self, mock_llm_provider):
        """The highest-risk deletion in the change: it is prompt-only now, so assert it exists."""
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "Never answer from memory" in prompt
        assert "time-sensitive" in prompt

    def test_differentiates_the_three_research_paths(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "Choosing a research path" in prompt
        for path in ("researcher-agent", "run_research_batch", "source tool directly"):
            assert path in prompt

    def test_lists_source_tools_as_directly_callable(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "web_search_tool" in prompt
        assert "You hold all of these directly" in prompt

    def test_delta_block_requires_planner_before_writer(self, mock_llm_provider):
        state = AutonomousResearchAgentState(
            messages=[HumanMessage(content="follow up")],
            files={"/shared/original_report.md": {"content": "# parent"}},
        )
        prompt = _build_and_capture(mock_llm_provider, state=state)["system_prompt"]
        assert "Parent-report delta" in prompt
        assert "planner-agent" in prompt
        assert "complete standalone revised report" in prompt

    def test_delta_block_absent_without_parent_context(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "Parent-report delta" not in prompt


class TestOrchestratorMiddleware:
    def test_writer_delegation_is_not_forced(self, mock_llm_provider):
        """RequiredWriterDelegationMiddleware would delete the valid inline exit."""
        assert "RequiredWriterDelegationMiddleware" not in _middleware_names(_build_and_capture(mock_llm_provider))

    def test_autonomous_seams_are_attached(self, mock_llm_provider):
        names = _middleware_names(_build_and_capture(mock_llm_provider))
        for required in (
            "DirectSourcePromotionMiddleware",
            "ResearcherTaskPersistenceMiddleware",
            "PlanBeforeWriterMiddleware",
            "AutonomousOrchestratorLoopGuardMiddleware",
            "AutonomousFinalizationMiddleware",
        ):
            assert required in names

    def test_no_tier_routing_middleware(self, mock_llm_provider):
        names = _middleware_names(_build_and_capture(mock_llm_provider))
        assert "ComplexityRouterMiddleware" not in names
        assert "SingleShotShallowDelegationMiddleware" not in names
        assert "SourceRoutingGuardMiddleware" not in names

    def test_direct_source_promotion_wraps_the_source_registry(self, mock_llm_provider):
        """Middleware compose first-is-outermost; promotion must see the registry's capture."""
        names = _middleware_names(_build_and_capture(mock_llm_provider))
        assert names.index("DirectSourcePromotionMiddleware") < names.index("SourceRegistryMiddleware")


class TestControlArmsAreUnaffected:
    """Building the autonomous agent must not mutate the deep or adaptive arms.

    This is the guard against the process-global harness-profile mechanism (option B in the
    design): if anyone ever swaps the zero-tool general-purpose spec for
    ``register_harness_profile(...)``, deepagents' module-level ``_HARNESS_PROFILES`` would leak
    into every agent sharing the model key — including the control arms this agent is measured
    against.
    """

    def test_deep_and_adaptive_subagents_are_byte_identical_after_building(self, mock_llm_provider):
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

        before = dict(GENERAL_PURPOSE_SUBAGENT)
        _build_and_capture(mock_llm_provider)
        assert dict(GENERAL_PURPOSE_SUBAGENT) == before

    def test_no_harness_profile_is_registered(self, mock_llm_provider):
        from deepagents.profiles.harness import harness_profiles

        before = dict(harness_profiles._HARNESS_PROFILES)
        _build_and_capture(mock_llm_provider)
        assert dict(harness_profiles._HARNESS_PROFILES) == before
