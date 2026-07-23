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

"""Tests for the adaptive researcher graph/prompt factory wiring."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.adaptive_researcher.agent import AGENT_DIR
from aiq_agent.agents.adaptive_researcher.agent import AdaptiveResearcherAgent
from aiq_agent.agents.adaptive_researcher.models import AdaptiveResearchAgentState
from aiq_agent.agents.adaptive_researcher.tiers import enabled_tier_profiles
from aiq_agent.agents.adaptive_researcher.tiers import sections_for_tier
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

ORCHESTRATOR_ANCHOR = "## Context"  # first below-KV-boundary section in the rendered prompt


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@pytest.fixture
def mock_llm_provider():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    provider = LLMProvider()
    provider.set_default(llm)
    for role in (LLMRole.ORCHESTRATOR, LLMRole.ROUTER, LLMRole.PLANNER, LLMRole.RESEARCHER, LLMRole.REPORT_WRITER):
        provider.configure(role, llm)
    return provider


class _FakeSummarizationMiddleware(AgentMiddleware):
    pass


def _build_and_capture(mock_llm_provider, *, state=None, **agent_kwargs):
    """Build the orchestrator graph and return the create_deep_agent kwargs."""
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    with (
        patch("aiq_agent.agents.adaptive_researcher.factory.create_deep_agent", return_value=graph) as create,
        patch("aiq_agent.agents.deep_researcher.factory.create_agent", return_value=graph),
        patch(
            "aiq_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=_FakeSummarizationMiddleware(),
        ),
    ):
        agent = AdaptiveResearcherAgent(llm_provider=mock_llm_provider, tools=[web_search_tool], **agent_kwargs)
        state = state or AdaptiveResearchAgentState(messages=[HumanMessage(content="q")])
        agent._build_orchestrator_agent(state)
    return create.call_args.kwargs


class TestOrchestratorWiring:
    def test_orchestrator_holds_finalize_tool_and_excludes_source_tools(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider)
        tool_names = [t.name for t in kwargs["tools"]]
        assert tool_names == ["think", "get_verified_sources", "run_research_batch", "submit_final_report"]
        assert "web_search_tool" not in tool_names

    def test_no_source_routing_guard_middleware(self, mock_llm_provider):
        # The guard would deadlock the shallow path; the adaptive orchestrator must omit it,
        # even when the source router is enabled.
        for enable in (False, True):
            kwargs = _build_and_capture(mock_llm_provider, enable_source_router=enable)
            names = {m.__class__.__name__ for m in kwargs["middleware"]}
            assert "SourceRoutingGuardMiddleware" not in names

    def test_subagents_default_no_source_router(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider)
        subagents = {s["name"] for s in kwargs["subagents"]}
        assert subagents == {"planner-agent", "writer-agent"}

    def test_subagents_include_source_router_when_enabled(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider, enable_source_router=True)
        subagents = {s["name"] for s in kwargs["subagents"]}
        assert subagents == {"source-router-agent", "planner-agent", "writer-agent"}

    def test_complexity_router_absent_by_default(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider)
        names = {m.__class__.__name__ for m in kwargs["middleware"]}
        assert "ComplexityRouterMiddleware" not in names

    def test_complexity_router_present_when_enforced(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider, enforce_tier_tools=True, enabled_tiers=["single_shot"])
        names = {m.__class__.__name__ for m in kwargs["middleware"]}
        assert "ComplexityRouterMiddleware" in names

    def test_delta_preserves_delegation_under_shallow_enforcement(self, mock_llm_provider):
        state = AdaptiveResearchAgentState(
            messages=[HumanMessage(content="Update the report")],
            files={"/shared/parent_report_context.json": {"content": "{}"}},
        )
        kwargs = _build_and_capture(
            mock_llm_provider,
            state=state,
            enforce_tier_tools=True,
            enabled_tiers=["single_shot"],
        )
        middleware = next(m for m in kwargs["middleware"] if m.__class__.__name__ == "ComplexityRouterMiddleware")
        assert "task" not in middleware._hidden_tool_names
        assert "write_todos" not in middleware._hidden_tool_names


class TestOrchestratorPromptRendering:
    """The rewritten orchestrator prompt keeps a strict KV-cache boundary."""

    def _render(self, **over):
        tmpl = load_prompt(AGENT_DIR / "prompts", "orchestrator")
        base = dict(
            current_datetime="2026-07-15 10:00:00",
            user_info=None,
            available_documents=[],
            execution_enabled=False,
            skills_enabled=False,
            sandbox_workdir="/sandbox",
            sandbox_artifact_dir="/sandbox/artifacts",
            enable_source_router=False,
            max_research_concurrency=6,
            parent_report_context_available=False,
            clarifier_result=None,
            triage_hint="",
            single_loop_single_shot=False,
            enabled_tiers=["direct", "single_shot", "standard", "deep"],
            tier_profiles=enabled_tier_profiles(["direct", "single_shot", "standard", "deep"]),
            retrieval_tools=[{"name": "web_search_tool", "description": "basic web search"}],
            tools=[{"name": "run_research_batch", "description": "fan out"}],
        )
        base.update(over)
        return render_prompt_template(tmpl, **base)

    def _render_mode(self, mode, *, enabled=("direct", "single_shot", "standard", "deep")):
        """Render the prompt for one dynamic-sections mode ("router", a tier, or "delta").

        Mirrors what factory._render_orchestrator does: collapse enabled_tiers to the mode (for
        the ## Workflow blocks) and pass the mode's section on/off map.
        """
        enabled = list(enabled)
        tiers_for_mode = enabled if mode in ("router", "delta") else [mode]
        return self._render(
            enabled_tiers=tiers_for_mode,
            tier_profiles=enabled_tier_profiles(tiers_for_mode),
            parent_report_context_available=(mode == "delta"),
            sections=sections_for_tier(mode, enabled=enabled),
        )

    def test_source_template_retains_boundary_marker(self):
        tmpl = load_prompt(AGENT_DIR / "prompts", "orchestrator")
        assert "=== KV CACHE BOUNDARY" in tmpl

    def test_kv_prefix_invariant_across_per_request_inputs(self):
        class Doc:
            file_name = "q3.pdf"
            summary = "Q3 financials"

        a = self._render()
        b = self._render(
            current_datetime="2030-01-01 00:00:00",
            user_info={"name": "Sam", "email": "s@x.com"},
            available_documents=[Doc()],
            parent_report_context_available=True,
            clarifier_result="focus on EU market",
            triage_hint="Assessed as: simple factual query.",
        )
        assert ORCHESTRATOR_ANCHOR in a and ORCHESTRATOR_ANCHOR in b
        # Everything above the first dynamic section must be byte-identical across requests.
        assert a.split(ORCHESTRATOR_ANCHOR)[0] == b.split(ORCHESTRATOR_ANCHOR)[0]

    def test_per_request_facts_render_below_boundary(self):
        b = self._render(parent_report_context_available=True, triage_hint="Assessed as: simple factual query.")
        prefix, below = b.split(ORCHESTRATOR_ANCHOR)[0], b.split(ORCHESTRATOR_ANCHOR, 1)[1]
        assert "## Delta Mode State" in below and "## Delta Mode State" not in prefix
        assert "Assessed as: simple factual query." in below

    def test_enabled_tiers_filter_which_tiers_are_described(self):
        prefix = self._render(
            enabled_tiers=["single_shot", "deep"],
            tier_profiles=enabled_tier_profiles(["single_shot", "deep"]),
        ).split(ORCHESTRATOR_ANCHOR)[0]
        assert "**single_shot**" in prefix and "**deep**" in prefix
        assert "**direct**" not in prefix and "**standard**" not in prefix

    def test_effort_summary_lists_only_enabled_levels_and_finalize_method(self):
        prefix = self._render(
            enabled_tiers=["single_shot", "deep"],
            tier_profiles=enabled_tier_profiles(["single_shot", "deep"]),
        ).split(ORCHESTRATOR_ANCHOR)[0]
        assert "## Path per Effort Level" not in prefix
        assert "submit_final_report(researched=true)" in prefix
        assert "return /shared/output.md marker" in prefix

    def test_workflow_procedures_gated_by_enabled_tiers(self):
        # deep-only config must not describe the direct/single_shot/standard procedures
        deep_only = self._render(
            enabled_tiers=["deep"],
            tier_profiles=enabled_tier_profiles(["deep"]),
        ).split(ORCHESTRATOR_ANCHOR)[0]
        assert "### `deep`" in deep_only
        assert "### `direct`" not in deep_only
        assert "### `single_shot`" not in deep_only
        assert "### `standard`" not in deep_only

    def test_meta_path_remains_available_when_direct_is_disabled(self):
        prefix = self._render(
            enabled_tiers=["single_shot", "deep"],
            tier_profiles=enabled_tier_profiles(["single_shot", "deep"]),
        ).split(ORCHESTRATOR_ANCHOR)[0]
        assert "### No-Research Meta / Capability Path" in prefix
        assert "meta messages never trigger pointless research" in prefix
        assert "### `direct`" not in prefix

    def test_standard_procedure_has_both_finalize_branches(self):
        prefix = self._render().split(ORCHESTRATOR_ANCHOR)[0]
        assert "### `standard`" in prefix
        standard = prefix.split("### `standard`", 1)[1].split("###", 1)[0]
        assert "Inline branch" in standard and "Writer branch" in standard
        assert "planner is mandatory" in standard
        assert "Never mix the branches" in standard

    def test_planned_writer_pipeline_is_complete_without_source_router(self):
        prefix = self._render(enable_source_router=False).split(ORCHESTRATOR_ANCHOR)[0]
        pipeline = prefix.split("### Planned Writer Pipeline", 1)[1].split("###", 1)[0]
        for n in (1, 2, 3, 4, 5):
            assert f"{n}." in pipeline
        assert "6." not in pipeline
        assert pipeline.index("planning to planner-agent") < pipeline.index("run_research_batch")
        assert pipeline.index("run_research_batch") < pipeline.index("writer-agent with task()")

    def test_planned_writer_pipeline_is_complete_with_source_router(self):
        prefix = self._render(enable_source_router=True).split(ORCHESTRATOR_ANCHOR)[0]
        pipeline = prefix.split("### Planned Writer Pipeline", 1)[1].split("###", 1)[0]
        for n in (1, 2, 3, 4, 5, 6):
            assert f"{n}." in pipeline

    def test_mechanism_based_finalize_protocol(self):
        prefix = self._render().split(ORCHESTRATOR_ANCHOR)[0]
        assert "Finalize Protocol" in prefix
        assert "You delegated to writer-agent" in prefix
        assert "You wrote the answer inline yourself" in prefix

    def test_retrieval_tools_render_below_boundary(self):
        rendered = self._render(retrieval_tools=[{"name": "exa_web_search", "description": "neural search"}])
        prefix, below = rendered.split(ORCHESTRATOR_ANCHOR)[0], rendered.split(ORCHESTRATOR_ANCHOR, 1)[1]
        assert "## Retrieval Tools" in below and "## Retrieval Tools" not in prefix
        assert "exa_web_search" in below

    def test_delta_uses_writer_pipeline_even_with_shallow_only_tiers(self):
        prefix = self._render(
            enabled_tiers=["single_shot"],
            tier_profiles=enabled_tier_profiles(["single_shot"]),
        ).split(ORCHESTRATOR_ANCHOR)[0]
        choosing = prefix.split("## Choosing Effort", 1)[1].split("##", 1)[0]
        assert "parent report context is present" in choosing
        assert "Planned Writer Pipeline" in choosing
        assert "### Planned Writer Pipeline" in prefix
        assert "### `deep`" not in prefix

    def test_escalation_runs_planned_research_before_writer(self):
        prefix = self._render().split(ORCHESTRATOR_ANCHOR)[0]
        escalation = prefix.split("## In-loop Escalation", 1)[1].split("##", 1)[0]
        assert "planner → planned research → writer" in escalation
        assert "without skipping the new research" in escalation

    def test_inline_citation_and_evidence_failure_contracts_are_explicit(self):
        prefix = self._render().split(ORCHESTRATOR_ANCHOR)[0]
        assert "## Inline Citation Contract" in prefix
        assert "Each citation number maps to exactly one verified URL" in prefix
        assert "Remove an unsupported claim or state it as an evidence gap" in prefix
        assert "## Stopping and Evidence Failure" in prefix
        assert "verified source registry remains empty" in prefix

    def test_effort_choice_does_not_require_user_facing_announcement(self):
        prefix = self._render().split(ORCHESTRATOR_ANCHOR)[0]
        assert "Do not add a user-facing tier announcement" in prefix


class TestDynamicSectionRendering:
    """Per-tier trimmed prompts (dynamic_orchestrator_sections): each mode renders only its
    sections. Reuses TestOrchestratorPromptRendering._render / _render_mode.
    """

    def setup_method(self):
        self._h = TestOrchestratorPromptRendering()

    def _prefix(self, mode):
        return self._h._render_mode(mode).split(ORCHESTRATOR_ANCHOR)[0]

    def test_router_teaches_selection_only(self):
        prefix = self._prefix("router")
        # keeps the tier-selection machinery ...
        assert "## Effort Levels" in prefix
        assert "## Choosing Effort" in prefix
        # ... and drops the execution machinery (swapped in after declare_effort_tier)
        assert "## Workflow" not in prefix
        assert "## Research Loop" not in prefix
        assert "## Available Subagents" not in prefix

    def test_direct_is_minimal(self):
        prefix = self._prefix("direct")
        assert "### `direct`" in prefix
        assert "## Effort Levels" not in prefix
        assert "## Research Loop" not in prefix
        # direct never plans/writes: no subagent pipeline
        assert "### Planned Writer Pipeline" not in prefix

    def test_single_shot_keeps_inline_research_and_citations(self):
        prefix = self._prefix("single_shot")
        assert "### `single_shot`" in prefix
        assert "## Research Loop" in prefix
        assert "## Inline Citation Contract" in prefix
        # single_shot answers inline: no subagents, no planner/writer pipeline, no tier catalog
        assert "## Available Subagents" not in prefix
        assert "### Planned Writer Pipeline" not in prefix
        assert "## Choosing Effort" not in prefix
        # only its own workflow block renders
        assert "### `deep`" not in prefix

    def test_deep_keeps_writer_pipeline_drops_inline_citation(self):
        prefix = self._prefix("deep")
        assert "### `deep`" in prefix
        assert "## Available Subagents" in prefix
        assert "### Planned Writer Pipeline" in prefix
        # writer-agent owns citations on the deep path; the inline contract is not needed
        assert "## Inline Citation Contract" not in prefix
        assert "## Choosing Effort" not in prefix

    def test_delta_keeps_delta_rule_and_pipeline(self):
        prefix = self._prefix("delta")
        assert "## Delta / Parent-Report Rule" in prefix
        assert "### Planned Writer Pipeline" in prefix
        assert "## Available Subagents" in prefix
        assert "## Inline Citation Contract" not in prefix

    def test_escalation_section_present_only_when_higher_tier_enabled(self):
        # single_shot with deeper tiers enabled can step up -> escalation guidance included
        can_step_up = self._h._render_mode("single_shot", enabled=("single_shot", "deep")).split(ORCHESTRATOR_ANCHOR)[0]
        assert "## In-loop Escalation" in can_step_up
        # single_shot as the deep-most enabled tier -> nothing to escalate to, section dropped
        top_tier = self._h._render_mode("single_shot", enabled=("single_shot",)).split(ORCHESTRATOR_ANCHOR)[0]
        assert "## In-loop Escalation" not in top_tier

    def test_every_mode_is_smaller_than_the_full_prompt(self):
        # The full prompt is what every model call pays today; each trimmed mode must be smaller.
        full = len(self._h._render())
        for mode in ("router", "direct", "single_shot", "standard", "deep", "delta"):
            assert len(self._h._render_mode(mode)) < full, mode


class TestDynamicSectionWiring:
    """factory wiring: the opt-in flag attaches ComplexityRouterMiddleware with a renderer, and
    the flag-off / delta paths behave as before.
    """

    def test_flag_attaches_complexity_router_with_renderer(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider, dynamic_orchestrator_sections=True)
        router = next((m for m in kwargs["middleware"] if m.__class__.__name__ == "ComplexityRouterMiddleware"), None)
        assert router is not None
        assert router._prompt_renderer is not None
        # build-time prompt is the minimal router prompt (no execution machinery yet)
        assert "## Research Loop" not in kwargs["system_prompt"].split(ORCHESTRATOR_ANCHOR)[0]

    def test_flag_off_renders_full_prompt_and_no_router_middleware(self, mock_llm_provider):
        kwargs = _build_and_capture(mock_llm_provider, dynamic_orchestrator_sections=False)
        names = {m.__class__.__name__ for m in kwargs["middleware"]}
        assert "ComplexityRouterMiddleware" not in names
        prefix = kwargs["system_prompt"].split(ORCHESTRATOR_ANCHOR)[0]
        assert "## Research Loop" in prefix and "## Available Subagents" in prefix

    def test_delta_run_uses_full_delta_prompt_without_swapping(self, mock_llm_provider):
        state = AdaptiveResearchAgentState(
            messages=[HumanMessage(content="Update the report")],
            files={"/shared/parent_report_context.json": {"content": "{}"}},
        )
        kwargs = _build_and_capture(mock_llm_provider, state=state, dynamic_orchestrator_sections=True)
        prefix = kwargs["system_prompt"].split(ORCHESTRATOR_ANCHOR)[0]
        # delta prompt carries the full pipeline, not the router prompt
        assert "### Planned Writer Pipeline" in prefix
        assert "## Delta / Parent-Report Rule" in prefix
        # no renderer-driven swap on a delta run
        router = next((m for m in kwargs["middleware"] if m.__class__.__name__ == "ComplexityRouterMiddleware"), None)
        if router is not None:
            assert router._prompt_renderer is None
