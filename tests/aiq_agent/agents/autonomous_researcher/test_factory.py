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
from aiq_agent.agents.autonomous_researcher.factory import build_planner_subagent_description
from aiq_agent.agents.autonomous_researcher.factory import build_writer_subagent_description
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


class TestDelegationGuidanceLivesInDescriptions:
    """The `# Subagents` / `# Subagent Delegation Instructions` prompt sections were folded into
    the subagent descriptions on 2026-08-18. These tests pin the union: the prompt must no longer
    restate the triggers, and every description must carry its own complete delegation contract.
    """

    def test_prompt_no_longer_restates_routing_or_briefs(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        for removed in (
            "# Subagents",
            "# Subagent Delegation Instructions",
            "## Planner Sub-agent Delegation",
            "## Researcher Sub-agent Delegation",
            "## Writer Sub-agent Delegation",
            "three or more distinct deliverables",  # planner trigger, now description-only
            "Their order is fixed",  # set-level ordering, now distributed per description
        ):
            assert removed not in prompt, removed

    def test_prompt_owns_the_research_loop(self, mock_llm_provider):
        """Loop control is orchestrator behavior across turns, not delegation mechanics, so it
        lives in the prompt. Delegation mechanics live in the tool/subagent descriptions.

        The section is written in deliberately plain language and compressed to three stages; these
        assertions pin the rules, not the phrasing, so a reword that drops a rule still fails.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "\n# The Research Loop\n" in prompt
        for stage in ("**Each pass.**", "**If a pass comes back thin.**", "**When to stop.**"):
            assert stage in prompt, stage
        for rule in (
            "Write down every query you send",  # ledger
            "a repeat is blocked",  # why the ledger matters
            "Read every result before deciding anything",
            "evidence_judgment",
            "list what the answer still needs",  # name the gaps
            "If you cannot name what is missing, stop",
            "Do not ask the same thing in new words",
            "keywords, not URLs",
            "resend only the queries that failed, never one that worked",
            "give the whole chain to `researcher-agent` once",
            "answer with what you have",  # honest partial beats nothing
            "Stop once the evidence is enough",
            "answer that part and say what is missing",
            "same year, the same definition and the same kind of source",  # superlative check
            "do not guess, and do not fall back to an unresearched answer",
            "`researched=true`",
            "call `get_verified_sources` before writing anything with citations",
        ):
            assert rule in prompt, rule

    def test_research_loop_stays_concise(self, mock_llm_provider):
        """It is the orchestrator's hot path, re-read on every turn, so terse beats exhaustive.

        The ceiling is the size of the ``The research loop:`` list this replaced (997 chars) plus
        headroom for the two blocks folded into it (the lookup-failure ladder and the stopping
        rules), which previously lived elsewhere in the prompt.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        section = prompt.split("# The Research Loop", 1)[1].split("\n# ", 1)[0]
        content_lines = [line for line in section.splitlines() if line.strip()]
        assert len(content_lines) <= 6, f"research loop grew to {len(content_lines)} content lines"
        assert len(section) <= 1600, f"research loop grew to {len(section)} chars"

    def test_research_loop_maintainer_note_is_not_sent_to_the_model(self, mock_llm_provider):
        """The scope boundary is a Jinja comment, so it costs zero tokens per turn."""
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "LOOP CONTROL" not in prompt

    def test_prompt_signposts_task_without_restating_it(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "Each subagent's entry in the `task` description is authoritative" in prompt

    @pytest.mark.parametrize(
        ("name", "required"),
        [
            (
                "researcher-agent",
                (
                    "WHEN TO CHOOSE IT",
                    "SEQUENCING",
                    "WHAT IT PRODUCES",
                    "DELEGATION BRIEF",
                    "PREREQUISITE CHAIN",
                    "Already attempted, do not repeat",
                ),
            ),
            (
                "planner-agent",
                (
                    "WHEN TO CHOOSE IT",
                    "SEQUENCING",
                    "WHAT IT PRODUCES",
                    "DELEGATION BRIEF",
                    "three or more distinct deliverables",
                    "runs FIRST, or not at all",
                    "Create a research plan for the following user request",
                ),
            ),
            (
                "writer-agent",
                (
                    "WHEN TO CHOOSE IT",
                    "SEQUENCING",
                    "WHAT IT PRODUCES",
                    "DELEGATION BRIEF",
                    "runs LAST",
                    "do NOT call submit_final_report",
                    "Write the final Markdown answer to /shared/output.md",
                ),
            ),
        ],
    )
    def test_each_description_carries_the_full_contract(self, mock_llm_provider, name, required):
        specs = _build_and_capture(mock_llm_provider)["subagents"]
        description = next(s for s in specs if s["name"] == name)["description"]
        for fragment in required:
            assert fragment in description, f"{name} lost: {fragment}"

    def test_multiline_descriptions_stay_inside_their_own_bullet(self, mock_llm_provider):
        """deepagents renders descriptions as `f"- {name}: {description}"` in both the task tool
        description and the system prompt. That assumes one line; without indentation every
        continuation line escapes to column 0 and visually merges with the NEXT agent's entry, so
        writer-agent's delegation brief can read as part of planner-agent.
        """
        specs = _build_and_capture(mock_llm_provider)["subagents"]
        rendered = "\n".join(f"- {spec['name']}: {spec['description']}" for spec in specs)
        escaped = [
            line
            for line in rendered.splitlines()
            if line.strip() and not line.startswith("- ") and not line.startswith("  ")
        ]
        assert not escaped, f"{len(escaped)} description line(s) escaped their bullet: {escaped[:2]}"
        assert len([line for line in rendered.splitlines() if line.startswith("- ")]) == len(specs)

    def test_writer_brief_omits_chart_rules_when_execution_is_disabled(self, mock_llm_provider):
        """Briefing the writer on artifacts it cannot produce is how the old prompt's
        `execution_enabled` conditional earned its keep; the description must preserve it."""
        with_exec = build_writer_subagent_description(parent_report_context_available=False, execution_enabled=True)
        without_exec = build_writer_subagent_description(parent_report_context_available=False, execution_enabled=False)
        assert "artifact://" in with_exec
        assert "artifact://" not in without_exec

    def test_planner_and_writer_briefs_gate_on_parent_report(self):
        planner_plain = build_planner_subagent_description(parent_report_context_available=False)
        planner_delta = build_planner_subagent_description(parent_report_context_available=True)
        assert "parent-report revision" not in planner_plain
        assert "parent-report revision" in planner_delta

        writer_plain = build_writer_subagent_description(parent_report_context_available=False, execution_enabled=False)
        writer_delta = build_writer_subagent_description(parent_report_context_available=True, execution_enabled=False)
        assert "original_report.md" not in writer_plain
        assert "original_report.md" in writer_delta

    def test_delta_mode_overrides_the_writer_length_test(self):
        """In delta mode the writer is mandatory: preserved parent citations only stay verifiable
        through the writer path, which contradicts the default "only for long deliverables" rule.
        """
        delta = build_writer_subagent_description(parent_report_context_available=True, execution_enabled=False)
        assert "REQUIRED when a parent report is mounted" in delta

    def test_research_batch_description_owns_the_delegation_contract(self, mock_llm_provider):
        """What one ResearchQuery must contain, and what the call returns — nothing about looping."""
        tools = {t.name: t for t in _build_and_capture(mock_llm_provider)["tools"]}
        description = tools["run_research_batch"].description
        for field in ("preferred_tools", "target_components", "rationale", "depth"):
            assert field in description, field
        # A plan component id is not a topic; the worker cannot resolve one.
        assert "latest_price_anchor" in description
        assert "evidence_judgment" in description


class TestOrchestratorPrompt:
    def test_carries_no_tier_artifacts(self, mock_llm_provider):
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert not [artifact for artifact in TIER_ARTIFACTS if artifact in prompt]

    def test_states_the_anti_memory_rule(self, mock_llm_provider):
        """The highest-risk deletion in the change: it is prompt-only now, so assert it exists."""
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "Never answer from memory" in prompt
        assert "time-sensitive" in prompt

    def test_decision_section_names_every_route(self, mock_llm_provider):
        """Every route must be reachable from the decision section.

        Asserting planner-agent specifically: the previous prompt described the planner only in an
        advisory block outside the routing menu, and it won zero routing decisions across 90 eval
        trials. Since writer-agent is gated on /shared/plan.json, that also forced zero writer calls.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "Deciding what to do" in prompt
        for route in ("submit_final_report", "run_research_batch", "researcher-agent", "planner-agent"):
            assert route in prompt, route

    def test_decision_section_is_not_a_tier_ladder(self, mock_llm_provider):
        """The whole point of this agent is that requests are not classified into effort levels.

        Guards the regression this prompt was rewritten twice to avoid: an enumerated set of
        request kinds is a tier system regardless of whether middleware enforces it.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        for ladder_artifact in ("opening move", "Shape A", "shape B", "Start at A", "climb"):
            assert ladder_artifact not in prompt, ladder_artifact

    def test_states_the_budgets_as_numbers(self, mock_llm_provider):
        """Budgets are prompt-only here (nothing enforces them), so assert the text carries them."""
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        # Top-level sections are `#`; `## Budgets` would also match a bare `# Budgets` substring
        # check, so anchor on the newline to assert the heading level as well as the section.
        assert "\n# Budgets\n" in prompt
        assert "at most 2 per request" in prompt

    def test_states_the_answer_set_contract(self, mock_llm_provider):
        """The precision fix: the answer may not enumerate rejected candidates."""
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "What goes in the answer" in prompt
        assert "only qualifying members" in prompt

    def test_source_tools_are_held_directly_but_demoted_to_verification(self, mock_llm_provider):
        """Source tools stay in the orchestrator's hands, and the prompt demotes them to
        verification without re-listing them: the names and descriptions reach the model through
        `bind_tools`, so the prompt only has to say how to use them.
        """
        captured = _build_and_capture(mock_llm_provider)
        assert {"web_search_tool", "knowledge_search"} <= {t.name for t in captured["tools"]}
        prompt = captured["system_prompt"]
        assert "counts against the 2-call budget" in prompt
        assert "raw results stay in this conversation" in prompt

    def test_prompt_does_not_duplicate_bound_tool_schemas(self, mock_llm_provider):
        """Every orchestrator tool is bound to the model with its name, description, and argument
        schema. Rendering name+description into the prompt too was a verbatim second copy of ~3.9k
        characters per turn, and the prompt half could not be withdrawn when the loop guard
        withdraws a tool. `task`, `write_todos`, and the filesystem tools were always schema-only.
        """
        captured = _build_and_capture(mock_llm_provider)
        prompt = captured["system_prompt"]
        assert "## Your Tools (callable)" not in prompt
        assert "## Retrieval Tools" not in prompt
        for bound in captured["tools"]:
            description = (bound.description or "").strip()
            if len(description) > 40:
                assert description not in prompt, f"{bound.name} description duplicated into prompt"

    def test_delta_block_requires_planner_before_writer(self, mock_llm_provider):
        """The prompt still frames delta mode; the per-agent briefs now live in the descriptions."""
        state = AutonomousResearchAgentState(
            messages=[HumanMessage(content="follow up")],
            files={"/shared/original_report.md": {"content": "# parent"}},
        )
        captured = _build_and_capture(mock_llm_provider, state=state)
        prompt = captured["system_prompt"]
        assert "Parent-report delta" in prompt
        assert "planner-agent" in prompt
        # The revised-report contract itself moved into writer-agent's delegation brief, which is
        # only rendered when a parent report is actually mounted.
        writer = next(s for s in captured["subagents"] if s["name"] == "writer-agent")
        assert "complete standalone revised report" in writer["description"]

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
