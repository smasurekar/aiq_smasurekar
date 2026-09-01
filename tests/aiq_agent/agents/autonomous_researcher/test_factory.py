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

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from deepagents.backends.state import create_file_data
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.autonomous_researcher.agent import AutonomousResearcherAgent
from aiq_agent.agents.autonomous_researcher.custom_middleware import AutonomousFinalReportCommitTracker
from aiq_agent.agents.autonomous_researcher.factory import GENERAL_PURPOSE_SUBAGENT_NAME
from aiq_agent.agents.autonomous_researcher.factory import _shallow_subagent_tools
from aiq_agent.agents.autonomous_researcher.factory import build_planner_subagent_description
from aiq_agent.agents.autonomous_researcher.factory import build_writer_subagent_description
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchAgentState
from aiq_agent.agents.autonomous_researcher.models import AutonomousResearchPlan
from aiq_agent.agents.autonomous_researcher.subagents.shallow import SHALLOW_ANSWER_CONTRACT
from aiq_agent.agents.autonomous_researcher.tools.research import build_autonomous_research_batch_tool
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole

# Strings that would prove some part of the tier machinery leaked into this agent.
#
# "shallow-researcher" is deliberately NOT here. The sub-agent of that name is a first-class
# delegation route in this agent, reached by description like every other one — what it is not is
# the adaptive arm's tier-gated, middleware-forced `single_shot` route. The tier vocabulary that
# would prove that machinery came with it ("single_shot", "declare_effort_tier", ...) is still
# listed, so a genuine leak still fails.
TIER_ARTIFACTS = (
    "declare_effort_tier",
    "effort tier",
    "Effort Levels",
    "Choosing Effort",
    "single_shot",
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


@tool
def fetch_url_tool(urls: list[str]) -> str:
    """Open pages and return their text."""
    return f"Opened: {urls}"


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
            tools=tools if tools is not None else [web_search_tool, knowledge_search, fetch_url_tool],
            **agent_kwargs,
        )
        state = state or AutonomousResearchAgentState(messages=[HumanMessage(content="q")])
        agent._build_orchestrator_agent(state, AutonomousFinalReportCommitTracker())
    return dict(create.call_args.kwargs)


def _middleware_names(captured: dict) -> list[str]:
    return [type(m).__name__ for m in captured["middleware"]]


def _build_run(mock_llm_provider, *, state=None, tools=None, **agent_kwargs):
    """Build the graph and return the ``AutonomousResearchGraphRun`` wrapper itself."""
    graph = MagicMock()
    graph.with_config = MagicMock(return_value=graph)
    with (
        patch("aiq_agent.agents.autonomous_researcher.factory.create_deep_agent", return_value=graph),
        patch("aiq_agent.agents.deep_researcher.factory.create_agent", return_value=graph),
        patch(
            "aiq_agent.agents.deep_researcher.factory.create_summarization_middleware",
            return_value=_FakeSummarizationMiddleware(),
        ),
    ):
        agent = AutonomousResearcherAgent(
            llm_provider=mock_llm_provider,
            tools=tools if tools is not None else [web_search_tool, knowledge_search, fetch_url_tool],
            **agent_kwargs,
        )
        state = state or AutonomousResearchAgentState(messages=[HumanMessage(content="q")])
        return agent._build_orchestrator_agent(state, AutonomousFinalReportCommitTracker())


def _delta_state() -> AutonomousResearchAgentState:
    """A state with a parent report mounted, which must suppress the shallow sub-agent."""
    return AutonomousResearchAgentState(
        messages=[HumanMessage(content="revise it")],
        files={"/shared/original_report.md": create_file_data("# Parent report")},
    )


class TestShallowSubagentWiring:
    """The shallow sub-agent is opt-out, delta-suppressed, and hands back a run-scoped capture."""

    def test_present_by_default(self, mock_llm_provider):
        names = [s["name"] for s in _build_and_capture(mock_llm_provider)["subagents"]]
        assert "shallow-researcher" in names

    def test_absent_when_disabled(self, mock_llm_provider):
        captured = _build_and_capture(mock_llm_provider, shallow_subagent=False)
        assert "shallow-researcher" not in [s["name"] for s in captured["subagents"]]
        assert "ShallowFinalizationMiddleware" not in _middleware_names(captured)
        assert _build_run(mock_llm_provider, shallow_subagent=False).shallow_capture is None

    def test_absent_for_a_parent_report_delta(self, mock_llm_provider):
        """A delta must keep the planner -> research -> writer path: it is the only one that
        carries preserved parent citations through in a verifiable form."""
        captured = _build_and_capture(mock_llm_provider, state=_delta_state())
        assert "shallow-researcher" not in [s["name"] for s in captured["subagents"]]
        assert "ShallowFinalizationMiddleware" not in _middleware_names(captured)

    def test_finalization_middleware_shares_the_returned_capture(self, mock_llm_provider):
        """One object: the adapter writes it, the middleware reads it, agent.run() cancels it."""
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
                tools=[web_search_tool, knowledge_search],
            )
            state = AutonomousResearchAgentState(messages=[HumanMessage(content="q")])
            built = agent._build_orchestrator_agent(state, AutonomousFinalReportCommitTracker())

        guard = next(
            m for m in create.call_args.kwargs["middleware"] if type(m).__name__ == "ShallowFinalizationMiddleware"
        )
        assert built.shallow_capture is not None
        assert guard._capture is built.shallow_capture

    def test_the_shallow_agent_receives_the_raw_tool_list(self, mock_llm_provider):
        """Not tool_set.researcher_tools: it must run exactly as it does standalone."""
        with patch("aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent") as shallow_cls:
            _build_and_capture(mock_llm_provider)
        passed = [t.name for t in shallow_cls.call_args.kwargs["tools"]]
        assert passed == ["web_search_tool", "knowledge_search", "fetch_url_tool"]
        assert "get_verified_sources" not in passed

    def test_the_shallow_sub_run_carries_the_answer_contract(self, mock_llm_provider):
        """The shallow exit needs the same answer-set discipline the inline exit already has.

        In DSQA-90 job 2026-08-20__21-44-00 the shallow exit emitted excessive items on 40.6% of
        its trials (mean 1.56) against 33.3% for the inline exit, and the difference is that
        orchestrator.j2 carries an answer-set rule while the shared shallow template carries none.
        Injecting per sub-run rather than editing that template keeps it byte-identical for the
        shipped configs that use the standalone shallow agent - see
        `test_default_model_profiles.test_shallow_profiles_use_the_shared_citation_prompt`.
        """
        with patch("aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent") as shallow_cls:
            _build_and_capture(mock_llm_provider)
        prompt = shallow_cls.call_args.kwargs["system_prompt"]
        assert prompt is not None
        # The shared template is still the base ...
        assert "You are a Shallow Research Agent" in prompt
        # ... with the contract appended on top of it.
        assert prompt.endswith(SHALLOW_ANSWER_CONTRACT)
        assert "### Considered and excluded" in prompt
        # Trial 0256: the grader read the chart, the takeaways block and the references section
        # as answer items, so the contract has to place them below the answer explicitly.
        assert "belong below the `## Answer`" in prompt

    def test_the_shallow_contract_is_jinja_inert(self):
        """`render_prompt_template` uses StrictUndefined and the shallow render site passes only
        four variables, so a stray `{{ }}` in the contract would raise at agent-node time rather
        than at build time - long after this test could have caught it.
        """
        for token in ("{{", "{%", "{#"):
            assert token not in SHALLOW_ANSWER_CONTRACT, token

    def test_loop_bounds_are_forwarded(self, mock_llm_provider):
        with patch("aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent") as shallow_cls:
            _build_and_capture(
                mock_llm_provider,
                shallow_subagent_max_llm_turns=3,
                shallow_subagent_max_tool_iterations=2,
            )
        assert shallow_cls.call_args.kwargs["max_llm_turns"] == 3
        assert shallow_cls.call_args.kwargs["max_tool_iterations"] == 2

    def test_budget_exhaustion_policy_is_forwarded(self, mock_llm_provider):
        """The knob is inert unless it reaches the sub-agent that enforces it."""
        with patch("aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent") as shallow_cls:
            _build_and_capture(mock_llm_provider)
        assert shallow_cls.call_args.kwargs["escalate_on_budget_exhaustion"] is True

        with patch("aiq_agent.agents.autonomous_researcher.subagents.shallow.ShallowResearcherAgent") as shallow_cls:
            _build_and_capture(mock_llm_provider, shallow_subagent_escalate_on_budget_exhaustion=False)
        assert shallow_cls.call_args.kwargs["escalate_on_budget_exhaustion"] is False

    def test_description_states_the_first_once_and_terminal_contract(self, mock_llm_provider):
        """These three rules have no runtime enforcement, so the text is the only thing carrying
        them. Assertions pin the rules, not the phrasing."""
        spec = next(s for s in _build_and_capture(mock_llm_provider)["subagents"] if s["name"] == "shallow-researcher")
        description = spec["description"]
        assert "FIRST, or not at all" in description
        assert "ONCE" in description
        assert "do NOT call submit_final_report" in description
        assert "run_research_batch" in description, "the failure/escalation route must be named"

    def test_description_survives_the_list_rendering(self, mock_llm_provider):
        """deepagents renders `- {name}: {description}`; unindented lines escape the bullet."""
        spec = next(s for s in _build_and_capture(mock_llm_provider)["subagents"] if s["name"] == "shallow-researcher")
        continuation = spec["description"].split("\n", 1)[1]
        assert all(not line or line.startswith("  ") for line in continuation.splitlines())


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
    """`task` must advertise exactly four usable delegation routes."""

    def test_the_default_roster_reaches_the_researcher_only_through_the_batch(self, mock_llm_provider):
        """Order is part of the contract: shallow-researcher renders first in the `task` listing.

        Its description says "FIRST, or not at all"; putting it first in the rendered menu is free
        reinforcement of that, since deepagents renders the specs in list order.

        `researcher-agent` is absent by default. It still executes every research question, as the
        `run_research_batch` worker — what the default withholds is a second, direct `task` door
        onto that same worker.
        """
        names = [s["name"] for s in _build_and_capture(mock_llm_provider)["subagents"]]
        assert names == [
            "shallow-researcher",
            "planner-agent",
            "writer-agent",
            GENERAL_PURPOSE_SUBAGENT_NAME,
        ]

    def test_the_direct_researcher_door_inserts_ahead_of_the_planner_when_asked_for(self, mock_llm_provider):
        names = [s["name"] for s in _build_and_capture(mock_llm_provider, researcher_subagent=True)["subagents"]]
        assert names == [
            "shallow-researcher",
            "researcher-agent",
            "planner-agent",
            "writer-agent",
            GENERAL_PURPOSE_SUBAGENT_NAME,
        ]

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

    def test_general_purpose_description_redirects_to_the_door_this_build_holds(self, mock_llm_provider):
        """Its description must not compete for research delegation, and must not name an absent
        door: a redirect to a route that was never built costs the orchestrator a turn."""
        for kwargs, expected, absent in (
            ({}, "run_research_batch", "researcher-agent"),
            ({"researcher_subagent": True}, "researcher-agent", None),
        ):
            specs = _build_and_capture(mock_llm_provider, **kwargs)["subagents"]
            gp = next(s for s in specs if s["name"] == GENERAL_PURPOSE_SUBAGENT_NAME)
            assert expected in gp["description"]
            if absent:
                assert absent not in gp["description"]
            assert "researching complex questions" not in gp["description"], "deepagents' default description leaked"

    def test_researcher_subagent_returns_structured_notes(self, mock_llm_provider):
        specs = _build_and_capture(mock_llm_provider, researcher_subagent=True)["subagents"]
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
            "answer with what you have",  # honest partial beats nothing
            "Stop once the evidence is enough",
            "answer that part and say what is missing",
            "same year, the same definition and the same kind of source",  # superlative check
            "do not guess, and do not fall back to an unresearched answer",
            "`researched=true`",
            "call `get_verified_sources` before writing anything with citations",
        ):
            assert rule in prompt, rule
        # The failure ladder's last rung names a door, so it is the one arm-dependent rule here.
        assert "do not send a third: record it as an explicit gap" in prompt
        opt_in = _build_and_capture(mock_llm_provider, researcher_subagent=True)["system_prompt"]
        assert "give the whole chain to `researcher-agent` once" in opt_in

    def test_research_loop_stays_concise(self, mock_llm_provider):
        """It is the orchestrator's hot path, re-read on every turn, so terse beats exhaustive.

        The ceiling is the size of the ``The research loop:`` list this replaced (997 chars) plus
        headroom for the two blocks folded into it (the lookup-failure ladder and the stopping
        rules), which previously lived elsewhere in the prompt.

        Raised 1600 -> 1750 on 2026-09-01 for one clause: once a shortlist exists, the next pass
        must test what is not on it. dsqa90 job 2026-09-01__11-56-18 measured 45% of deep-path
        queries re-pricing names already held, and the gold answer absent from the shortlist is the
        single most replicated T3 failure. Line count is unchanged; the clause is folded into an
        existing paragraph rather than added as one. See
        misc/autonomous_researcher/autonomous-researcher-t3-consistency-analysis.md sections 4.1 and 9.3.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        section = prompt.split("# The Research Loop", 1)[1].split("\n# ", 1)[0]
        content_lines = [line for line in section.splitlines() if line.strip()]
        assert len(content_lines) <= 6, f"research loop grew to {len(content_lines)} content lines"
        assert len(section) <= 1750, f"research loop grew to {len(section)} chars"

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
        # researcher-agent is an opt-in door, so open every door to inspect all four contracts.
        specs = _build_and_capture(mock_llm_provider, researcher_subagent=True)["subagents"]
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

    def test_the_per_batch_query_cap_is_stated_next_to_the_schema_that_enforces_it(self, mock_llm_provider):
        """The one ceiling worth telling the model, because exceeding it is rejected outright.

        It lived in the prompt's `# Budgets` section as a hard-coded "1-5"; it now renders from the
        same `max_research_concurrency` the tool validates against, so the two cannot drift.
        """
        for concurrency in (3, 7):
            built = build_autonomous_research_batch_tool(
                researcher_runnable=None, callbacks=[], max_research_concurrency=concurrency
            )
            assert f"Send 1-{concurrency} queries in one call" in built.description


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
        for route in ("submit_final_report", "run_research_batch", "planner-agent"):
            assert route in prompt, route
        # researcher-agent is off by default, and an absent door must not be named anywhere.
        assert "researcher-agent" not in prompt
        opt_in = _build_and_capture(mock_llm_provider, researcher_subagent=True)["system_prompt"]
        assert "researcher-agent" in opt_in

    def test_decision_section_is_not_a_tier_ladder(self, mock_llm_provider):
        """The whole point of this agent is that requests are not classified into effort levels.

        Guards the regression this prompt was rewritten twice to avoid: an enumerated set of
        request kinds is a tier system regardless of whether middleware enforces it.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        for ladder_artifact in ("opening move", "Shape A", "shape B", "Start at A", "climb"):
            assert ladder_artifact not in prompt, ladder_artifact

    def test_bound_tool_descriptions_state_no_budget_counts(self, mock_llm_provider):
        """A tool description is model input exactly like the system prompt.

        Deleting `# Budgets` from the prompt while `run_research_batch`'s description still said
        "Issue ONE batch per request" and "`high`: at most one per request" just moved the drift
        somewhere less visible: configuring max_batch_calls or max_high_depth_queries differently
        recreates the mismatch the change exists to remove. The per-call query cap is the one
        allowed number, because it is interpolated from the same max_research_concurrency the tool
        validates against.
        """
        captured = _build_and_capture(mock_llm_provider)
        forbidden = ("ONE batch per request", "at most one\n", "at most one per request", "per request")
        for bound in captured["tools"]:
            description = bound.description or ""
            for claim in forbidden:
                assert claim not in description, f"{bound.name}: {claim!r}"

    def test_the_prompt_states_no_budget_numbers(self, mock_llm_provider):
        """Budgets belong to the middleware, which enforces them and explains itself when one fires.

        The prompt used to carry a `# Budgets` section stating four ceilings. Three were enforced by
        nothing and one contradicted the config by 6x, because a prompt copy of a number cannot be
        kept in step with the object that enforces it. The section is gone; every ceiling now lives
        in AutonomousOrchestratorLoopGuardMiddleware and reaches the model as a blocked ToolMessage
        or the pre-withdrawal nudge, at the moment it actually matters.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "\n# Budgets\n" not in prompt
        for budget_claim in (
            "at most 2 per request",
            "one per request",
            "2-call budget",
            "max_batch_calls",
            "budget reached",
            "runtime enforces",
        ):
            assert budget_claim not in prompt, budget_claim

    def test_states_the_answer_set_contract(self, mock_llm_provider):
        """The precision fix: rejected candidates may not sit where the answer sits."""
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        assert "What goes in the answer" in prompt
        assert "only qualifying members" in prompt
        # Rejected candidates are relocated, not banned - naming why something failed is good
        # research writing, it just may not appear inside the answer section.
        assert "### Considered and excluded" in prompt
        # The self-check that catches a filter the model never actually applied (trial 0824).
        assert "name the filter it satisfies" in prompt

    def test_answer_shape_is_query_driven_not_length_capped(self, mock_llm_provider):
        """Shape follows the question; nothing here may turn into a brevity rule.

        This is the constraint under which the answer-set contract was accepted: the agent keeps
        its long-report capability, and only questions that name a discrete target get an answer
        section. The measurement backs it - in DSQA-90 job 2026-08-20__21-44-00, 3-6k-char answers
        were the best-scoring band (FC 0.538 vs 0.488 for the shortest), so length was never the
        defect. A future edit that reintroduces a word limit or a "be brief" instruction has
        broken that contract, and this test is what catches it.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        section = prompt.split("## What goes in the answer", 1)[1].split("\n## ", 1)[0]
        # Both arms of the query-driven rule are stated.
        assert "**Does the question name a discrete target?**" in section
        assert "`## Answer` section" in section
        assert "**When the question does not name a discrete target**" in section
        # Ambiguity resolves toward the fuller answer, never toward the shorter one.
        assert "treat it as NOT discrete and write the fuller answer" in section
        # And the section says so in as many words.
        disclaimer = "no length target, and nothing here asking you to be brief"
        assert disclaimer in section
        # Scan everything except that disclaimer, which necessarily quotes the phrasing it forbids.
        scanned = section.replace(disclaimer, "").lower()
        for brevity_artifact in (
            "be brief",
            "be concise",
            "keep it short",
            "word limit",
            "no more than",
            "at most 500",
            "maximum length",
        ):
            assert brevity_artifact not in scanned, brevity_artifact

    def test_source_tools_are_held_directly_but_demoted_to_verification(self, mock_llm_provider):
        """Source tools stay in the orchestrator's hands, and the prompt demotes them to
        verification without re-listing them: the names and descriptions reach the model through
        `bind_tools`, so the prompt only has to say how to use them.
        """
        captured = _build_and_capture(mock_llm_provider)
        assert {"web_search_tool", "knowledge_search"} <= {t.name for t in captured["tools"]}
        prompt = captured["system_prompt"]
        # Why a direct call is expensive, and that verifying a worker's finding is a legitimate
        # use of one rather than a duplicate. Both are judgment, not ceilings.
        assert "for verification, never for primary research" in prompt
        assert "raw results stay in this conversation" in prompt
        assert "is not a repeat" in prompt

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


class TestShallowSubagentToolNarrowing:
    """`shallow_subagent_tools` / `shallow_subagent_exclude_tools` scope retrieval to the sub-run.

    The knobs exist because the global `tools` / `exclude_tools` cannot express "the orchestrator
    keeps both web tools but the shallow sub-run only gets the wide one" — they decide what the
    whole agent can reach. These tests pin the two properties that make the distinction real:
    the narrowing applies, and it applies *only* to the sub-run.
    """

    @staticmethod
    def _tools(*names):
        # Not MagicMock: its `name` kwarg configures the mock's repr rather than setting a
        # `.name` attribute, which is the exact attribute the filter reads.
        return [SimpleNamespace(name=n) for n in names]

    def test_unset_inherits_the_full_tool_set(self):
        tools = self._tools("web_search_tool", "advanced_web_search_tool")
        assert _shallow_subagent_tools(tools, None, None) == tools
        assert _shallow_subagent_tools(tools, [], []) == tools

    def test_allowlist_narrows_to_the_named_tools(self):
        tools = self._tools("web_search_tool", "advanced_web_search_tool", "knowledge_search")
        selected = _shallow_subagent_tools(tools, ["web_search_tool"], None)
        assert [t.name for t in selected] == ["web_search_tool"]

    def test_page_fetch_survives_the_shallow_narrowing(self):
        """config_autonomous_frag.yml pins the sub-run to the wide search tool plus the page fetch.

        Case 1 (Shallow-Researcher) was 32 of the 90 eval trials — the largest single shape — so a
        narrowing that dropped fetch_url_tool would forfeit most of the capability's upside on the
        very path that uses it most.
        """
        tools = self._tools("web_search_tool", "advanced_web_search_tool", "fetch_url_tool")
        selected = _shallow_subagent_tools(tools, ["web_search_tool", "fetch_url_tool"], None)
        assert [t.name for t in selected] == ["web_search_tool", "fetch_url_tool"]

    def test_exclude_list_applies_after_the_allowlist(self):
        tools = self._tools("web_search_tool", "advanced_web_search_tool")
        selected = _shallow_subagent_tools(
            tools,
            ["web_search_tool", "advanced_web_search_tool"],
            ["advanced_web_search_tool"],
        )
        assert [t.name for t in selected] == ["web_search_tool"]

    def test_empty_result_falls_back_to_the_full_set(self):
        """A sub-run with zero tools answers from memory — the one outcome this path prevents.

        Reachable without a typo: startup validates the names against the agent's tool set, but a
        request's `data_sources` can still exclude every allowed tool.
        """
        tools = self._tools("web_search_tool")
        assert _shallow_subagent_tools(tools, ["knowledge_search"], None) == tools

    def test_narrowing_does_not_touch_the_orchestrator_tool_set(self, mock_llm_provider):
        """The orchestrator keeps every tool the agent was built with."""
        captured = _build_and_capture(mock_llm_provider)
        orchestrator_tool_names = {getattr(t, "name", "") for t in captured["tools"]}
        assert "web_search_tool" in orchestrator_tool_names


# =================================================================================================
# Delegated-research door flags (eval A/B)
# =================================================================================================

# (arm id, research_batch_tool, researcher_subagent). Both-off is not an arm; it is rejected.
RESEARCH_ARMS = [
    ("both", True, True),
    ("batch_only", True, False),
    ("subagent_only", False, True),
]

# What each arm must NOT mention anywhere the model can read it.
_ABSENT_ROUTE = {"batch_only": "researcher-agent", "subagent_only": "run_research_batch"}


def _model_visible_text(captured: dict) -> str:
    """Concatenate every string this build sends to the model as routing input.

    Three separate mechanisms carry route names — ``render_prompt`` for the system prompt,
    ``SubAgentMiddleware`` for subagent descriptions, and ``bind_tools`` for tool schemas — so a
    per-file audit is not enough to prove a disabled door is gone.
    """
    parts = [captured["system_prompt"]]
    parts += [str(spec.get("description", "")) for spec in captured["subagents"]]
    parts += [str(spec.get("system_prompt", "")) for spec in captured["subagents"]]
    parts += [str(tool.description or "") for tool in captured["tools"]]
    return "\n".join(parts)


class TestResearchDoorFlags:
    """`research_batch_tool` / `researcher_subagent` each remove one delegated-research door."""

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_only_the_configured_doors_are_built(self, mock_llm_provider, arm, batch, subagent):
        captured = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        tool_names = [t.name for t in captured["tools"]]
        subagent_names = [s["name"] for s in captured["subagents"]]
        assert ("run_research_batch" in tool_names) is batch
        assert ("researcher-agent" in subagent_names) is subagent
        # The rest of the menu is untouched in every arm: only the door moves.
        assert {"think", "get_verified_sources", "submit_final_report"} <= set(tool_names)
        assert {"planner-agent", "writer-agent", GENERAL_PURPOSE_SUBAGENT_NAME} <= set(subagent_names)

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_no_disabled_route_is_named_anywhere_the_model_can_read_it(self, mock_llm_provider, arm, batch, subagent):
        """The keystone test for this feature.

        In this agent descriptions ARE the routing logic, so a surviving mention of a door that was
        not built does not fail loudly — the model emits a call for an unbound tool, gets a generic
        dispatch error, and burns an orchestrator turn against ``max_orchestrator_turns`` with no
        recovery instruction. Repeated across an eval that reads as a quality difference between
        arms rather than as the bug it is. Every itemized gate elsewhere is a means to this end.
        """
        captured = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        absent = _ABSENT_ROUTE.get(arm)
        if absent is None:
            return
        assert absent not in _model_visible_text(captured)

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_the_surviving_door_is_still_named(self, mock_llm_provider, arm, batch, subagent):
        """The negative test above passes trivially if the routing text is simply gone."""
        captured = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        text = _model_visible_text(captured)
        assert ("run_research_batch" in text) is batch
        assert ("researcher-agent" in text) is subagent
        assert "Deciding what to do" in text
        for always in ("submit_final_report", "planner-agent", "writer-agent"):
            assert always in text, always

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_the_research_loop_stays_concise_in_every_arm(self, mock_llm_provider, arm, batch, subagent):
        """The hot-path ceiling is not a default-arm-only guarantee; it is re-read every turn."""
        prompt = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )["system_prompt"]
        section = prompt.split("# The Research Loop", 1)[1].split("\n# ", 1)[0]
        content_lines = [line for line in section.splitlines() if line.strip()]
        assert len(content_lines) <= 6, f"{arm}: research loop grew to {len(content_lines)} content lines"
        assert len(section) <= 1750, f"{arm}: research loop grew to {len(section)} chars"

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_the_decision_section_states_the_dependency_rule_generically(self, mock_llm_provider, arm, batch, subagent):
        """The prompt states the dependency SHAPE; the descriptions name the route for it.

        The section used to branch three ways on the door flags and name a subagent per shape,
        which made the orchestrator prompt carry per-subagent routing text — the thing this agent
        keeps in the descriptions. The rule that survives here is arm-independent, so it must be
        stated exactly once and must not name a door.
        """
        captured = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        prompt = captured["system_prompt"]
        rule = "**Send independent unknowns out together, dependent ones in order.**"
        assert prompt.count(rule) == 1
        assert "Never aim two research calls at the same unresolved fact" in prompt
        # The chain still has a home; it is just named by a description now, not by the prompt.
        if subagent:
            assert "PREREQUISITE CHAIN" in _model_visible_text(captured)

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_researcher_task_persistence_stays_wired_in_every_arm(self, mock_llm_provider, arm, batch, subagent):
        """It is a no-op without the subagent, and the ONLY evidence seam without the batch."""
        captured = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        assert "ResearcherTaskPersistenceMiddleware" in _middleware_names(captured)

    @pytest.mark.parametrize(("arm", "batch", "subagent"), RESEARCH_ARMS)
    def test_the_sanitizer_allowlist_tracks_the_batch_tool(self, mock_llm_provider, arm, batch, subagent):
        captured = _build_and_capture(
            mock_llm_provider,
            research_batch_tool=batch,
            researcher_subagent=subagent,
        )
        sanitizer = next(m for m in captured["middleware"] if type(m).__name__ == "ToolNameSanitizationMiddleware")
        assert ("run_research_batch" in set(sanitizer.valid_tool_names)) is batch
        assert {"web_search_tool", "knowledge_search", "submit_final_report"} <= set(sanitizer.valid_tool_names)

    def test_the_batch_worker_runnable_is_not_built_when_the_batch_is_off(self, mock_llm_provider):
        """`researcher_runnable` exists only to feed the batch tool; the task spec builds its own."""
        with patch("aiq_agent.agents.autonomous_researcher.factory.build_researcher_runnable") as build:
            _build_and_capture(mock_llm_provider, research_batch_tool=False, researcher_subagent=True)
        build.assert_not_called()

    def test_both_doors_off_is_rejected_by_the_factory(self, mock_llm_provider):
        """The factory is called directly by tests and by any caller bypassing the NAT config layer."""
        with pytest.raises(ValueError, match="at least one delegated research path"):
            _build_and_capture(mock_llm_provider, research_batch_tool=False, researcher_subagent=False)

    def test_the_default_arm_is_unchanged(self, mock_llm_provider):
        """The shipped arm's routing text must survive the flags byte for byte.

        Everything the A/B measures is a delta against this arm, so a stray rewording here would
        silently move the baseline. The shipped arm is batch-only: `run_research_batch` is the
        research path and `task` advertises no direct researcher door.
        """
        prompt = _build_and_capture(mock_llm_provider)["system_prompt"]
        for sentence in (
            "- **`run_research_batch`** is your primary research path;",
            "your only route to `shallow-researcher`, `planner-agent`, and `writer-agent`",
            "name `fetch_url_tool` in that query's `preferred_tools`",
            "steer them by naming them (exact names) in a `ResearchQuery.preferred_tools`",
            "If a target fails twice, do not send a third: record it as an explicit gap",
            "**Send independent unknowns out together, dependent ones in order.**",
        ):
            assert sentence in prompt, sentence

    def test_fetch_url_passages_vanish_when_the_tool_is_absent(self, mock_llm_provider):
        """A prompt that names a tool the config removed sends the model after something it cannot do.

        dsqa90 job 2026-09-01__11-56-18 excluded fetch_url_tool and still rendered all four
        passages; 40% of the deep path's delegated queries were attempts to retrieve a document,
        and one searched for a PDF filename it had already resolved. Every other door in this agent
        is gated on the flag that provides it.
        """
        with_tool = _build_and_capture(mock_llm_provider)["system_prompt"]
        without = _build_and_capture(mock_llm_provider, tools=[web_search_tool, knowledge_search])["system_prompt"]
        assert "fetch_url_tool" in with_tool
        assert "fetch_url_tool" not in without, "the prompt must not name a tool the agent does not hold"
        assert "no tool here can open a page" in without, "and it must say so, so the model records a gap"

    def test_the_both_doors_arm_is_unchanged(self, mock_llm_provider):
        """The opt-in arm is the A/B comparison, so pin its routing text the same way."""
        prompt = _build_and_capture(mock_llm_provider, researcher_subagent=True)["system_prompt"]
        for sentence in (
            "- **`run_research_batch`** is your primary research path;",
            "your only route to `shallow-researcher`, `planner-agent`, `researcher-agent`, and `writer-agent`",
            "If a target fails twice, give the whole chain to `researcher-agent` once",
            "**Send independent unknowns out together, dependent ones in order.**",
        ):
            assert sentence in prompt, sentence
