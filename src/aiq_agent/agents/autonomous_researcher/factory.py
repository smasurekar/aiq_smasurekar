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

"""Graph and middleware factory for the autonomous researcher agent.

One ``create_deep_agent`` call, one prompt render, no routing layer. Compared with
``adaptive_researcher.factory`` this file has no ``_render_orchestrator``, no ``TierResolver``,
no ``ComplexityRouterMiddleware``, no ``SingleShotShallowDelegationMiddleware``, and no
``hidden_tools_for_ceiling`` — the four machines the tier design used to decide what the model
was allowed to see.

In their place, two things carry the adaptivity:

**The orchestrator holds the full menu, always.** ``orchestrator_tools`` is
``[*helper_tools, run_research_batch, submit_final_report, *research_source_tools]``, and
``task`` reaches every subagent. Nothing is hidden by anything (the request-wide loop guard
withdraws research tools once the budget is spent, which is termination, not routing).

**Descriptions do the routing.** ``SubAgentMiddleware`` renders each subagent's ``description``
into the ``task`` tool and retrieval tool descriptions render into the prompt's context block, so
describing a capability well *is* the routing logic. The description strings below are
load-bearing and should be reviewed as carefully as code.

Everything expensive is imported rather than forked: the graph context, tool groupings, the
researcher runnable builder, the planner/writer subagent specs, and the shared middleware stack
all come from ``deep_researcher.factory``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore

from aiq_agent.agents.adaptive_researcher.custom_middleware import ConsecutiveThinkGuardMiddleware
from aiq_agent.agents.adaptive_researcher.custom_middleware import ResearcherLoopGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import EmptyContentFixMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ExecuteTimeoutClampMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import FilesystemToolCallGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import FinalReportOwnershipGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import StateMutationGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import StructuredResponseTextFallbackMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import TodoQuotaMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import TodoSuppressionMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolNameSanitizationMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolResultPruningMiddleware
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
from aiq_agent.agents.deep_researcher.factory import FILESYSTEM_TOOL_NAMES
from aiq_agent.agents.deep_researcher.factory import ORCHESTRATOR_AGENT
from aiq_agent.agents.deep_researcher.factory import PLANNER_AGENT
from aiq_agent.agents.deep_researcher.factory import RESEARCHER_AGENT
from aiq_agent.agents.deep_researcher.factory import WRITER_AGENT
from aiq_agent.agents.deep_researcher.factory import DeepResearchGraphContext
from aiq_agent.agents.deep_researcher.factory import DeepResearchMiddlewareSet
from aiq_agent.agents.deep_researcher.factory import DeepResearchToolSet
from aiq_agent.agents.deep_researcher.factory import build_common_middleware
from aiq_agent.agents.deep_researcher.factory import build_deep_research_subagents
from aiq_agent.agents.deep_researcher.factory import build_deep_research_tool_set
from aiq_agent.agents.deep_researcher.factory import build_researcher_runnable
from aiq_agent.agents.deep_researcher.factory import runtime_visibility_middleware
from aiq_agent.agents.deep_researcher.resource_limits import DeepResearchResourceLimits
from aiq_agent.agents.deep_researcher.resource_limits import StateBudgetLedger
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole

from .custom_middleware import AutonomousFinalizationMiddleware
from .custom_middleware import AutonomousFinalReportCommitTracker
from .custom_middleware import AutonomousOrchestratorLoopGuardMiddleware
from .custom_middleware import DirectSourcePromotionMiddleware
from .custom_middleware import PlanBeforeWriterMiddleware
from .custom_middleware import ResearcherTaskPersistenceMiddleware
from .models import AutonomousRequestTerminationConfig
from .models import AutonomousResearchAgentState
from .models import AutonomousResearchPlan
from .models import ResearcherLoopGuardConfig
from .models import ResearchNotes
from .tools.finalize import build_submit_final_report_tool
from .tools.research import build_autonomous_research_batch_tool

logger = logging.getLogger(__name__)

FINALIZE_TOOL_NAME = "submit_final_report"
GENERAL_PURPOSE_SUBAGENT_NAME = "general-purpose"

# The autonomous tool set is identical to the deep researcher's; alias the builder so agent.py
# reads clearly and future divergence has a single seam.
build_autonomous_research_tool_set = build_deep_research_tool_set


# =================================================================================================
# Subagent descriptions — the routing mechanism AND the delegation contract
# =================================================================================================
# SubAgentMiddleware renders each description in two places: into the `task` tool description
# (subagents.py `_build_task_tool`) and appended to the orchestrator system prompt under
# "Available subagent types:". In a design with no tier table and no tool hiding, these strings
# ARE the routing logic.
#
# As of 2026-08-18 they are also the *complete* delegation contract. The orchestrator prompt
# previously carried a parallel `# Subagents` + `# Subagent Delegation Instructions` pair that
# restated every trigger verbatim and added the `task()` brief templates and the ordering rules —
# three copies of the same advice, in violation of the prompt's own "never restate
# middleware-supplied text" rule. Those sections were folded in here, so each description now
# answers four questions in a fixed order:
#
#   WHEN TO CHOOSE IT  — the request properties that select this agent (the original routing text)
#   SEQUENCING         — where it sits relative to the other two, stated from its own perspective
#                        (the prompt's set-level "their order is fixed" list, distributed)
#   WHAT IT PRODUCES   — the artifact and where the runtime persists it
#   DELEGATION BRIEF   — the verbatim template, since a subagent cannot see the conversation
#
# Triggers are stated as PROPERTIES of the request, not as request categories: this agent exists
# to drop the tier ladder, and an enumerated set of request kinds would reintroduce it under
# another name. The earlier wording keyed off style ("comprehensive", "deep dive") instead, and
# won zero planner routing decisions across 90 DeepSearchQA trials; since writer-agent requires
# /shared/plan.json, that forced zero writer calls too. See
# misc/autonomous_researcher/autonomous-orchestrator-prompt-redesign-plan.md §D2.
#
# COST NOTE: because deepagents renders descriptions twice (tool schema + system prompt), text
# moved here is paid for twice per orchestrator turn, whereas the prompt sections it replaced were
# paid for once. Consolidating still removes two of the three copies, but keep these strings tight
# and never restate anything already carried by a tool description. See
# misc/autonomous_researcher/autonomous-researcher-review-feedback-analysis.md §1.5.
#
# The planner and writer briefs are request-conditional (`parent_report_context_available`,
# `execution_enabled`), so those two are built by functions rather than being module constants.

RESEARCHER_SUBAGENT_DESCRIPTION = (
    "Investigate ONE topic end-to-end in an isolated context and return structured, cited findings.\n"
    "\n"
    "WHEN TO CHOOSE IT: the question is a PREREQUISITE CHAIN — you must resolve one fact before you can even write "
    "the next query — because parallel workers cannot pass results to each other. Also choose it when a lookup has "
    "already failed twice and you want a fresh, isolated attempt: give it the whole chain plus what you already "
    "tried, so it does not repeat you. Give it exactly one topic, stated with full standalone context. For several "
    "INDEPENDENT questions use run_research_batch instead. Prefer either path over searching yourself, even for a "
    "single fact: a worker's search trail is digested into notes before it reaches you instead of accumulating in "
    "your context.\n"
    "\n"
    "SEQUENCING: runs after planning and before writing. A prerequisite chain is a dependent step, so send one "
    "delegation per assistant turn and wait for its result before the next step. Never fan out two queries aimed at "
    "the same unresolved fact, and never fan out a query whose text you cannot write until another has answered.\n"
    "\n"
    "WHAT IT PRODUCES: structured `ResearchNotes` for the one topic, carrying its own `evidence_judgment`. It is "
    "also the worker behind every `run_research_batch` query. The runtime persists its notes under /shared/ and "
    "registers their source locators exactly as the batch path does, so both paths leave the same evidence for "
    "`writer-agent` and for `get_verified_sources`.\n"
    "\n"
    "DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:\n"
    "    Research the following topic and return structured, cited notes:\n"
    "    <the single topic in full standalone context — entities, timeframe, units, and what a complete answer must "
    "contain>\n"
    "    Already attempted, do not repeat: <the queries or targets already tried, and how each failed>\n"
    "    Resolve the steps in order, carrying each answer into the next search. Return ResearchNotes with a source "
    "locator for every claim, and state explicitly anything you could not verify."
)


def build_planner_subagent_description(*, parent_report_context_available: bool) -> str:
    """Build ``planner-agent``'s description, including its request-conditional delegation brief.

    Args:
        parent_report_context_available: True when a parent report is mounted in ``/shared/`` for
            this request, which adds the delta-revision line to the brief and is itself one of the
            three routing triggers.
    """
    delta_line = (
        "\n    This is a parent-report revision. Plan only the delta research needed to revise "
        "/shared/original_report.md; do not plan a fresh report unless the user asked for one."
        if parent_report_context_available
        else ""
    )
    return (
        "Turn a compound request into an explicit answer strategy plus a set of ResearchQuery objects, persisted to "
        "/shared/plan.json.\n"
        "\n"
        "WHEN TO CHOOSE IT: ANY of these is true — (1) the request contains three or more distinct deliverables; "
        "(2) the answer's structure must be fixed before research — a sectioned report, a comparison matrix, a "
        "briefing — which also means you intend to publish through writer-agent, since writer-agent reads its "
        "output contract from the plan; (3) a parent report is mounted for this request. For a multi-part RESEARCH "
        "request this supersedes write_todos: delegate here rather than writing a todo list and researching it "
        "yourself. Skip it when one batch of queries and an inline answer would fully satisfy the request.\n"
        "\n"
        "SEQUENCING: runs FIRST, or not at all. Planning after results are in hand is too late — the plan cannot "
        "account for what you found, so you either re-run work or break the output contract the writer reads. If "
        "you did not plan first, finish the run yourself. Never re-delegate here once research has begun.\n"
        "\n"
        "WHAT IT PRODUCES: the runtime persists the returned plan to /shared/plan.json. Read it before starting "
        "research. Its existence is also a hard prerequisite of the writer-agent path.\n"
        "\n"
        "DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:\n"
        "    Create a research plan for the following user request:\n"
        f"    <paste the user's complete request here verbatim>{delta_line}\n"
        "    Use the search tools to establish what information exists and whether it is internal or external, then "
        "return the plan with answer_strategy, constraints, and queries. Every query needs full standalone context "
        "and a depth of low, medium, or high."
    )


def build_writer_subagent_description(
    *,
    parent_report_context_available: bool,
    execution_enabled: bool,
) -> str:
    """Build ``writer-agent``'s description, including its request-conditional delegation brief.

    Args:
        parent_report_context_available: True when a parent report is mounted, which adds the
            preserve-and-merge instruction to the brief.
        execution_enabled: True when the sandbox is available, which adds the chart-embedding
            rules. Omitting them when execution is off keeps the orchestrator from briefing the
            writer on artifacts it cannot produce.
    """
    delta_line = (
        "\n    Also read /shared/original_report.md and /shared/source_summary.md. Preserve supported "
        "parent-report material where it still applies, incorporate the new delta evidence, and write a "
        "complete standalone revised report. Do not produce an insertion plan, patch, diff, or "
        "explanation of the rewrite."
        if parent_report_context_available
        else ""
    )
    # In delta mode the writer stops being optional: preserved parent citations only stay
    # verifiable if they travel through the writer path, so this overrides the length test above.
    delta_requirement = (
        " REQUIRED when a parent report is mounted, whatever the length — this is the only path that "
        "carries preserved parent citations through in a verifiable form, so use it even when the "
        "requested change looks small."
        if parent_report_context_available
        else ""
    )
    chart_line = (
        "\n    Embed each earned chart exactly once with ![<caption>](artifact://<filename>); never paste "
        "sandbox paths or base64 data. Only generate or embed a chart when its data is source-anchored "
        "and reasonably complete; otherwise present the table with explicit gaps and state the limitation."
        if execution_enabled
        else ""
    )
    return (
        "Synthesize a long-form cited report from /shared/plan.json and the research notes under /shared/, writing "
        "the result to /shared/output.md.\n"
        "\n"
        "WHEN TO CHOOSE IT: only when the deliverable has named sections the user asked for and is long enough that "
        "composing it inline would degrade it. For anything you can write well yourself, do that and call "
        f"submit_final_report instead.{delta_requirement}\n"
        "\n"
        "SEQUENCING: runs LAST — it is the run's final action. Requires /shared/plan.json to exist first; the "
        "runtime rejects the call otherwise, so planner-agent is a hard prerequisite. After it returns, report its "
        "short completion marker as your entire reply: do NOT call submit_final_report, and do not research, "
        "verify, delegate, rewrite, summarize, or comment on the report afterwards.\n"
        "\n"
        "WHAT IT PRODUCES: the final Markdown answer at /shared/output.md, plus the completion marker it returns to "
        "you.\n"
        "\n"
        "DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:\n"
        "    Synthesize the final answer for the user request using the files already written to /shared/.\n"
        f"    Read /shared/plan.json, every research note file under /shared/, and the verified sources.{delta_line}\n"
        "    Before writing, inspect Available Skills. If an applicable writer skill exists, read its SKILL.md "
        "first and treat it as the controlling synthesis protocol; do not draft the answer until you have read it.\n"
        "    For broad reports, produce a cross-synthesized narrative with developed paragraphs. Do not compress "
        "the answer into a checklist of short component summaries unless the user asked for that "
        f"format.{chart_line}\n"
        "    Cite every material claim with numeric citations drawn only from the verified sources, and end with a "
        "compact Sources section.\n"
        "    Write the final Markdown answer to /shared/output.md.\n"
        "    Return only the short completion marker `Wrote /shared/output.md` once the file is written. Do not "
        "return JSON and do not echo the full Markdown."
    )


# Not a real delegation route. Supplying a spec under this name is what suppresses deepagents'
# auto-injected general-purpose subagent (graph.py only auto-adds it when no inline spec claims
# the name). The default is actively unsafe here: it would inherit the parent's ENTIRE tool list —
# including submit_final_report (return_direct=True) and run_research_batch — and run on a fresh
# default middleware stack with no SourceRegistryMiddleware, so any citation it produced could
# never verify. Its shipped description also advertises it for "researching complex questions",
# competing head-on with researcher-agent in a design where descriptions are the routing logic.
#
# This is the per-agent mechanism, deliberately chosen over HarnessProfile(...enabled=False):
# harness profiles are process-global (_HARNESS_PROFILES is a module-level dict) and all three
# research arms resolve to the same model key, so disabling it there would silently mutate the
# deep and adaptive control arms too.
GENERAL_PURPOSE_STUB_DESCRIPTION = (
    "NOT AVAILABLE in this agent — it has no tools and cannot do anything. Never delegate to it. "
    "For research use `researcher-agent`; for planning use `planner-agent`; for report writing use "
    "`writer-agent`."
)
GENERAL_PURPOSE_STUB_PROMPT = (
    "You have no tools and no role in this agent. Immediately reply with exactly: "
    "'general-purpose is not available; use researcher-agent, planner-agent, or writer-agent instead.'"
)


def _as_list_item(description: str) -> str:
    """Indent a multi-line description so it survives deepagents' list rendering.

    Both places that render a description do it as ``f"- {name}: {description}"``
    (``subagents.py`` ``_build_task_tool`` and ``SubAgentMiddleware.__init__``). That assumes a
    one-line string: every line after the first lands at column 0, escapes the list item, and
    visually merges with the *next* agent's entry — so the writer's delegation brief can read as
    part of planner-agent. The descriptions here are deliberately multi-line, so continuation
    lines are indented two spaces to stay inside their own bullet.

    Applied at spec-build time rather than baked into the constants, which keeps the source
    strings readable and keeps the indentation concern in one place.
    """
    first, _, rest = description.partition("\n")
    if not rest:
        return description
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in rest.split("\n"))
    return f"{first}\n{indented}"


def build_autonomous_orchestrator_middleware(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    research_batch_tool_name: str = "run_research_batch",
    finalize_tool_name: str = FINALIZE_TOOL_NAME,
) -> list[Any]:
    """Middleware for the autonomous orchestrator.

    Same shape as ``deep_researcher.build_orchestrator_middleware`` with three differences:

    - ``SourceRoutingGuardMiddleware`` is omitted. That guard blocks every orchestrator tool call
      until the source-router writes ``/shared/source_routing.json``; the autonomous agent has no
      source-router subagent at all (rich tool descriptions are exactly what it substituted for),
      so the guard would deadlock every run.
    - ``submit_final_report`` **and every source tool** join the tool-name sanitizer allowlist.
      Upstream deliberately excludes source tools there because the deep orchestrator routes all
      retrieval through ``run_research_batch``; here it holds them directly, so a source-tool name
      it emits is a legitimate call and must not be rewritten.
    - ``DirectSourcePromotionMiddleware`` is inserted immediately **before**
      ``source_registry_middleware``. Middleware compose first-is-outermost, so this ordering is
      what lets the promotion middleware observe sources the registry captured during the same
      tool call; listing it later would make it inner and it would see nothing.
    """
    valid_tool_names = {tool.name for tool in tool_set.helper_tools}
    valid_tool_names.add(research_batch_tool_name)
    valid_tool_names.add(finalize_tool_name)
    valid_tool_names.update(FILESYSTEM_TOOL_NAMES)
    valid_tool_names.update(tool.name for tool in tool_set.research_source_tools)
    return [
        EmptyContentFixMiddleware(),
        ToolNameSanitizationMiddleware(valid_tool_names=sorted(valid_tool_names)),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        DirectSourcePromotionMiddleware(
            source_registry_middleware=source_registry_middleware,
            source_tool_names={tool.name for tool in tool_set.research_source_tools},
        ),
        source_registry_middleware,
        ToolResultPruningMiddleware(keep_last_n=10, max_chars=2000),
        ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
        ConsecutiveThinkGuardMiddleware(),
    ]


def build_autonomous_research_middleware_set(
    *,
    tool_set: DeepResearchToolSet,
    source_registry_middleware: SourceRegistryMiddleware,
    researcher_loop_guard: ResearcherLoopGuardConfig,
    artifact_manager: object | None = None,
) -> DeepResearchMiddlewareSet:
    """Build researcher, planner, writer, and orchestrator middleware stacks."""

    def common(extra_valid_tool_names: Sequence[str] = ()) -> list[Any]:
        return build_common_middleware(
            tool_set=tool_set,
            source_registry_middleware=source_registry_middleware,
            artifact_manager=artifact_manager,
            extra_valid_tool_names=extra_valid_tool_names,
        )

    researcher_middleware = common()
    tool_retry_index = next(
        index for index, middleware in enumerate(researcher_middleware) if isinstance(middleware, ToolRetryMiddleware)
    )
    researcher_middleware[tool_retry_index:tool_retry_index] = [
        ResearcherLoopGuardMiddleware(
            source_tool_names=tool_set.source_tool_names,
            config=researcher_loop_guard,
        ),
        ConsecutiveThinkGuardMiddleware(
            max_consecutive_thinks=researcher_loop_guard.max_consecutive_thinks,
        ),
    ]

    return DeepResearchMiddlewareSet(
        researcher=researcher_middleware,
        planner=[*common(), ConsecutiveThinkGuardMiddleware()],
        writer=[*common(), ConsecutiveThinkGuardMiddleware()],
        orchestrator=build_autonomous_orchestrator_middleware(
            tool_set=tool_set,
            source_registry_middleware=source_registry_middleware,
        ),
    )


def _researcher_subagent_spec(
    context: DeepResearchGraphContext,
    *,
    system_prompt: str,
) -> dict[str, Any]:
    """Build the ``task``-reachable ``researcher-agent`` spec.

    New relative to the tier design, where the researcher existed only as the runnable behind
    ``run_research_batch``. Exposing it through ``task`` as well gives the orchestrator a
    single-topic, context-isolated research path without inventing a second researcher: this spec
    carries the same tools, the same prompt, the same loop guards, and the same structured
    ``ResearchNotes`` contract as the batch workers.

    One difference is unavoidable and deliberate: ``run_research_batch`` sets
    ``CURRENT_RESEARCHER_GUARD_STATE`` per worker (seeded from the query's ``depth``), whereas a
    ``task`` delegation has no ``ResearchQuery`` and therefore no depth to seed it with. With no
    guard state, ``ResearcherLoopGuardMiddleware`` passes through and the sub-run is bounded by
    the request-wide guard and ``DeepResearchResourceLimits`` instead of a per-depth source-call
    budget. That is the correct trade: this path exists precisely for open-ended multi-hop work
    whose number of hops is not knowable in advance.
    """
    return {
        "name": RESEARCHER_AGENT,
        "description": _as_list_item(RESEARCHER_SUBAGENT_DESCRIPTION),
        "system_prompt": system_prompt,
        "tools": list(context.tool_set.researcher_tools),
        "model": context.llm_provider.get(LLMRole.RESEARCHER),
        "permissions": context.permissions(RESEARCHER_AGENT),
        "middleware": context.middleware(
            [
                *context.middleware_set.researcher,
                FinalReportOwnershipGuardMiddleware(),
                StateMutationGuardMiddleware(
                    writer=False,
                    sandbox_enabled=context.runtime.execution_enabled,
                ),
                TodoSuppressionMiddleware(),
                StructuredResponseTextFallbackMiddleware(ResearchNotes),
            ]
        ),
        "response_format": ResearchNotes,
        "skills": context.skill_sources(RESEARCHER_AGENT),
    }


def build_autonomous_subagents(context: DeepResearchGraphContext, *, researcher_prompt: str) -> list[dict[str, Any]]:
    """Build exactly four subagent specs: researcher, planner, writer, and the inert stub.

    Planner and writer are reused verbatim from ``build_deep_research_subagents`` — same prompts,
    same persistence and commit middleware — with two edits: the planner's structured output is
    retyped to ``AutonomousResearchPlan`` so planner-authored queries carry the per-query
    ``depth`` hint, and both descriptions are replaced with autonomous-specific routing text.

    Those two descriptions are *built*, not looked up, because their delegation briefs vary with
    the request: the planner's on ``parent_report_context_available`` and the writer's on that
    plus ``runtime.execution_enabled``. This is what lets the orchestrator prompt drop its
    ``# Subagent Delegation Instructions`` section, whose Jinja conditionals were the only reason
    the briefs had to live in the prompt rather than alongside the routing triggers.

    The source-router subagent is never built (``context.enable_source_router`` is pinned False
    at construction): its advisory domain routing is precisely what rich tool descriptions
    replace, and every adaptive config already disabled it.
    """
    planner_description = _as_list_item(
        build_planner_subagent_description(
            parent_report_context_available=context.parent_report_context_available,
        )
    )
    writer_description = _as_list_item(
        build_writer_subagent_description(
            parent_report_context_available=context.parent_report_context_available,
            execution_enabled=context.runtime.execution_enabled,
        )
    )
    subagents = build_deep_research_subagents(context)
    for spec in subagents:
        if spec["name"] == PLANNER_AGENT:
            spec["response_format"] = AutonomousResearchPlan
            spec["description"] = planner_description
        elif spec["name"] == WRITER_AGENT:
            spec["description"] = writer_description

    subagents.insert(0, _researcher_subagent_spec(context, system_prompt=researcher_prompt))
    subagents.append(
        {
            "name": GENERAL_PURPOSE_SUBAGENT_NAME,
            "description": GENERAL_PURPOSE_STUB_DESCRIPTION,
            "system_prompt": GENERAL_PURPOSE_STUB_PROMPT,
            # An explicit empty list, not an omitted key: deepagents inherits the parent's tools
            # when "tools" is absent from the spec, which is the hazard this stub exists to avoid.
            "tools": [],
        }
    )
    return subagents


def build_autonomous_research_graph(
    *,
    llm_provider: LLMProvider,
    state: AutonomousResearchAgentState,
    prompts: dict[str, str],
    tools: Sequence[BaseTool],
    runtime: DeepAgentsRuntime,
    tool_set: DeepResearchToolSet,
    middleware_set: DeepResearchMiddlewareSet,
    source_registry_middleware: SourceRegistryMiddleware,
    final_report_tracker: AutonomousFinalReportCommitTracker,
    callbacks: list[Any],
    max_research_concurrency: int,
    researcher_loop_guard: ResearcherLoopGuardConfig,
    request_termination: AutonomousRequestTerminationConfig | None = None,
    resource_limits: DeepResearchResourceLimits | None = None,
    state_budget: StateBudgetLedger | None = None,
) -> Any:
    """Build the full DeepAgents graph for one autonomous research run.

    Args:
        final_report_tracker: The run's dual-exit tracker. Owned by the agent (one per request)
            because both the writer's ``FinalReportCommitMiddleware`` and the orchestrator's
            ``submit_final_report`` must record onto the same instance.

    Returns:
        The compiled runnable, configured with the request's recursion ceiling.
    """
    cross_cutting_middleware = [
        FilesystemToolCallGuardMiddleware(),
        *runtime_visibility_middleware(runtime),
    ]
    execute_ceiling = runtime.execute_timeout_seconds
    if execute_ceiling:
        # Agent-supplied execute timeouts are unreliable (models pass milliseconds or arbitrarily
        # large values); clamp them to the configured sandbox lifetime.
        cross_cutting_middleware = [
            ExecuteTimeoutClampMiddleware(max_timeout_seconds=execute_ceiling),
            *cross_cutting_middleware,
        ]

    # Upstream 2.2.0 made resource_limits / final_report_tracker / state_budget required on
    # DeepResearchGraphContext. The ledger takes the run's real state.files and the runtime's
    # actual sandbox flag — not empty/True placeholders — so byte accounting starts from what the
    # request actually carries.
    limits = resource_limits or DeepResearchResourceLimits()
    budget = state_budget or StateBudgetLedger(
        limits=limits,
        files=state.files,
        sandbox_enabled=runtime.execution_enabled,
    )
    context = DeepResearchGraphContext(
        llm_provider=llm_provider,
        state=state,
        prompts=prompts,
        tools=tools,
        runtime=runtime,
        tool_set=tool_set,
        middleware_set=middleware_set,
        domain_catalog_path=None,
        current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        max_research_concurrency=max_research_concurrency,
        resource_limits=limits,
        # Pinned off: the autonomous agent has no source-router subagent and no domain catalog.
        enable_source_router=False,
        backend=runtime.backend,
        visibility_middleware=cross_cutting_middleware,
        final_report_tracker=final_report_tracker,
        state_budget=budget,
    )
    request_termination = request_termination or AutonomousRequestTerminationConfig()

    # One researcher prompt render, shared by the batch workers and the task-reachable subagent,
    # so the two paths cannot drift.
    researcher_prompt = context.render_prompt(
        "researcher",
        tools=context.tool_set.tools_info,
        execution_enabled=context.runtime.execution_enabled,
        researcher_source_call_budgets=researcher_loop_guard.source_call_budgets.model_dump(),
        researcher_max_identical_source_calls=researcher_loop_guard.max_identical_source_calls,
        researcher_loop_guard_enabled=researcher_loop_guard.enabled,
    )
    researcher_runnable = build_researcher_runnable(
        researcher_model=context.llm_provider.get(LLMRole.RESEARCHER),
        researcher_tools=context.tool_set.researcher_tools,
        system_prompt=researcher_prompt,
        researcher_middleware=[
            *context.middleware_set.researcher,
            FinalReportOwnershipGuardMiddleware(),
            StateMutationGuardMiddleware(
                writer=False,
                sandbox_enabled=context.runtime.execution_enabled,
            ),
        ],
        skill_sources=context.skill_sources(RESEARCHER_AGENT),
        backend=context.backend,
        visibility_middleware=context.visibility_middleware,
        filesystem_permissions=context.permissions(RESEARCHER_AGENT),
    )
    research_batch_tool = build_autonomous_research_batch_tool(
        researcher_runnable=researcher_runnable,
        backend=context.backend,
        callbacks=callbacks,
        max_research_concurrency=max_research_concurrency,
        resource_limits=context.resource_limits,
        state_budget=context.state_budget,
        source_registry_middleware=source_registry_middleware,
    )
    submit_final_report_tool = build_submit_final_report_tool(
        backend=context.backend,
        tracker=final_report_tracker,
    )

    # The full menu, unconditionally. Source tools sit alongside run_research_batch and task, and
    # the model decides how to research from the descriptions alone.
    research_source_tools = list(context.tool_set.research_source_tools)
    orchestrator_tools = [
        *context.tool_set.helper_tools,
        research_batch_tool,
        submit_final_report_tool,
        *research_source_tools,
    ]
    source_tool_names = frozenset(tool.name for tool in research_source_tools)

    # No `tools=` / `retrieval_tools=` here on purpose. Every tool the orchestrator holds is bound
    # to the model through `bind_tools`, which already carries each tool's name, description, and
    # argument schema. Rendering name+description into the prompt as well was a verbatim second
    # copy — measured at ~3.9k characters per turn against 3.8k of bound schema — and it is the
    # prompt half that goes stale: AutonomousOrchestratorLoopGuardMiddleware withdraws
    # run_research_batch, think, and the source tools once the request is finalizing, and a prompt
    # list cannot be withdrawn with them. `task`, `write_todos`, and the six filesystem tools were
    # already schema-only and have never needed a prompt entry. See
    # misc/autonomous_researcher/autonomous-researcher-review-feedback-analysis.md §1.10.
    #
    # No `request_budgets=` and no `max_research_concurrency=` here on purpose. Budgets are the
    # middleware's to own and to state: AutonomousOrchestratorLoopGuardMiddleware enforces every
    # ceiling, explains itself in the blocked ToolMessage when one fires, and warns in-context via
    # the nudge before withdrawing the source tools. A prompt copy could only drift from it — that
    # is exactly how the prompt came to promise "one batch per request" against a configured 6 —
    # and it is re-sent on every turn whether or not any budget is close. The per-batch query
    # ceiling moved to the run_research_batch description, next to the schema that enforces it.
    orchestrator_system_prompt = context.render_prompt(
        "orchestrator",
        clarifier_result=context.state.clarifier_result,
        execution_enabled=context.runtime.execution_enabled,
        parent_report_context_available=context.parent_report_context_available,
    )

    orchestrator_middleware = [
        *context.middleware_set.orchestrator,
        FinalReportOwnershipGuardMiddleware(),
        StateMutationGuardMiddleware(
            writer=False,
            sandbox_enabled=context.runtime.execution_enabled,
        ),
        TodoQuotaMiddleware(resource_limits=context.resource_limits),
        # Evidence-state seams: make the three research paths equivalent in what they leave behind.
        ResearcherTaskPersistenceMiddleware(
            backend=context.backend,
            state_budget=context.state_budget,
            resource_limits=context.resource_limits,
            source_registry_middleware=source_registry_middleware,
        ),
        PlanBeforeWriterMiddleware(),
    ]
    if request_termination.enabled:
        orchestrator_middleware.append(
            AutonomousOrchestratorLoopGuardMiddleware(
                config=request_termination,
                source_tool_names=source_tool_names,
            )
        )
    # Deliberately NOT RequiredWriterDelegationMiddleware: that would force every run through the
    # writer and eliminate the valid inline exit. This guard accepts either.
    orchestrator_middleware.append(AutonomousFinalizationMiddleware(tracker=final_report_tracker))

    agent = create_deep_agent(
        model=context.llm_provider.get(LLMRole.ORCHESTRATOR),
        tools=orchestrator_tools,
        system_prompt=orchestrator_system_prompt,
        subagents=build_autonomous_subagents(context, researcher_prompt=researcher_prompt),
        store=InMemoryStore(),
        middleware=context.middleware(orchestrator_middleware),
        permissions=context.permissions(ORCHESTRATOR_AGENT),
        backend=context.backend,
    )
    return agent.with_config({"recursion_limit": request_termination.recursion_limit})
