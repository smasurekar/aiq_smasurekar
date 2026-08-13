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
# Subagent descriptions — the routing mechanism
# =================================================================================================
# SubAgentMiddleware appends "Available subagent types:" plus each description verbatim to the
# `task` tool. In a design with no tier table and no tool hiding, these strings ARE the routing
# logic, so each one answers "when should the orchestrator pick me?" without reference to any
# effort level. They differentiate on context isolation and scale, matching the prompt's
# "Choosing a research path" section.

RESEARCHER_SUBAGENT_DESCRIPTION = (
    "Investigate ONE topic end-to-end in an isolated context and return structured, cited findings. "
    "Choose this when a single question needs iterative multi-hop work — resolve one fact, then use it "
    "to find the next — and you want the search trail digested rather than dumped into your own context. "
    "Give it exactly one topic, stated with full standalone context. For several INDEPENDENT questions at "
    "once use run_research_batch instead; for one quick lookup whose raw results you want to see, call a "
    "source tool directly."
)

PLANNER_SUBAGENT_DESCRIPTION = (
    "Turn a complex or multi-part request into an explicit answer strategy plus a set of ResearchQuery "
    "objects, persisted to /shared/plan.json. Choose this when the request has several interacting parts, "
    "an output shape that must be decided up front (a report, a comparison matrix, a briefing), or when you "
    "intend to publish through writer-agent — writer-agent reads its output contract from the plan, so "
    "planning is mandatory before any writer delegation. Skip it when you can answer inline."
)

WRITER_SUBAGENT_DESCRIPTION = (
    "Synthesize a long-form cited report from /shared/plan.json and the research notes under /shared/, "
    "writing the result to /shared/output.md. Choose this only when the answer is genuinely report-shaped "
    "and long enough that composing it inline would degrade it. Requires /shared/plan.json to exist first; "
    "the call is rejected otherwise. For anything you can write yourself, do that and call "
    "submit_final_report."
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
        "description": RESEARCHER_SUBAGENT_DESCRIPTION,
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

    The source-router subagent is never built (``context.enable_source_router`` is pinned False
    at construction): its advisory domain routing is precisely what rich tool descriptions
    replace, and every adaptive config already disabled it.
    """
    subagents = build_deep_research_subagents(context)
    for spec in subagents:
        if spec["name"] == PLANNER_AGENT:
            spec["response_format"] = AutonomousResearchPlan
            spec["description"] = PLANNER_SUBAGENT_DESCRIPTION
        elif spec["name"] == WRITER_AGENT:
            spec["description"] = WRITER_SUBAGENT_DESCRIPTION

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

    orchestrator_system_prompt = context.render_prompt(
        "orchestrator",
        clarifier_result=context.state.clarifier_result,
        # Advertised as callable, minus the source tools, which get their own richer
        # "Retrieval Tools" block in the prompt's context section.
        tools=[
            {"name": t.name, "description": t.description}
            for t in orchestrator_tools
            if t.name not in source_tool_names
        ],
        retrieval_tools=context.tool_set.tools_info,
        max_research_concurrency=context.max_research_concurrency,
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
