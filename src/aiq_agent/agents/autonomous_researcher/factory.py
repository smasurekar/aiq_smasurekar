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
from dataclasses import dataclass
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
from .custom_middleware import ShallowFinalizationMiddleware
from .models import AutonomousRequestTerminationConfig
from .models import AutonomousResearchAgentState
from .models import AutonomousResearchPlan
from .models import ResearcherLoopGuardConfig
from .models import ResearchNotes
from .subagents import ShallowSubagentCapture
from .subagents import build_shallow_researcher_subagent
from .subagents import last_human_text
from .tools.finalize import build_submit_final_report_tool
from .tools.research import build_autonomous_research_batch_tool

logger = logging.getLogger(__name__)

FINALIZE_TOOL_NAME = "submit_final_report"
GENERAL_PURPOSE_SUBAGENT_NAME = "general-purpose"

# Loop bounds handed to the shallow sub-agent. Same defaults the standalone
# ``shallow_research_agent`` uses, so the delegated run behaves as it does on its own.
DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS = 10
DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS = 5

# The autonomous tool set is identical to the deep researcher's; alias the builder so agent.py
# reads clearly and future divergence has a single seam.
build_autonomous_research_tool_set = build_deep_research_tool_set


@dataclass(frozen=True)
class AutonomousResearchGraphRun:
    """The compiled graph plus the run-scoped state the graph result cannot carry.

    ``build_autonomous_research_graph`` used to return the runnable alone. The shallow sub-agent
    needs two things that a returned graph state cannot provide: a handle to cancel an in-flight
    shallow run during request teardown (it executes in a detached ``asyncio.Task``, so cancelling
    the awaiting coroutine does not stop it), and a way to recover a completed report after
    ``ainvoke`` raised on the timeout / recursion paths, where no state update reaches the caller.
    """

    runnable: Any
    shallow_capture: ShallowSubagentCapture | None = None


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
# `execution_enabled`). Their bodies are still module-level templates; the conditional fragments are
# separate constants that `build_planner_subagent_description` and `build_writer_subagent_description`
# splice into the `{...}` placeholders, so every string the model sees is readable in one piece.

# ------------------------------------------------------------------------------------------------
# Prompt-text convention for every string below
# ------------------------------------------------------------------------------------------------
# These are triple-quoted so they read as prose in the source, but the text the model receives is
# NOT wrapped: each paragraph is one long logical line, and the only real newlines are the blank
# lines between sections and the indented DELEGATION BRIEF lines. A trailing ``\`` is a source-only
# soft wrap — Python drops it together with the newline — so keep the space that separates the two
# words BEFORE the backslash and start the continuation at column 0. Re-wrapping the source this
# way is free; reflowing a paragraph into real newlines changes what the model reads.

# The cheapest rung of the ladder, and the only subagent whose output is the run's answer rather
# than an input to a later step. Every constraint stated below is carried by this text alone:
# nothing hides this subagent after the first turn and nothing forces the model to reach for it,
# because in this agent descriptions are the routing logic (see the section header above).
#
# The one thing the runtime *does* enforce is the exit: ShallowFinalizationMiddleware commits a
# successful report and ends the run before the model is called again. So "there is nothing to do
# afterwards" is a fact rather than an instruction — the wording below tells the model that
# plainly so it does not plan follow-up work it will never get to perform.
SHALLOW_SUBAGENT_DESCRIPTION = """\
Answer the whole request end-to-end in one bounded run and return the finished, cited answer.

WHEN TO CHOOSE IT: the request is one a single agent can finish by itself — the answer is a fact, \
a short list, a definition, a current value, or a couple of related points, and you can already \
picture the finished reply as a paragraph or two. Judge the REQUEST, not the topic: a question can \
be about a specialist subject and still be easy. Choose it whenever nothing about the request \
requires splitting the work up — you do not need a fixed set of sections, you are not chasing \
several unrelated unknowns, and no answer has to be resolved before another can be asked. When \
that is true this is always the right first move: it costs one run instead of a research cycle \
plus a composition turn.

SEQUENCING: FIRST, or not at all, and exactly ONCE. It is the opening move of the run or it is \
never used. Once you have searched, batched, planned, or delegated anything else, this is no \
longer available to you and re-delegating here only wastes a turn. Do not pair it with any other \
tool call in the same turn.

WHAT IT PRODUCES: the complete answer for the user, already written and already cited — not \
notes, not a draft, not evidence for you to synthesize. The runtime records it as the run's final \
report the moment it returns, and the run ends there. You will not be asked to do anything with \
it: do not review, verify, reformat, summarize, or comment on it, and do NOT call \
submit_final_report afterwards.

IF IT FAILS: it answers instead with a short notice saying it could not complete the request. \
That is the one and only case where research continues after this agent — treat the notice as \
your signal to research the request yourself from scratch with run_research_batch and finish \
normally. A returned answer is never a failure notice; do not escalate on one.

DELEGATION BRIEF: none needed. The runtime hands it the user's original request verbatim, so pass \
the user's request as `description` and add nothing else."""


RESEARCHER_SUBAGENT_DESCRIPTION = """\
Investigate ONE topic end-to-end in an isolated context and return structured, cited findings.

WHEN TO CHOOSE IT: the question is a PREREQUISITE CHAIN — you must resolve one fact before you can even write \
the next query — because parallel workers cannot pass results to each other. Also choose it when a lookup has \
already failed twice and you want a fresh, isolated attempt: give it the whole chain plus what you already \
tried, so it does not repeat you. Give it exactly one topic, stated with full standalone context. For several \
INDEPENDENT questions use run_research_batch instead. Prefer either path over searching yourself, even for a \
single fact: a worker's search trail is digested into notes before it reaches you instead of accumulating in \
your context.

SEQUENCING: runs after planning and before writing. A prerequisite chain is a dependent step, so send one \
delegation per assistant turn and wait for its result before the next step. Never fan out two queries aimed at \
the same unresolved fact, and never fan out a query whose text you cannot write until another has answered.

WHAT IT PRODUCES: structured `ResearchNotes` for the one topic, carrying its own `evidence_judgment`. It is \
also the worker behind every `run_research_batch` query. The runtime persists its notes under /shared/ and \
registers their source locators exactly as the batch path does, so both paths leave the same evidence for \
`writer-agent` and for `get_verified_sources`.

DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:
    Research the following topic and return structured, cited notes:
    <the single topic in full standalone context — entities, timeframe, units, and what a complete answer must \
contain>
    Already attempted, do not repeat: <the queries or targets already tried, and how each failed>
    Resolve the steps in order, carrying each answer into the next search. Return ResearchNotes with a source \
locator for every claim, and state explicitly anything you could not verify."""


# Spliced into ``{delta_line}`` when a parent report is mounted. That flag is also one of the three
# routing triggers, so it appears in WHEN TO CHOOSE IT as well.
_PLANNER_DELTA_BRIEF_LINE = """
    This is a parent-report revision. Plan only the delta research needed to revise \
/shared/original_report.md; do not plan a fresh report unless the user asked for one."""

_PLANNER_SUBAGENT_DESCRIPTION = """\
Turn a compound request into an explicit answer strategy plus a set of ResearchQuery objects, persisted to \
/shared/plan.json.

WHEN TO CHOOSE IT: ANY of these is true — (1) the request contains three or more distinct deliverables; \
(2) the answer's structure must be fixed before research — a sectioned report, a comparison matrix, a \
briefing — which also means you intend to publish through writer-agent, since writer-agent reads its \
output contract from the plan; (3) a parent report is mounted for this request. For a multi-part RESEARCH \
request this supersedes write_todos: delegate here rather than writing a todo list and researching it \
yourself. Skip it when one batch of queries and an inline answer would fully satisfy the request.

SEQUENCING: runs FIRST, or not at all. Planning after results are in hand is too late — the plan cannot \
account for what you found, so you either re-run work or break the output contract the writer reads. If \
you did not plan first, finish the run yourself. Never re-delegate here once research has begun.

WHAT IT PRODUCES: the runtime persists the returned plan to /shared/plan.json. Read it before starting \
research. Its existence is also a hard prerequisite of the writer-agent path.

DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:
    Create a research plan for the following user request:
    <paste the user's complete request here verbatim>{delta_line}
    Use the search tools to establish what information exists and whether it is internal or external, then \
return the plan with answer_strategy, constraints, and queries. Every query needs full standalone context \
and a depth of low, medium, or high."""


def build_planner_subagent_description(*, parent_report_context_available: bool) -> str:
    """Build ``planner-agent``'s description, including its request-conditional delegation brief.

    Args:
        parent_report_context_available: True when a parent report is mounted in ``/shared/`` for
            this request, which adds the delta-revision line to the brief and is itself one of the
            three routing triggers.
    """
    return _PLANNER_SUBAGENT_DESCRIPTION.format(
        delta_line=_PLANNER_DELTA_BRIEF_LINE if parent_report_context_available else "",
    )


# Spliced into ``{delta_line}`` when a parent report is mounted.
_WRITER_DELTA_BRIEF_LINE = """
    Also read /shared/original_report.md and /shared/source_summary.md. Preserve supported \
parent-report material where it still applies, incorporate the new delta evidence, and write a \
complete standalone revised report. Do not produce an insertion plan, patch, diff, or \
explanation of the rewrite."""

# Spliced into ``{delta_requirement}``. In delta mode the writer stops being optional: preserved
# parent citations only stay verifiable if they travel through the writer path, so this overrides
# the length test in WHEN TO CHOOSE IT.
_WRITER_DELTA_REQUIREMENT = """ REQUIRED when a parent report is mounted, whatever the length — this is the only \
path that carries preserved parent citations through in a verifiable form, so use it even when the \
requested change looks small."""

# Spliced into ``{chart_line}`` only when the sandbox is available. Omitting it when execution is
# off keeps the orchestrator from briefing the writer on artifacts it cannot produce.
_WRITER_CHART_BRIEF_LINE = """
    Embed each earned chart exactly once with ![<caption>](artifact://<filename>); never paste \
sandbox paths or base64 data. Only generate or embed a chart when its data is source-anchored \
and reasonably complete; otherwise present the table with explicit gaps and state the limitation."""

_WRITER_SUBAGENT_DESCRIPTION = """\
Synthesize a long-form cited report from /shared/plan.json and the research notes under /shared/, writing \
the result to /shared/output.md.

WHEN TO CHOOSE IT: only when the deliverable has named sections the user asked for and is long enough that \
composing it inline would degrade it. For anything you can write well yourself, do that and call \
submit_final_report instead.{delta_requirement}

SEQUENCING: runs LAST — it is the run's final action. Requires /shared/plan.json to exist first; the \
runtime rejects the call otherwise, so planner-agent is a hard prerequisite. After it returns, report its \
short completion marker as your entire reply: do NOT call submit_final_report, and do not research, \
verify, delegate, rewrite, summarize, or comment on the report afterwards.

WHAT IT PRODUCES: the final Markdown answer at /shared/output.md, plus the completion marker it returns to \
you.

DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:
    Synthesize the final answer for the user request using the files already written to /shared/.
    Read /shared/plan.json, every research note file under /shared/, and the verified sources.{delta_line}
    Before writing, inspect Available Skills. If an applicable writer skill exists, read its SKILL.md \
first and treat it as the controlling synthesis protocol; do not draft the answer until you have read it.
    For broad reports, produce a cross-synthesized narrative with developed paragraphs. Do not compress \
the answer into a checklist of short component summaries unless the user asked for that \
format.{chart_line}
    Cite every material claim with numeric citations drawn only from the verified sources, and end with a \
compact Sources section.
    Write the final Markdown answer to /shared/output.md.
    Return only the short completion marker `Wrote /shared/output.md` once the file is written. Do not \
return JSON and do not echo the full Markdown."""


def build_writer_subagent_description(
    *,
    parent_report_context_available: bool,
    execution_enabled: bool,
) -> str:
    """Build ``writer-agent``'s description, including its request-conditional delegation brief.

    Args:
        parent_report_context_available: True when a parent report is mounted, which makes the
            writer mandatory and adds the preserve-and-merge instruction to the brief.
        execution_enabled: True when the sandbox is available, which adds the chart-embedding
            rules.
    """
    return _WRITER_SUBAGENT_DESCRIPTION.format(
        delta_requirement=_WRITER_DELTA_REQUIREMENT if parent_report_context_available else "",
        delta_line=_WRITER_DELTA_BRIEF_LINE if parent_report_context_available else "",
        chart_line=_WRITER_CHART_BRIEF_LINE if execution_enabled else "",
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
GENERAL_PURPOSE_STUB_DESCRIPTION = """\
NOT AVAILABLE in this agent — it has no tools and cannot do anything. Never delegate to it. \
For research use `researcher-agent`; for planning use `planner-agent`; for report writing use \
`writer-agent`."""
GENERAL_PURPOSE_STUB_PROMPT = """\
You have no tools and no role in this agent. Immediately reply with exactly: \
'general-purpose is not available; use researcher-agent, planner-agent, or writer-agent instead.'"""


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


def build_autonomous_subagents(
    context: DeepResearchGraphContext,
    *,
    researcher_prompt: str,
    shallow_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the subagent specs: shallow, researcher, planner, writer, and the inert stub.

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

    Args:
        context: Shared graph-build inputs.
        researcher_prompt: The rendered researcher system prompt.
        shallow_spec: The ``shallow-researcher`` ``CompiledSubAgent`` spec, or ``None`` when the
            sub-agent is disabled for this request. It is placed FIRST in the returned list so it
            renders first in the ``task`` tool description and in the orchestrator's "Available
            subagent types" block — free reinforcement of the "call it first, or not at all"
            contract its description states.
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
    if shallow_spec is not None:
        subagents.insert(0, shallow_spec)
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


def _shallow_subagent_tools(
    tools: Sequence[BaseTool],
    allowed: Sequence[str] | None,
    excluded: Sequence[str] | None,
) -> list[BaseTool]:
    """Narrow ``tools`` to the set the shallow sub-run may use.

    Applied to the request's already-resolved tools, so it composes with the ``data_sources``
    filter rather than fighting it: a tool the request never selected cannot be re-added here.

    Falls back to the unnarrowed list when the result would be empty. A shallow agent with no
    tools does not fail loudly - it answers the question from memory, which is the one outcome
    this whole path exists to prevent - so a config that over-narrows for a given request is
    better served by a warning and full retrieval than by a confidently uncited answer. Names
    are validated against the agent's tool set at startup (``register.py``), so reaching this
    branch means the request's ``data_sources`` excluded them, not that they were misspelt.
    """
    allow, deny = set(allowed or []), set(excluded or [])
    if not allow and not deny:
        return list(tools)

    selected = [
        tool
        for tool in tools
        if (not allow or getattr(tool, "name", "") in allow) and getattr(tool, "name", "") not in deny
    ]
    if not selected:
        logger.warning(
            "Shallow sub-agent tool narrowing removed every tool for this request; "
            "falling back to the full tool set (allowed=%s excluded=%s available=%s)",
            sorted(allow),
            sorted(deny),
            sorted(getattr(t, "name", "") for t in tools),
        )
        return list(tools)

    if len(selected) != len(tools):
        logger.info(
            "Shallow sub-agent restricted to %d of %d tool(s): %s",
            len(selected),
            len(tools),
            ", ".join(sorted(getattr(t, "name", "") for t in selected)),
        )
    return selected


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
    shallow_subagent: bool = True,
    shallow_subagent_max_llm_turns: int = DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS,
    shallow_subagent_max_tool_iterations: int = DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS,
    shallow_subagent_tools: Sequence[str] | None = None,
    shallow_subagent_exclude_tools: Sequence[str] | None = None,
) -> AutonomousResearchGraphRun:
    """Build the full DeepAgents graph for one autonomous research run.

    Args:
        final_report_tracker: The run's dual-exit tracker. Owned by the agent (one per request)
            because both the writer's ``FinalReportCommitMiddleware`` and the orchestrator's
            ``submit_final_report`` must record onto the same instance.
        shallow_subagent: Whether to offer the ``shallow-researcher`` sub-agent for this request.
        shallow_subagent_max_llm_turns: LLM-turn bound inside the shallow sub-run.
        shallow_subagent_max_tool_iterations: Tool-call bound inside the shallow sub-run.

    Returns:
        The compiled runnable plus the run-scoped shallow capture, as an
        :class:`AutonomousResearchGraphRun`.
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

    # --- Shallow-researcher sub-agent (opt-in, default on) ---------------------------------------
    # Parent-report deltas are excluded here, at the canonical mode definition, so every later
    # subagent / middleware branch inherits the safety decision: a delta rewrite must keep the
    # citation-safe planner -> research -> writer pipeline, which is the only path that carries
    # preserved parent citations through in a verifiable form. It can never be answered by a
    # single shallow run.
    shallow_mode = shallow_subagent and not context.parent_report_context_available
    shallow_capture = ShallowSubagentCapture() if shallow_mode else None
    shallow_spec = None
    if shallow_mode:
        shallow_spec = build_shallow_researcher_subagent(
            llm_provider=llm_provider,
            # The request's raw NAT tools (already filtered by data_sources upstream), NOT
            # tool_set.researcher_tools: the shallow researcher must run exactly as it does
            # standalone, where it receives the plain tool list. Optionally narrowed further by
            # shallow_subagent_tools / shallow_subagent_exclude_tools, which apply to this
            # sub-run alone - orchestrator_tools below is built from the unnarrowed `tools`.
            tools=_shallow_subagent_tools(tools, shallow_subagent_tools, shallow_subagent_exclude_tools),
            callbacks=callbacks,
            capture=shallow_capture,
            source_registry_middleware=source_registry_middleware,
            # Captured once, at build time, from the request's own messages, so the sub-agent
            # always receives the user's actual question rather than an orchestrator paraphrase.
            original_query=last_human_text(state) or "",
            description=_as_list_item(SHALLOW_SUBAGENT_DESCRIPTION),
            max_llm_turns=shallow_subagent_max_llm_turns,
            max_tool_iterations=shallow_subagent_max_tool_iterations,
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
    if shallow_mode:
        # Ends the run on a successful shallow report without a further model turn. Listed before
        # the loop guard and the finalization guard so its `before_model` jump is decided first;
        # by that point the report is already committed, so neither of those has anything to do.
        orchestrator_middleware.append(
            ShallowFinalizationMiddleware(
                capture=shallow_capture,
                backend=context.backend,
                tracker=final_report_tracker,
                source_registry_middleware=source_registry_middleware,
            )
        )
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
        subagents=build_autonomous_subagents(
            context,
            researcher_prompt=researcher_prompt,
            shallow_spec=shallow_spec,
        ),
        store=InMemoryStore(),
        middleware=context.middleware(orchestrator_middleware),
        permissions=context.permissions(ORCHESTRATOR_AGENT),
        backend=context.backend,
    )
    return AutonomousResearchGraphRun(
        runnable=agent.with_config({"recursion_limit": request_termination.recursion_limit}),
        shallow_capture=shallow_capture,
    )
