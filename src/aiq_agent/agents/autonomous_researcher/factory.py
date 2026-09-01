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

The one exception is *configuration*, not routing: ``research_batch_tool`` and
``researcher_subagent`` each remove one of the two delegated-research doors for an eval A/B arm.
That is a build-time decision - the door is never advertised and never appears - and every string
that names a door is gated on the same flags, so the model is not told about a path it does not
hold. They cannot both be false; ``AutonomousResearchAgentConfig`` and
``build_autonomous_research_graph`` both reject that.

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
from aiq_agent.agents.deep_researcher.agent import DEFAULT_MAX_RESEARCHER_MODEL_CALLS
from aiq_agent.agents.deep_researcher.custom_middleware import EmptyContentFixMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ExecuteTimeoutClampMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import FilesystemToolCallGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import FinalReportOwnershipGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import StateMutationGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import StructuredOutputRetryGuardMiddleware
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
# ARE the routing logic, and they are also the complete delegation contract: the orchestrator
# prompt deliberately carries no per-subagent triggers, ordering rules, or `task()` briefs.
#
# Every description answers the same six questions, in this fixed order, in hyphen bullets:
#
#   WHEN TO CHOOSE IT      — the request properties that select this agent
#   WHEN NOT TO CHOOSE IT  — the request properties that hand it to a different agent
#   SEQUENCING             — where it sits in the run, stated from its own perspective
#   WHAT IT PRODUCES       — the artifact and where the runtime persists it
#   IF IT FAILS            — what comes back and what to do next
#   DELEGATION BRIEF       — what to paste, since a subagent cannot see the conversation
#
# Two rules keep the set coherent:
#
# 1. WHEN TO CHOOSE IT and WHEN NOT TO CHOOSE IT are MUTUALLY EXCLUSIVE across the four agents. A
#    property that selects one agent must appear as an exclusion in the others, worded the same
#    way. If you add a trigger here, add the matching exclusion everywhere else.
# 2. Triggers are stated as PROPERTIES of the request, not as request categories: this agent
#    exists to drop the tier ladder, and an enumerated set of request kinds would reintroduce it
#    under another name.
#
# The default route. `shallow-researcher` is the cheap turn-one path and it is deliberately given
# a wide mouth (see the evidence note above _SHALLOW_SUBAGENT_DESCRIPTION). Anything it declines
# falls through to the deep-researcher shape this repo already ships: `planner-agent` fixes the
# strategy, research fans out against the plan, and `writer-agent` publishes. That fall-through is
# a preference, not a gate — a request that is genuinely two independent lookups still goes
# straight to research — but it is what the descriptions steer toward when the choice is unclear.
#
# COST NOTE: because deepagents renders descriptions twice (tool schema + system prompt), text
# here is paid for twice per orchestrator turn, whereas the prompt sections it replaced were paid
# for once. Keep these strings tight and never restate anything a tool description already
# carries. See misc/autonomous_researcher/autonomous-researcher-review-feedback-analysis.md §1.5.
#
# The planner and writer briefs are request-conditional (`parent_report_context_available`,
# `execution_enabled`). Their bodies are module-level templates; the conditional fragments are
# separate constants that `build_planner_subagent_description` and
# `build_writer_subagent_description` splice into the `{...}` placeholders, so every string the
# model sees is readable in one piece.

# ------------------------------------------------------------------------------------------------
# Prompt-text convention for every string below
# ------------------------------------------------------------------------------------------------
# These are triple-quoted so they read as prose in the source, but the text the model receives is
# NOT wrapped: each bullet is one long logical line, and the only real newlines are the blank lines
# between sections and the line breaks between bullets. A trailing ``\`` is a source-only soft wrap
# — Python drops it together with the newline — so keep the space that separates the two words
# BEFORE the backslash and start the continuation at column 0. Re-wrapping the source this way is
# free; turning a bullet into several real lines changes what the model reads.

# The cheapest rung of the ladder, and the only subagent whose output is the run's answer rather
# than an input to a later step. Every constraint stated below is carried by this text alone:
# nothing hides this subagent after the first turn and nothing forces the model to reach for it.
#
# The one thing the runtime *does* enforce is the exit: ShallowFinalizationMiddleware commits a
# successful report and ends the run before the model is called again. So "there is nothing to do
# afterwards" is a fact rather than an instruction.
#
# WHEN TO CHOOSE IT keys on the REQUEST, never on how short the answer looks: keying on the answer
# ("a paragraph or two") is true of essentially every eval question and made this the unconditional
# turn-one winner — 55 of 90 trials in job 2026-08-20__09-11-27. It is a single sequential loop
# capped at `shallow_subagent_max_tool_iterations`, and past ~12 searches its F1 fell to 0.32
# against 0.625 below that: the capped runs published correct partial work ("no data for the
# remaining ten") as the final answer, because this exit is unconditional. Hence WHEN NOT TO
# CHOOSE IT exists at all — nothing else can decline this route once it is taken.
#
# The one property that genuinely splits the two paths is the SHAPE OF THE ANSWER SET. Pooled
# across all 14 Fetch-disabled dsqa90 jobs (1,259 trials, 90 questions, question + job fixed
# effects):
#
#     answer shape                     shallow   deep    shallow - deep
#     enumerate all qualifying members   0.646   0.605        +0.053
#     select one winner by comparison    0.325   0.515        -0.169
#
# interaction +0.222, permutation p=0.0037, replicated across both halves of the job history.
# The cause is scoring geometry, not capability. A select-one-winner request is all-or-nothing:
# 98% of those trials score exactly 0 or exactly 1, because naming a winner requires a value for
# EVERY candidate on one definition, and a bounded run that covers most of them answers *wrongly*
# rather than incompletely. An enumeration is graded (47% land strictly between 0 and 1), so the
# same bounded coverage still earns its share — and the deep path's extra ~30 searches buy nothing
# there (0.605 against 0.646).
#
# WHEN TO CHOOSE IT is keyed on that interaction through one request property: is the candidate
# set CLOSED (nameable now, a standard roster, or returned whole by one lookup against a source the
# request names) or OPEN (buildable only by traversing a source exhaustively before any condition
# applies)? That property, not the answer's shape, is what the interaction above is really
# measuring: "select one winner" only hurt the shallow path when the candidates had to be
# discovered first, because that is the case where a bounded run cannot know it missed one.
#
# The discriminator comes from the per-query report, which grouped all 90 questions into solvability
# tiers: every one of the 48 T1/T2 questions has a closed set (roster given, standard roster, or one
# named source yields it), while the open-set questions concentrate in T3/T4. It also subsumes two
# shapes the old wording sent to the planner by mistake - a superlative that only scopes a subset
# ("of the top 5 states, which ...") and a tie-break cascade ("if more than one remains, then ..."),
# neither of which changes the population being filtered. See
# misc/autonomous_researcher/autonomous-researcher-per-query-path-analysis.md sections 5.2 and 5.5.
#
# The WIDE clause in WHEN NOT TO CHOOSE IT is the guard that makes this safe rather than a way to
# overload a capped loop: a set can be closed and still have far more members than
# `shallow_subagent_max_tool_iterations` can price one at a time, which is exactly the >~12-search
# regime measured above. It names the fan-out door and not the planner, because a known set has
# nothing left to plan.
#
# Chained lookups are routed here too, and that is a cost decision rather than an accuracy claim:
# a short chain the shallow loop can walk on its own costs one sub-run, where the deep path costs
# a plan, a delegation per link, and a synthesis pass. `researcher-agent` still owns chains, but
# only the ones that turn up INSIDE a planned run; the exclusion in each agent says so.
#
# Note what none of this claims: route explains 0.6% of F1 variance overall and 28.9% of token
# variance, so this is worth ~+0.011 F1 and ~-47% tokens against job 2026-08-27__10-57-20 — a cost
# win with an accuracy tiebreak, not an accuracy fix. See
# misc/autonomous_researcher/autonomous-researcher-f1-and-token-recommendations.md sections
# 8.3-8.4 (recommendation N2). Do not re-derive the old "narrows a set in steps" rule from a raw
# case table: the gradient that motivated it is difficulty selection, and it disappears under
# question fixed effects.
_SHALLOW_SUBAGENT_DESCRIPTION = """\
Answer the whole request in one bounded run and hand back the finished, cited answer.

WHEN TO CHOOSE IT:
- The request asks for one standing fact — a value, a date, a name, a definition.
- The set of candidates is CLOSED: you can name every member now — the request lists them, or \
they are a standard published set — or ONE lookup against a source the request names returns the \
whole set. Once the set is closed it does not matter what the answer looks like: one winner, a \
filtered subset, or a cascade of tie-breakers are all one pass over the same members.
- Narrowing is only a STAGE when the SET ITSELF changes — when answering one part tells you which \
DIFFERENT population to go and find next. Applying more conditions to the same closed set is not a \
stage, however many conditions stack and however the request numbers them.
- The request asks for EVERY member that meets stated conditions. Extra conditions do not \
disqualify it: cutting a group down by three thresholds is still one list, and each member you \
confirm counts.
- The request is a short chain of lookups — you need one fact before you can search for the next \
— and roughly a dozen searches will walk it. This is the cheapest way to answer a chain, so \
prefer it here.
- Judge the REQUEST, not the topic. A specialist subject can still be one lookup.
- If you are torn between this and planning the request out, and the set of candidates is \
closed, choose this.

WHEN NOT TO CHOOSE IT:
- The set of candidates is OPEN: no list is given, none is standard, and no single named source \
returns it — you would have to build the set yourself by working through a source exhaustively \
before any condition applies. You can never be sure you have every member, so a partial set gives \
a WRONG answer rather than a short one, and your answer ends the run. Send these to planner-agent \
or {staged_route}.
- The request narrows in STAGES in the sense above: a later part needs a DIFFERENT population that \
only an earlier answer identifies. That is planner-agent's trigger.
- The set is closed but WIDE: checking it means a separate lookup per member and there are more \
members than roughly a dozen searches can cover. Send it to {staged_route} — not planner-agent, \
because the set is already known and there is nothing to plan.
- The user fixed the shape of the deliverable — named sections, a comparison matrix, a briefing, \
a report. That is planner-agent's trigger.
- The request carries three or more separate deliverables, or a parent report is mounted for it. \
Both are planner-agent's triggers.
- Anything has already happened this run: you have searched, batched, planned, or delegated. \
Re-delegating here then only wastes a turn.

SEQUENCING:
- It runs FIRST, or not at all, and exactly ONCE.
- Send it as the turn's only tool call; do not pair it with anything else.
- After it returns an answer there is nothing left for you to do.

WHAT IT PRODUCES:
- The complete answer for the user, already written and already cited — not notes, not a draft, \
not evidence for you to synthesize.
- The runtime records it as the run's final report the moment it returns, and the run ends there.
- Do not review, verify, reformat, summarize, or comment on it, and do NOT call \
submit_final_report afterwards.

IF IT FAILS:
- It replies with a short notice saying it could not complete the request. That notice is the one \
and only case where research continues after this agent.
- Treat the notice as your signal to research the request yourself from scratch with \
{escalation_route}, then finish normally.
- A returned answer is never a failure notice. Do not escalate on one.

DELEGATION BRIEF:
- Pass the user's request verbatim as `description` and add nothing else. The runtime hands it the \
original request, so it needs no extra context from you."""


def build_shallow_subagent_description(*, research_batch_enabled: bool) -> str:
    """Build ``shallow-researcher``'s description for the configured research doors.

    Three clauses name a door and must follow it. WHEN NOT TO CHOOSE IT points open-set requests
    at a fan-out path and closed-but-wide ones at the same door, and IF IT FAILS names the
    escalation route - and that last one is the only escalation path in the whole design, because
    on success the runtime ends the run and the orchestrator never gets another turn.

    Args:
        research_batch_enabled: Whether ``run_research_batch`` is offered on the orchestrator.

    Returns:
        The rendered description.
    """
    route = "run_research_batch" if research_batch_enabled else "researcher-agent"
    return _SHALLOW_SUBAGENT_DESCRIPTION.format(
        staged_route=route,
        escalation_route=("run_research_batch" if research_batch_enabled else "researcher-agent delegations"),
    )


# The research worker of the deep path. Its exclusions mirror shallow-researcher's triggers: a
# chain that IS the whole request belongs to the cheap turn-one path, and a chain that turns up
# inside a planned run belongs here. Nothing else in the set claims either.
_RESEARCHER_SUBAGENT_DESCRIPTION = """\
Investigate ONE topic in an isolated context and hand back structured, cited findings.

WHEN TO CHOOSE IT:
- You are researching a request that is already under way — planned, or broken into unknowns you \
have written down — and one of those unknowns needs to be worked out properly.
- The unknown is a PREREQUISITE CHAIN inside that larger run: you must resolve one fact before you \
can even write the next query. Parallel workers cannot pass results to each other, so a chain \
needs one worker holding all of it.
- A lookup has already failed twice and you want a fresh, isolated attempt. Give it the whole \
chain plus what you already tried, so it does not repeat you.
- Prefer delegating over searching yourself, even for a single fact: a worker's search trail is \
digested into notes before it reaches you instead of piling up in your context.

WHEN NOT TO CHOOSE IT:
- Nothing has happened yet this run and the whole request is one lookup, one short chain, or one \
pass over a CLOSED set of candidates. That is shallow-researcher's trigger, and it is available on \
turn one only.
- You have several INDEPENDENT questions. {independent_route}.
- Two delegations would aim at the same unresolved fact, or you cannot write a delegation's text \
until another one has answered. Wait instead.
- You have more than one topic to give it. Send one topic per delegation, each stated with full \
standalone context.

SEQUENCING:
- It runs after planning and before writing.
- {sequencing_rule}

WHAT IT PRODUCES:
- Structured `ResearchNotes` for the one topic, carrying its own `evidence_judgment`.
- {evidence_clause}

IF IT FAILS:
- It returns notes that say what it could not verify rather than raising. Read the \
`evidence_judgment`, record the gap, and do not re-send the same topic in new words.
- If this was already the isolated retry of a target that had failed twice, stop there: record it \
as an explicit gap and answer with what you have.

DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:
- Open with: "Research the following topic and return structured, cited notes:"
- Then the single topic in full standalone context — entities, timeframe, units, and what a \
complete answer must contain.
- Then: "Already attempted, do not repeat:" followed by the queries or targets already tried and \
how each failed.
- Close with: "Resolve the steps in order, carrying each answer into the next search. Return \
ResearchNotes with a source locator for every claim, and state explicitly anything you could not \
verify\""""


def build_researcher_subagent_description(*, research_batch_enabled: bool) -> str:
    """Build ``researcher-agent``'s description for the configured research doors.

    Only called when the subagent itself is offered, so only the batch flag varies.

    The SEQUENCING clause is the load-bearing one. With both doors open, "one delegation per
    assistant turn" is correct: the batch tool owns parallelism and this path owns chains, so
    fanning out here would duplicate the batch badly. With the batch gone this subagent is the
    ONLY research path, and that same sentence would force the arm strictly serial - which would
    make an A/B between the two doors measure serialization instead of architecture. So the
    batch-off wording keeps the chain sequential while letting independent topics go out together
    in one turn, which the request-wide guard already supports (it increments before awaiting, so
    parallel calls in one turn share the ceiling).

    Args:
        research_batch_enabled: Whether ``run_research_batch`` is offered on the orchestrator.

    Returns:
        The rendered description.
    """
    if research_batch_enabled:
        return _RESEARCHER_SUBAGENT_DESCRIPTION.format(
            independent_route="Fan them out with run_research_batch instead, which owns parallel research",
            sequencing_rule=(
                "A prerequisite chain is a dependent step, so send one delegation per assistant "
                "turn and wait for its result before the next step."
            ),
            evidence_clause=(
                "It is also the worker behind every `run_research_batch` query. The runtime "
                "persists its notes under /shared/ and registers their source locators exactly as "
                "the batch path does, so both paths leave the same evidence for `writer-agent` "
                "and for `get_verified_sources`."
            ),
        )
    return _RESEARCHER_SUBAGENT_DESCRIPTION.format(
        independent_route="Send one delegation per question in the SAME turn",
        sequencing_rule=(
            "A prerequisite chain is a dependent step, so wait for each step's result before "
            "sending the next step of that same chain; independent topics may go out together in "
            "one turn."
        ),
        evidence_clause=(
            "The runtime persists its notes under /shared/ and registers their source locators, "
            "so its evidence reaches `writer-agent` and `get_verified_sources` unchanged."
        ),
    )


# Spliced into ``{delta_line}`` when a parent report is mounted. That flag is also one of the four
# routing triggers, so it appears in WHEN TO CHOOSE IT as well.
_PLANNER_DELTA_BRIEF_LINE = """
- Add: "This is a parent-report revision. Plan only the delta research needed to revise \
/shared/original_report.md; do not plan a fresh report unless the user asked for one\""""

# The entry point of the deep path, and the one that decides whether the run ends through
# `writer-agent` or through an inline `submit_final_report`. Trigger (3) is the mirror image of
# shallow-researcher's exclusion and must stay worded the same way; triggers (1), (2) and (4) are
# the report-shaped ones and are what `writer-agent` reads its output contract from.
_PLANNER_SUBAGENT_DESCRIPTION = """\
Turn a compound request into an explicit answer strategy plus a set of ResearchQuery objects, \
persisted to /shared/plan.json.

WHEN TO CHOOSE IT — any ONE of these is enough:
- The request contains three or more distinct deliverables.
- The answer's structure has to be fixed before research: a sectioned report, a comparison matrix, \
a briefing. This also means you intend to publish through writer-agent, which reads its output \
contract from the plan.
- The set of candidates is OPEN: the request names no list, there is no standard one, and no \
single named source returns it, so the set has to be built by exhaustive traversal before any \
condition applies. Or the request narrows in STAGES: a later part needs a DIFFERENT population \
that only an earlier answer identifies. Both fail quietly in a bounded run: a winner chosen from a \
partial set is wrong rather than incomplete, and a wrong first stage spoils every stage after it.
- A parent report is mounted for this request.

WHEN NOT TO CHOOSE IT:
- The request asks for every member meeting stacked conditions. That is an enumeration and belongs \
to shallow-researcher, not here.
- The set of candidates is CLOSED, however the answer is shaped. One winner, a filtered subset, \
and a cascade of tie-breakers over one known set all belong to shallow-researcher. A closed set \
that is too wide for one bounded pass still needs no plan: shallow-researcher's WIDE clause names \
the fan-out door for it.
- You can already write every query the request needs. {fan_out_route}; a plan buys nothing there \
and costs a full sub-agent run before any evidence arrives.
- Research has already begun. A plan written after results are in hand cannot account for what you \
found, so you either re-run work or break the output contract the writer reads.

SEQUENCING:
- It runs FIRST, or not at all. Weigh it against shallow-researcher on turn one and pick one of \
the two.
- Never re-delegate here once research has begun. If you did not plan first, finish the run \
yourself.
- For a multi-part RESEARCH request this supersedes write_todos: delegate here rather than writing \
a todo list and researching it yourself.

WHAT IT PRODUCES:
- A plan the runtime persists to /shared/plan.json. Read it before starting research.
- Its queries are what you fan out; its answer_strategy is what tells you when the set is complete.
- Which exit follows is decided by the trigger that brought you here, not by the plan existing. A \
report-shaped, three-deliverable, or parent-report request publishes through writer-agent; a \
winner-selection request is usually a line or two, so research the plan's queries and finish with \
submit_final_report yourself.

IF IT FAILS:
- It returns a thin or empty plan rather than raising. Do not re-delegate: write the queries \
yourself from the request and research them, then finish with submit_final_report.
- Without /shared/plan.json the runtime rejects writer-agent, so a failed plan means you write the \
answer yourself.

DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:
- Open with: "Create a research plan for the following user request:"
- Then the user's complete request, verbatim.{delta_line}
- Add: "Use the search tools to establish what information exists and whether it is internal or \
external, then return the plan with answer_strategy, constraints, and queries. Every query needs \
full standalone context and a depth of low, medium, or high."
- If the request must price a candidate group before naming a winner, add: "Make the steps \
explicit: a query that establishes which members qualify, then a query per member for the value \
that decides between them, plus the exact threshold or ranking rule that picks the final answer\""""


def build_planner_subagent_description(
    *,
    parent_report_context_available: bool,
    research_batch_enabled: bool = True,
) -> str:
    """Build ``planner-agent``'s description, including its request-conditional delegation brief.

    Args:
        parent_report_context_available: True when a parent report is mounted in ``/shared/`` for
            this request, which adds the delta-revision line to the brief and is itself one of the
            routing triggers.
        research_batch_enabled: Whether ``run_research_batch`` is offered. The WHEN NOT TO CHOOSE IT
            clause names the fan-out path to prefer over planning, so it has to name a door that
            exists.

    Returns:
        The rendered description.
    """
    return _PLANNER_SUBAGENT_DESCRIPTION.format(
        delta_line=_PLANNER_DELTA_BRIEF_LINE if parent_report_context_available else "",
        fan_out_route=(
            "Skip planning and fan them out with run_research_batch"
            if research_batch_enabled
            else "Skip planning and send them straight to researcher-agent, one delegation per query"
        ),
    )


# Spliced into ``{delta_line}`` when a parent report is mounted.
_WRITER_DELTA_BRIEF_LINE = """
- Add: "Also read /shared/original_report.md and /shared/source_summary.md. Preserve supported \
parent-report material where it still applies, incorporate the new delta evidence, and write a \
complete standalone revised report. Do not produce an insertion plan, patch, diff, or explanation \
of the rewrite\""""

# Spliced into ``{delta_requirement}``. In delta mode the writer stops being optional: preserved
# parent citations only stay verifiable if they travel through the writer path, so this overrides
# the length test in WHEN TO CHOOSE IT.
_WRITER_DELTA_REQUIREMENT = """
- REQUIRED when a parent report is mounted, whatever the length. This is the only path that \
carries preserved parent citations through in a verifiable form, so use it even when the requested \
change looks small."""

# Spliced into ``{chart_line}`` only when the sandbox is available. Omitting it when execution is
# off keeps the orchestrator from briefing the writer on artifacts it cannot produce.
_WRITER_CHART_BRIEF_LINE = """
- Add: "Embed each earned chart exactly once with ![<caption>](artifact://<filename>); never paste \
sandbox paths or base64 data. Only generate or embed a chart when its data is source-anchored and \
reasonably complete; otherwise present the table with explicit gaps and state the limitation.\""""

# The exit of the deep path. Its WHEN TO CHOOSE IT is keyed on the plan's deliverable rather than
# on the plan merely existing, because planner trigger (3) legitimately ends in a one-line inline
# answer; the two exits must not both claim the same run.
_WRITER_SUBAGENT_DESCRIPTION = """\
Synthesize a long-form cited report from /shared/plan.json and the research notes under /shared/, \
writing the result to /shared/output.md.

WHEN TO CHOOSE IT:
- The deliverable has named sections the user asked for and is long enough that composing it \
inline would degrade it — a report, a briefing, a full comparison write-up.
- The run went through planner-agent for one of those report-shaped reasons, so a plan already \
states the output contract this agent reads.{delta_requirement}

WHEN NOT TO CHOOSE IT:
- You can write the answer well yourself. Do that and call submit_final_report instead.
- The answer is a fact, a list, or one winner. Those finish inline even when a plan exists.
- /shared/plan.json does not exist. The runtime rejects the call, so planner-agent is a hard \
prerequisite.

SEQUENCING:
- It runs LAST — it is the run's final action.
- After it returns, report its short completion marker as your entire reply: do NOT call \
submit_final_report.
- Do not research, verify, delegate, rewrite, summarize, or comment on the report afterwards.

WHAT IT PRODUCES:
- The final Markdown answer at /shared/output.md.
- The short completion marker it returns to you, which is what you reply with.

IF IT FAILS:
- It returns an error rather than the completion marker. If you are unsure whether the file was \
written, confirm with `ls /shared/`.
- Then write the answer yourself from the notes under /shared/ and finish with \
submit_final_report. Do not re-delegate here.

DELEGATION BRIEF — it cannot see this conversation, so paste full standalone context:
- Open with: "Synthesize the final answer for the user request using the files already written to \
/shared/."
- Add: "Read /shared/plan.json, every research note file under /shared/, and the verified \
sources."{delta_line}
- Add: "Before writing, inspect Available Skills. If an applicable writer skill exists, read its \
SKILL.md first and treat it as the controlling synthesis protocol; do not draft the answer until \
you have read it."
- Add: "For broad reports, produce a cross-synthesized narrative with developed paragraphs. Do not \
compress the answer into a checklist of short component summaries unless the user asked for that \
format."{chart_line}
- Add: "Cite every material claim with numeric citations drawn only from the verified sources, and \
end with a compact Sources section."
- Close with: "Write the final Markdown answer to /shared/output.md. Return only the short \
completion marker `Wrote /shared/output.md` once the file is written. Do not return JSON and do \
not echo the full Markdown\""""


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
_GENERAL_PURPOSE_STUB_DESCRIPTION = """\
NOT AVAILABLE in this agent — it has no tools and cannot do anything. Never delegate to it. \
For research use {research_route}; for planning use `planner-agent`; for report writing use \
`writer-agent`."""
_GENERAL_PURPOSE_STUB_PROMPT = """\
You have no tools and no role in this agent. Immediately reply with exactly: \
'general-purpose is not available; use {research_route_plain}, planner-agent, or writer-agent instead.'"""


def build_general_purpose_stub_description(*, researcher_subagent_enabled: bool) -> str:
    """Build the inert stub's description, redirecting only to routes this agent actually holds.

    Worth gating carefully despite being three lines: subagent descriptions render TWICE per
    orchestrator turn (into the ``task`` schema and into the system prompt's "Available subagent
    types" block), so a route name that no longer exists is re-sent to the model on every turn of
    every run.

    Args:
        researcher_subagent_enabled: Whether ``researcher-agent`` is offered through ``task``.

    Returns:
        The rendered description.
    """
    return _GENERAL_PURPOSE_STUB_DESCRIPTION.format(
        research_route="`researcher-agent`" if researcher_subagent_enabled else "`run_research_batch`",
    )


def build_general_purpose_stub_prompt(*, researcher_subagent_enabled: bool) -> str:
    """Build the inert stub's system prompt, naming only routes this agent actually holds.

    Args:
        researcher_subagent_enabled: Whether ``researcher-agent`` is offered through ``task``.

    Returns:
        The rendered prompt.
    """
    return _GENERAL_PURPOSE_STUB_PROMPT.format(
        research_route_plain="researcher-agent" if researcher_subagent_enabled else "run_research_batch",
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
    research_batch_tool_name: str | None = "run_research_batch",
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
    # ``None`` when run_research_batch is disabled for this deployment. Allowlisting a tool that is
    # not bound cannot cause a misroute today (ToolNameSanitizationMiddleware does suffix-stripping
    # plus a three-entry alias table, with no fuzzy matching), but a name in the allowlist asserts
    # the tool exists, and that assertion should not outlive the tool.
    if research_batch_tool_name:
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
    research_batch_tool: bool = True,
) -> DeepResearchMiddlewareSet:
    """Build researcher, planner, writer, and orchestrator middleware stacks.

    Args:
        research_batch_tool: Whether ``run_research_batch`` is offered on the orchestrator. Only
            reaches the sanitizer allowlist; every other use of the flag lives in the graph
            builder.
    """

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
            research_batch_tool_name="run_research_batch" if research_batch_tool else None,
        ),
    )


def _researcher_subagent_spec(
    context: DeepResearchGraphContext,
    *,
    system_prompt: str,
    research_batch_enabled: bool = True,
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
        "description": _as_list_item(
            build_researcher_subagent_description(research_batch_enabled=research_batch_enabled)
        ),
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
                StructuredOutputRetryGuardMiddleware(),
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
    research_batch_tool: bool = True,
    researcher_subagent: bool = False,
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
        research_batch_tool: Whether ``run_research_batch`` is offered. Does not add or remove a
            spec here; it selects which door the planner's and researcher's descriptions name.
        researcher_subagent: Whether to offer ``researcher-agent`` as a DIRECT ``task`` route.
            Defaults False: the researcher still runs every question behind ``run_research_batch``,
            and this only decides whether a second door onto it is advertised. When False its spec
            is not built, so neither the ``task`` schema nor the "Available subagent types" block
            mentions it, and the inert stub stops redirecting research there.

    Returns:
        The subagent specs, in render order.
    """
    planner_description = _as_list_item(
        build_planner_subagent_description(
            parent_report_context_available=context.parent_report_context_available,
            research_batch_enabled=research_batch_tool,
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

    if researcher_subagent:
        subagents.insert(
            0,
            _researcher_subagent_spec(
                context,
                system_prompt=researcher_prompt,
                research_batch_enabled=research_batch_tool,
            ),
        )
    if shallow_spec is not None:
        subagents.insert(0, shallow_spec)
    subagents.append(
        {
            "name": GENERAL_PURPOSE_SUBAGENT_NAME,
            "description": build_general_purpose_stub_description(
                researcher_subagent_enabled=researcher_subagent,
            ),
            "system_prompt": build_general_purpose_stub_prompt(
                researcher_subagent_enabled=researcher_subagent,
            ),
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
    research_batch_tool: bool = True,
    researcher_subagent: bool = False,
    shallow_subagent: bool = True,
    shallow_subagent_max_llm_turns: int = DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS,
    shallow_subagent_max_tool_iterations: int = DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS,
    shallow_subagent_escalate_on_budget_exhaustion: bool = True,
    shallow_subagent_tools: Sequence[str] | None = None,
    shallow_subagent_exclude_tools: Sequence[str] | None = None,
) -> AutonomousResearchGraphRun:
    """Build the full DeepAgents graph for one autonomous research run.

    Args:
        final_report_tracker: The run's dual-exit tracker. Owned by the agent (one per request)
            because both the writer's ``FinalReportCommitMiddleware`` and the orchestrator's
            ``submit_final_report`` must record onto the same instance.
        research_batch_tool: Whether to offer ``run_research_batch`` on the orchestrator.
        researcher_subagent: Whether to offer ``researcher-agent`` as a direct ``task`` route, in
            addition to its always-on role as the ``run_research_batch`` worker. Defaults False.
            At least one of these two must be True.
        shallow_subagent: Whether to offer the ``shallow-researcher`` sub-agent for this request.
        shallow_subagent_max_llm_turns: LLM-turn bound inside the shallow sub-run.
        shallow_subagent_max_tool_iterations: Tool-call bound inside the shallow sub-run.
        shallow_subagent_escalate_on_budget_exhaustion: Whether an exhausted shallow tool-call budget
            fails and escalates instead of synthesizing a partial answer that ends the run.

    Returns:
        The compiled runnable plus the run-scoped shallow capture, as an
        :class:`AutonomousResearchGraphRun`.

    Raises:
        ValueError: If both delegated-research doors are disabled.
    """
    # Duplicated from AutonomousResearchAgentConfig's validator on purpose: this factory is called
    # directly by the test suite and by any future caller that does not go through the NAT config
    # layer, and a graph with neither door would fail silently at answer time rather than here.
    if not (research_batch_tool or researcher_subagent):
        raise ValueError(
            "build_autonomous_research_graph requires at least one delegated research path: "
            "research_batch_tool or researcher_subagent."
        )
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
        # Upstream added ``max_researcher_model_calls`` as a required field: it sizes the per-worker
        # model-call budget that ``ResearcherFinalizationMiddleware`` reserves one turn out of.
        # Neither agent exposes a knob for it, so both take the deep researcher's default.
        max_researcher_model_calls=DEFAULT_MAX_RESEARCHER_MODEL_CALLS,
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
        researcher_max_consecutive_blocked_source_calls=researcher_loop_guard.max_consecutive_blocked_source_calls,
    )
    # `researcher_runnable` exists ONLY to be the worker behind run_research_batch: the
    # task-reachable researcher-agent spec builds its own agent from `context`. So when the batch
    # door is closed both are skipped, rather than building a runnable nothing will ever invoke.
    # The researcher prompt above is still rendered unconditionally - the spec needs it, and the
    # both-off guard means at least one consumer always exists.
    research_batch_tool_obj = None
    if research_batch_tool:
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
            max_researcher_model_calls=context.max_researcher_model_calls,
            skill_sources=context.skill_sources(RESEARCHER_AGENT),
            backend=context.backend,
            visibility_middleware=context.visibility_middleware,
            filesystem_permissions=context.permissions(RESEARCHER_AGENT),
        )
        research_batch_tool_obj = build_autonomous_research_batch_tool(
            researcher_runnable=researcher_runnable,
            backend=context.backend,
            callbacks=callbacks,
            max_research_concurrency=max_research_concurrency,
            resource_limits=context.resource_limits,
            state_budget=context.state_budget,
            source_registry_middleware=source_registry_middleware,
            researcher_subagent_enabled=researcher_subagent,
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
            description=_as_list_item(build_shallow_subagent_description(research_batch_enabled=research_batch_tool)),
            # The failure notice is the ONLY way research continues after a shallow attempt, so it
            # has to name a door this deployment actually holds.
            escalation_route=(
                "run_research_batch" if research_batch_tool else 'task(subagent_type="researcher-agent", ...)'
            ),
            max_llm_turns=shallow_subagent_max_llm_turns,
            max_tool_iterations=shallow_subagent_max_tool_iterations,
            escalate_on_budget_exhaustion=shallow_subagent_escalate_on_budget_exhaustion,
        )

    # The full menu the configured doors allow. Source tools sit alongside run_research_batch and
    # task, and the model decides how to research from the descriptions alone. Nothing is hidden at
    # runtime; a missing entry here means the door was never built for this deployment.
    research_source_tools = list(context.tool_set.research_source_tools)
    orchestrator_tools = [
        *context.tool_set.helper_tools,
        *([research_batch_tool_obj] if research_batch_tool_obj is not None else []),
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
        # The prompt names both delegated-research doors by hand, so it has to know which exist.
        # render_prompt uses StrictUndefined: a `{% if %}` on either name that is not supplied here
        # raises at agent-node time, long after startup and every build-time test.
        research_batch_enabled=research_batch_tool,
        researcher_subagent_enabled=researcher_subagent,
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
        # Wired in every arm on purpose. It is a no-op when researcher-agent is not offered (it
        # gates on subagent_type), and when run_research_batch is not offered it is the ONLY thing
        # persisting research notes and registering their locators.
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
                research_batch_enabled=research_batch_tool,
                researcher_subagent_enabled=researcher_subagent,
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
            research_batch_tool=research_batch_tool,
            researcher_subagent=researcher_subagent,
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
