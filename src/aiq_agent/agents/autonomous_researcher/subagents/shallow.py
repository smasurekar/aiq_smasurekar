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

"""Shallow-researcher ``CompiledSubAgent`` for the autonomous orchestrator.

The shallow researcher is reused exactly as it ships (:class:`ShallowResearcherAgent`): same
graph, same prompt, same bounded tool loop, same citation post-processing. This module only
adapts its I/O to the DeepAgents sub-agent contract and captures its post-processed output so
``ShallowFinalizationMiddleware`` can end the run on it without an extra orchestrator turn.

Relationship to the adaptive copy
---------------------------------
This is a deliberate fork of ``adaptive_researcher/subagents/shallow.py`` rather than an import.
The adaptive version is tier-keyed at every seam (``declared_tier``, ``tier="single_shot"`` in the
persisted metadata, tier vocabulary in its notices), and the autonomous agent is tier-free by
construction — ``tests/aiq_agent/agents/autonomous_researcher/test_factory.py`` pins that no tier
artifact reaches the model. The three load-bearing behaviours below are carried over unchanged and
are documented at their implementation sites:

* the original user query is used as the sub-agent input, not the orchestrator's paraphrase;
* the citation registry is shared with the parent for the duration of the sub-run;
* the adapter never raises - failures become an ordinary return value with a spent attempt.

What is different here:

* no ``declared_tier`` field and no ``tier`` key in the persisted metadata;
* the attempt budget short-circuits *before* executing once it is spent. In the adaptive agent a
  routing middleware refused further delegations; the autonomous agent enforces nothing through
  middleware (descriptions do the routing), so this is the only thing standing between a
  systematically failing configuration and a repeated-failure loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from deepagents.backends.state import create_file_data
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.shallow_researcher.agent import AGENT_DIR as SHALLOW_AGENT_DIR
from aiq_agent.agents.shallow_researcher.agent import ShallowResearcherAgent
from aiq_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from aiq_agent.common import LLMProvider
from aiq_agent.common import load_prompt
from aiq_agent.common.citation_verification import get_session_registry
from aiq_agent.common.citation_verification import reset_session_registry
from aiq_agent.common.citation_verification import set_session_registry

from ..tools.finalize import FINAL_REPORT_META_PATH
from ..tools.finalize import FINAL_REPORT_PATH

logger = logging.getLogger(__name__)

SHALLOW_RESEARCHER_SUBAGENT = "shallow-researcher"

# Hard cap on shallow research attempts per request. Every attempt is a full shallow run, and the
# failures this guards against (an unusable source, a missing API key, a zero-result retrieval
# config) are systematic rather than transient - retrying them buys nothing and costs the workflow
# budget. Once the cap is spent, `_execute_shallow_once` stops executing and only returns the
# notice, so the orchestrator falls back to ordinary research instead of burning its turn budget
# on a path that cannot succeed.
MAX_SHALLOW_ATTEMPTS = 2

# Answer-set discipline for the shallow sub-run, appended to the shallow agent's own template.
#
# Why it lives here and not in `shallow_researcher/prompts/researcher.j2`: that template is shared
# by every shipped config that offers a standalone shallow agent, and this contract is tuned for
# the autonomous researcher's grading surface. Appending it per sub-run keeps the shared template
# byte-identical for those configs (`test_default_model_profiles` pins that invariant).
#
# Why it exists at all: in DSQA-90 job 2026-08-20__21-44-00 the shallow exit produced 40.6% of its
# answers with excessive items (mean 1.56) against 33.3% for the orchestrator's inline exit - the
# gap being that the inline path carries an answer-set rule in orchestrator.j2 and this path
# carried none. Trial 0256 is the clearest case: the grader counted the chart, the "Key Takeaways"
# summary and the references section themselves as excessive answers.
#
# Must stay Jinja-inert. `render_prompt_template` uses StrictUndefined and the shallow agent's
# render site passes only tools/user_info/current_datetime/available_documents, so any `{{ }}`,
# `{% %}` or `{#` here would raise at agent-node time.
SHALLOW_ANSWER_CONTRACT = """## Answering discipline

Does the question name a discrete target? "Which X", "list all X", "how many X" and "identify X" \
name one; "write a report on X" and "assess X" do not. When unsure, treat it as NOT discrete and \
write the fuller answer.

When the question names a discrete target, open with an `## Answer` section holding exactly the \
entities that pass every filter the question states, and nothing else. Rejected candidates, close \
alternatives, near-misses and "commonly confused with" entries never appear there, not even \
flagged as excluded - anything named in that section is read as one of your answers. Put those \
under a `### Considered and excluded` heading in the body instead, each with the reason it fails.

Charts, tables, key-takeaway summaries and the references section belong below the `## Answer` \
section, never inside it.

Before answering, for each entity in the `## Answer` section, name the filter it satisfies. If you \
cannot, remove it.

When the question does not name a discrete target, answer at whatever length it warrants. There is \
no length target here."""


def _shallow_system_prompt() -> str | None:
    """Return the shallow template with the answer contract appended, or ``None`` on failure.

    Returning ``None`` hands construction back to ``ShallowResearcherAgent._load_system_prompt``,
    which has its own inline fallback - so a missing or unreadable template degrades to today's
    behaviour instead of failing the whole request over a prompt suffix.
    """
    try:
        base = load_prompt(SHALLOW_AGENT_DIR / "prompts", "researcher")
    except Exception:  # noqa: BLE001 - any load failure should degrade, never break the run
        logger.warning("Shallow researcher template unavailable; sub-run falls back to its own default prompt")
        return None
    return f"{base}\n\n{SHALLOW_ANSWER_CONTRACT}"


def _failure_notice(capture: ShallowSubagentCapture) -> str:
    """Render the orchestrator-facing notice for a failed shallow attempt.

    Deliberately category-level: it names the exception *type* and the remaining budget but never
    the exception message, which can carry retrieved source content or credential fragments.

    The "what to do next" half of the notice is the escalation instruction. It is the *only*
    escalation path in this design: on success the run ends inside the runtime and the
    orchestrator never gets another turn, so a failure notice is the sole signal that ordinary
    research is still needed.
    """
    remaining = max(0, MAX_SHALLOW_ATTEMPTS - capture.attempts)
    if remaining:
        next_step = (
            f"You may retry the shallow-researcher {remaining} more time(s). If it fails again, "
            "research the request yourself with run_research_batch."
        )
    else:
        next_step = (
            "No further shallow-researcher attempts are available. Research the request yourself "
            "with run_research_batch and finish through submit_final_report."
        )
    return f"The shallow researcher did not complete this request ({capture.error_type}). {next_step}"


def last_human_text(state: Any) -> str | None:
    """Return the text of the last human message in ``state``, or ``None``.

    Accepts both the dict state DeepAgents hands to a sub-agent runnable and the Pydantic state
    the autonomous agent holds, so the same helper serves the sub-agent adapter and the factory.
    """
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if not messages:
        return None
    for message in reversed(list(messages)):
        # Match on type name rather than isinstance so a HumanMessage subclass (or a message
        # rebuilt by upstream middleware) is still recognised.
        if getattr(message, "type", None) != "human":
            continue
        content = getattr(message, "content", None)
        text = content if isinstance(content, str) else str(content) if content else ""
        if text.strip():
            return text
    return None


@dataclass
class ShallowSubagentCapture:
    """Per-request result, attempt budget, and at-most-once coordination for the shallow run.

    Created once per autonomous run (in ``build_autonomous_research_graph``); separate requests
    never share one. Calls *within* one request may still execute concurrently - ToolNode
    dispatches a turn's tool calls together - so :meth:`run_once` coalesces duplicates onto a
    single shallow execution.

    Coordination state is reached only through this class's own methods; callers never touch the
    lock or the task handle directly.
    """

    markdown: str | None = None
    researched: bool = True
    status: Literal["not_started", "running", "completed", "failed"] = "not_started"
    # Metadata only: the exception *type* name. Never the message - it can carry source content.
    error_type: str | None = None
    attempts: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def invoked(self) -> bool:
        """Whether a shallow invocation is running or has succeeded."""
        return self.status in ("running", "completed")

    @property
    def exhausted(self) -> bool:
        """Whether the attempt budget is spent with no usable report."""
        return self.status != "completed" and self.attempts >= MAX_SHALLOW_ATTEMPTS

    @property
    def has_report(self) -> bool:
        """Whether a completed shallow report is available to finalize or recover the run."""
        return self.status == "completed" and bool(self.markdown)

    async def run_once(self, factory: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """Execute ``factory()`` once, sharing its result with concurrently dispatched callers.

        The task is created before the first await so siblings from the same turn find it. It is
        stored on the capture rather than in a closure so ``AutonomousResearcherAgent.run`` can
        cancel it: ``asyncio.create_task`` detaches the coroutine, and cancelling the awaiter does
        not stop the work.

        A finished task is retained only when the attempt *succeeded*, so duplicate calls within
        one turn share the report. Any other finished outcome clears the slot so a later,
        budget-permitting delegation is a genuine new attempt rather than a replay. Note that the
        deciding signal is ``status``, not whether the task raised: the adapter deliberately
        converts failures into an ordinary return value (see the module docstring), so a failed
        attempt completes the task normally.
        """
        async with self._lock:
            if self._task is None:
                self.status = "running"
                self._task = asyncio.create_task(factory())
            task = self._task
        try:
            return await task
        finally:
            async with self._lock:
                # An awaiter can reach this line while the task is still running (its own
                # cancellation), in which case the handle must stay so `cancel()` can reach it.
                if self._task is task and task.done() and self.status != "completed":
                    self._task = None

    def cancel(self) -> None:
        """Cancel an unfinished shallow run. Idempotent, and a no-op once the task has finished."""
        task = self._task
        if task is not None and not task.done():
            logger.info("Cancelling in-flight shallow-researcher run (request teardown)")
            task.cancel()


def build_shallow_researcher_subagent(
    *,
    llm_provider: LLMProvider,
    tools: Sequence[BaseTool],
    callbacks: list[Any],
    capture: ShallowSubagentCapture,
    source_registry_middleware: SourceRegistryMiddleware,
    original_query: str | None,
    description: str,
    max_llm_turns: int,
    max_tool_iterations: int,
) -> dict[str, Any]:
    """Build the ``shallow-researcher`` ``CompiledSubAgent`` spec for ``create_deep_agent``.

    Args:
        llm_provider: Shared role-based provider; the shallow agent resolves ``LLMRole.RESEARCHER``
            from it exactly as it does standalone.
        tools: The request's raw NAT tools (already filtered by ``data_sources``), so the shallow
            agent sees the same tool list it would receive from its own register function.
        callbacks: Parent callbacks, forwarded so the frontend sees the shallow draft as it lands.
        capture: Run-scoped capture that carries the report, attempt budget, and task handle.
        source_registry_middleware: Parent registry, bound as the session registry for the sub-run
            so citations verify against one shared registry.
        original_query: The user's question, captured at graph-build time. Authoritative input.
        description: The routing/delegation contract rendered into the ``task`` tool description
            and the orchestrator system prompt. Supplied by the factory rather than defined here
            because in this agent the description *is* the routing logic and belongs beside the
            other subagent descriptions it competes with.
        max_llm_turns: Shallow agent LLM-turn bound.
        max_tool_iterations: Shallow agent tool-call bound.

    Returns:
        A DeepAgents ``CompiledSubAgent`` spec (``name`` / ``description`` / ``runnable``).
    """
    # Built once per run, mirroring how the shallow agent is built per request in its own
    # register.py: the active tool set depends on this request's data_sources / MCP tools.
    shallow_agent = ShallowResearcherAgent(
        llm_provider=llm_provider,
        tools=list(tools),
        system_prompt=_shallow_system_prompt(),
        max_llm_turns=max_llm_turns,
        max_tool_iterations=max_tool_iterations,
        callbacks=callbacks,
    )

    async def _execute_shallow_once(state: dict[str, Any]) -> dict[str, Any]:
        # ---- 0. Refuse once the budget is spent ------------------------------------------
        # Nothing hides this sub-agent from `task` after a failure - the autonomous agent routes
        # by description and never by tool filtering - so without this check a model that keeps
        # re-delegating would spend a full shallow run per orchestrator turn on a path that has
        # already failed systematically. Returning the notice costs nothing and repeats the
        # escalation instruction.
        if capture.exhausted:
            return {"messages": [AIMessage(content=_failure_notice(capture))]}

        # ---- 1. Input query -------------------------------------------------------------
        # DeepAgents removes the parent's `messages` key and replaces it with one HumanMessage
        # containing the task description; it does not forward parent conversation history. The
        # orchestrator authors that description itself here (there is no middleware rewriting it),
        # so the build-time snapshot of the user's own words stays authoritative and the
        # description is only a fallback.
        query = original_query or last_human_text(state) or ""

        shallow_state = ShallowResearchAgentState(
            messages=[HumanMessage(content=query)],
            data_sources=state.get("data_sources"),
            user_info=state.get("user_info"),
            available_documents=state.get("available_documents"),
            collection_name=state.get("collection_name"),
        )

        # ---- 2. Share one citation registry ---------------------------------------------
        # Both agents resolve their registry as `get_session_registry() or <own registry>`. Outside
        # the chat path nothing sets a session registry, so the two would diverge: retrieval would
        # populate the shallow agent's registry while the autonomous layer verified against an
        # empty one and raised EmptySourceRegistryError on a perfectly good answer. Binding the
        # parent's instance registry for the sub-run makes both sides use one object. When a
        # session registry already exists, both already resolve to it and no binding is needed.
        token = None
        if get_session_registry() is None:
            token = set_session_registry(source_registry_middleware.registry)
        try:
            result = await shallow_agent.run(shallow_state)
            markdown = str(result.messages[-1].content).strip()
            if not markdown:
                raise ValueError("shallow researcher returned an empty final report")
        except asyncio.CancelledError:
            # Never swallow cancellation - the request is being torn down (workflow deadline or
            # client disconnect). Leave `status="running"` so no partial capture is recoverable.
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately terminal; see the module docstring
            # Deliberately NOT raised. The orchestrator stack runs ToolRetryMiddleware with the
            # default `retry_on=(Exception,)`, and ToolNode converts survivors into error
            # ToolMessages - so raising here would cost four full shallow runs and still not reach
            # AutonomousResearcherAgent.run. Record the failure, spend one attempt, and hand the
            # orchestrator a notice it can act on. This return value is the escalation trigger.
            capture.status = "failed"
            capture.error_type = type(exc).__name__
            capture.attempts += 1
            logger.warning(
                "shallow researcher failed (%s); attempt %d/%d",
                capture.error_type,
                capture.attempts,
                MAX_SHALLOW_ATTEMPTS,
            )
            return {"messages": [AIMessage(content=_failure_notice(capture))]}
        finally:
            if token is not None:
                reset_session_registry(token)

        capture.markdown = markdown
        capture.researched = True
        capture.error_type = None
        capture.attempts += 1
        capture.status = "completed"
        logger.info(
            "shallow researcher completed | attempt=%d length=%d characters",
            capture.attempts,
            len(markdown),
        )

        # ---- 3. Return to the parent -----------------------------------------------------
        # `messages` is mandatory (DeepAgents reads the last AIMessage as the ToolMessage content).
        # `files` is not excluded from sub-agent state updates, so this write merges into the
        # parent's files channel - which is what makes /shared/final_report.md the run's answer
        # for AutonomousResearcherAgent.run(). ShallowFinalizationMiddleware additionally commits
        # the same content through the backend so the write survives paths that read files from
        # the backend rather than from graph state.
        #
        # No `tier` key: this agent has no effort tiers, and test_factory.py pins that no tier
        # vocabulary reaches the model or the persisted metadata.
        meta = json.dumps({"researched": True, "source": SHALLOW_RESEARCHER_SUBAGENT})
        return {
            "messages": [AIMessage(content=markdown)],
            "files": {
                FINAL_REPORT_PATH: create_file_data(markdown),
                FINAL_REPORT_META_PATH: create_file_data(meta),
            },
        }

    async def _run_shallow(state: dict[str, Any]) -> dict[str, Any]:
        """Entry point for the ``task`` tool: at-most-once execution per attempt."""
        return await capture.run_once(lambda: _execute_shallow_once(state))

    def _run_shallow_sync(_state: dict[str, Any]) -> dict[str, Any]:
        """Reject the synchronous sub-agent path with an actionable error.

        AI-Q always drives the graph through ``ainvoke`` (``AutonomousResearcherAgent.run``), so
        DeepAgents uses its async ``atask`` path. Raising here turns a hypothetical sync
        invocation into a clear message instead of an opaque LangChain coroutine error.
        """
        raise RuntimeError(
            "shallow-researcher subagent requires the async path; invoke the autonomous graph with ainvoke()"
        )

    return {
        "name": SHALLOW_RESEARCHER_SUBAGENT,
        "description": description,
        "runnable": RunnableLambda(_run_shallow_sync, afunc=_run_shallow),
    }
