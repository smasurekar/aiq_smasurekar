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

"""Shallow-researcher ``CompiledSubAgent`` for the adaptive orchestrator's ``single_shot`` tier.

The shallow researcher is reused exactly as it ships (:class:`ShallowResearcherAgent`): same
graph, same prompt, same bounded tool loop, same citation post-processing. This module only
adapts its I/O to the DeepAgents sub-agent contract and captures its post-processed output so
the adaptive finalize seam can make that output authoritative without orchestrator rewriting.

Three behaviours here are load-bearing and are documented at their implementation sites:

* the original user query is used as the sub-agent input, not the orchestrator's paraphrase;
* the citation registry is shared with the parent for the duration of the sub-run;
* the adapter never raises - failures become an ordinary return value with a spent attempt.
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
from aiq_agent.agents.shallow_researcher.agent import ShallowResearcherAgent
from aiq_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from aiq_agent.common import LLMProvider
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
# budget. Once the cap is spent, SingleShotShallowDelegationMiddleware stops accepting `task` and
# opens the finalize escape hatch, so the run ends through the normal seam instead of grinding to
# the orchestrator turn budget or the workflow deadline.
MAX_SHALLOW_ATTEMPTS = 2

SHALLOW_SUBAGENT_DESCRIPTION = (
    "Shallow researcher - answers one bounded, factual question with a short, citation-backed "
    "Markdown report. Available only for the single_shot effort tier; the routing middleware "
    "supplies the original user query."
)


def _failure_notice(capture: ShallowSubagentCapture) -> str:
    """Render the orchestrator-facing notice for a failed shallow attempt.

    Deliberately category-level: it names the exception *type* and the remaining budget but never
    the exception message, which can carry retrieved source content or credential fragments.
    """
    remaining = max(0, MAX_SHALLOW_ATTEMPTS - capture.attempts)
    if remaining:
        next_step = (
            f"You may retry the shallow-researcher {remaining} more time(s). If it fails again, "
            "finalize with whatever evidence has been gathered."
        )
    else:
        next_step = (
            "No further shallow-researcher attempts are available. Call submit_final_report with "
            "whatever the gathered evidence supports, and state plainly what could not be verified."
        )
    return f"The shallow researcher did not complete this request ({capture.error_type}). {next_step}"


def last_human_text(state: Any) -> str | None:
    """Return the text of the last human message in ``state``, or ``None``.

    Accepts both the dict state DeepAgents hands to a sub-agent runnable and the Pydantic state
    the adaptive agent holds, so the same helper serves the sub-agent adapter and the factory.
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

    Created once per adaptive run (in ``build_adaptive_research_graph``); separate requests never
    share one. Calls *within* one request may still execute concurrently - ToolNode dispatches a
    turn's tool calls together - so :meth:`run_once` coalesces duplicates onto a single shallow
    execution rather than relying on next-turn tool hiding, which lands one turn too late.

    Coordination state is reached only through this class's own methods; callers never touch the
    lock or the task handle directly.
    """

    markdown: str | None = None
    researched: bool = True
    # Kept current by SingleShotShallowDelegationMiddleware, including on escalation, so the
    # finalizer override and the timeout/recursion recovery can both require that the run is
    # still on the single_shot tier before reusing a captured report.
    declared_tier: str | None = None
    status: Literal["not_started", "running", "completed", "failed"] = "not_started"
    # Metadata only: the exception *type* name. Never the message - it can carry source content.
    error_type: str | None = None
    attempts: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def invoked(self) -> bool:
        """Whether a shallow invocation is running or has succeeded (drives tool hiding)."""
        return self.status in ("running", "completed")

    @property
    def exhausted(self) -> bool:
        """Whether the attempt budget is spent with no usable report (opens the escape hatch)."""
        return self.status != "completed" and self.attempts >= MAX_SHALLOW_ATTEMPTS

    @property
    def has_report(self) -> bool:
        """Whether a completed shallow report is available for the finalizer / recovery paths."""
        return self.status == "completed" and bool(self.markdown)

    async def run_once(self, factory: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """Execute ``factory()`` once, sharing its result with concurrently dispatched callers.

        The task is created before the first await so siblings from the same turn find it. It is
        stored on the capture rather than in a closure so ``AdaptiveResearcherAgent.run`` can
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
        max_llm_turns=max_llm_turns,
        max_tool_iterations=max_tool_iterations,
        callbacks=callbacks,
    )

    async def _execute_shallow_once(state: dict[str, Any]) -> dict[str, Any]:
        # ---- 1. Input query -------------------------------------------------------------
        # DeepAgents removes the parent's `messages` key and replaces it with one HumanMessage
        # containing the task description; it does not forward parent conversation history. The
        # delegation middleware also overwrites that description with the original query, but this
        # closure remains the authoritative source and the task description is only a fallback.
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
        # populate the shallow agent's registry while the adaptive layer verified against an empty
        # one and raised EmptySourceRegistryError on a perfectly good answer. Binding the parent's
        # instance registry for the sub-run makes both sides use one object. When a session
        # registry already exists, both already resolve to it and no binding is needed.
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
            # AdaptiveResearcherAgent.run. Record the failure, spend one attempt, and hand the
            # orchestrator a notice it can act on.
            capture.status = "failed"
            capture.error_type = type(exc).__name__
            capture.attempts += 1
            logger.warning(
                "single_shot shallow researcher failed (%s); attempt %d/%d",
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
            "single_shot shallow researcher completed | attempt=%d length=%d characters",
            capture.attempts,
            len(markdown),
        )

        # ---- 3. Return to the parent -----------------------------------------------------
        # `messages` is mandatory (DeepAgents reads the last AIMessage as the ToolMessage content).
        # `files` is not excluded from sub-agent state updates, so this write merges into the
        # parent's files channel - a safety net that makes /shared/final_report.md exist even if
        # the orchestrator never reaches submit_final_report. Exception recovery cannot use it
        # (an `ainvoke` that raises returns no state) and goes through `capture` instead.
        meta = json.dumps({"researched": True, "tier": "single_shot", "source": SHALLOW_RESEARCHER_SUBAGENT})
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

        AI-Q always drives the graph through ``ainvoke`` (``AdaptiveResearcherAgent.run``), so
        DeepAgents uses its async ``atask`` path. Raising here turns a hypothetical sync
        invocation into a clear message instead of an opaque LangChain coroutine error.
        """
        raise RuntimeError(
            "shallow-researcher subagent requires the async path; invoke the adaptive graph with ainvoke()"
        )

    return {
        "name": SHALLOW_RESEARCHER_SUBAGENT,
        "description": SHALLOW_SUBAGENT_DESCRIPTION,
        "runnable": RunnableLambda(_run_shallow_sync, afunc=_run_shallow),
    }
