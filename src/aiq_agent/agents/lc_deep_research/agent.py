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

"""Graph construction and output extraction for the LangChain DeepAgents deep-research example.

Ported from ``deepagents/examples/deep_research/agent.py``. The topology is upstream's, unchanged:
one ``create_deep_agent`` holding ``[tavily_search, think_tool]``, one ``research-agent`` subagent
with the same tools and ``RESEARCHER_INSTRUCTIONS``, and an orchestrator prompt built by
concatenating ``RESEARCH_WORKFLOW_INSTRUCTIONS`` and ``SUBAGENT_DELEGATION_INSTRUCTIONS`` across an
80-character rule.

Three deliberate deviations, none of which touch what the model sees:

1. **The model is injected.** Upstream hard-codes ``init_chat_model("anthropic:claude-...")``. Here
   the caller passes the NAT-resolved LangChain chat model, so the config's ``llms:`` block picks
   the model (Nemotron Ultra for the first iteration).
2. **The graph is built per call, not at import.** Upstream is a script whose module body *is* the
   construction. ``current_date`` therefore moves from process start to graph build, which matters
   for a long-running server and is otherwise equivalent.
3. **Output extraction is added.** Upstream is driven from a notebook where a human reads
   ``/final_report.md`` out of the resulting state. AI-Q has to resolve that file into a single
   string, so ``extract_final_report`` lives here.

Everything AI-Q normally layers on a research agent -- citation verification, report sanitization,
loop guards, the source registry, tier routing -- is deliberately absent. This agent exists to
measure the upstream design; post-processing that rewrites the report text would defeat that.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from aiq_agent.agents.deep_researcher.models import DeepResearchAgentState

from .research_agent.prompts import RESEARCH_WORKFLOW_INSTRUCTIONS
from .research_agent.prompts import RESEARCHER_INSTRUCTIONS
from .research_agent.prompts import SUBAGENT_DELEGATION_INSTRUCTIONS
from .research_agent.tools import tavily_search
from .research_agent.tools import think_tool

logger = logging.getLogger(__name__)

# Alias, not a subclass: this agent adds no state fields, and the shared class is what the UI, the
# workflow wrapper, and the job runner already understand. The name matters -- aiq_api's
# `_get_agent_state_class` discovers an agent's state by trying `<AgentName>State`-style names in
# the agent's own module, so `LcDeepResearchAgent` must be able to find `LcDeepResearchAgentState`
# here or the job runner falls back to an untyped dict state.
LcDeepResearchAgentState = DeepResearchAgentState

# Upstream limits, kept as the defaults. `max_concurrent_research_units` and
# `max_researcher_iterations` are not enforced by any middleware -- they are rendered into
# SUBAGENT_DELEGATION_INSTRUCTIONS and enforced by the orchestrator's own compliance, exactly as
# upstream intends.
DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS = 3
DEFAULT_MAX_RESEARCHER_ITERATIONS = 3

# LangGraph's own default is 25 steps, which a three-round, three-sub-agent research run reliably
# exceeds -- and it surfaces as a GraphRecursionError, i.e. a crashed run rather than a degraded
# one. Raising the ceiling permits the graph to finish; it changes no reasoning step and no prompt.
DEFAULT_RECURSION_LIMIT = 100

# Virtual-filesystem path the orchestrator writes the report to (RESEARCH_WORKFLOW_INSTRUCTIONS
# step 5). The final chat message is usually a short acknowledgement, so this file -- not the
# message -- is the answer.
FINAL_REPORT_PATH = "/final_report.md"

# Subagent identity. The description is LLM-facing: SubAgentMiddleware renders it into the `task`
# tool, so it is part of the prompt surface and is quoted verbatim from upstream.
RESEARCH_SUBAGENT_NAME = "research-agent"
RESEARCH_SUBAGENT_DESCRIPTION = (
    "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time."
)


def build_orchestrator_instructions(
    *,
    max_concurrent_research_units: int = DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
    max_researcher_iterations: int = DEFAULT_MAX_RESEARCHER_ITERATIONS,
) -> str:
    """Build the orchestrator system prompt exactly as upstream ``agent.py`` does.

    ``RESEARCHER_INSTRUCTIONS`` is intentionally *not* included -- upstream scopes it to the
    sub-agent only, so the orchestrator never sees the researcher's tool-call budgets.

    Args:
        max_concurrent_research_units: Parallel sub-agents allowed per delegation round.
        max_researcher_iterations: Delegation rounds allowed before the orchestrator must stop.

    Returns:
        The concatenated orchestrator system prompt.
    """
    return (
        RESEARCH_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_research_units=max_concurrent_research_units,
            max_researcher_iterations=max_researcher_iterations,
        )
    )


def build_lc_deep_research_graph(
    model: BaseChatModel,
    *,
    max_concurrent_research_units: int = DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
    max_researcher_iterations: int = DEFAULT_MAX_RESEARCHER_ITERATIONS,
    current_date: str | None = None,
) -> Any:
    """Build the upstream deep-research DeepAgent over a NAT-provided chat model.

    Args:
        model: LangChain chat model resolved by NAT from the config's ``llms:`` block.
        max_concurrent_research_units: Parallel sub-agents per delegation round.
        max_researcher_iterations: Delegation rounds before the orchestrator must stop.
        current_date: Date rendered into ``RESEARCHER_INSTRUCTIONS``; defaults to today. Injectable
            so tests can assert the prompt without depending on the clock.

    Returns:
        The compiled DeepAgents graph.
    """
    date = current_date or datetime.now().strftime("%Y-%m-%d")

    research_sub_agent = {
        "name": RESEARCH_SUBAGENT_NAME,
        "description": RESEARCH_SUBAGENT_DESCRIPTION,
        "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=date),
        "tools": [tavily_search, think_tool],
    }

    return create_deep_agent(
        model=model,
        tools=[tavily_search, think_tool],
        system_prompt=build_orchestrator_instructions(
            max_concurrent_research_units=max_concurrent_research_units,
            max_researcher_iterations=max_researcher_iterations,
        ),
        subagents=[research_sub_agent],
    )


def _read_state_file(files: Any, path: str) -> str | None:
    """Return the text at ``path`` in a DeepAgents file map, or ``None`` when it carries no content.

    DeepAgents backends put one of three shapes under ``files[path]`` depending on version and
    backend: a plain ``str``, raw ``bytes``, or a dict whose ``"content"`` key holds either. All
    three are handled here for the same reason ``autonomous_researcher.agent`` handles them --
    guessing wrong silently degrades the answer to the fallback path.
    """
    if not isinstance(files, dict):
        return None
    entry = files.get(path)
    if isinstance(entry, dict):
        entry = entry.get("content")
    if isinstance(entry, bytes):
        entry = entry.decode("utf-8", errors="replace")
    if isinstance(entry, str) and entry.strip():
        return entry.strip()
    return None


def _last_message_content(result: dict | Any) -> str | None:
    """Return the final message's text content, or ``None`` when there is none."""
    messages = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", None)
    if not messages:
        return None
    content = getattr(messages[-1], "content", None)
    if isinstance(content, list):
        # Some providers return content blocks rather than a flat string; join the text parts.
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        content = "".join(parts)
    if not isinstance(content, str):
        return None
    return content.strip() or None


def extract_final_report(result: dict | Any) -> str:
    """Resolve the run's answer: ``/final_report.md`` first, final chat message as a fallback.

    The workflow prompt tells the orchestrator to write the report to ``/final_report.md`` and then
    verify it, so on a well-behaved run the file is the answer and the closing message is a short
    acknowledgement. The fallback covers runs that answered inline instead -- a greeting, a
    capability question, or a research run that stopped early.

    Args:
        result: The state returned by the compiled DeepAgents graph.

    Returns:
        The final Markdown report.

    Raises:
        ValueError: If neither the report file nor the final message carries any text. Raising here
            produces an actionable error; letting an empty string through fails later in the eval
            harness with a far less useful message.
    """
    files = result.get("files") if isinstance(result, dict) else getattr(result, "files", None)
    report = _read_state_file(files, FINAL_REPORT_PATH)
    if report is not None:
        return report

    fallback = _last_message_content(result)
    if fallback is not None:
        logger.info(
            "LC deep research produced no %s; falling back to the final inline message (%d characters).",
            FINAL_REPORT_PATH,
            len(fallback),
        )
        return fallback

    raise ValueError(f"LC deep research produced neither {FINAL_REPORT_PATH} nor a non-empty final message")


class LcDeepResearchAgent:
    """One request's worth of the upstream deep-research DeepAgent.

    This class exists because AI-Q has *two* entry paths into a research agent and they do not
    share a mechanism:

    - The **NAT function** path (``register.py``, used by ``nat run`` / ``nat serve`` and the eval
      harnesses) resolves the agent through ``@register_function``.
    - The **async job** path (``frontends/aiq_api``, used by the UI and ``/jobs`` submissions)
      bypasses NAT function resolution entirely: ``aiq_api.jobs.runner`` looks the agent up in
      ``aiq_api.registry.AGENT_REGISTRY``, imports the *class* named there, instantiates it, and
      calls ``run(state)`` directly.

    Both paths route through this one class so they cannot drift apart. The constructor signature
    is dictated by the job runner: ``_create_agent_instance`` selects a construction pattern by
    inspecting which kwargs the constructor declares, and declaring both ``config`` and ``job_id``
    selects the generic ``(llm_provider, tools, verbose, callbacks, config, job_id)`` pattern.

    ``tools`` is accepted and ignored. The job runner resolves the config's tool refs and passes
    them to every agent it builds, but upstream's search path is its own ``tavily_search``, which
    calls Tavily directly. Accepting-and-ignoring keeps this class compatible with the runner
    without pretending the tools are wired.
    """

    def __init__(
        self,
        llm_provider: Any = None,
        tools: Any = None,
        *,
        verbose: bool = True,
        callbacks: list[Any] | None = None,
        config: Any = None,
        job_id: str | None = None,
    ) -> None:
        """Build one agent instance.

        Args:
            llm_provider: ``aiq_agent.common.LLMProvider``; its default LLM drives both the
                orchestrator and the research sub-agent, as upstream passes one model to
                ``create_deep_agent``.
            tools: Accepted for job-runner compatibility and ignored -- see the class docstring.
            verbose: Retained for symmetry with the sibling agents; callbacks are passed in
                already resolved.
            callbacks: LangChain callbacks to attach to the graph invocation.
            config: The ``LcDeepResearchAgentConfig`` for this run. Optional so tests and the
                NAT path can construct with defaults.
            job_id: Async job identifier, kept for logging parity. This agent has no sandbox to
                scope, so nothing else consumes it.
        """
        from aiq_agent.common import LLMRole

        self.llm_provider = llm_provider
        self.tools = list(tools) if tools else []
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.config = config
        self.job_id = job_id

        self.model = llm_provider.get(LLMRole.ORCHESTRATOR) if llm_provider is not None else None

        self.max_concurrent_research_units = getattr(
            config, "max_concurrent_research_units", DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
        )
        self.max_researcher_iterations = getattr(config, "max_researcher_iterations", DEFAULT_MAX_RESEARCHER_ITERATIONS)
        self.recursion_limit = getattr(config, "recursion_limit", DEFAULT_RECURSION_LIMIT)
        self.workflow_timeout_seconds = getattr(config, "workflow_timeout_seconds", None)

        if self.tools:
            # Loud but non-fatal: the job runner always hands over the config's resolved tools, and
            # a reader comparing this arm to the sibling arms will otherwise assume they are wired.
            logger.info(
                "LC deep research ignores the %d NAT tool(s) passed by the caller; its search tool "
                "calls Tavily directly.",
                len(self.tools),
            )

    async def run(self, state: LcDeepResearchAgentState) -> LcDeepResearchAgentState:
        """Run one deep-research request and return the state carrying the final report."""
        if getattr(state, "data_sources", None):
            # The upstream agent's only retrieval path is its own Tavily-backed tool, so a UI
            # source toggle cannot affect it. Say so rather than appearing to honour the request.
            logger.info(
                "LC deep research ignores data_sources (%s); its search tool calls Tavily directly.",
                ", ".join(state.data_sources),
            )

        # Built per request so `current_date` in the researcher prompt stays accurate on a
        # long-running server, matching how the adaptive and autonomous agents build their graphs.
        graph = build_lc_deep_research_graph(
            self.model,
            max_concurrent_research_units=self.max_concurrent_research_units,
            max_researcher_iterations=self.max_researcher_iterations,
        )

        if state.messages:
            content = state.messages[-1].content
            query = content if isinstance(content, str) else str(content)
            logger.info("=" * 80)
            logger.info("LC Deep Research: Starting workflow")
            logger.info("Query: %s...", query[:100])
            logger.info("=" * 80)

        invoke_config: dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if self.callbacks:
            invoke_config["callbacks"] = self.callbacks

        graph_input = {"messages": list(state.messages)}
        try:
            if self.workflow_timeout_seconds is None:
                result = await graph.ainvoke(graph_input, config=invoke_config)
            else:
                async with asyncio.timeout(self.workflow_timeout_seconds):
                    result = await graph.ainvoke(graph_input, config=invoke_config)
        except Exception:
            logger.error("LC Deep Research failed", exc_info=True)
            raise

        final_report = extract_final_report(result)

        # Re-emit through any callback that streams reports to the frontend. Duck-typed on purpose:
        # VerboseTraceCallback has no such hook, but the async-job callbacks do, and this mirrors
        # the pattern in autonomous_researcher/agent.py.
        for cb in self.callbacks:
            if hasattr(cb, "emit_final_report"):
                cb.emit_final_report(final_report)
                break

        logger.info("=" * 80)
        logger.info("LC Deep Research: Workflow complete")
        logger.info("Final answer length : %d characters", len(final_report))
        logger.info("=" * 80)

        # Return the caller's history plus one assistant message holding the report. The graph's
        # own internal message trace stays out of AI-Q state: the report file is the deliverable,
        # and `messages[-1].content` is the contract the workflow wrapper, the job runner's
        # `_extract_result`, and the UI all read.
        return state.model_copy(
            update={
                "messages": list(state.messages) + [AIMessage(content=final_report)],
                "files": result.get("files", {}) if isinstance(result, dict) else {},
            }
        )
