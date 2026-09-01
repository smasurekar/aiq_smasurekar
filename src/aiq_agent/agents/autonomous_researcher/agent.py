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

"""Autonomous research agent: one description-driven DeepAgents loop, no effort tiers.

Adapted from ``adaptive_researcher.agent``. The report-extraction order, citation-verification
boundary, sanitization, artifact post-processing, partial-result fallback, and response shape are
ported **verbatim** — that is the API and report contract the eval harnesses and the UI depend on
(see the plan's §3.1), and it is deliberately not a place this agent innovates.

What is removed: ``_read_tier`` (there is no tier to read) and the shallow-subagent capture and
recovery path (there is no shallow subagent). What is added: the run owns one
``AutonomousFinalReportCommitTracker``, so the writer exit and the inline exit are recorded on the
same object and the finalization guard can accept either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepagents.backends.state import create_file_data
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepAgentsRuntime
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSkillsConfig
from aiq_agent.agents.deep_researcher.tools.source_tool_batching import DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS
from aiq_agent.agents.deep_researcher.tools.source_tool_batching import DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE
from aiq_agent.common import LLMProvider
from aiq_agent.common import load_prompt
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import sanitize_report
from aiq_agent.common.citation_verification import source_entries_from_parent_context
from aiq_agent.common.citation_verification import verify_citations

from .custom_middleware import AutonomousFinalReportCommitTracker
from .factory import DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS
from .factory import DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS
from .factory import AutonomousResearchGraphRun
from .factory import build_autonomous_research_graph
from .factory import build_autonomous_research_middleware_set
from .factory import build_autonomous_research_tool_set
from .models import AutonomousRequestTerminationConfig
from .models import AutonomousResearchAgentState
from .models import ResearcherLoopGuardConfig
from .subagents import ShallowSubagentCapture
from .tools.finalize import FINAL_REPORT_META_PATH
from .tools.finalize import FINAL_REPORT_PATH

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESEARCH_CONCURRENCY = 6
PARENT_REPORT_CONTEXT_PATH = "/shared/parent_report_context.json"

# Path to this agent's directory (for loading prompts)
AGENT_DIR = Path(__file__).parent

# Salvage gate: when the orchestrator ends a run without either finalize exit (no writer
# /shared/output.md and no inline /shared/final_report.md), we accept its final inline message as
# a last resort. Unlike the deep_researcher — which sits behind an intent classifier and always
# runs the writer — the autonomous agent handles greetings and capability questions itself, so
# this path must accept a short conversational reply. We reject only an empty message or the
# writer completion marker (that marker without an output.md is a different, real failure we must
# not mask).
_WRITER_COMPLETION_MARKER = "Wrote /shared/output.md"


class AutonomousResearcherAgent:
    """Autonomous research agent: one undifferentiated DeepAgents loop, description-driven."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        verbose: bool = True,
        callbacks: list[Any] | None = None,
        enable_citation_verification: bool = True,
        researcher_loop_guard: ResearcherLoopGuardConfig | None = None,
        request_termination: AutonomousRequestTerminationConfig | None = None,
        skills: DeepResearchSkillsConfig | None = None,
        sandbox: DeepResearchSandboxConfig | None = None,
        job_id: str | None = None,
        artifact_db_url: str | None = None,
        artifact_emit: Callable[[dict[str, Any]], None] | None = None,
        max_research_concurrency: int = DEFAULT_MAX_RESEARCH_CONCURRENCY,
        max_concurrent_source_tool_calls: int = DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
        max_source_tool_batch_size: int = DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
        research_batch_tool: bool = True,
        researcher_subagent: bool = False,
        shallow_subagent: bool = True,
        shallow_subagent_max_llm_turns: int = DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS,
        shallow_subagent_max_tool_iterations: int = DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS,
        shallow_subagent_tools: Sequence[str] | None = None,
        shallow_subagent_exclude_tools: Sequence[str] | None = None,
    ) -> None:
        """Initialize the autonomous researcher agent.

        Args:
            llm_provider: LLMProvider for role-based LLM access.
            tools: Optional sequence of LangChain tools. The orchestrator holds all of them
                directly, alongside run_research_batch and task.
            verbose: Enable detailed logging.
            callbacks: Optional list of callbacks.
            enable_citation_verification: Verify generated citations against the captured source
                registry.
            researcher_loop_guard: Hard per-researcher source-call, repeated-call, and
                consecutive-think limits. Defaults to enabled budgets aligned with the researcher
                prompt.
            request_termination: Request-wide batch/query/turn budgets, the hard workflow
                deadline, and the graph recursion ceiling — one flat set, always applied.
            skills: Optional DeepAgents skills config.
            sandbox: Optional DeepAgents sandbox config.
            job_id: Optional async job identifier used to scope sandbox backends.
            artifact_db_url: Optional artifact store URL for sandbox artifact harvesting.
            artifact_emit: Optional callback invoked when an artifact is checkpointed.
            max_research_concurrency: Maximum ResearchQuery items accepted and run concurrently
                per run_research_batch call.
            max_concurrent_source_tool_calls: Shared source-tool concurrency limit across
                researcher workers.
            max_source_tool_batch_size: Maximum concrete inputs per batch-capable source tool call.
            research_batch_tool: Offer ``run_research_batch``, which fans several independent
                questions out to isolated workers in one call. Disabling it removes the tool and
                every string that names it. Cannot be false together with ``researcher_subagent``.
            researcher_subagent: Also offer the ``researcher-agent`` sub-agent directly through
                ``task``, the only single-call path for a prerequisite chain. Off by default: the
                researcher already runs every research question as the ``run_research_batch``
                worker, so this opens a second door onto that same worker. Enabling it adds the
                spec and the strings that name it. Cannot be false together with
                ``research_batch_tool``.
            shallow_subagent: Offer the ``shallow-researcher`` sub-agent, which answers an easy
                request end to end and whose report finishes the run without a further
                orchestrator turn. Automatically suppressed for parent-report deltas.
            shallow_subagent_max_llm_turns: LLM-turn bound inside the shallow sub-run.
            shallow_subagent_max_tool_iterations: Tool-call bound inside the shallow sub-run.
            shallow_subagent_tools: Tool names the shallow sub-run may use. ``None``/empty keeps
                the agent's full tool set. Narrows the sub-run only; the orchestrator and the
                other subagents are unaffected.
            shallow_subagent_exclude_tools: Tool names withheld from the shallow sub-run only,
                applied after ``shallow_subagent_tools``.
        """
        self.llm_provider = llm_provider
        self.tools = list(tools) if tools else []
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.max_research_concurrency = max_research_concurrency
        self.max_concurrent_source_tool_calls = max_concurrent_source_tool_calls
        self.max_source_tool_batch_size = max_source_tool_batch_size
        self.research_batch_tool = research_batch_tool
        self.researcher_subagent = researcher_subagent
        self.shallow_subagent = shallow_subagent
        self.shallow_subagent_max_llm_turns = shallow_subagent_max_llm_turns
        self.shallow_subagent_max_tool_iterations = shallow_subagent_max_tool_iterations
        self.shallow_subagent_tools = list(shallow_subagent_tools or [])
        self.shallow_subagent_exclude_tools = list(shallow_subagent_exclude_tools or [])
        self.enable_citation_verification = enable_citation_verification
        self.researcher_loop_guard = researcher_loop_guard or ResearcherLoopGuardConfig()
        self.request_termination = request_termination or AutonomousRequestTerminationConfig()
        self.job_id = str(job_id) if job_id is not None else str(uuid4())

        self.deepagents_runtime = DeepAgentsRuntime(
            skills=skills,
            sandbox=sandbox,
            job_id=self.job_id,
            artifact_db_url=artifact_db_url,
            artifact_emit=artifact_emit,
        )

        try:
            self._prompts = self._load_prompts()
            source_tool_names = {tool.name for tool in self.tools}
            self.source_registry_middleware = SourceRegistryMiddleware(source_tool_names=source_tool_names)
            self.tool_set = build_autonomous_research_tool_set(
                self.tools,
                source_registry_middleware=self.source_registry_middleware,
                max_concurrent_source_tool_calls=self.max_concurrent_source_tool_calls,
                max_source_tool_batch_size=self.max_source_tool_batch_size,
            )
            self.middleware_set = build_autonomous_research_middleware_set(
                tool_set=self.tool_set,
                source_registry_middleware=self.source_registry_middleware,
                researcher_loop_guard=self.researcher_loop_guard,
                artifact_manager=self.deepagents_runtime.artifact_manager,
                # The orchestrator's ToolNameSanitizationMiddleware allowlist is built here, once,
                # and the graph builder only ever consumes it - so a disabled run_research_batch
                # has to be dropped from the allowlist at this call, not in the graph.
                research_batch_tool=self.research_batch_tool,
            )

            self.source_tool_names = self.tool_set.source_tool_names
            self.tools_info = self.tool_set.tools_info
            self.non_search_tools = self.tool_set.helper_tools
            self.all_tools = self.tool_set.all_tools
            self.research_source_tools = self.tool_set.research_source_tools
            self.researcher_tools = self.tool_set.researcher_tools
            self.writer_tools = self.tool_set.writer_tools
            self.researcher_middleware = self.middleware_set.researcher
            self.writer_middleware = self.middleware_set.writer
            self.orchestrator_middleware = self.middleware_set.orchestrator
        except Exception:
            try:
                cleanup_succeeded = self.deepagents_runtime.finalize(interrupted=False)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the original construction failure
                logger.warning(
                    "Autonomous research runtime cleanup failed during agent construction (%s)",
                    type(cleanup_error).__name__,
                )
            else:
                if not cleanup_succeeded:
                    logger.warning("Autonomous research runtime cleanup reported failure during agent construction")
            raise

    def finalize(self, *, interrupted: bool) -> bool:
        """Release this request's sandbox runtime exactly once."""
        return self.deepagents_runtime.finalize(interrupted=interrupted)

    def _load_prompts(self) -> dict[str, str]:
        """Load the four prompts this agent renders.

        No ``source_router`` prompt: the source-router subagent is dropped. No
        ``source_registry`` prompt either — the adaptive package shipped one that
        ``agent.py`` never loaded, and ``SourceRegistryMiddleware`` renders its own from the
        deep researcher's copy.
        """
        return {
            name: load_prompt(AGENT_DIR / "prompts", name)
            for name in ("planner", "researcher", "orchestrator", "writer")
        }

    def _build_orchestrator_agent(
        self,
        state: AutonomousResearchAgentState,
        tracker: AutonomousFinalReportCommitTracker,
    ) -> AutonomousResearchGraphRun:
        """Build the orchestrator graph for one run, bound to that run's dual-exit tracker.

        Returns the run wrapper rather than the bare runnable because the shallow sub-agent's
        capture is not reachable from graph state on the two paths that need it: teardown
        cancellation and post-exception report recovery.
        """
        return build_autonomous_research_graph(
            llm_provider=self.llm_provider,
            state=state,
            prompts=self._prompts,
            tools=self.tools,
            runtime=self.deepagents_runtime,
            tool_set=self.tool_set,
            middleware_set=self.middleware_set,
            source_registry_middleware=self.source_registry_middleware,
            final_report_tracker=tracker,
            callbacks=self.callbacks,
            max_research_concurrency=self.max_research_concurrency,
            researcher_loop_guard=self.researcher_loop_guard,
            request_termination=self.request_termination,
            research_batch_tool=self.research_batch_tool,
            researcher_subagent=self.researcher_subagent,
            shallow_subagent=self.shallow_subagent,
            shallow_subagent_max_llm_turns=self.shallow_subagent_max_llm_turns,
            shallow_subagent_max_tool_iterations=self.shallow_subagent_max_tool_iterations,
            shallow_subagent_tools=self.shallow_subagent_tools,
            shallow_subagent_exclude_tools=self.shallow_subagent_exclude_tools,
        )

    @staticmethod
    def _read_researched_flag(result: dict | Any) -> bool:
        """Return the ``researched`` flag written by ``submit_final_report``.

        Defaults to ``True`` when the sidecar meta file is absent or unreadable, so the writer
        path and the last-resort salvage path preserve the behaviour of raising on an empty
        source registry. Only a deliberate ``submit_final_report(researched=False)`` flips this to
        ``False`` to skip citation verification for a no-research answer.
        """
        files = result.get("files", None) if isinstance(result, dict) else getattr(result, "files", None)
        if not isinstance(files, dict):
            return True
        entry = files.get(FINAL_REPORT_META_PATH)
        if isinstance(entry, dict):
            entry = entry.get("content")
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8")
        if not isinstance(entry, str) or not entry.strip():
            return True
        try:
            data = json.loads(entry)
        except (ValueError, TypeError):
            return True
        return bool(data.get("researched", True))

    def _resolve_output_file_markdown(self, result: dict | Any, files: dict[str, Any] | None = None) -> str | None:
        """Resolve final Markdown from a known output file, or ``None`` if none exists.

        Resolution order is the contract shared with the deep and adaptive agents: the writer's
        ``/shared/output.md`` first, then the orchestrator's inline ``/shared/final_report.md``
        written by ``submit_final_report``. Returns ``None`` when no output file carries content,
        which signals the caller to fall back to inline salvage.
        """
        output_paths = ("/shared/output.md", "/output.md", FINAL_REPORT_PATH)
        # Resolve result files first, then fall back to the passed-in files (state.files) and
        # finally an empty dict. Without the explicit grouping, `or files or {}` would bind only
        # to the else branch, so a dict result lacking a usable "files" key would silently discard
        # the fallback and skip straight to inline salvage even when output files existed.
        result_files = result.get("files", None) if isinstance(result, dict) else getattr(result, "files", None)
        files = result_files or files or {}
        if isinstance(files, dict):
            for output_path in output_paths:
                output_entry = files.get(output_path)
                if isinstance(output_entry, dict):
                    output_entry = output_entry.get("content")
                if isinstance(output_entry, bytes):
                    output_entry = output_entry.decode("utf-8")
                if isinstance(output_entry, str) and output_entry.strip():
                    return output_entry.strip()
        return None

    def _extract_final_markdown(self, result: dict | Any, files: dict[str, Any] | None = None) -> str | None:
        """Return the final Markdown: an output file if present, else the inline salvage."""
        return self._resolve_output_file_markdown(result, files) or self._salvage_inline_report(result)

    @staticmethod
    def _salvage_inline_report(result: dict | Any) -> str | None:
        """Salvage the orchestrator's final inline message as a last resort.

        Reached only when neither the writer's ``/shared/output.md`` nor the inline
        ``/shared/final_report.md`` exists — i.e. the orchestrator ended the run without either
        finalize exit, and ``AutonomousFinalizationMiddleware`` already spent its one corrective
        turn. Accept any non-empty final message; reject only an empty message or the writer
        completion marker (that marker without an ``output.md`` is a different, real failure we
        must not mask). ``run()`` downgrades a source-less salvaged reply to ``researched=False``
        so citation checks do not fail a conversational answer.
        """
        messages = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", None)
        if not messages:
            return None
        content = getattr(messages[-1], "content", None)
        if not isinstance(content, str):
            return None
        stripped = content.strip()
        if not stripped or stripped == _WRITER_COMPLETION_MARKER:
            return None
        return stripped

    @staticmethod
    def _read_seed_file_text(files: dict[str, Any], path: str) -> str | None:
        entry = files.get(path)
        if isinstance(entry, dict):
            entry = entry.get("content")
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8")
        return entry if isinstance(entry, str) and entry.strip() else None

    def _seed_parent_sources(self, files: dict[str, Any]) -> None:
        """Register parent report sources so preserved citations verify in delta reports."""
        context_text = self._read_seed_file_text(files, PARENT_REPORT_CONTEXT_PATH)
        if not context_text:
            return
        parent_sources = source_entries_from_parent_context(context_text)
        seeded = self.source_registry_middleware.register_compact_sources(parent_sources)
        if seeded:
            logger.info("Seeded %d parent report source(s) into citation registry", seeded)

    @staticmethod
    def _replace_last_message_content(result: dict | Any, content: str) -> None:
        """Overwrite the final message content in-place with post-processed Markdown."""
        messages = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", None)
        if not messages:
            return
        last_msg = messages[-1]
        if hasattr(last_msg, "model_copy"):
            messages[-1] = last_msg.model_copy(update={"content": content})
        else:
            messages[-1] = type(last_msg)(content=content)

    @staticmethod
    def _persisted_notes_and_gaps(files: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        """Best-effort harvest of note summaries and gap descriptions from persisted ResearchNotes.

        Scans ``/shared/research_note_*.json`` entries — which both the batch path and the
        ``task(researcher-agent)`` path write — for the researcher-authored ``summary`` and
        ``gaps[].description`` fields. Used only by the deterministic partial-result path, so it
        is deliberately forgiving: any unreadable or malformed note is skipped rather than
        raising. Returns ``([], [])`` when no notes are available (the common timeout case, since
        the mid-run graph state is not returned) — the caller then falls back to a sources-only
        answer.
        """
        summaries: list[str] = []
        gaps: list[str] = []
        if not isinstance(files, dict):
            return summaries, gaps
        for path, entry in files.items():
            if not (isinstance(path, str) and "research_note_" in path and path.endswith(".json")):
                continue
            content = entry.get("content") if isinstance(entry, dict) else entry
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                data = json.loads(content)
            except (ValueError, TypeError):
                continue
            summary = data.get("summary") if isinstance(data, dict) else None
            if isinstance(summary, str) and summary.strip():
                summaries.append(summary.strip())
            for gap in (data.get("gaps") if isinstance(data, dict) else None) or []:
                description = gap.get("description") if isinstance(gap, dict) else None
                if isinstance(description, str) and description.strip():
                    gaps.append(description.strip())
        return summaries, gaps

    def _render_deterministic_partial(self, state: AutonomousResearchAgentState, reason: str) -> str:
        """Render a citation-safe partial report from already-gathered evidence — no model call.

        Uses only what is durably available after a forced termination: the in-memory verified
        source registry and any persisted ResearchNotes reachable from ``state.files``. It never
        fabricates: with no notes and no sources it returns a bounded failure explaining the
        configured sources did not provide enough evidence.
        """
        summaries, gaps = self._persisted_notes_and_gaps(state.files)
        sources = (
            self.source_registry_middleware.get_source_entries(mode="compact")
            if self.source_registry_middleware.has_sources()
            else []
        )

        if not summaries and not gaps and not sources:
            return (
                "# Research could not be completed\n\n"
                f"Research stopped because {reason}, and the configured sources did not return enough "
                "evidence to produce even a partial answer. Please try again, narrow the question, or "
                "confirm that the required documents are available in the knowledge base."
            )

        parts = [
            "# Partial research result",
            "",
            (
                f"Research stopped before completion because {reason}. The summary below is assembled "
                "from the evidence gathered so far and is intentionally conservative."
            ),
            "",
            "## What was found",
            "",
        ]
        if summaries:
            parts.extend(f"- {summary}" for summary in summaries)
        elif sources:
            parts.append(
                f"- No fully synthesized findings were available when research stopped; "
                f"{len(sources)} source(s) were consulted and are listed below."
            )
        else:
            parts.append("- No synthesized findings were available when research stopped.")

        parts.extend(["", "## Evidence gaps", ""])
        if gaps:
            parts.extend(f"- {gap}" for gap in gaps)
        else:
            parts.append(
                "- Some requested information could not be confirmed from the configured sources "
                "within the research budget."
            )

        if sources:
            parts.extend(["", "## Sources", ""])
            for index, entry in enumerate(sources, start=1):
                label = entry.title or entry.citation_key or entry.url or entry.tool_name or "source"
                locator = entry.url or entry.citation_key or entry.tool_name or ""
                suffix = f" — {locator}" if locator and locator != label else ""
                parts.append(f"{index}. {label}{suffix}")

        return "\n".join(parts)

    @staticmethod
    def _state_with_completed_shallow_capture(
        state: AutonomousResearchAgentState,
        capture: ShallowSubagentCapture | None,
    ) -> AutonomousResearchAgentState:
        """Return ``state`` with the shallow report merged in, when one is safely reusable.

        Reached only from the timeout / recursion handlers. On those paths ``ainvoke`` raised, so
        the sub-agent's ``files`` update — which reaches the caller on a normal completion — is
        unavailable, and the run-scoped capture is the only way to recover a report the shallow
        researcher already finished producing. Without this, a run whose only remaining work was
        to end would return a deterministic partial instead of the answer it already had.

        ``ShallowFinalizationMiddleware`` normally commits the report through the backend before
        the run ends, so this path is reached only when the deadline landed in the narrow window
        between the sub-agent completing and that commit — or when the commit itself failed.
        """
        if capture is None or not capture.has_report:
            return state
        logger.info(
            "Recovering the completed shallow-researcher report (%d characters) after a forced exit",
            len(capture.markdown),
        )
        files = {
            **state.files,
            FINAL_REPORT_PATH: create_file_data(capture.markdown),
            FINAL_REPORT_META_PATH: create_file_data(json.dumps({"researched": capture.researched})),
        }
        return state.model_copy(update={"files": files})

    def _build_partial_result(
        self, state: AutonomousResearchAgentState, *, reason: str
    ) -> AutonomousResearchAgentState:
        """Return a terminal partial-result state after a forced termination (deadline/recursion).

        Ordered fallback: (1) reuse an already-completed report if one is present in the input
        state files, running it through the normal citation-verification path; otherwise (2) build
        a deterministic partial from gathered evidence. The result is always sanitized, re-emitted
        to the frontend, and returned as a terminal state — never left running.
        """
        final_message = self._resolve_output_file_markdown({}, state.files)
        reused_completed_report = final_message is not None
        if not reused_completed_report:
            final_message = self._render_deterministic_partial(state, reason)

        # Verify citations only when reusing a real report that carries [n] markers against a
        # populated registry. The deterministic body has no inline citations — its Sources section
        # is rendered directly from verified entries — so it is sanitized but not re-verified.
        verify_reused = (
            reused_completed_report
            and self.enable_citation_verification
            and self.source_registry_middleware.has_sources()
        )
        if verify_reused:
            verification = verify_citations(
                final_message,
                self.source_registry_middleware.active_registry(),
                reference_sources=self.source_registry_middleware.get_source_entries(mode="compact"),
            )
            final_message = verification.verified_report

        final_message = sanitize_report(final_message).sanitized_report

        for cb in self.callbacks:
            if hasattr(cb, "emit_final_report"):
                cb.emit_final_report(final_message)
                break

        logger.info("=" * 80)
        logger.info("Autonomous Research: Returning partial result (%s)", reason)
        logger.info(
            "Reused completed report: %s | Final answer length: %d characters",
            reused_completed_report,
            len(final_message),
        )
        logger.info("=" * 80)

        messages = list(state.messages) + [AIMessage(content=final_message)]
        return state.model_copy(update={"messages": messages})

    async def run(self, state: AutonomousResearchAgentState) -> AutonomousResearchAgentState:
        """Execute one autonomous research request end to end."""
        prepared_files = self.deepagents_runtime.prepare_state_files(dict(state.files))
        if prepared_files != state.files:
            state = state.model_copy(update={"files": prepared_files})
        self._seed_parent_sources(state.files)

        # One tracker per request. Both exits record onto it: the writer through the upstream
        # FinalReportCommitMiddleware, and submit_final_report through its inline digest.
        final_report_tracker = AutonomousFinalReportCommitTracker()
        built = self._build_orchestrator_agent(state, final_report_tracker)
        runnable = built.runnable

        messages = state.messages
        if messages:
            query_content = messages[-1].content
            query = query_content if isinstance(query_content, str) else str(query_content)
            logger.info("=" * 80)
            logger.info("Autonomous Research: Starting workflow")
            logger.info("Query: %s...", query[:100])
            logger.info("=" * 80)

        # Hard workflow deadline: bound the entire graph invocation (orchestrator + planner +
        # researchers + writer + source tools + synthesis) so a request can never remain active
        # indefinitely. A per-source-tool timeout is not a workflow timeout. On the deadline — or
        # if the graph hits its recursion ceiling — we drop into a deterministic partial result
        # built from evidence already gathered rather than raising an opaque server error.
        timeout_seconds = self.request_termination.workflow_timeout_seconds
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await runnable.ainvoke(state, config={"callbacks": self.callbacks} if self.callbacks else None)
        except TimeoutError:
            logger.warning(
                "Autonomous Research exceeded the %ds workflow deadline; returning a deterministic partial result.",
                timeout_seconds,
            )
            recovery_state = self._state_with_completed_shallow_capture(state, built.shallow_capture)
            return self._build_partial_result(
                recovery_state, reason=f"the {timeout_seconds}s workflow time limit was reached"
            )
        except GraphRecursionError:
            logger.warning(
                "Autonomous Research reached the graph recursion limit (%d); returning a deterministic partial result.",
                self.request_termination.recursion_limit,
            )
            recovery_state = self._state_with_completed_shallow_capture(state, built.shallow_capture)
            return self._build_partial_result(recovery_state, reason="the maximum research step limit was reached")
        except Exception as ex:
            # Preserve observability: any other invocation failure is logged with a traceback and
            # re-raised (post-processing errors are handled by the block below).
            logger.error("Autonomous Research failed: %s", ex, exc_info=True)
            raise
        finally:
            # The shallow sub-agent runs in a detached asyncio task so concurrent `task` calls can
            # share it, which means cancelling this coroutine does NOT stop it. Without this the
            # workflow deadline would return a partial result while an orphaned shallow run kept
            # issuing LLM and source-tool calls. Runs on every exit path — normal completion
            # (a no-op, the task is done), timeout, recursion abort, and the CancelledError a
            # client disconnect raises, which this method deliberately does not catch.
            if built.shallow_capture is not None:
                built.shallow_capture.cancel()

        try:
            # Resolve via the two sub-methods (not the combined extractor) so we know whether the
            # answer came from a known output file or from the last-resort inline salvage.
            final_message = self._resolve_output_file_markdown(result, state.files)
            from_salvage = False
            if final_message is None:
                final_message = self._salvage_inline_report(result)
                from_salvage = final_message is not None
            if final_message is None:
                raise ValueError("agent did not produce a final Markdown answer")

            # A deliberate no-research answer signals researched=False via submit_final_report so
            # citation verification is skipped for it. A genuine tried-to-research-but-found-
            # nothing run leaves this True and still raises below.
            researched = self._read_researched_flag(result)

            # A salvaged reply means the orchestrator ended without either finalize exit, so
            # `researched` defaulted to True. With no sources gathered this is a conversational /
            # capability answer (e.g. a greeting), not a failed research run — treat it as
            # no-research so we return it instead of raising EmptySourceRegistryError below.
            if from_salvage and researched and not self.source_registry_middleware.has_sources():
                logger.info("Salvaged inline reply with empty source registry; treating as no-research answer.")
                researched = False

            # Post-process: verify citations against the source registry. Compact mode is the
            # path-independent union of direct, task-delegated, and batched evidence — see
            # DirectSourcePromotionMiddleware and ResearcherTaskPersistenceMiddleware.
            if self.enable_citation_verification and self.source_registry_middleware.has_sources():
                registry = self.source_registry_middleware.active_registry()
                verification = verify_citations(
                    final_message,
                    registry,
                    reference_sources=self.source_registry_middleware.get_source_entries(mode="compact"),
                )
                if verification.removed_citations:
                    removed_details = []
                    for c in verification.removed_citations:
                        url_match = re.search(r"https?://\S+", c.get("line", ""))
                        url_str = url_match.group(0).rstrip(".,;)") if url_match else "(no url)"
                        removed_details.append(f"[{c['number']}] {c['reason']}: {url_str}")
                    logger.info(
                        "Citation verification removed %d invalid citation(s):\n  %s",
                        len(verification.removed_citations),
                        "\n  ".join(removed_details),
                    )
                final_message = verification.verified_report
                if not verification.valid_citations:
                    logger.warning(
                        "Citation verification found no valid citations in the final output; "
                        "returning the generated report without failing the job. "
                        "This may indicate unsupported citation formatting or over-aggressive verification."
                    )
            elif self.enable_citation_verification and researched:
                from aiq_agent.common.tool_validation import validate_tool_availability

                _, available_count, unavailable = validate_tool_availability(
                    self.tools,
                    research_type="autonomous research",
                    enable_logging=False,
                )
                raise EmptySourceRegistryError(
                    "autonomous research",
                    unavailable_tools=unavailable,
                    available_count=available_count,
                )

            # Post-process: sanitize report (strip body URLs, shortened URLs, unsafe URLs)
            sanitization = sanitize_report(final_message)
            final_message = sanitization.sanitized_report

            # Post-process: harvest sandbox artifacts and resolve artifact:// references so
            # generated charts/files render in the report. Inert (manager is None) unless a
            # sandbox + artifact_capture + db_url are configured. Blocking I/O off the loop.
            manager = self.deepagents_runtime.artifact_manager
            if manager is not None:
                try:
                    await asyncio.to_thread(manager.final_harvest)
                    produced = await asyncio.to_thread(manager.store.list, manager.job_id)
                    final_message = await asyncio.to_thread(manager.resolve_report_references, final_message, produced)
                    final_message = await asyncio.to_thread(
                        manager.ensure_inline_artifacts_embedded, final_message, produced
                    )
                    final_message = await asyncio.to_thread(manager.append_artifact_index, final_message, produced)
                except Exception:
                    # Best-effort: never discard an already verified/sanitized report because
                    # artifact harvest or embedding failed. final_message stays as-is.
                    logger.warning(
                        "Artifact post-processing failed; returning report without embedded artifacts",
                        exc_info=True,
                    )

            # Re-emit the verified/sanitized report so the frontend overwrites the raw version
            # that on_llm_end auto-emitted during ainvoke().
            for cb in self.callbacks:
                if hasattr(cb, "emit_final_report"):
                    cb.emit_final_report(final_message)
                    break

            self._replace_last_message_content(result, final_message)

            exit_path = "writer" if final_report_tracker.digest is not None else "inline"
            if from_salvage:
                exit_path = "salvaged inline message"
            logger.info("=" * 80)
            logger.info("Autonomous Research: Workflow complete")
            logger.info("Finalize exit       : %s", exit_path)
            logger.info("Final answer length : %d characters", len(final_message))
            logger.info("=" * 80)
            return AutonomousResearchAgentState.model_validate(result)

        except Exception as ex:
            logger.error("Autonomous Research failed: %s", ex, exc_info=True)
            raise
