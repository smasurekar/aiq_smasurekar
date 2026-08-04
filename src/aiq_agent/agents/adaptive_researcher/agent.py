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

"""Adaptive research agent using deepagents library for one adaptive (shallow<->deep) workflow.

Cloned from ``deep_researcher`` with rewritten prompts and a small finalize seam. Shared
machinery (middleware, runtime, source-tool batching) is imported from ``deep_researcher`` to
avoid drift; only the report-extraction order and the no-research safeguard diverge here.
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

from .custom_middleware import _DEFAULT_SINGLE_SHOT_SEARCH_BUDGET
from .factory import DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS
from .factory import DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS
from .factory import AdaptiveResearchGraphRun
from .factory import build_adaptive_research_graph
from .factory import build_adaptive_research_middleware_set
from .factory import build_adaptive_research_tool_set
from .models import AdaptiveRequestTerminationConfig
from .models import AdaptiveResearchAgentState
from .models import ResearcherLoopGuardConfig
from .subagents import ShallowSubagentCapture
from .tools.finalize import EFFORT_TIER_PATH
from .tools.finalize import FINAL_REPORT_META_PATH
from .tools.finalize import FINAL_REPORT_PATH

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESEARCH_CONCURRENCY = 6
# Re-export the middleware's single_shot search-budget default so register.py and callers have a
# public name to reference (the middleware constant is the single source of truth for its value).
DEFAULT_SINGLE_SHOT_SEARCH_BUDGET = _DEFAULT_SINGLE_SHOT_SEARCH_BUDGET
PARENT_REPORT_CONTEXT_PATH = "/shared/parent_report_context.json"

# Path to this agent's directory (for loading prompts)
AGENT_DIR = Path(__file__).parent

# Salvage gate: when the orchestrator ends a run without the finalize protocol (no writer
# /shared/output.md and no inline /shared/final_report.md from submit_final_report), we accept
# its final inline message as a last resort. Unlike the deep_researcher — which is only reached
# after an upstream intent_classifier and always runs the writer pipeline — the adaptive agent
# handles greetings / chit-chat itself, so this path must accept a short conversational reply.
# We reject only an empty message or the writer completion marker (that marker without an
# output.md is a different, real failure we must not mask).
_WRITER_COMPLETION_MARKER = "Wrote /shared/output.md"


class AdaptiveResearcherAgent:
    """
    Adaptive research agent using deepagents library for one adaptive (shallow<->deep) workflow.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        verbose: bool = True,
        callbacks: list[Any] | None = None,
        domain_catalog_path: str | None = None,
        enable_source_router: bool = False,
        enable_citation_verification: bool = True,
        enabled_tiers: list[str] | None = None,
        enforce_tier_tools: bool = False,
        single_loop_single_shot: bool = False,
        single_shot_search_budget: int = DEFAULT_SINGLE_SHOT_SEARCH_BUDGET,
        single_shot_shallow_subagent: bool = False,
        shallow_subagent_max_llm_turns: int = DEFAULT_SHALLOW_SUBAGENT_MAX_LLM_TURNS,
        shallow_subagent_max_tool_iterations: int = DEFAULT_SHALLOW_SUBAGENT_MAX_TOOL_ITERATIONS,
        dynamic_orchestrator_sections: bool = False,
        researcher_loop_guard: ResearcherLoopGuardConfig | None = None,
        request_termination: AdaptiveRequestTerminationConfig | None = None,
        skills: DeepResearchSkillsConfig | None = None,
        sandbox: DeepResearchSandboxConfig | None = None,
        job_id: str | None = None,
        artifact_db_url: str | None = None,
        artifact_emit: Callable[[dict[str, Any]], None] | None = None,
        max_research_concurrency: int = DEFAULT_MAX_RESEARCH_CONCURRENCY,
        max_concurrent_source_tool_calls: int = DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
        max_source_tool_batch_size: int = DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
    ) -> None:
        """
        Initialize the adaptive researcher agent.

        Args:
            llm_provider: LLMProvider for role-based LLM access.
            tools: Optional sequence of LangChain tools for research.
            verbose: Enable detailed logging.
            callbacks: Optional list of callbacks.
            domain_catalog_path: Optional YAML/JSON domain catalog path for source-router-agent.
            enable_source_router: Enable the advisory source-router-agent before planning.
            enable_citation_verification: Verify generated citations against the captured source registry.
            enforce_tier_tools: Enable Layer-B ceiling-based tool hiding via ComplexityRouterMiddleware.
            single_loop_single_shot: Collapse single_shot to a direct-tool single loop, bypassing
                the researcher subagent. Requires enforce_tier_tools or auto-enables the middleware.
            single_shot_search_budget: Max direct source-tool calls the single_loop_single_shot
                single_shot path may make before ComplexityRouterMiddleware withdraws the search
                tools and forces finalize. Caps runaway search loops on cheap lookups.
            single_shot_shallow_subagent: Route the single_shot tier to the shallow researcher,
                registered as a DeepAgents CompiledSubAgent. Tier selection is unchanged; when
                single_shot is declared the orchestrator delegates the original user query and the
                shallow report becomes the authoritative answer. Takes precedence over
                single_loop_single_shot.
            shallow_subagent_max_llm_turns: LLM-turn bound inside the shallow sub-agent.
            shallow_subagent_max_tool_iterations: Tool-call bound inside the shallow sub-agent.
            dynamic_orchestrator_sections: Render the orchestrator prompt trimmed per declared tier
                (minimal router prompt on turn 1, per-tier prompt swapped in after declare_effort_tier).
                Off by default renders the full prompt once at build time, exactly as before.
            researcher_loop_guard: Hard per-researcher source-call, repeated-call, and consecutive-think
                limits. Defaults to enabled budgets aligned with the researcher prompt.
            request_termination: Request-wide batch/query/turn budgets, the hard workflow deadline, and
                the graph recursion ceiling. Guarantees every request reaches a terminal state.
                Defaults to enabled, finite budgets.
            skills: Optional DeepAgents skills config.
            sandbox: Optional DeepAgents sandbox config.
            job_id: Optional async job identifier used to scope sandbox backends.
            max_research_concurrency: Maximum ResearchQuery items accepted and run concurrently per
                run_research_batch call.
            max_concurrent_source_tool_calls: Shared source-tool concurrency limit across researcher workers.
            max_source_tool_batch_size: Maximum concrete inputs per batch-capable source tool call.
        """
        self.llm_provider = llm_provider
        self.tools = list(tools) if tools else []
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.max_research_concurrency = max_research_concurrency
        self.max_concurrent_source_tool_calls = max_concurrent_source_tool_calls
        self.max_source_tool_batch_size = max_source_tool_batch_size
        self.domain_catalog_path = domain_catalog_path
        self.enable_source_router = enable_source_router
        self.enable_citation_verification = enable_citation_verification
        self.enabled_tiers = list(enabled_tiers) if enabled_tiers else ["direct", "single_shot", "standard", "deep"]
        self.enforce_tier_tools = enforce_tier_tools
        self.single_loop_single_shot = single_loop_single_shot
        self.single_shot_search_budget = single_shot_search_budget
        self.single_shot_shallow_subagent = single_shot_shallow_subagent
        self.shallow_subagent_max_llm_turns = shallow_subagent_max_llm_turns
        self.shallow_subagent_max_tool_iterations = shallow_subagent_max_tool_iterations
        self.dynamic_orchestrator_sections = dynamic_orchestrator_sections
        self.researcher_loop_guard = researcher_loop_guard or ResearcherLoopGuardConfig()
        self.request_termination = request_termination or AdaptiveRequestTerminationConfig()
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
            self.tool_set = build_adaptive_research_tool_set(
                self.tools,
                source_registry_middleware=self.source_registry_middleware,
                max_concurrent_source_tool_calls=self.max_concurrent_source_tool_calls,
                max_source_tool_batch_size=self.max_source_tool_batch_size,
            )
            direct_source_tool_names: frozenset[str] = (
                frozenset(t.name for t in self.tool_set.research_source_tools)
                if self.single_loop_single_shot
                else frozenset()
            )
            self.middleware_set = build_adaptive_research_middleware_set(
                tool_set=self.tool_set,
                source_registry_middleware=self.source_registry_middleware,
                researcher_loop_guard=self.researcher_loop_guard,
                enable_source_router=self.enable_source_router,
                artifact_manager=self.deepagents_runtime.artifact_manager,
                direct_source_tool_names=direct_source_tool_names,
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
            self.middleware = self.researcher_middleware
        except Exception:
            try:
                cleanup_succeeded = self.deepagents_runtime.finalize(interrupted=False)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the original construction failure
                logger.warning(
                    "Deep research runtime cleanup failed during agent construction (%s)",
                    type(cleanup_error).__name__,
                )
            else:
                if not cleanup_succeeded:
                    logger.warning("Deep research runtime cleanup reported failure during agent construction")
            raise

    def finalize(self, *, interrupted: bool) -> bool:
        """Release this request's sandbox runtime exactly once."""
        return self.deepagents_runtime.finalize(interrupted=interrupted)

    def _load_prompts(self) -> dict[str, str]:
        """Load all prompts for subagents."""
        prompts = {}
        prompt_names = ["planner", "researcher", "orchestrator", "writer", "source_router"]

        for name in prompt_names:
            prompts[name] = load_prompt(AGENT_DIR / "prompts", name)

        return prompts

    def _build_orchestrator_agent(self, state: AdaptiveResearchAgentState) -> AdaptiveResearchGraphRun:
        """Build the orchestrator graph bundle (runnable + run-scoped shallow capture)."""
        return build_adaptive_research_graph(
            llm_provider=self.llm_provider,
            state=state,
            prompts=self._prompts,
            tools=self.tools,
            runtime=self.deepagents_runtime,
            tool_set=self.tool_set,
            middleware_set=self.middleware_set,
            source_registry_middleware=self.source_registry_middleware,
            callbacks=self.callbacks,
            domain_catalog_path=self.domain_catalog_path,
            enable_source_router=self.enable_source_router,
            max_research_concurrency=self.max_research_concurrency,
            enabled_tiers=self.enabled_tiers,
            enforce_tier_tools=self.enforce_tier_tools,
            single_loop_single_shot=self.single_loop_single_shot,
            single_shot_search_budget=self.single_shot_search_budget,
            single_shot_shallow_subagent=self.single_shot_shallow_subagent,
            shallow_subagent_max_llm_turns=self.shallow_subagent_max_llm_turns,
            shallow_subagent_max_tool_iterations=self.shallow_subagent_max_tool_iterations,
            dynamic_orchestrator_sections=self.dynamic_orchestrator_sections,
            researcher_loop_guard=self.researcher_loop_guard,
            request_termination=self.request_termination,
        )

    @staticmethod
    def _read_tier(result: dict | Any) -> str | None:
        """Return the effort tier from the run's shared filesystem, or ``None`` if absent.

        Checks two sources in priority order:
        1. ``/shared/effort_tier.json`` — written by ``declare_effort_tier`` at the very start
           of the run, before any research or delegation. Present on all paths.
        2. ``/shared/final_report_meta.json`` — written by ``submit_final_report`` at the end
           of shallow paths. Absent on the deep / writer-agent path and when the orchestrator
           answers inline without calling ``submit_final_report``.
        """
        files = result.get("files", None) if isinstance(result, dict) else getattr(result, "files", None)
        if not isinstance(files, dict):
            return None

        def _extract(path: str) -> str | None:
            entry = files.get(path)
            if isinstance(entry, dict):
                entry = entry.get("content")
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8")
            if not isinstance(entry, str) or not entry.strip():
                return None
            try:
                return json.loads(entry).get("tier")
            except (ValueError, TypeError):
                return None

        return _extract(EFFORT_TIER_PATH) or _extract(FINAL_REPORT_META_PATH)

    @staticmethod
    def _read_researched_flag(result: dict | Any) -> bool:
        """Return the ``researched`` flag written by ``submit_final_report``.

        Defaults to ``True`` when the sidecar meta file is absent or unreadable, so the
        writer/deep path and the last-resort salvage path preserve today's behaviour of raising
        on an empty source registry. Only a deliberate ``submit_final_report(researched=False)``
        flips this to ``False`` to skip citation verification for a no-research answer.
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

        Resolution order: the writer's ``/shared/output.md`` (deep path) first, then the
        orchestrator's inline ``/shared/final_report.md`` written by ``submit_final_report``
        (shallow/direct/meta path). Returns ``None`` when no output file carries content, which
        signals the caller to fall back to inline salvage.
        """
        output_paths = ("/shared/output.md", "/output.md", FINAL_REPORT_PATH)
        # Resolve result files first, then fall back to the passed-in files (state.files) and
        # finally an empty dict. Without the explicit grouping, `or files or {}` bound only to
        # the else branch, so a dict result lacking a usable "files" key silently discarded the
        # fallback and skipped straight to inline salvage even when output files existed.
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
        ``/shared/final_report.md`` exists — i.e. the orchestrator ended the run without the
        finalize protocol (for example a greeting answered conversationally without calling
        ``submit_final_report``). Accept any non-empty final message; reject only an empty
        message or the writer completion marker (that marker without an ``output.md`` is a
        different, real failure we must not mask). ``run()`` downgrades a source-less salvaged
        reply to ``researched=False`` so citation checks do not fail a conversational answer.
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
    def _state_with_completed_shallow_capture(
        state: AdaptiveResearchAgentState,
        capture: ShallowSubagentCapture | None,
    ) -> AdaptiveResearchAgentState:
        """Return ``state`` with the shallow report merged in, when one is safely reusable.

        Reached only from the timeout / recursion handlers. On those paths ``ainvoke`` raised, so
        the sub-agent's ``files`` update — which reaches the caller on a normal completion — is
        unavailable, and the run-scoped capture is the only way to recover a report the shallow
        researcher already finished producing.

        Three conditions must all hold, otherwise the caller keeps today's behaviour and builds a
        deterministic partial: the shallow run completed, it produced content, and the run is
        still on the ``single_shot`` tier (an escalated run must not silently return a shallow
        answer authored for a lower effort level).
        """
        if capture is None or not capture.has_report or capture.declared_tier != "single_shot":
            return state
        logger.info(
            "Recovering the completed shallow-researcher report (%d characters) after a forced exit",
            len(capture.markdown),
        )
        files = {
            **state.files,
            FINAL_REPORT_PATH: create_file_data(capture.markdown),
            FINAL_REPORT_META_PATH: create_file_data(
                json.dumps({"researched": capture.researched, "tier": "single_shot"})
            ),
        }
        return state.model_copy(update={"files": files})

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

        Scans ``/shared/research_note_*.json`` entries for the researcher-authored ``summary`` and
        ``gaps[].description`` fields. Used only by the deterministic partial-result path, so it is
        deliberately forgiving: any unreadable or malformed note is skipped rather than raising.
        Returns ``([], [])`` when no notes are available (the common timeout case, since the
        mid-run graph state is not returned) — the caller then falls back to a sources-only answer.
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

    def _render_deterministic_partial(self, state: AdaptiveResearchAgentState, reason: str) -> str:
        """Render a citation-safe partial report from already-gathered evidence — no model call.

        Uses only what is durably available after a forced termination: the in-memory verified
        source registry (``source_registry_middleware``) and any persisted ResearchNotes reachable
        from ``state.files``. It never fabricates: with no notes and no sources it returns a bounded
        failure explaining the configured sources did not provide enough evidence.
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

    def _build_partial_result(self, state: AdaptiveResearchAgentState, *, reason: str) -> AdaptiveResearchAgentState:
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
        logger.info("Adaptive Research: Returning partial result (%s)", reason)
        logger.info(
            "Reused completed report: %s | Final answer length: %d characters",
            reused_completed_report,
            len(final_message),
        )
        logger.info("=" * 80)

        messages = list(state.messages) + [AIMessage(content=final_message)]
        return state.model_copy(update={"messages": messages})

    async def run(self, state: AdaptiveResearchAgentState) -> AdaptiveResearchAgentState:
        """
        Execute adaptive research: the model self-selects effort and self-limits its tool use.
        """
        prepared_files = self.deepagents_runtime.prepare_state_files(dict(state.files))
        if prepared_files != state.files:
            state = state.model_copy(update={"files": prepared_files})
        self._seed_parent_sources(state.files)
        built = self._build_orchestrator_agent(state)

        messages = state.messages
        if messages:
            query_content = messages[-1].content
            query = query_content if isinstance(query_content, str) else str(query_content)
            logger.info("=" * 80)
            logger.info("Adaptive Research: Starting workflow")
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
                result = await built.runnable.ainvoke(
                    state, config={"callbacks": self.callbacks} if self.callbacks else None
                )
        except TimeoutError:
            logger.warning(
                "Adaptive Research exceeded the %ds workflow deadline; returning a deterministic partial result.",
                timeout_seconds,
            )
            recovery_state = self._state_with_completed_shallow_capture(state, built.shallow_capture)
            return self._build_partial_result(
                recovery_state, reason=f"the {timeout_seconds}s workflow time limit was reached"
            )
        except GraphRecursionError:
            logger.warning(
                "Adaptive Research reached the graph recursion limit (%d); returning a deterministic partial result.",
                self.request_termination.recursion_limit,
            )
            recovery_state = self._state_with_completed_shallow_capture(state, built.shallow_capture)
            return self._build_partial_result(recovery_state, reason="the maximum research step limit was reached")
        except Exception as ex:
            # Preserve the original observability: any other invocation failure is logged with a
            # traceback and re-raised (post-processing errors are handled by the block below).
            logger.error("Adaptive Research failed: %s", ex, exc_info=True)
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

            # A deliberate no-research answer (direct / meta) signals researched=False via
            # submit_final_report so citation verification is skipped for it. A genuine
            # tried-to-research-but-found-nothing run leaves this True and still raises below.
            researched = self._read_researched_flag(result)

            # A salvaged reply means the orchestrator ended without submit_final_report, so
            # `researched` defaulted to True. With no sources gathered this is a conversational /
            # meta answer (e.g. a greeting), not a failed research run — treat it as no-research so
            # we return it instead of raising EmptySourceRegistryError below.
            if from_salvage and researched and not self.source_registry_middleware.has_sources():
                logger.info("Salvaged inline reply with empty source registry; treating as no-research answer.")
                researched = False

            # Post-process: verify citations against source registry
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
                        "Citation verification found no valid citations in writer-agent output; "
                        "returning the generated report without failing the job. "
                        "This may indicate unsupported citation formatting or over-aggressive verification."
                    )
            elif self.enable_citation_verification and researched:
                from aiq_agent.common.tool_validation import validate_tool_availability

                _, available_count, unavailable = validate_tool_availability(
                    self.tools,
                    research_type="adaptive research",
                    enable_logging=False,
                )
                raise EmptySourceRegistryError(
                    "adaptive research",
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

            # Re-emit the verified/sanitized report so the frontend overwrites
            # the raw version that on_llm_end auto-emitted during ainvoke().
            for cb in self.callbacks:
                if hasattr(cb, "emit_final_report"):
                    cb.emit_final_report(final_message)
                    break

            self._replace_last_message_content(result, final_message)

            tier = self._read_tier(result)
            if tier is None:
                # Writer-agent path (deep or standard — writer branch) does not call
                # submit_final_report, so the meta JSON is absent. Infer from output.md.
                if self._resolve_output_file_markdown(result, state.files) is not None:
                    tier = "deep/standard (writer branch)"
            logger.info("=" * 80)
            logger.info("Adaptive Research: Workflow complete")
            logger.info("Effort tier selected : %s", tier or "unknown")
            logger.info("Final answer length  : %d characters", len(final_message))
            logger.info("=" * 80)
            return AdaptiveResearchAgentState.model_validate(result)

        except Exception as ex:
            logger.error("Adaptive Research failed: %s", ex, exc_info=True)
            raise
