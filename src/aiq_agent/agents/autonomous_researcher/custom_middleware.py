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

"""Autonomous-researcher-specific evidence and integrity middleware.

The autonomous agent reuses every middleware in ``deep_researcher.custom_middleware`` and the
two tier-independent guards from ``adaptive_researcher.custom_middleware``
(``ResearcherLoopGuardMiddleware``, ``ConsecutiveThinkGuardMiddleware``). It deliberately drops
all four tier machines — ``ComplexityRouterMiddleware``, ``TierResolver``,
``SingleShotShallowDelegationMiddleware``, and ``hidden_tools_for_ceiling``.

What is left is the price of the architecture's central decision. Because the orchestrator holds
the *full* retrieval menu — every source tool directly, plus ``run_research_batch``, plus
``task`` into every subagent — three different tool calls can mean "go find this out", and there
is no router left to normalize what each one does to run state. These five seams close that gap:

``AutonomousFinalReportCommitTracker``
    One tracker recording *either* valid exit — a writer-owned ``/shared/output.md`` commit or an
    orchestrator-owned ``/shared/final_report.md`` commit.

``DirectSourcePromotionMiddleware``
    Promotes sources captured by a *direct* orchestrator source call into the compact
    verified-source set, so direct evidence does not disappear from compact
    ``get_verified_sources`` once a batch has established a compact subset.

``ResearcherTaskPersistenceMiddleware``
    Gives ``task(subagent_type="researcher-agent")`` the same evidence side effects
    ``run_research_batch`` already has: the returned ``ResearchNotes`` is persisted under
    ``/shared/`` and its source locators are registered as compact sources.

``PlanBeforeWriterMiddleware``
    Rejects ``task(subagent_type="writer-agent")`` until ``/shared/plan.json`` exists.
    ``writer.j2`` hard-requires ``answer_strategy.*`` and ``constraints`` from that file, so
    writer-without-planner is a guaranteed bad report rather than a style choice.

``AutonomousFinalizationMiddleware``
    Accepts *either* tracked exit. It deliberately replaces upstream's
    ``RequiredWriterDelegationMiddleware``, which would force every run through the writer and
    eliminate the valid inline exit.

``AutonomousOrchestratorLoopGuardMiddleware``
    The flat-budget variant of the adaptive orchestrator loop guard: no ``budgets_for_tier()``
    lookup, one budget set that always applies.

Logging in this module is metadata-only (counts, hashed signatures, paths) — never prompts,
model responses, ``ResearchNotes`` bodies, or tool payloads. See ``common.logging_utils``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import hook_config
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from aiq_agent.agents.adaptive_researcher.custom_middleware import _normalize_text
from aiq_agent.agents.deep_researcher.custom_middleware import FINAL_REPORT_STATE_PATHS
from aiq_agent.agents.deep_researcher.custom_middleware import FinalReportCommitTracker
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import _request_tool_name

from .models import AutonomousRequestTerminationConfig
from .models import ResearchNotes
from .tools.finalize import FINAL_REPORT_PATH as INLINE_FINAL_REPORT_PATH

logger = logging.getLogger(__name__)

# Tool / subagent names this module keys on. Kept as literals rather than imported from
# ``factory.py``, which imports this module.
THINK_TOOL = "think"
RUN_RESEARCH_BATCH_TOOL = "run_research_batch"
TASK_TOOL = "task"
FINALIZE_TOOL = "submit_final_report"
PLANNER_SUBAGENT = "planner-agent"
RESEARCHER_SUBAGENT = "researcher-agent"
WRITER_SUBAGENT = "writer-agent"
PLAN_PATH = "/shared/plan.json"

# Argument names of the DeepAgents ``task`` tool (``TaskToolSchema``).
_SUBAGENT_TYPE_ARG = "subagent_type"
_TASK_DESCRIPTION_ARG = "description"

# Marker used to bound corrective turns, matching the upstream convention so a retry injected
# here is not mistaken for user input.
_GENERATED_RETRY_MARKER = "aiq_generated_retry"

_NOTE_SLUG_MAX_LENGTH = 40

# Appended (not overwritten) to the direct source-tool result that spends the last of the
# orchestrator's own search budget. Appending preserves that call's evidence — the model still
# needs it — while explaining in-context why the source tools are about to disappear. The
# withdrawal in ``_filter_tools`` is the hard guarantee; this is the explanation. Mirrors
# ``_SINGLE_SHOT_BUDGET_NUDGE`` in the adaptive arm, but points at delegation rather than at
# finalization: direct search closing does not mean research is over here.
_DIRECT_SOURCE_BUDGET_NUDGE = (
    "\n\n[SYSTEM — direct-search budget reached: you have used your own source-tool calls for "
    "this request, and those tools are now withdrawn. Research is NOT over. Delegate any "
    "remaining lookup through `run_research_batch`, or give a dependent chain to "
    '`task(subagent_type="researcher-agent", ...)` — a worker\'s search trail is digested before '
    "it reaches you instead of accumulating in your context. Only finalize once the evidence is "
    "sufficient.]"
)


# =================================================================================================
# Dual-exit commit tracking
# =================================================================================================


class AutonomousFinalReportCommitTracker(FinalReportCommitTracker):
    """Run-local proof that *one of the two* valid final-report exits actually committed.

    The base class records only the writer's mutation digest, which upstream's
    ``RequiredWriterDelegationMiddleware`` then treats as the sole proof a run may finish. That
    is correct for the deep researcher, where the writer pipeline is mandatory, and wrong here:
    the autonomous orchestrator may legitimately answer inline, in which case no writer ever
    runs and ``/shared/output.md`` never exists.

    So this subclass adds a second, independent digest for the inline exit written by
    ``submit_final_report``. The writer path is untouched — ``FinalReportCommitMiddleware`` and
    ``RequiredOutputFileMiddleware`` keep calling ``record`` / ``committed_text`` exactly as they
    do for the deep researcher — and ``any_exit_committed`` is the union the autonomous
    finalization guard checks.
    """

    def __init__(self) -> None:
        """Initialize with neither exit committed."""
        super().__init__()
        self._inline_digest: str | None = None

    def record_inline(self, content: str) -> str:
        """Record the digest of an inline report committed by ``submit_final_report``."""
        digest = self._digest_text(content)
        # The base class guards ``_digest`` with a lock because the writer subagent can commit
        # from another thread. The inline exit is only ever written by the orchestrator's own
        # tool call, so a plain assignment is sufficient and the base lock is left alone.
        self._inline_digest = digest
        return digest

    @property
    def inline_digest(self) -> str | None:
        """Return the digest of the inline report committed this run, if any."""
        return self._inline_digest

    def inline_committed_text(self, files: object) -> str | None:
        """Return ``/shared/final_report.md`` when it matches the recorded inline digest."""
        digest = self._inline_digest
        if digest is None or not isinstance(files, dict):
            return None
        entry = files.get(INLINE_FINAL_REPORT_PATH)
        if isinstance(entry, dict):
            entry = entry.get("content")
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8", errors="replace")
        if isinstance(entry, str) and entry.strip() and self._digest_text(entry) == digest:
            return entry
        return None

    def any_exit_committed(self, files: object) -> bool:
        """Return True when either the writer exit or the inline exit committed this run.

        The digest-only fallbacks matter: ``submit_final_report`` is ``return_direct=True``, so
        the run can end before the tool's state update is visible in the ``files`` mapping this
        guard is handed. Recording the digest is itself proof the backend write succeeded (the
        tool records only after ``upload_files`` returned without errors), so a matching digest
        with no visible file is a commit, not a failure.
        """
        if self.committed_text(files, paths=FINAL_REPORT_STATE_PATHS) is not None:
            return True
        if self.inline_committed_text(files) is not None:
            return True
        return self.digest is not None or self._inline_digest is not None


# =================================================================================================
# Evidence-state seams: making the three research paths equivalent
# =================================================================================================


class DirectSourcePromotionMiddleware(AgentMiddleware):
    """Promote directly-retrieved sources into the compact verified-source set.

    ``SourceRegistryMiddleware`` already *captures* every source a tool result yields, but
    ``get_verified_sources`` defaults to **compact** mode, which returns only the subset carried
    forward by ``ResearchNotes``. In the tier design that was complete by construction: the
    orchestrator never held source tools on any path that also ran a batch. Here it holds all of
    them all the time, so a run that calls ``web_search_tool`` directly *and* runs a batch would
    silently drop the direct evidence from the compact set the moment the batch established a
    compact subset — and the finalizer would then strip any citation pointing at it.

    This middleware closes that: around every direct orchestrator source call it snapshots the
    registry and promotes whatever is new into the compact set, so compact
    ``get_verified_sources`` is the path-independent union of direct, delegated, and batched
    evidence.
    """

    def __init__(
        self,
        *,
        source_registry_middleware: SourceRegistryMiddleware,
        source_tool_names: set[str] | frozenset[str],
    ) -> None:
        """Scope promotion to the orchestrator's own source tools."""
        self._registry_middleware = source_registry_middleware
        self._source_tool_names = frozenset(source_tool_names)

    def _snapshot_keys(self) -> set[str]:
        """Return the comparable keys of every source currently in the active registry."""
        registry = self._registry_middleware.active_registry()
        keys: set[str] = set()
        for entry in registry.all_sources():
            key = SourceRegistryMiddleware._entry_key(entry)
            if key:
                keys.add(key)
        return keys

    async def awrap_tool_call(self, request, handler):
        """Promote sources newly captured by this direct source call into the compact set."""
        tool_call = getattr(request, "tool_call", None)
        name = tool_call.get("name") if isinstance(tool_call, dict) else None
        if name not in self._source_tool_names:
            return await handler(request)

        before = self._snapshot_keys()
        result = await handler(request)
        try:
            registry = self._registry_middleware.active_registry()
            promoted = [
                entry
                for entry in registry.all_sources()
                if (key := SourceRegistryMiddleware._entry_key(entry)) is not None and key not in before
            ]
            if promoted:
                # register_compact_sources re-adds through SourceRegistry.add, which dedupes by
                # normalized locator, so re-registering an already-present entry is a no-op
                # beyond marking it compact.
                count = self._registry_middleware.register_compact_sources(promoted)
                logger.info(
                    "[CitationRegistry] Promoted %d directly-retrieved source(s) from %s into the compact set",
                    count,
                    name,
                )
        except Exception:  # noqa: BLE001 - promotion is best-effort; never fail a good tool result
            logger.warning("Failed to promote directly-retrieved sources into the compact set", exc_info=True)
        return result


def _tool_message_from_result(result: object) -> ToolMessage | None:
    """Return the ``ToolMessage`` carried by a tool result, whether bare or inside a ``Command``.

    The deepagents ``task`` tool returns a ``Command`` whose ``update["messages"]`` holds the
    single ``ToolMessage`` the parent will see; ordinary tools return the message directly.
    """
    if isinstance(result, ToolMessage):
        return result
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], ToolMessage):
            return messages[0]
    return None


def _research_note_slug(text: str) -> str:
    """Return a compact filesystem-safe slug for a research note."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").lower()).strip("_")
    slug = slug[:_NOTE_SLUG_MAX_LENGTH].strip("_")
    return slug or "research_note"


class ResearcherTaskPersistenceMiddleware(AgentMiddleware):
    """Give ``task(researcher-agent)`` the evidence side effects ``run_research_batch`` has.

    ``researcher-agent`` is exposed twice in this design: as the runnable behind
    ``run_research_batch`` and as a ``task``-reachable subagent for a single topic that needs
    iterative, multi-hop investigation. Both use the same runnable and the same loop guards, but
    only the batch path persists notes and registers sources — ``task`` just hands the parent a
    ``ToolMessage`` and forgets. Without this wrapper, evidence gathered through the ``task``
    path would never reach ``/shared/`` (so the writer could not read it) and its locators would
    never enter the compact source set (so citations drawn from it would fail verification).

    The middleware therefore does what the batch tool does, for one note: validate the returned
    JSON as ``ResearchNotes``, persist it under a collision-safe ``/shared/research_note_*.json``
    path, and register its source locators as compact sources. Every step is best-effort — a
    malformed or oversized note degrades to "no persistence" and the orchestrator still receives
    the researcher's answer, because losing a bookkeeping side effect must never lose the
    research itself.
    """

    def __init__(
        self,
        *,
        backend: Any | None,
        state_budget: Any,
        resource_limits: Any,
        source_registry_middleware: SourceRegistryMiddleware | None = None,
    ) -> None:
        """Wire the shared filesystem backend, byte budgets, and the source registry."""
        self._backend = backend
        self._state_budget = state_budget
        self._limits = resource_limits
        self._registry_middleware = source_registry_middleware
        self._note_index = 0

    def _note_path(self, note: ResearchNotes, description: str) -> str:
        """Build a collision-safe ``/shared/`` path for one task-delegated research note.

        Unique by construction across both research paths: the ``task_`` infix keeps it out of
        the batch tool's numbering, the counter separates repeated ``task`` calls, and the digest
        of the delegation text separates concurrent ones.
        """
        self._note_index += 1
        digest = hashlib.sha1(description.encode("utf-8")).hexdigest()[:8]
        slug = _research_note_slug(note.query_topic or description)
        return f"/shared/research_note_task_{self._note_index:02d}_{slug}_{digest}.json"

    def _persist(self, note: ResearchNotes, description: str) -> str | None:
        """Persist one note and return its path, or ``None`` when persistence was skipped."""
        if self._backend is None:
            return None
        payload = json.dumps(note.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False).encode(
            "utf-8"
        )
        if len(payload) > self._limits.max_research_note_bytes:
            logger.warning(
                "Skipping persistence of a task-delegated research note: %d bytes exceeds the %d-byte per-note limit",
                len(payload),
                self._limits.max_research_note_bytes,
            )
            return None
        path = self._note_path(note, description)
        note_files = [(path, payload)]
        reservation = self._state_budget.reserve(note_files)
        try:
            responses = self._backend.upload_files(note_files)
        except Exception:
            self._state_budget.rollback(reservation)
            raise
        errors = [f"{r.path}: {r.error}" for r in responses if getattr(r, "error", None)]
        if errors:
            self._state_budget.rollback(reservation)
            raise RuntimeError(f"failed to persist task research note: {'; '.join(errors)}")
        return path

    async def awrap_tool_call(self, request, handler):
        """Persist and register the ``ResearchNotes`` a ``task(researcher-agent)`` call returned."""
        tool_call = getattr(request, "tool_call", None)
        if not isinstance(tool_call, dict) or tool_call.get("name") != TASK_TOOL:
            return await handler(request)
        args = tool_call.get("args")
        if not isinstance(args, dict) or args.get(_SUBAGENT_TYPE_ARG) != RESEARCHER_SUBAGENT:
            return await handler(request)

        result = await handler(request)
        message = _tool_message_from_result(result)
        if message is None or getattr(message, "status", None) == "error":
            return result

        try:
            note = ResearchNotes.model_validate_json(str(message.content))
        except Exception:  # noqa: BLE001 - a non-structured answer is still a usable answer
            logger.info(
                "task(%s) returned a non-ResearchNotes payload; skipping note persistence",
                RESEARCHER_SUBAGENT,
            )
            return result

        description = str(args.get(_TASK_DESCRIPTION_ARG) or "")
        try:
            path = self._persist(note, description)
        except Exception:  # noqa: BLE001 - never discard research because bookkeeping failed
            logger.warning("Failed to persist a task-delegated research note", exc_info=True)
            path = None

        if self._registry_middleware is not None:
            # Same method the batch path uses, on the run's registry instance, so a source is
            # compact-eligible regardless of which research path produced it.
            self._registry_middleware.register_research_note_sources([note])

        if path is not None:
            logger.info("Persisted task-delegated research note to %s", path)
            try:
                message.content = f"{message.content}\n\n[Research note persisted to {path}]"
            except Exception:  # noqa: BLE001 - annotation is cosmetic
                pass
        return result


class PlanBeforeWriterMiddleware(AgentMiddleware):
    """Reject ``task(writer-agent)`` until ``/shared/plan.json`` exists.

    Adaptivity in this design is emergent: the orchestrator decides for itself whether an answer
    needs the writer at all. But *once* it chooses writer publication, planner-first sequencing
    is not a preference — ``writer.j2`` reads ``answer_strategy.answer_type``,
    ``answer_strategy.title``, ``answer_strategy.required_components``, and ``constraints``
    straight out of ``/shared/plan.json``. Delegating without it produces a writer run with no
    output contract, which is a broken report rather than a cheaper one. The tier agent got this
    for free (its writer branch was reachable only through the planned pipeline); with the tier
    machinery gone, this middleware is the enforcement point.

    Plan availability is established two ways because neither alone is reliable: a completed
    ``task(planner-agent)`` observed by this middleware, or ``/shared/plan.json`` present in the
    state handed to the tool call (which also covers a plan carried in from a previous turn).
    """

    def __init__(self, *, plan_path: str = PLAN_PATH) -> None:
        """Track plan availability for one request."""
        self._plan_path = plan_path
        self._plan_seen = False

    def _plan_available(self, state: object) -> bool:
        """Return True when a plan exists, from either the observed planner run or state files."""
        if self._plan_seen:
            return True
        files = state.get("files") if isinstance(state, dict) else getattr(state, "files", None)
        if isinstance(files, dict) and files.get(self._plan_path):
            self._plan_seen = True
            return True
        return False

    @staticmethod
    def _blocked_result(tool_call: dict) -> ToolMessage:
        return ToolMessage(
            content=(
                f"writer-agent was not invoked: {PLAN_PATH} does not exist yet. writer-agent reads its output "
                "contract (answer_strategy.answer_type, .title, .required_components, and constraints) from that "
                "file and cannot produce a correctly-shaped report without it. Delegate to planner-agent first "
                'with task(subagent_type="planner-agent", ...), let it persist the plan, run the research it '
                "asks for, and only then delegate to writer-agent. If this answer does not warrant a plan, write "
                "it yourself and call submit_final_report instead."
            ),
            tool_call_id=tool_call.get("id", "plan-before-writer"),
            name=tool_call.get("name", TASK_TOOL),
            status="error",
        )

    async def awrap_tool_call(self, request, handler):
        """Gate writer delegation on plan existence and note successful planner runs."""
        tool_call = getattr(request, "tool_call", None)
        if not isinstance(tool_call, dict) or tool_call.get("name") != TASK_TOOL:
            return await handler(request)
        args = tool_call.get("args")
        subagent = args.get(_SUBAGENT_TYPE_ARG) if isinstance(args, dict) else None

        if subagent == WRITER_SUBAGENT and not self._plan_available(getattr(request, "state", None)):
            logger.warning("Blocked task(%s): %s has not been written yet", WRITER_SUBAGENT, self._plan_path)
            return self._blocked_result(tool_call)

        result = await handler(request)

        if subagent == PLANNER_SUBAGENT:
            message = _tool_message_from_result(result)
            if message is not None and getattr(message, "status", None) != "error":
                # PlanPersistenceMiddleware writes /shared/plan.json in the planner's after_agent
                # hook, so a planner run that returned without error has committed the plan.
                self._plan_seen = True
                logger.info("Planner run completed; writer delegation is now permitted")
        return result


# =================================================================================================
# Finalization: two valid exits, one guard
# =================================================================================================


class AutonomousFinalizationMiddleware(AgentMiddleware):
    """Give one bounded corrective turn when a run ends without committing *either* exit.

    This replaces upstream's ``RequiredWriterDelegationMiddleware`` on the orchestrator. That
    middleware treats a missing ``/shared/output.md`` as proof the run is unfinished, which is
    correct only when the writer pipeline is mandatory. Attaching it here would force every
    autonomous run through the writer and delete the inline exit outright — the single most
    important behaviour this architecture is meant to allow.

    The guard fires only when the orchestrator returns plain text with no tool calls **and**
    neither exit is committed: the writer never wrote ``/shared/output.md`` and
    ``submit_final_report`` never ran. One corrective turn is requested, phrased so the model
    picks whichever exit matches the work it already did, rather than being pushed toward the
    writer.

    After the retry budget is spent the run is allowed to end rather than raising. That is
    deliberate: unlike the deep researcher — which sits behind an intent classifier and always
    runs the writer — the autonomous agent handles greetings and capability questions itself, and
    ``AutonomousResearcherAgent._salvage_inline_report`` accepts a conversational final message as
    a last resort. Raising here would convert a recoverable answer into a request failure.
    """

    def __init__(
        self,
        *,
        tracker: AutonomousFinalReportCommitTracker,
        max_retries: int = 1,
    ) -> None:
        """Configure the dual-exit tracker and the bounded corrective-turn budget."""
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.tracker = tracker
        self.max_retries = max_retries
        self._retry_message = (
            "This run has not finished: no final report was committed. Do not perform or retry research. "
            "Finish now using whichever exit matches the work you already did:\n"
            "- If you wrote the answer yourself, call submit_final_report(markdown=<the complete final answer>, "
            "researched=<true if any source tool, researcher-agent delegation, or run_research_batch ran>).\n"
            f"- If writer-agent already wrote {FINAL_REPORT_STATE_PATHS[0]}, return only its completion marker.\n"
            "- If you delegated to writer-agent and it did not commit, delegate once more with the plan, research "
            "notes, verified sources, and explicit evidence gaps already available.\n"
            "Do not reply with plain text again without one of these."
        )

    def _retry_count(self, messages: list[object]) -> int:
        return sum(
            isinstance(message, HumanMessage)
            and message.additional_kwargs.get(_GENERATED_RETRY_MARKER) == "autonomous_finalization"
            for message in messages
        )

    def _check_after_model(self, state: object) -> dict[str, object] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        files = state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})
        if not isinstance(messages, list) or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or last_message.tool_calls:
            return None
        if self.tracker.any_exit_committed(files):
            return None
        if self._retry_count(messages) >= self.max_retries:
            logger.warning(
                "Orchestrator ended without committing either final-report exit after %d corrective turn(s); "
                "falling through to inline salvage",
                self.max_retries,
            )
            return None

        logger.warning("Orchestrator ended without committing a final report; requesting one corrective turn")
        return {
            "messages": [
                HumanMessage(
                    content=self._retry_message,
                    additional_kwargs={_GENERATED_RETRY_MARKER: "autonomous_finalization"},
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        """Verify synchronous dual-exit completion and request one local repair when needed."""
        return self._check_after_model(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        """Verify asynchronous dual-exit completion and request one local repair when needed."""
        return self._check_after_model(state)


# =================================================================================================
# Request-wide termination
# =================================================================================================


def _canonical_request_query_signature(query: object) -> str:
    """Hash a ResearchQuery for the *request-wide* duplicate ledger, ignoring ``depth``.

    This is deliberately not ``adaptive_researcher``'s ``_canonical_research_query_signature``,
    which folds ``depth`` into the hash. Two reasons, and the first is a correctness bug the
    shared helper would reintroduce here:

    1. This guard clamps ``high`` down to ``medium`` past the per-request allowance. If ``depth``
       were part of the signature, the same question re-sent at ``high`` in a later batch would be
       clamped, hash differently from its own earlier run, and execute again — the clamp itself
       would become a duplicate bypass.
    2. Asking the same question harder is still asking the same question. Re-running a query for
       "more depth" is exactly the behavior the research loop tells the orchestrator not to do,
       so it should collide rather than count as new work.

    Everything else matches the shared helper: normalized query text, ordered normalized
    subqueries (order is meaningful), sorted target components, sorted preferred tools. Free-form
    ``rationale`` and ``fallback_tools`` are omitted so padding a query cannot bypass detection.
    Accepts a dict (raw LLM tool args) or a Pydantic model. Only the hash is retained.
    """

    def _get(field: str, default: object) -> object:
        if isinstance(query, dict):
            return query.get(field, default)
        return getattr(query, field, default)

    canonical = {
        "query": _normalize_text(_get("query", "")),
        "subqueries": [_normalize_text(s) for s in (_get("subqueries", []) or [])],
        "target_components": sorted(_normalize_text(c) for c in (_get("target_components", []) or [])),
        "preferred_tools": sorted(_normalize_text(t) for t in (_get("preferred_tools", []) or [])),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_direct_source_signature(tool_name: str, args: object) -> str:
    """Hash a direct source-tool call into a normalized, content-free signature.

    ``adaptive_researcher``'s ``_canonical_source_signature`` sorts JSON keys but leaves string
    *values* untouched, so ``"same"`` and ``"  SAME  "`` hash differently and both execute. That
    is tolerable inside one researcher invocation, where the budget is small and short-lived, but
    this ledger spans the whole request and exists specifically to stop an orchestrator re-issuing
    near-identical searches into its own context. Values are therefore normalized with the same
    ``_normalize_text`` (NFKC, whitespace-collapsed, casefolded) the query signature uses.

    Only the hash is retained — raw argument text is never kept.
    """

    def _norm(value: object) -> object:
        if isinstance(value, str):
            return _normalize_text(value)
        if isinstance(value, dict):
            return {str(key): _norm(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_norm(item) for item in value]
        return value

    payload = json.dumps(
        {"tool": _normalize_text(tool_name), "args": _norm(args)},
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AutonomousOrchestratorLoopGuardMiddleware(AgentMiddleware):
    """Bound the whole request: research batches, delegated queries, and orchestrator turns.

    Adapted from ``adaptive_researcher.custom_middleware.OrchestratorLoopGuardMiddleware`` with
    the tier machinery removed. The adaptive guard looked its budgets up per declared tier via
    ``budgets_for_tier()``, which returned ``None`` — i.e. *no request-wide guard at all* — for
    ``single_shot``, ``direct``, ``meta``, and the whole pre-declaration window. Here one flat
    budget set always applies, which is strictly safer as well as simpler.

    The per-researcher ``ResearcherLoopGuardMiddleware`` bounds one delegated invocation, but its
    state resets for every new invocation, so an orchestrator that keeps authoring fresh
    ``run_research_batch`` calls can run indefinitely while every per-researcher guard fires
    correctly. This middleware closes that gap at the orchestrator boundary.

    Lifetime and concurrency: exactly one instance is built per top-level request in
    ``build_autonomous_research_graph``, so request-scoped counters live safely on ``self``.
    Counters are mutated *before* awaiting the tool handler, so parallel batch calls in a single
    turn cannot race past a limit.

    Enforcement:

    - A batch that would exceed ``max_batch_calls`` or ``max_total_research_queries``, or that
      repeats a normalized query beyond ``max_identical_research_queries``, is **not executed** —
      a deterministic error ``ToolMessage`` is returned and the request enters ``finalizing``.
    - ``task(subagent_type="researcher-agent")`` spends the same ``max_batch_calls`` ceiling as a
      one-query batch. The two are the same capability behind two doors; budgeting only one of
      them leaves the other as a free escape hatch, including after the request has begun
      finalizing.
    - Queries past ``max_high_depth_queries`` are **clamped** from ``high`` to ``medium`` rather
      than rejected. Clamping loses no work and costs no corrective turn; rejecting would do both.
    - The orchestrator's own direct source-tool calls are capped at ``max_direct_source_calls``,
      with repeats of one normalized call signature capped at
      ``max_identical_direct_source_calls``. Spending this budget withdraws the source tools but
      deliberately does **not** finalize the request — see below.
    - Once finalizing (or once model turns exceed ``max_orchestrator_turns``),
      ``run_research_batch``, ``think``, and every direct source tool are withdrawn from later
      model calls so the orchestrator can only finalize from evidence already collected.

    Withdrawing the source tools is new relative to the adaptive guard, which did not need to:
    there the orchestrator held source tools only on the ``single_loop_single_shot`` path, where
    a separate per-tier budget capped them. Here it always holds them, so leaving them visible
    while "finalizing" would let it keep researching one direct call at a time.

    Two asymmetries in here are deliberate, and both would be easy to "fix" into a regression:

    1. **Exhausting the direct-search budget does not finalize the request.** Finalizing withdraws
       ``run_research_batch`` and ``think`` as well, which would push the model to answer when the
       intended response is to *delegate instead*. Direct search closes; research continues.
    2. **Direct-call duplicate detection is scoped to the direct path only.** A direct call that
       repeats a question a worker already researched is verification, which is precisely what the
       direct budget is reserved for. Only direct-against-direct repeats are blocked — the shape
       seen in the runaway trials, where the orchestrator re-issued near-identical searches into
       its own context.

    Logging is metadata-only (request tag, phase, counts, truncated hashed signature) — never raw
    query arguments.
    """

    def __init__(self, *, config: AutonomousRequestTerminationConfig, source_tool_names: frozenset[str]) -> None:
        """Create the request-scoped guard for one autonomous run."""
        self._config = config
        self._source_tool_names = frozenset(source_tool_names)
        # Short opaque per-request tag for correlating log lines without leaking content.
        self._request_tag = uuid4().hex[:12]
        self._phase: str = "active"
        self._exhaustion_reason: str | None = None
        self._batch_call_count = 0
        self._total_query_count = 0
        self._model_turn_count = 0
        self._query_signature_counts: dict[str, int] = {}
        # The orchestrator's own source-tool calls, budgeted separately from delegated research
        # because their results land in the parent conversation and are re-sent every later turn.
        self._direct_source_call_count = 0
        self._direct_source_signature_counts: dict[str, int] = {}
        # ``depth: "high"`` queries admitted so far; later ones are clamped, not rejected.
        self._high_depth_query_count = 0

    # --- introspection helpers (used by tests and by operators reading logs) ------------------

    @property
    def phase(self) -> str:
        """Return ``active``, ``finalizing``, or ``terminal`` for this request."""
        return self._phase

    @property
    def exhaustion_reason(self) -> str | None:
        """Return why the request transitioned out of ``active``, if it has."""
        return self._exhaustion_reason

    def _mark_finalizing(self, reason: str) -> None:
        if self._phase == "active":
            self._phase = "finalizing"
        self._exhaustion_reason = reason

    def _log_block(self, reason: str, *, signature: str | None = None, budget: int | None = None) -> None:
        logger.warning(
            "Autonomous loop guard blocked research | request=%s phase=%s reason=%s "
            "batches=%d queries=%d turns=%d limit=%s signature=%s",
            self._request_tag,
            self._phase,
            reason,
            self._batch_call_count,
            self._total_query_count,
            self._model_turn_count,
            budget if budget is not None else "-",
            signature[:12] if signature else "-",
        )

    @staticmethod
    def _blocked_result(tool_call: dict, message: str) -> ToolMessage:
        return ToolMessage(
            content=message,
            tool_call_id=tool_call.get("id", "autonomous-loop-guard"),
            name=tool_call.get("name", RUN_RESEARCH_BATCH_TOOL),
            status="error",
        )

    @staticmethod
    def _extract_queries(tool_call: dict) -> list[object]:
        args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
        if isinstance(args, dict):
            queries = args.get("queries")
            if isinstance(queries, list):
                return queries
        return []

    def _direct_budget_spent(self) -> bool:
        """Return True once the orchestrator has used its own direct source-call budget."""
        return self._direct_source_call_count >= self._config.max_direct_source_calls

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Withdraw the research affordances this request has spent the budget for.

        Two independent withdrawals, and the narrower one must not be widened into the other:

        - Finalizing withdraws everything research-related, leaving only the finalize path.
        - A spent *direct* budget withdraws only the source tools. ``run_research_batch`` and
          ``think`` stay, because the intended next move is to delegate rather than to answer.
        """
        if not self._config.enabled:
            return tools
        if self._phase != "active":
            hidden = {RUN_RESEARCH_BATCH_TOOL, THINK_TOOL, *self._source_tool_names}
        elif self._direct_budget_spent():
            hidden = set(self._source_tool_names)
        else:
            return tools
        return [tool for tool in tools if _request_tool_name(tool) not in hidden]

    def _maybe_force_finalize_on_turns(self) -> None:
        if self._model_turn_count > self._config.max_orchestrator_turns and self._phase == "active":
            self._mark_finalizing("orchestrator turn budget")
            self._log_block("orchestrator_turn_budget", budget=self._config.max_orchestrator_turns)

    def wrap_model_call(self, request, handler):
        """Count the turn, force finalize on turn overflow, and withdraw tools when finalizing."""
        if not self._config.enabled:
            return handler(request)
        self._model_turn_count += 1
        self._maybe_force_finalize_on_turns()
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(self, request, handler):
        """Async counterpart of ``wrap_model_call``."""
        if not self._config.enabled:
            return await handler(request)
        self._model_turn_count += 1
        self._maybe_force_finalize_on_turns()
        return await handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_tool_call(self, request, handler):
        """Route each tool call to the budget that governs it.

        Four paths, and the order matters only in that the two research doors must both be
        reached before the catch-all passthrough:

        - a direct source tool -> the orchestrator's own search budget;
        - ``task`` -> the shared research budget, but only for ``researcher-agent``;
        - ``run_research_batch`` -> the batch/query/duplicate budgets and the depth clamp;
        - anything else (``think``, ``get_verified_sources``, filesystem tools, the finalizer)
          -> untouched.
        """
        tool_call = getattr(request, "tool_call", None)
        if not self._config.enabled or not isinstance(tool_call, dict):
            return await handler(request)
        name = tool_call.get("name")
        if name in self._source_tool_names:
            return await self._guard_direct_source_call(name, tool_call, request, handler)
        if name == TASK_TOOL:
            return await self._guard_delegation(tool_call, request, handler)
        if name != RUN_RESEARCH_BATCH_TOOL:
            return await handler(request)
        return await self._guard_research_batch(tool_call, request, handler)

    async def _guard_direct_source_call(self, name, tool_call, request, handler):
        """Bound the source-tool calls the orchestrator makes itself, in its own context.

        This is the budget the eval identified as the dominant token lever: 480 direct calls
        against the adaptive arm's 34, with single trials reaching 61 and 62. Unlike a worker's
        search, a direct result stays in the parent conversation and is re-sent on every later
        turn, so the cost of one call compounds across the rest of the run.

        Spending this budget withdraws the source tools (``_filter_tools``) but leaves
        ``run_research_batch`` and ``task`` in place: the intended response is to delegate the
        remaining lookups, not to stop researching.
        """
        if self._phase != "active":
            self._log_block("already_finalizing")
            return self._blocked_result(
                tool_call,
                "Source research is closed: the request has reached its research budget and is finalizing. "
                "Call get_verified_sources and submit_final_report to write your final answer from the "
                "evidence already gathered; represent any missing components as explicit gaps.",
            )

        # --- Reserve before awaiting so a turn of parallel searches shares one ceiling. ---
        if self._direct_budget_spent():
            self._log_block("direct_source_budget", budget=self._config.max_direct_source_calls)
            return self._blocked_result(
                tool_call,
                f"Direct-search budget reached "
                f"({self._direct_source_call_count}/{self._config.max_direct_source_calls} calls). Your own "
                "source-tool calls are spent for this request, but research is not over. Delegate the "
                "remaining lookups through run_research_batch, or give a dependent chain to "
                'task(subagent_type="researcher-agent", ...). Finalize only once the evidence is sufficient.',
            )

        signature = _canonical_direct_source_signature(name, tool_call.get("args", {}))
        if self._direct_source_signature_counts.get(signature, 0) >= self._config.max_identical_direct_source_calls:
            self._log_block("duplicate_direct_source_call", signature=signature)
            return self._blocked_result(
                tool_call,
                "Duplicate direct search blocked: you have already run this exact search in this request, and "
                "repeating it returns the same results while doubling their cost in your context. Change the "
                "target rather than the wording — the source organization, a page that quotes the figure, or a "
                "mirror — or delegate the lookup through run_research_batch.",
            )

        self._direct_source_call_count += 1
        self._direct_source_signature_counts[signature] = self._direct_source_signature_counts.get(signature, 0) + 1
        logger.info(
            "Autonomous loop guard: request=%s direct_source=%d/%d turns=%d",
            self._request_tag,
            self._direct_source_call_count,
            self._config.max_direct_source_calls,
            self._model_turn_count,
        )
        result = await handler(request)

        # ``>=`` rather than ``==``: parallel calls in one turn can overshoot the budget together,
        # and every one of them should still carry the explanation.
        if self._direct_budget_spent():
            try:
                result = result.model_copy(update={"content": f"{result.content}{_DIRECT_SOURCE_BUDGET_NUDGE}"})
            except Exception:
                # Non-Pydantic or immutable result: the withdrawal in _filter_tools still enforces
                # the cap, so a missing nudge is a soft degradation rather than a failure.
                pass
        return result

    async def _guard_delegation(self, tool_call, request, handler):
        """Spend the shared research budget on ``task(researcher-agent)``.

        ``run_research_batch`` and ``task(researcher-agent)`` are one capability behind two doors.
        Budgeting only the batch left the delegation door free — including after the request had
        entered ``finalizing``, when every other research affordance is withdrawn but ``task``
        cannot be, because ``task(writer-agent)`` is one of the two ways a run legitimately ends.
        The gate therefore lives here, keyed on ``subagent_type``, rather than in ``_filter_tools``.
        """
        args = tool_call.get("args")
        subagent = args.get(_SUBAGENT_TYPE_ARG) if isinstance(args, dict) else None
        if subagent != RESEARCHER_SUBAGENT:
            # planner-agent and writer-agent do no research fan-out; writer-agent is an exit.
            return await handler(request)

        if self._phase != "active":
            self._log_block("already_finalizing")
            return self._blocked_result(
                tool_call,
                "Source research is closed: the request has reached its research budget and is finalizing. "
                "Do not delegate further research. Call get_verified_sources and submit_final_report to write "
                "your final answer from the evidence already gathered; represent missing components as gaps.",
            )

        # A researcher delegation is one research call carrying one question, so it spends the
        # batch ceiling and one query slot — the same cost as a one-query batch.
        if self._batch_call_count + 1 > self._config.max_batch_calls:
            self._mark_finalizing("research batch-call budget")
            self._log_block("batch_call_budget", budget=self._config.max_batch_calls)
            return self._blocked_result(
                tool_call,
                f"Research budget reached ({self._batch_call_count}/{self._config.max_batch_calls} research "
                "calls, counting both run_research_batch and researcher-agent delegations). No further "
                "research will run. Call get_verified_sources and submit_final_report to finalize from the "
                "evidence already gathered; record unsupported requirements as gaps.",
            )
        if self._total_query_count + 1 > self._config.max_total_research_queries:
            self._mark_finalizing("total delegated-query budget")
            self._log_block("total_query_budget", budget=self._config.max_total_research_queries)
            return self._blocked_result(
                tool_call,
                f"Delegated-query budget reached ({self._config.max_total_research_queries} queries for this "
                "request). The delegation was not run. Call get_verified_sources and submit_final_report to "
                "finalize from the evidence already gathered; record unsupported requirements as gaps.",
            )

        self._batch_call_count += 1
        self._total_query_count += 1
        logger.info(
            "Autonomous loop guard: request=%s research=%d/%d (researcher-agent delegation) queries=%d/%d turns=%d",
            self._request_tag,
            self._batch_call_count,
            self._config.max_batch_calls,
            self._total_query_count,
            self._config.max_total_research_queries,
            self._model_turn_count,
        )
        return await handler(request)

    def _clamp_high_depth(self, queries: list[object]) -> tuple[list[object], int, int]:
        """Clamp ``high``-depth queries past the request allowance down to ``medium``.

        Returns the (possibly rewritten) query list, how many ``high`` queries this batch keeps,
        and how many were clamped. Deliberately pure with respect to ``self``: the running count
        is committed only once the batch is admitted, so a batch rejected by a later check does
        not silently consume the allowance.

        Clamping rather than rejecting is the point. ``high`` was declared on 42% of delegated
        queries with no aggregate F1 return, so the cost is real, but rejecting a batch over it
        would discard four good queries to correct one and spend a model turn saying so.

        Clamping is invisible to duplicate detection by design:
        ``_canonical_request_query_signature`` excludes ``depth``, so a rewritten query still
        hashes to the same value as its unclamped self. An earlier version signed the clamped
        query with ``depth`` included, which let the same ``high`` question re-run in a later
        batch as a "new" ``medium`` one — the clamp silently became a duplicate bypass.
        """
        allowance = self._config.max_high_depth_queries - self._high_depth_query_count
        kept = 0
        clamped = 0
        result: list[object] = []
        for query in queries:
            # Queries arrive as raw tool-call args (dicts) at middleware time; anything else is a
            # shape we do not recognize and must pass through untouched.
            if not isinstance(query, dict) or query.get("depth") != "high":
                result.append(query)
                continue
            if kept < allowance:
                kept += 1
                result.append(query)
                continue
            result.append({**query, "depth": "medium"})
            clamped += 1
        return result, kept, clamped

    async def _guard_research_batch(self, tool_call, request, handler):
        """Enforce request-wide batch, query, and duplicate budgets before a batch runs."""
        if self._phase != "active":
            self._log_block("already_finalizing")
            return self._blocked_result(
                tool_call,
                "Source research is closed: the request has reached its research budget and is finalizing. "
                "Do not call run_research_batch again. Use get_verified_sources and submit_final_report to "
                "write your final answer from the evidence already gathered; represent any missing "
                "components as explicit gaps.",
            )

        # Clamp order is not load-bearing for duplicate detection: the request-wide signature
        # excludes ``depth`` precisely so that clamping cannot change it (see
        # _canonical_request_query_signature).
        queries, high_kept, high_clamped = self._clamp_high_depth(self._extract_queries(tool_call))
        incoming = len(queries)

        # --- Count and check BEFORE awaiting the handler so concurrent batch calls in one turn
        # share one hard ceiling (no await between the checks and the increments). ---
        if self._batch_call_count + 1 > self._config.max_batch_calls:
            self._mark_finalizing("research batch-call budget")
            self._log_block("batch_call_budget", budget=self._config.max_batch_calls)
            return self._blocked_result(
                tool_call,
                f"Research batch budget reached ({self._batch_call_count}/{self._config.max_batch_calls} calls). "
                "No further research batches will run. Call get_verified_sources and submit_final_report to "
                "finalize from the evidence already gathered; record unsupported requirements as gaps.",
            )

        if incoming and self._total_query_count + incoming > self._config.max_total_research_queries:
            self._mark_finalizing("total delegated-query budget")
            self._log_block("total_query_budget", budget=self._config.max_total_research_queries)
            remaining = self._config.max_total_research_queries - self._total_query_count
            return self._blocked_result(
                tool_call,
                f"Delegated-query budget reached: this batch of {incoming} would exceed the remaining "
                f"{max(remaining, 0)} of {self._config.max_total_research_queries} queries for this request. The "
                "batch was not run. Call get_verified_sources and submit_final_report to finalize from the "
                "evidence already gathered; record unsupported requirements as gaps.",
            )

        # Count occurrences within this batch as well as against earlier batches. Checking the
        # whole list against the committed ledger first, and only then incrementing, let a batch
        # carrying the same query twice through: neither copy had been recorded yet when the
        # other was checked. Fanning two workers at one question is the exact waste this guards.
        signatures = [_canonical_request_query_signature(q) for q in queries]
        seen_in_batch: dict[str, int] = {}
        for signature in signatures:
            already_run = self._query_signature_counts.get(signature, 0) + seen_in_batch.get(signature, 0)
            if already_run >= self._config.max_identical_research_queries:
                self._mark_finalizing("repeated delegated-query signature")
                self._log_block("duplicate_query", signature=signature)
                return self._blocked_result(
                    tool_call,
                    "Duplicate research query blocked: this batch repeats a query it already contains, or one "
                    "this request has already researched. Asking the same question again — including at a "
                    "greater depth — will not surface new evidence. Call get_verified_sources and "
                    "submit_final_report to finalize; if a required period or component is unavailable in the "
                    "configured sources, state it as an explicit evidence gap instead of searching again.",
                )
            seen_in_batch[signature] = seen_in_batch.get(signature, 0) + 1

        # Reserve the budget atomically (still before the first await).
        self._batch_call_count += 1
        self._total_query_count += incoming
        self._high_depth_query_count += high_kept
        for signature in signatures:
            self._query_signature_counts[signature] = self._query_signature_counts.get(signature, 0) + 1
        logger.info(
            "Autonomous loop guard: request=%s batch=%d/%d queries=%d/%d high_depth=%d/%d turns=%d",
            self._request_tag,
            self._batch_call_count,
            self._config.max_batch_calls,
            self._total_query_count,
            self._config.max_total_research_queries,
            self._high_depth_query_count,
            self._config.max_high_depth_queries,
            self._model_turn_count,
        )
        if high_clamped:
            logger.info(
                "Autonomous loop guard: request=%s clamped %d query(ies) from depth=high to depth=medium "
                "(allowance %d per request already used)",
                self._request_tag,
                high_clamped,
                self._config.max_high_depth_queries,
            )
            args = tool_call.get("args")
            if isinstance(args, dict):
                request = request.override(tool_call={**tool_call, "args": {**args, "queries": queries}})
        return await handler(request)
