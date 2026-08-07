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

"""Adaptive-researcher-specific middleware.

The adaptive agent reuses all of ``deep_researcher.custom_middleware`` verbatim. This module
adds only the optional Layer-B enforcement — ``ComplexityRouterMiddleware`` — which hides
heavier tools from the orchestrator's model requests based on the statically-derived
enabled-tiers ceiling.

Layer A (the orchestrator prompt describing only the enabled tiers) is the primary
enforcement and is always on. Layer B is a belt-and-suspenders hardening that is wired only
when ``enforce_tier_tools=True`` (default off), honoring the POC's "only if eval shows drift"
guidance and keeping behaviour changes minimal for the first iteration.

When ``single_loop_single_shot=True``, the middleware also performs a dynamic tool swap after
``declare_effort_tier`` fires: for ``single_shot`` tiers it removes ``run_research_batch`` and
exposes direct source tools so the orchestrator can search inline; for all other tiers it hides
the source tools to prevent accidental direct calls.

The tier is captured via ``awrap_tool_call`` (intercepting the ``declare_effort_tier`` call
directly) rather than by scanning ``request.messages`` in ``awrap_model_call``. The messages
approach is unreliable because framework middleware (e.g. SummarizationMiddleware) that sit
earlier in the chain may transform or omit prior AIMessages by the time this middleware runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

# Reuse the tool-name reader from deep_researcher so tool shapes are handled identically.
from aiq_agent.agents.deep_researcher.custom_middleware import _request_tool_name
from aiq_agent.agents.deep_researcher.researcher_context import CURRENT_RESEARCHER_GUARD_STATE

from .models import AdaptiveRequestTerminationConfig
from .models import ResearcherLoopGuardConfig
from .subagents import MAX_SHALLOW_ATTEMPTS
from .subagents import SHALLOW_RESEARCHER_SUBAGENT
from .tiers import tier_ceiling
from .tools.finalize import EFFORT_TIER_PATH

logger = logging.getLogger(__name__)

_THINK_TOOL = "think"
_DEFAULT_MAX_CONSECUTIVE_THINKS = 3

# Default single_shot search budget: the maximum number of direct source-tool calls the
# orchestrator may make on the ``single_loop_single_shot`` ``single_shot`` path before it is
# forced to finalize. single_shot is meant to be a 1-3 query lookup, but the "1-3 queries"
# prompt guidance is soft and was observed being ignored (a bounded factual query ran 6+
# sequential searches, inflating every later prompt with re-sent retrieval context). This cap
# is the deterministic Layer-B backstop; it is configurable via ``single_shot_search_budget``.
_DEFAULT_SINGLE_SHOT_SEARCH_BUDGET = 2

# Appended (not overwritten) to the source-tool result that spends the last of the budget. We
# append so the retrieved evidence from that final allowed search is preserved — the model
# still needs it to synthesize — while being told, in-context, to stop searching and finalize.
# The tool-hiding in ``_filter_tools`` is the hard guarantee; this nudge explains the why.
_SINGLE_SHOT_BUDGET_NUDGE = (
    "\n\n[SYSTEM — single_shot search budget reached: you have used your allotted "
    "search calls. Do NOT search again (the search tools are now withdrawn). Call "
    "`get_verified_sources` to obtain the citation whitelist, then write your final cited "
    "Markdown answer and call `submit_final_report(markdown, researched=true, "
    'tier="single_shot")`. If the gathered evidence is incomplete, answer only what it '
    "supports and clearly note the gaps — do not keep searching for missing facts.]"
)

# Tools considered "heavier" than a given effort ceiling. When the deep-most enabled tier is
# below these thresholds, the corresponding tools are hidden from the orchestrator's model
# requests. These are the same knobs the POC's per-tier exposure table describes (§4.7 C).
_ADVANCED_WEB_SEARCH_TOOL = "advanced_web_search_tool"
_TASK_TOOL = "task"
_WRITE_TODOS_TOOL = "write_todos"
_DELEGATION_TOOLS = (_TASK_TOOL, _WRITE_TODOS_TOOL)
_RUN_RESEARCH_BATCH_TOOL = "run_research_batch"
_DECLARE_EFFORT_TIER_TOOL = "declare_effort_tier"
# Kept as a literal rather than imported from factory.py, which imports this module.
_FINALIZE_TOOL = "submit_final_report"
# Argument names of the DeepAgents ``task`` tool (``TaskToolSchema``).
_SUBAGENT_TYPE_ARG = "subagent_type"
_TASK_DESCRIPTION_ARG = "description"


def hidden_tools_for_ceiling(
    ceiling: str,
    *,
    allow_delegation: bool = False,
    allow_shallow_subagent: bool = False,
) -> set[str]:
    """Return the tool names to hide from the orchestrator for a given enabled-tiers ceiling.

    - ceiling below ``deep``  -> hide ``advanced_web_search_tool`` (deep-only heavy retrieval).
    - ceiling below ``standard`` (i.e. shallow-only: ``single_shot`` / ``direct``) -> also hide
      ``task`` and ``write_todos`` so the orchestrator cannot delegate or plan, unless
      ``allow_delegation`` preserves them for a parent-report delta request.

    ``allow_shallow_subagent`` keeps ``task`` visible (but still hides ``write_todos``) when the
    ``single_shot`` shallow sub-agent is wired in. Without it, a shallow-only fast lane
    (``enabled_tiers: [single_shot]``) — the most likely place to enable that feature — would hide
    the very tool the orchestrator needs to reach the sub-agent.

    Note: the orchestrator does not hold source tools directly (they live on the researcher and
    are reached via ``run_research_batch``), so hiding ``advanced_web_search_tool`` here is a
    no-op unless a future config binds it to the orchestrator; it is included so the ceiling
    logic is complete and ready if Layer-B exposure is extended to the researcher surface.
    """
    hidden: set[str] = set()
    if ceiling in ("direct", "single_shot", "standard"):
        hidden.add(_ADVANCED_WEB_SEARCH_TOOL)
    if ceiling in ("direct", "single_shot") and not allow_delegation:
        hidden.update(_DELEGATION_TOOLS)
        if allow_shallow_subagent:
            hidden.discard(_TASK_TOOL)
    return hidden


def _declared_tiers_in_current_tool_batch(state: object) -> set[str]:
    """Return tiers declared by ``declare_effort_tier`` calls in the tool batch being executed.

    ToolNode dispatches every tool call of one assistant turn together and their wrappers may run
    in any order, so a middleware that consults only its own cached tier can observe a *stale*
    tier when ``declare_effort_tier`` happens to be scheduled after its sibling. Reading the tool
    calls of the AIMessage currently being executed makes a same-turn ``declare_effort_tier`` +
    ``task`` (or + ``submit_final_report``) deterministic regardless of scheduling order.

    Only the last message is inspected — never history, which upstream middleware (e.g.
    SummarizationMiddleware) may have rewritten. Returns an empty set for an unrecognised state
    shape, so callers fall back to their cached tier rather than failing.
    """
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if not messages:
        return set()
    last_message = list(messages)[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return set()
    tiers: set[str] = set()
    for tool_call in tool_calls:
        name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
        if name != _DECLARE_EFFORT_TIER_TOOL:
            continue
        args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
        tier = args.get("tier") if isinstance(args, dict) else None
        if isinstance(tier, str) and tier.strip():
            tiers.add(tier.strip())
    return tiers


# ---------------------------------------------------------------------------------------------
# Catalog-mode tier resolution
# ---------------------------------------------------------------------------------------------
# Under catalog mode the orchestrator no longer spends a turn declaring its effort tier: it
# emits ``declare_effort_tier`` *alongside* the first action of the chosen tier (or, for the
# terminal direct/meta paths, a lone ``submit_final_report`` carrying ``tier=``). That means the
# tier must be derived from a whole tool-call batch, deterministically, before any sibling call
# executes — hence one resolver shared by every tier-aware middleware.
#
# See misc/adaptive-orchestrator-skip-tier-turn-plan.md (§2, §9) for the design and caveats.

_PLANNER_SUBAGENT = "planner-agent"
_SOURCE_ROUTER_SUBAGENT = "source-router-agent"
_WRITER_SUBAGENT = "writer-agent"

# Coarse "what is this call trying to do" labels. Tier inference and the compatibility matrix
# both key on these rather than on raw tool names, so a config that renames or swaps a source
# tool does not need a new rule.
_KIND_FINALIZE_DIRECT = "finalize_direct"
_KIND_FINALIZE_META = "finalize_meta"
_KIND_FINALIZE_RESEARCHED = "finalize_researched"
_KIND_SOURCE_TOOL = "source_tool"
_KIND_SHALLOW_TASK = "shallow_task"
_KIND_PLANNER_TASK = "planner_task"
_KIND_WRITER_TASK = "writer_task"
_KIND_RESEARCH_BATCH = "research_batch"
_KIND_TODOS = "todos"

# Kinds that constitute "real work". A batch containing none of these (only think / file reads /
# get_verified_sources / a bare declaration) must never establish a tier by itself.
_SUBSTANTIVE_KINDS = frozenset(
    {
        _KIND_FINALIZE_DIRECT,
        _KIND_FINALIZE_META,
        _KIND_FINALIZE_RESEARCHED,
        _KIND_SOURCE_TOOL,
        _KIND_SHALLOW_TASK,
        _KIND_PLANNER_TASK,
        _KIND_WRITER_TASK,
        _KIND_RESEARCH_BATCH,
    }
)

# The normal effort tiers, least to most effort. Mirrors tiers._TIER_ORDER; duplicated as a local
# tuple so this module does not depend on a private name.
_NORMAL_TIER_ORDER: tuple[str, ...] = ("direct", "single_shot", "standard", "deep")
# Terminal, non-research pseudo-tier for chit-chat / capability answers. Never rank-compared.
_META_TIER = "meta"


@dataclass(frozen=True)
class BatchTierDecision:
    """The single, immutable tier decision for one assistant tool-call batch.

    ``source`` records how the tier was established, which drives two behaviours: an
    ``inferred`` tier is persisted by the resolver (no ``declare_effort_tier`` ran, so nothing
    else would write ``/shared/effort_tier.json``), and it is logged at WARNING so evals can
    report how often the model skipped its declaration. ``error`` being set means no sibling
    substantive call in this batch may execute.
    """

    tier: str | None
    source: str  # "declared" | "cached" | "inferred" | "unresolved"
    error: str | None = None


def _tool_call_name_args(tool_call: object) -> tuple[str | None, dict]:
    """Return ``(name, args)`` from a tool call in either dict or attribute form."""
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        args = tool_call.get("args")
    else:
        name = getattr(tool_call, "name", None)
        args = getattr(tool_call, "args", None)
    return name, args if isinstance(args, dict) else {}


def _last_message(state: object):
    """Return the last message of ``state``, or ``None`` for an unrecognised/empty state.

    Only the last message is ever inspected. History is unreliable here: middleware earlier in
    the stack (e.g. ``SummarizationMiddleware``) may have rewritten or dropped prior AIMessages
    by the time this code runs.
    """
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if not messages:
        return None
    return list(messages)[-1]


class TierResolver:
    """Resolve the effort tier for a whole tool-call batch, once, for every middleware.

    ``ToolNode`` dispatches all tool calls of one assistant turn together and their wrappers may
    run in any order. A middleware that caches the tier from its own ``awrap_tool_call`` can
    therefore observe a *stale* tier when ``declare_effort_tier`` happens to be scheduled after
    its sibling. This resolver removes that ordering dependence: it reads every sibling call of
    the current AIMessage, computes one ``BatchTierDecision``, and memoizes it by message
    identity, so whichever wrapper asks first fixes the answer for all of them.

    Resolution priority:

    1. exactly one valid ``declare_effort_tier`` in the current batch;
    2. the tier already accepted on an earlier turn;
    3. inference from the complete current action batch (the fallback for models that will not
       emit parallel tool calls).

    Instances are per-request (built in ``build_adaptive_research_graph``), so mutable state on
    ``self`` is concurrency-safe, and all mutation happens before the first ``await`` in the
    calling middleware — the same atomicity rule the budget guards use.
    """

    def __init__(
        self,
        *,
        enabled_tiers: list[str] | None,
        single_loop_single_shot: bool = False,
        shallow_mode: bool = False,
        direct_source_tool_names: frozenset[str] | set[str] = frozenset(),
        backend: object | None = None,
    ) -> None:
        self._enabled: tuple[str, ...] = tuple(t for t in _NORMAL_TIER_ORDER if t in (enabled_tiers or ()))
        if not self._enabled:
            self._enabled = _NORMAL_TIER_ORDER
        self._single_loop = single_loop_single_shot
        self._shallow_mode = shallow_mode
        self._source_tool_names = frozenset(direct_source_tool_names)
        self._backend = backend
        self._tier: str | None = None
        self._decisions: dict[object, BatchTierDecision] = {}
        self._persisted = False

    # --- public surface ----------------------------------------------------------------------

    @property
    def tier(self) -> str | None:
        """The tier accepted so far for this request, or ``None`` before the first batch."""
        return self._tier

    @property
    def enabled_tiers(self) -> tuple[str, ...]:
        return self._enabled

    def decide(self, state: object) -> BatchTierDecision:
        """Return the memoized decision for the batch currently being executed."""
        message = _last_message(state)
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if not tool_calls:
            # No batch to read (unusual shape, or a model call rather than a tool call): fall
            # back to whatever has already been accepted rather than failing the run.
            return BatchTierDecision(self._tier, "cached" if self._tier else "unresolved")
        key = getattr(message, "id", None) or id(message)
        cached = self._decisions.get(key)
        if cached is not None:
            return cached
        decision = self._resolve(tool_calls)
        self._decisions[key] = decision
        return decision

    def commit(self, decision: BatchTierDecision) -> None:
        """Accept a decision. Idempotent, so every sibling wrapper may call it."""
        if decision.error or decision.tier is None:
            return
        if decision.tier == self._tier:
            return
        self._tier = decision.tier
        if decision.source == "inferred":
            logger.warning(
                "TierResolver: no declare_effort_tier in the batch — tier inferred as %s. "
                "The model is not honouring the co-declaration contract.",
                decision.tier,
            )
            self._persist(decision.tier)
        else:
            logger.debug("TierResolver: tier=%s (%s)", decision.tier, decision.source)

    # --- internals ---------------------------------------------------------------------------

    def _persist(self, tier: str) -> None:
        """Write an inferred tier to ``/shared/effort_tier.json`` so ``run()`` can report it.

        ``declare_effort_tier`` already persists explicitly declared tiers; only the inference
        fallback would otherwise leave the file missing. Best-effort: a failed write must not
        break a run that is otherwise proceeding correctly.
        """
        if self._backend is None:
            return
        try:
            self._backend.upload_files([(EFFORT_TIER_PATH, json.dumps({"tier": tier}).encode("utf-8"))])
            self._persisted = True
        except Exception:  # pragma: no cover - observability only
            logger.warning("TierResolver: could not persist inferred tier %s", tier, exc_info=True)

    def _rank(self, tier: str | None) -> int:
        return _NORMAL_TIER_ORDER.index(tier) if tier in _NORMAL_TIER_ORDER else -1

    def _lowest_enabled(self, candidates: tuple[str, ...]) -> str | None:
        for tier in _NORMAL_TIER_ORDER:
            if tier in candidates and tier in self._enabled:
                return tier
        return None

    def _highest_enabled(self, candidates: tuple[str, ...]) -> str | None:
        for tier in reversed(_NORMAL_TIER_ORDER):
            if tier in candidates and tier in self._enabled:
                return tier
        return None

    def classify(self, name: str | None, args: dict) -> str | None:
        """Map one tool call to an action kind, or ``None`` when it implies nothing about tier."""
        if name is None:
            return None
        if name == _FINALIZE_TOOL:
            if not args.get("researched", True):
                return _KIND_FINALIZE_META if args.get("tier") == _META_TIER else _KIND_FINALIZE_DIRECT
            return _KIND_FINALIZE_RESEARCHED
        if name == _RUN_RESEARCH_BATCH_TOOL:
            return _KIND_RESEARCH_BATCH
        if name == _WRITE_TODOS_TOOL:
            return _KIND_TODOS
        if name == _TASK_TOOL:
            subagent = args.get(_SUBAGENT_TYPE_ARG)
            if subagent == SHALLOW_RESEARCHER_SUBAGENT:
                return _KIND_SHALLOW_TASK
            if subagent == _WRITER_SUBAGENT:
                return _KIND_WRITER_TASK
            if subagent in (_PLANNER_SUBAGENT, _SOURCE_ROUTER_SUBAGENT):
                return _KIND_PLANNER_TASK
            return _KIND_PLANNER_TASK
        if name in self._source_tool_names:
            return _KIND_SOURCE_TOOL
        # think / file tools / get_verified_sources / declare_effort_tier: no tier implication.
        return None

    def _implied_tier(self, kind: str) -> tuple[str | None, str | None]:
        """Return ``(tier, error)`` implied by one action kind under the current execution mode."""
        if kind == _KIND_FINALIZE_META:
            return _META_TIER, None
        if kind == _KIND_FINALIZE_DIRECT:
            if "direct" in self._enabled:
                return "direct", None
            return None, (
                "The `direct` tier is not enabled in this configuration. Use the lowest enabled "
                "research tier and verify the answer instead of answering from memory."
            )
        if kind == _KIND_SOURCE_TOOL:
            if self._single_loop and "single_shot" in self._enabled:
                return "single_shot", None
            return None, (
                "Source tools are not callable directly on this path. Delegate research through "
                "`run_research_batch` instead."
            )
        if kind == _KIND_SHALLOW_TASK:
            if self._shallow_mode and "single_shot" in self._enabled:
                return "single_shot", None
            return None, "The shallow-researcher subagent is not available in this configuration."
        if kind == _KIND_RESEARCH_BATCH:
            # Never infer `single_shot` while a tier with real request budgets is enabled:
            # budgets_for_tier() returns None for single_shot, which makes the request-wide
            # loop guard completely inert (no batch cap, no query cap, no turn cap). Guessing
            # low here would remove every research bound in exactly the situation where the
            # model is already misbehaving. See plan doc §9.1.
            guarded = self._lowest_enabled(("standard", "deep"))
            if guarded is not None:
                return guarded, None
            if self._single_loop or self._shallow_mode:
                # A fast lane owns single_shot, and neither variant reaches research through
                # run_research_batch — so with standard/deep disabled there is no tier that can
                # legally perform this call.
                return None, (
                    "`run_research_batch` is not available in this configuration: the single_shot "
                    + (
                        "path delegates to the shallow-researcher subagent."
                        if self._shallow_mode
                        else "path calls source tools directly."
                    )
                )
            if "single_shot" in self._enabled:
                # Only reachable when single_shot is the sole research tier *and* it researches
                # through run_research_batch, where the guard would have been inert regardless.
                return "single_shot", None
            return None, "No research-capable tier is enabled in this configuration."
        if kind == _KIND_PLANNER_TASK:
            # A planner / source-router hand-off means the planned writer pipeline is starting.
            # Map it to the deepest enabled writer tier so the run gets the more generous
            # budgets: truncating a genuine `deep` run mid-research is the worse failure.
            ceiling = self._highest_enabled(("standard", "deep"))
            if ceiling is not None:
                return ceiling, None
            return None, "Planning and delegation are not available under the enabled tiers."
        if kind == _KIND_WRITER_TASK:
            return None, (
                "writer-agent cannot be the first action: it requires `/shared/plan.json` and the "
                "planned research notes. Run the Planned Writer Pipeline in order."
            )
        # _KIND_TODOS and _KIND_FINALIZE_RESEARCHED imply nothing on their own.
        return None, None

    def _allowed(self, kind: str, tier: str) -> bool:
        """Return True when ``kind`` is a legal action for an already-resolved ``tier``."""
        if kind in (_KIND_FINALIZE_DIRECT, _KIND_FINALIZE_META, _KIND_FINALIZE_RESEARCHED):
            # Finalizing is always structurally legal; the finalize protocol and the shallow
            # delegation middleware police *what* may be finalized.
            return True
        if tier == _META_TIER:
            return False
        if tier == "direct":
            return False
        if tier == "single_shot":
            if self._shallow_mode:
                return kind == _KIND_SHALLOW_TASK
            if self._single_loop:
                return kind == _KIND_SOURCE_TOOL
            return kind in (_KIND_RESEARCH_BATCH, _KIND_TODOS)
        # standard / deep
        if kind in (_KIND_SOURCE_TOOL, _KIND_SHALLOW_TASK):
            return False
        return kind in (_KIND_RESEARCH_BATCH, _KIND_PLANNER_TASK, _KIND_WRITER_TASK, _KIND_TODOS)

    def _promotable(self, kind: str, tier: str) -> bool:
        """Return True when a tier/action mismatch may be absorbed as an upward escalation.

        Two mismatches are deliberately *not* promotable, because promoting would silently
        convert the run into a different (more expensive) shape rather than making the model
        correct itself:

        - a fast-lane ``single_shot`` calling ``run_research_batch`` — the fast lane exists
          precisely to avoid the delegated researcher loop;
        - ``standard``/``deep`` reaching for a direct source tool or the shallow sub-agent —
          those belong to the single-shot lane only.
        """
        if tier == "single_shot" and kind == _KIND_RESEARCH_BATCH and (self._single_loop or self._shallow_mode):
            return False
        if tier in ("standard", "deep") and kind in (_KIND_SOURCE_TOOL, _KIND_SHALLOW_TASK):
            return False
        return True

    def _infer(self, kinds: list[str]) -> tuple[str | None, str | None]:
        """Infer a tier from the complete batch of substantive action kinds."""
        implied: list[str] = []
        for kind in kinds:
            tier, error = self._implied_tier(kind)
            if error:
                return None, error
            if tier is not None:
                implied.append(tier)
        distinct = set(implied)
        if not distinct:
            return None, None
        if len(distinct) == 1:
            return implied[0], None
        if distinct <= {"standard", "deep"}:
            # e.g. write_todos + planner hand-off: same pipeline, take the deeper reading.
            return self._highest_enabled(tuple(distinct)), None
        return None, (
            "These actions belong to different effort levels and cannot run in one turn. Take the "
            "first step of a single effort level, and declare it with `declare_effort_tier`."
        )

    def _resolve(self, tool_calls: list) -> BatchTierDecision:
        """Compute the decision for one batch: declaration, then cache, then inference."""
        declared: set[str] = set()
        kinds: list[str] = []
        for call in tool_calls:
            name, args = _tool_call_name_args(call)
            if name == _DECLARE_EFFORT_TIER_TOOL:
                tier = args.get("tier")
                if isinstance(tier, str) and tier.strip():
                    declared.add(tier.strip())
                continue
            kind = self.classify(name, args)
            if kind is not None:
                kinds.append(kind)

        substantive = [k for k in kinds if k in _SUBSTANTIVE_KINDS]

        if len(declared) > 1:
            return BatchTierDecision(
                None,
                "unresolved",
                "Conflicting effort tiers were declared in one turn. Declare exactly one tier.",
            )

        tier: str | None = None
        source = "unresolved"
        if declared:
            tier = next(iter(declared))
            error = self._validate_declaration(tier, substantive)
            if error:
                return BatchTierDecision(None, "unresolved", error)
            source = "declared"
        elif self._tier is not None:
            tier, source = self._tier, "cached"
        elif substantive:
            tier, error = self._infer(kinds)
            if error:
                return BatchTierDecision(None, "unresolved", error)
            source = "inferred" if tier is not None else "unresolved"

        if tier is None:
            if substantive:
                return BatchTierDecision(
                    None,
                    "unresolved",
                    "The effort tier for this request is not established. Call "
                    "`declare_effort_tier(tier=...)` together with the first action of that tier.",
                )
            # Only helpers ran (think / file reads / a bare declaration): nothing to resolve yet.
            return BatchTierDecision(None, "unresolved")

        return self._check_actions(tier, source, kinds)

    def _validate_declaration(self, tier: str, substantive: list[str]) -> str | None:
        """Validate an explicit declaration against the enabled set and the run's history."""
        if tier == _META_TIER:
            if all(k in (_KIND_FINALIZE_META, _KIND_FINALIZE_DIRECT) for k in substantive) and substantive:
                return None
            return (
                'tier="meta" is only valid on the No-Research Meta / Capability Path, which ends '
                'immediately with `submit_final_report(..., researched=false, tier="meta")`.'
            )
        if tier not in _NORMAL_TIER_ORDER:
            return f"Unknown effort tier {tier!r}. Choose one of the enabled levels: {', '.join(self._enabled)}."
        if tier not in self._enabled:
            return f"The {tier!r} tier is not enabled in this configuration. Choose one of: {', '.join(self._enabled)}."
        if self._tier is not None and self._tier != _META_TIER and self._rank(tier) < self._rank(self._tier):
            return (
                f"Cannot downgrade from {self._tier!r} to {tier!r} mid-run. Continue on the current "
                "level, or escalate to a higher enabled level."
            )
        return None

    def _check_actions(self, tier: str, source: str, kinds: list[str]) -> BatchTierDecision:
        """Verify every action in the batch against the resolved tier, promoting where allowed."""
        resolved = tier
        for kind in kinds:
            if self._allowed(kind, resolved):
                continue
            if not self._promotable(kind, resolved):
                return BatchTierDecision(None, "unresolved", self._incompatible_message(kind, resolved))
            implied, error = self._implied_tier(kind)
            if error:
                return BatchTierDecision(None, "unresolved", error)
            if implied is None or implied == _META_TIER or implied not in self._enabled:
                return BatchTierDecision(None, "unresolved", self._incompatible_message(kind, resolved))
            if self._rank(implied) <= self._rank(resolved):
                return BatchTierDecision(None, "unresolved", self._incompatible_message(kind, resolved))
            logger.info(
                "TierResolver: promoting %s -> %s (action %s implies a higher level)",
                resolved,
                implied,
                kind,
            )
            resolved = implied
        return BatchTierDecision(resolved, source if resolved == tier else "inferred")

    def _incompatible_message(self, kind: str, tier: str) -> str:
        """Return the corrective text for an action that the resolved tier cannot perform."""
        if kind == _KIND_RESEARCH_BATCH and tier == "single_shot":
            if self._shallow_mode:
                return (
                    "`run_research_batch` is not available on the single_shot path. Delegate to the "
                    "shallow-researcher subagent with `task`, or declare a higher tier first."
                )
            return (
                "`run_research_batch` is not available on the single_shot path. Call the source "
                "tools directly, or declare `standard`/`deep` before delegating research."
            )
        if kind == _KIND_SOURCE_TOOL:
            return (
                f"Source tools cannot be called directly on the {tier!r} path. Name them in each "
                "`ResearchQuery.preferred_tools` and delegate through `run_research_batch`."
            )
        if kind == _KIND_SHALLOW_TASK:
            return "shallow-researcher is available only on the single_shot tier; follow the declared tier's workflow."
        return (
            f"That action is not part of the {tier!r} workflow. Follow the procedure for the tier "
            "you selected, or declare a higher enabled tier before using it."
        )


class ComplexityRouterMiddleware(AgentMiddleware):
    """Hide heavier tools from the orchestrator based on the enabled-tiers ceiling (Layer B).

    Wired when ``enforce_tier_tools=True`` or ``single_loop_single_shot=True``. With both off
    it is never attached, so the agent behaves exactly as the prompt-driven Layer-A design
    intends. The ceiling comes from static ``enabled_tiers``; a parent-report request may
    preserve delegation tools because its citation-safe writer workflow is mandatory.

    When ``single_loop_single_shot=True`` and ``direct_source_tools`` are provided, the
    middleware performs an additional dynamic swap keyed on the declared effort tier:

    - Before ``declare_effort_tier`` fires (tier unknown): hide source tools, keep
      ``run_research_batch`` so the orchestrator's first turn has a research path.
    - After ``declare_effort_tier(tier="single_shot")`` fires: expose source tools, remove
      ``run_research_batch`` — the orchestrator searches inline from its own loop.
    - After any other tier is declared: hide source tools, keep ``run_research_batch`` — the
      two-loop architecture is preserved for ``standard`` / ``deep``.

    Without a ``tier_resolver`` (the legacy, declaration-first path) the tier is captured via
    ``awrap_tool_call`` by intercepting the ``declare_effort_tier`` execution rather than by
    scanning ``request.messages``. The messages-scan approach is unreliable because framework
    middleware earlier in the stack (e.g. SummarizationMiddleware) may have transformed or
    omitted prior AIMessages by the time this middleware's ``awrap_model_call`` runs. Each
    middleware instance is created per request (in ``build_adaptive_research_graph`` called from
    ``AdaptiveResearcherAgent.run``), so caching the tier on ``self`` is safe with concurrent
    requests.

    With a ``tier_resolver`` (catalog mode) the tier instead comes from the shared
    ``TierResolver``, which resolves the *whole* tool batch at once. That is what lets the
    orchestrator declare its tier and take that tier's first action in the same turn: whichever
    wrapper runs first, every middleware sees the same decision. This middleware is also the
    enforcement point for that decision — a batch the resolver rejects has each of its
    substantive calls replaced by a corrective error result.

    When a ``prompt_renderer`` is supplied (``dynamic_orchestrator_sections=True``), the
    middleware also performs a dynamic *prompt* swap: the graph is built with the turn-1 catalog
    system prompt, and once a tier is resolved this middleware replaces the system message on
    every subsequent model call with ``prompt_renderer(tier)`` — the prompt trimmed to just that
    tier's sections. Renders are memoized per tier so the swapped prompt is byte-stable across a
    run's model calls (KV-cache friendly) and each tier renders at most once. Escalating to a
    higher tier simply renders and caches that tier's larger prompt.

    Finally, on the ``single_loop_single_shot`` ``single_shot`` path it enforces a **search
    budget**: it counts direct source-tool calls (in ``awrap_tool_call``) and, once
    ``single_shot_search_budget`` calls have been made, (a) appends a corrective nudge to the
    result of the search that spent the budget and (b) withdraws the source tools from every
    later model call (in ``_filter_tools``) so the model can only call ``get_verified_sources`` /
    ``submit_final_report``. This turns the soft "1-3 queries" prompt guidance into a hard cap,
    the dominant token-cost lever for cheap lookups. The budget is scoped to ``single_shot`` and
    never affects ``standard`` / ``deep``, which research through ``run_research_batch``.
    """

    def __init__(
        self,
        *,
        enabled_tiers: list[str] | None,
        allow_delegation: bool = False,
        direct_source_tools: list[object] | None = None,
        single_loop_single_shot: bool = False,
        single_shot_search_budget: int = _DEFAULT_SINGLE_SHOT_SEARCH_BUDGET,
        prompt_renderer: Callable[[str], str] | None = None,
        shallow_subagent_capture: object | None = None,
        tier_resolver: TierResolver | None = None,
    ) -> None:
        """Compute hidden tools, preserving the citation-safe delta writer path when needed."""
        # The shallow sub-agent is reached through `task`, so a shallow-only ceiling must not hide
        # it. Typed loosely (``object``) to avoid importing the subagents package here — factory.py
        # imports both, and this module must stay importable on its own.
        self._shallow_capture = shallow_subagent_capture
        self._hidden_tool_names = hidden_tools_for_ceiling(
            tier_ceiling(enabled_tiers),
            allow_delegation=allow_delegation,
            allow_shallow_subagent=shallow_subagent_capture is not None,
        )
        self._direct_source_tool_names: frozenset[str] = frozenset(
            name for t in (direct_source_tools or []) if (name := _request_tool_name(t)) is not None
        )
        self._single_loop_single_shot = single_loop_single_shot
        # Hard cap on single_shot direct source-tool calls before finalize is forced.
        self._search_budget = single_shot_search_budget
        # Populated by awrap_tool_call the moment declare_effort_tier executes.
        self._declared_tier: str | None = None
        # Running count of direct source-tool calls made on the single_shot path. Per-request
        # (instances are built per run in build_adaptive_research_graph), so mutating it on self
        # is concurrency-safe — same rationale as _declared_tier.
        self._source_call_count = 0
        # Per-tier prompt swap (opt-in). None -> tools-only behavior, no system-prompt swap.
        self._prompt_renderer = prompt_renderer
        # Memoize rendered prompts per tier: byte-stable across turns, and render each tier once.
        self._rendered_prompt_cache: dict[str, str] = {}
        # Catalog mode (opt-in): the shared batch resolver replaces this middleware's own tier
        # cache and gates the batch. None -> legacy declaration-first behavior, unchanged.
        self._resolver = tier_resolver

    @property
    def _tier(self) -> str | None:
        """The run's effort tier: from the shared resolver in catalog mode, else the local cache."""
        if self._resolver is not None:
            return self._resolver.tier
        return self._declared_tier

    async def awrap_tool_call(self, request, handler):
        """Resolve the batch's tier, gate incompatible calls, and enforce the search budget.

        Three responsibilities, in order:

        1. **Tier.** In catalog mode the shared resolver decides the tier for the whole tool
           batch; a batch it rejects has every substantive sibling blocked with the same
           corrective message. Outside catalog mode the legacy behavior is kept: cache the tier
           when ``declare_effort_tier`` fires.
        2. **Search budget.** On the ``single_loop_single_shot`` ``single_shot`` path, reserve a
           budget slot *before* awaiting the handler and refuse the call outright once the budget
           is spent. Reserving up-front is what makes the cap hard for a batch of parallel
           searches issued in one turn — counting afterwards would let the whole batch through.
        3. **Nudge.** The search that spends the last slot gets a corrective note appended to its
           result, so the model is told in-context why the search tools are about to vanish; the
           tool withdrawal in ``_filter_tools`` remains the guarantee.
        """
        tool_call = getattr(request, "tool_call", None)
        name = None
        if tool_call is not None:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)

        if self._resolver is not None:
            decision = self._resolver.decide(getattr(request, "state", None))
            if decision.error and self._is_substantive(name, tool_call):
                logger.info("ComplexityRouterMiddleware: blocked %s — %s", name, decision.error)
                return self._blocked(tool_call, decision.error)
            self._resolver.commit(decision)
        elif name == _DECLARE_EFFORT_TIER_TOOL:
            args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
            if isinstance(args, dict) and args.get("tier"):
                self._declared_tier = args["tier"]
                logger.debug("ComplexityRouterMiddleware: declared tier = %s", self._declared_tier)

        # Reserve the budget slot before running the call (and before any await) so parallel
        # searches in one turn share one hard ceiling.
        is_budgeted_search = (
            self._single_loop_single_shot
            and self._tier == "single_shot"
            and name is not None
            and name in self._direct_source_tool_names
        )
        if is_budgeted_search:
            if self._source_call_count >= self._search_budget:
                logger.info(
                    "ComplexityRouterMiddleware: refused source call beyond the single_shot search budget (%d/%d)",
                    self._source_call_count,
                    self._search_budget,
                )
                return self._blocked(
                    tool_call,
                    "single_shot search budget exhausted: no further searches will run. Call "
                    "`get_verified_sources`, then finalize with "
                    '`submit_final_report(markdown, researched=true, tier="single_shot")`, '
                    "noting any evidence gaps.",
                )
            self._source_call_count += 1

        result = await handler(request)

        # If this search spent the last of the budget, append the finalize nudge to its result.
        # `>=` (not `==`) so parallel calls in one turn that overshoot the budget still nudge.
        if is_budgeted_search and self._source_call_count >= self._search_budget:
            logger.info(
                "ComplexityRouterMiddleware: single_shot search budget reached "
                "(%d/%d) — withdrawing search tools and nudging to finalize",
                self._source_call_count,
                self._search_budget,
            )
            try:
                result = result.model_copy(update={"content": f"{result.content}{_SINGLE_SHOT_BUDGET_NUDGE}"})
            except Exception:
                # Non-Pydantic / immutable result: the tool-hiding in _filter_tools still
                # enforces the cap, so a missing nudge is a soft degradation, not a failure.
                pass
        return result

    def _is_substantive(self, name: str | None, tool_call: object) -> bool:
        """Return True when this call does real work (and so must not run on a rejected batch).

        Helpers (``think``, file reads, ``get_verified_sources``) and the declaration itself are
        allowed through on a rejected batch: blocking them adds noise without preventing the
        thing we care about, which is research or delegation running under an invalid tier.
        """
        if self._resolver is None or name is None:
            return False
        _, args = _tool_call_name_args(tool_call)
        return self._resolver.classify(name, args) in _SUBSTANTIVE_KINDS

    @staticmethod
    def _blocked(tool_call: object, message: str) -> ToolMessage:
        """Return a corrective error result, keeping the rejected call inside the agent loop."""
        call = tool_call if isinstance(tool_call, dict) else {}
        return ToolMessage(
            content=message,
            tool_call_id=call.get("id", "complexity-router"),
            name=call.get("name", "tool"),
            status="error",
        )

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Return the tool list filtered by ceiling rules and the single-shot swap when active."""
        if self._resolver is not None and self._tier is None:
            # Catalog turn 1: the tier is not known yet, so expose the union of the paths the
            # model may legitimately take (ceiling hiding still applies). The compatibility
            # matrix in TierResolver — not tool hiding — is what keeps the model on one path.
            return [t for t in tools if _request_tool_name(t) not in self._hidden_tool_names]
        if self._shallow_capture is not None and self._tier == "single_shot":
            # single_shot is delegated wholesale to the shallow researcher: hide
            # run_research_batch and every direct source tool (retrieval belongs to the
            # sub-agent), and hide `task` on later model turns once execution has started or
            # completed. The adapter's in-flight coalescing — not this next-turn filter — is the
            # at-most-once guard for several `task` calls emitted together in one turn.
            hidden = {_RUN_RESEARCH_BATCH_TOOL} | self._direct_source_tool_names
            if getattr(self._shallow_capture, "invoked", False):
                hidden.add(_TASK_TOOL)
            return [t for t in tools if _request_tool_name(t) not in hidden]
        if self._single_loop_single_shot and self._direct_source_tool_names:
            if self._tier == "single_shot":
                if self._source_call_count >= self._search_budget:
                    # Budget spent: also withdraw the source tools so the model *cannot* search
                    # again. Only get_verified_sources + submit_final_report (and the other
                    # helper/finalize tools) remain, forcing the run to synthesize and finish.
                    hidden = {_RUN_RESEARCH_BATCH_TOOL} | self._direct_source_tool_names
                    return [t for t in tools if _request_tool_name(t) not in hidden]
                # Collapse path (within budget): expose source tools, remove run_research_batch
                return [t for t in tools if _request_tool_name(t) != _RUN_RESEARCH_BATCH_TOOL]
            else:
                # Tier not yet declared or non-single_shot: hide source tools,
                # keep run_research_batch so the first turn has a research path.
                hidden = self._hidden_tool_names | self._direct_source_tool_names
                return [t for t in tools if _request_tool_name(t) not in hidden]
        # Default: static ceiling-based hiding only
        if not self._hidden_tool_names:
            return tools
        return [tool for tool in tools if _request_tool_name(tool) not in self._hidden_tool_names]

    def _model_overrides(self, request) -> dict[str, object]:
        """Build the ``request.override(...)`` kwargs applied before each model call.

        Always filters tools (ceiling hiding + single-shot swap). Additionally, when a prompt
        renderer is configured and a tier has been resolved, swaps the system message to that
        tier's trimmed prompt — mirroring ``TodoSuppressionMiddleware._clean_request`` in
        deep_researcher (one overrides dict, a freshly built ``SystemMessage``). Before the tier
        is resolved (catalog turn 1) it is None, so the baked-in catalog prompt is left intact.
        """
        overrides: dict[str, object] = {"tools": self._filter_tools(request.tools)}
        tier = self._tier
        if self._prompt_renderer is not None and tier is not None:
            if tier not in self._rendered_prompt_cache:
                self._rendered_prompt_cache[tier] = self._prompt_renderer(tier)
            overrides["system_message"] = SystemMessage(content=self._rendered_prompt_cache[tier])
        return overrides

    def wrap_model_call(self, request, handler):
        """Hide/swap tools and (when active) swap the system prompt before a sync model call."""
        return handler(request.override(**self._model_overrides(request)))

    async def awrap_model_call(self, request, handler):
        """Hide/swap tools and (when active) swap the system prompt before an async model call."""
        return await handler(request.override(**self._model_overrides(request)))


class SingleShotShallowDelegationMiddleware(AgentMiddleware):
    """Make shallow delegation and finalization deterministic on the ``single_shot`` tier.

    Attached only when the shallow sub-agent is wired in. ``ComplexityRouterMiddleware`` controls
    which *tool names* the model sees, but it cannot choose among the sub-agent names advertised
    inside the shared ``task`` tool — so this middleware is the correctness boundary for
    tier-aware routing. Responsibilities:

    1. Resolve the effective tier from a declaration in the current tool-call batch, falling back
       to the cached prior-turn tier; reject conflicting same-turn declarations.
    2. Cache every accepted declared tier on both this middleware and the capture (including
       escalation, so a later tier disables the finalizer override and the recovery path).
    3. Reject ``task`` when no tier is known. On ``single_shot``, force the sub-agent type and
       description to the shallow researcher and the original user query, and reject further
       delegation once the attempt budget is spent. On every other tier, reject an attempted
       shallow sub-type while leaving planner / writer / source-router delegation untouched.
    4. Reject premature ``single_shot`` finalization while a shallow attempt is still viable.
    5. Replace the accepted finalizer's ``markdown`` / ``researched`` / ``tier`` arguments with the
       captured values — or, once the attempt budget is spent, let the finalizer through with
       ``researched=True`` forced so the existing empty-registry gate decides the outcome.
       Rejecting forever would livelock the run: with ``task`` exhausted and finalize blocked
       there is no terminal action left, and only the turn budget or the workflow deadline would
       end it.

    Lifetime and concurrency match ``ComplexityRouterMiddleware``: one instance per top-level
    request, built in ``build_adaptive_research_graph``, so request-scoped state lives on ``self``.
    """

    def __init__(self, *, capture: object, original_query: str, tier_resolver: TierResolver | None = None) -> None:
        """Store the run-scoped capture, the authoritative user query, and the shared resolver."""
        self._capture = capture
        self._original_query = original_query
        self._declared_tier: str | None = None
        # Catalog mode: defer to the one batch-wide decision instead of this middleware's own
        # same-turn scan, so a tier co-declared with `task` is seen identically everywhere.
        self._resolver = tier_resolver

    @staticmethod
    def _tool_call_of(request) -> dict:
        """Return the request's tool call as a dict, or ``{}`` for an unrecognised shape."""
        tool_call = getattr(request, "tool_call", None)
        return tool_call if isinstance(tool_call, dict) else {}

    def _effective_tier(self, request) -> tuple[str | None, str | None]:
        """Return ``(tier, error)`` from the current tool-call batch, else the cached tier.

        Scanning full message history remains unreliable (upstream middleware may rewrite it), but
        the state ToolNode is executing does contain the AIMessage whose sibling tool calls are in
        flight — so a same-turn ``declare_effort_tier`` is visible here no matter which wrapper
        runs first.

        In catalog mode the shared resolver has already made that determination for the whole
        batch (including the inference fallback and the enabled/downgrade validation), so this
        defers to it rather than re-deriving a second, possibly different answer.
        """
        if self._resolver is not None:
            decision = self._resolver.decide(getattr(request, "state", None))
            return (None, decision.error) if decision.error else (decision.tier, None)
        declared_here = _declared_tiers_in_current_tool_batch(getattr(request, "state", None))
        if len(declared_here) > 1:
            return None, (
                "Conflicting effort tiers were declared in one turn. Declare exactly one tier "
                "before delegating or finalizing."
            )
        if declared_here:
            return next(iter(declared_here)), None
        return self._declared_tier, None

    def _blocked(self, request, message: str) -> ToolMessage:
        """Return a corrective error result without executing the tool.

        Mirrors ``ResearcherLoopGuardMiddleware._blocked_result`` / ``OrchestratorLoopGuard`` — the
        rejected call stays inside the agent loop so the model can correct itself.
        """
        tool_call = self._tool_call_of(request)
        return ToolMessage(
            content=message,
            tool_call_id=tool_call.get("id", "shallow-delegation-guard"),
            name=tool_call.get("name", _TASK_TOOL),
            status="error",
        )

    async def awrap_tool_call(self, request, handler):
        """Route ``task`` and rewrite ``submit_final_report`` according to the effective tier."""
        tool_call = self._tool_call_of(request)
        if not tool_call:
            return await handler(request)
        name = tool_call.get("name")
        args = dict(tool_call.get("args") or {})

        effective_tier, tier_error = self._effective_tier(request)
        if tier_error and name in (_TASK_TOOL, _FINALIZE_TOOL):
            logger.warning("single_shot shallow delegation: unusable tier decision — %s", tier_error)
            return self._blocked(request, tier_error)

        # Mirror the resolved tier onto the capture. AdaptiveResearcherAgent.run checks it before
        # reusing a captured shallow report after a timeout, and in catalog mode the tier may have
        # been inferred rather than declared — in which case the declaration branch below never
        # runs and the capture would otherwise keep a stale tier.
        if effective_tier is not None and self._capture.declared_tier != effective_tier:
            self._capture.declared_tier = effective_tier

        if name == _DECLARE_EFFORT_TIER_TOOL:
            tier = args.get("tier")
            if isinstance(tier, str) and tier.strip():
                self._declared_tier = tier.strip()
                # Mirrored onto the capture so AdaptiveResearcherAgent.run can tell whether the
                # run is still on single_shot before reusing a captured report after a timeout.
                self._capture.declared_tier = self._declared_tier
            return await handler(request)

        if name == _TASK_TOOL:
            if effective_tier is None:
                return self._blocked(request, "Declare the effort tier before delegating to a subagent.")
            if effective_tier == "single_shot":
                if self._capture.exhausted:
                    # Attempt budget spent. Stop re-delegating (each attempt is a full shallow
                    # run) and point the model at the now-open finalize path.
                    return self._blocked(
                        request,
                        f"The shallow researcher could not complete this request after "
                        f"{MAX_SHALLOW_ATTEMPTS} attempts. Do not delegate again; call "
                        "submit_final_report with whatever the gathered evidence supports.",
                    )
                args[_SUBAGENT_TYPE_ARG] = SHALLOW_RESEARCHER_SUBAGENT
                args[_TASK_DESCRIPTION_ARG] = self._original_query
                return await handler(request.override(tool_call={**tool_call, "args": args}))
            if args.get(_SUBAGENT_TYPE_ARG) == SHALLOW_RESEARCHER_SUBAGENT:
                return self._blocked(
                    request,
                    "shallow-researcher is available only on the single_shot tier; follow the "
                    "declared tier's existing workflow.",
                )
            return await handler(request)

        if name == _FINALIZE_TOOL and effective_tier == "single_shot":
            if not self._capture.has_report:
                if not self._capture.exhausted:
                    # A shallow attempt is still viable: make the model delegate first.
                    return self._blocked(
                        request,
                        "The single_shot shallow researcher has not completed. Call task with the "
                        "shallow-researcher subagent before finalizing.",
                    )
                # Escape hatch: the budget is spent and there is no report to enforce, so let the
                # finalizer run — but force researched=True so a failed research attempt cannot be
                # recorded as a deliberate no-research answer. With no sources captured this
                # reaches the existing EmptySourceRegistryError path, exactly as a failed
                # single_shot run does today.
                args["researched"] = True
                args["tier"] = "single_shot"
                logger.warning(
                    "single_shot: shallow researcher exhausted after %d attempt(s) (last error: %s); "
                    "allowing orchestrator-authored finalize",
                    self._capture.attempts,
                    self._capture.error_type or "unknown",
                )
                return await handler(request.override(tool_call={**tool_call, "args": args}))

            submitted = args.get("markdown")
            submitted_text = submitted.strip() if isinstance(submitted, str) else ""
            if submitted_text != self._capture.markdown:
                logger.info(
                    "single_shot: replacing orchestrator-authored final report (%d chars) with the "
                    "shallow-researcher report (%d chars)",
                    len(submitted_text),
                    len(self._capture.markdown),
                )
            args["markdown"] = self._capture.markdown
            args["researched"] = self._capture.researched
            args["tier"] = "single_shot"
            return await handler(request.override(tool_call={**tool_call, "args": args}))

        return await handler(request)


_RESEARCHER_BUDGET_NUDGE = (
    "\n\n[SYSTEM — researcher source budget exhausted. Stop searching and return "
    "ResearchNotes now using the evidence already gathered. Represent unsupported target "
    "components as ResearchGap entries; do not guess.]"
)


def _canonical_source_signature(tool_name: str, args: object) -> str:
    """Hash a source-tool name and canonical arguments without retaining argument content."""
    try:
        canonical_args = json.dumps(args, sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):
        canonical_args = repr(args)
    payload = f"{tool_name}:{canonical_args}".encode()
    return hashlib.sha256(payload).hexdigest()


class ResearcherLoopGuardMiddleware(AgentMiddleware):
    """Hard-limit source calls and repeated requests within one researcher invocation."""

    def __init__(
        self,
        *,
        source_tool_names: set[str] | frozenset[str],
        config: ResearcherLoopGuardConfig,
    ) -> None:
        self._source_tool_names = frozenset(source_tool_names)
        self._config = config

    @staticmethod
    def _mark_exhausted(state, reason: str) -> None:
        state.exhausted = True
        state.exhaustion_reason = reason

    @staticmethod
    def _append_nudge(result):
        try:
            return result.model_copy(update={"content": f"{result.content}{_RESEARCHER_BUDGET_NUDGE}"})
        except Exception:
            return result

    @staticmethod
    def _blocked_result(tool_call: dict, reason: str) -> ToolMessage:
        return ToolMessage(
            content=(
                f"Source tool not executed: researcher loop guard reached {reason}. "
                "Stop searching and return structured ResearchNotes using gathered evidence; "
                "record unsupported requirements as ResearchGap entries."
            ),
            tool_call_id=tool_call.get("id", "researcher-loop-guard"),
            name=tool_call.get("name", "source-tool"),
            status="error",
        )

    def _filter_tools(self, tools: list[object]) -> list[object]:
        state = CURRENT_RESEARCHER_GUARD_STATE.get()
        if not self._config.enabled or state is None:
            return tools
        hidden = set()
        if state.exhausted:
            hidden.update(self._source_tool_names)
            hidden.add(_THINK_TOOL)
        elif state.think_blocked:
            hidden.add(_THINK_TOOL)
        if not hidden:
            return tools
        return [tool for tool in tools if _request_tool_name(tool) not in hidden]

    def wrap_model_call(self, request, handler):
        """Withdraw exhausted source/think tools before a synchronous model call."""
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(self, request, handler):
        """Withdraw exhausted source/think tools before an asynchronous model call."""
        return await handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_tool_call(self, request, handler):
        """Count logical source calls, block repeats, and preserve the last allowed result."""
        state = CURRENT_RESEARCHER_GUARD_STATE.get()
        tool_call = getattr(request, "tool_call", None)
        if (
            not self._config.enabled
            or state is None
            or not isinstance(tool_call, dict)
            or tool_call.get("name") not in self._source_tool_names
        ):
            return await handler(request)

        tool_name = tool_call["name"]
        budget = self._config.source_call_budgets.for_depth(state.depth)
        if state.exhausted or state.source_call_count >= budget:
            self._mark_exhausted(state, "total source-call budget")
            logger.warning(
                "Researcher loop guard blocked source call | "
                "invocation=%s depth=%s tool=%s calls=%d/%d reason=total_budget",
                state.invocation_id,
                state.depth,
                tool_name,
                state.source_call_count,
                budget,
            )
            return self._blocked_result(tool_call, "the total source-call budget")

        signature = _canonical_source_signature(tool_name, tool_call.get("args", {}))
        identical_count = state.source_signature_counts.get(signature, 0)
        if identical_count >= self._config.max_identical_source_calls:
            self._mark_exhausted(state, "repeated source-call signature")
            logger.warning(
                "Researcher loop guard blocked repeated source call | "
                "invocation=%s depth=%s tool=%s repeats=%d/%d reason=repeated_signature",
                state.invocation_id,
                state.depth,
                tool_name,
                identical_count,
                self._config.max_identical_source_calls,
            )
            return self._blocked_result(tool_call, "the repeated source-call limit")

        # Count before awaiting so parallel tool calls in this researcher share one hard ceiling.
        state.source_call_count += 1
        state.source_signature_counts[signature] = identical_count + 1
        result = await handler(request)

        if state.source_call_count >= budget:
            self._mark_exhausted(state, "total source-call budget")
            logger.info(
                "Researcher source-call budget reached | invocation=%s depth=%s tool=%s calls=%d/%d",
                state.invocation_id,
                state.depth,
                tool_name,
                state.source_call_count,
                budget,
            )
            return self._append_nudge(result)
        return result


class ConsecutiveThinkGuardMiddleware(AgentMiddleware):
    """Nudge pure think-loops; researcher counts are isolated per invocation.

    This guard intentionally detects only uninterrupted ``think`` calls. Alternating source
    calls are bounded separately by ``ResearcherLoopGuardMiddleware``.
    """

    def __init__(self, *, max_consecutive_thinks: int = _DEFAULT_MAX_CONSECUTIVE_THINKS) -> None:
        self._max = max_consecutive_thinks
        # Fallback for orchestrator/planner/writer instances that have no researcher context.
        self._consecutive_think_count = 0

    async def awrap_tool_call(self, request, handler):
        """Track consecutive think calls and inject a corrective message when the threshold is hit."""
        tool_call = getattr(request, "tool_call", None)
        name = None
        if tool_call is not None:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)

        state = CURRENT_RESEARCHER_GUARD_STATE.get()
        if state is not None:
            if name == _THINK_TOOL:
                state.consecutive_think_count += 1
            else:
                state.consecutive_think_count = 0
            count = state.consecutive_think_count
        else:
            if name == _THINK_TOOL:
                self._consecutive_think_count += 1
            else:
                self._consecutive_think_count = 0
            count = self._consecutive_think_count

        result = await handler(request)

        if name == _THINK_TOOL and count >= self._max:
            if state is not None:
                state.think_blocked = True
            warning = (
                f"Thought recorded. WARNING: You have called 'think' {count} times in a row "
                "without taking action. You MUST now call a real tool or return your structured "
                "response instead of thinking again."
            )
            logger.warning(
                "ConsecutiveThinkGuardMiddleware: %d consecutive think calls — injecting corrective nudge",
                count,
            )
            try:
                result = result.model_copy(update={"content": warning})
            except Exception:
                pass
        return result


def _normalize_text(value: object) -> str:
    """Unicode-normalize, casefold, and whitespace-collapse text for stable signatures."""
    text = value if isinstance(value, str) else str(value)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).casefold()


def _canonical_research_query_signature(query: object) -> str:
    """Hash a delegated ResearchQuery into a normalized, content-free signature.

    The signature intentionally covers only the fields that make one query *materially the
    same* as another: the (normalized) main query text, the ordered normalized subqueries (order
    is meaningful), the sorted target components, the sorted preferred tool names, and the depth.
    Free-form ``rationale`` and ``fallback_tools`` are omitted so re-explaining or padding a query
    cannot bypass duplicate detection. The query may arrive as a dict (raw LLM tool args) or a
    Pydantic model; both are handled. Only the hash is retained — raw argument text is never kept.
    """

    def _get(field: str, default: object) -> object:
        if isinstance(query, dict):
            return query.get(field, default)
        return getattr(query, field, default)

    subqueries = _get("subqueries", []) or []
    target_components = _get("target_components", []) or []
    preferred_tools = _get("preferred_tools", []) or []
    canonical = {
        "query": _normalize_text(_get("query", "")),
        # Ordered: distinct search angles are order-sensitive.
        "subqueries": [_normalize_text(s) for s in subqueries],
        # Unordered sets: sort so ordering differences do not create a "new" query.
        "target_components": sorted(_normalize_text(c) for c in target_components),
        "preferred_tools": sorted(_normalize_text(t) for t in preferred_tools),
        "depth": _normalize_text(_get("depth", "")),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OrchestratorLoopGuardMiddleware(AgentMiddleware):
    """Bound the *whole* adaptive request: research batches, delegated queries, and model turns.

    The per-researcher ``ResearcherLoopGuardMiddleware`` bounds one delegated invocation, but its
    state resets for every new invocation, so an orchestrator that keeps authoring fresh
    ``run_research_batch`` calls can run indefinitely while every per-researcher guard fires
    correctly. This middleware closes that gap at the orchestrator boundary.

    Lifetime and concurrency: exactly one instance is built per top-level request in
    ``build_adaptive_research_graph`` (the same lifetime as ``ComplexityRouterMiddleware``), so
    request-scoped counters live safely on ``self``. Counters are mutated *before* awaiting the
    tool handler, so parallel batch calls in a single turn cannot race past a limit.

    Enforcement, keyed on the tier captured from ``declare_effort_tier`` (independently of
    ``ComplexityRouterMiddleware``, which captures it via its own ``awrap_tool_call``):

    - ``standard`` / ``deep`` / delta: count batches, total delegated queries, and normalized
      per-query signatures. A batch that would exceed ``max_batch_calls`` or
      ``max_total_research_queries``, or that repeats a normalized query beyond
      ``max_identical_research_queries``, is **not executed** — a deterministic error
      ``ToolMessage`` is returned and the request transitions to ``finalizing``.
    - Once ``finalizing`` (or once model turns exceed ``max_orchestrator_turns``),
      ``run_research_batch`` and ``think`` are withdrawn from every later model call so the
      orchestrator can only finalize from evidence already collected.
    - ``single_shot`` / ``direct`` / ``meta`` are inert here: they self-limit (single_shot's own
      search budget) or perform no delegated research.

    Logging is metadata-only (request tag, tier, phase, counts, hashed signature) — never raw
    query arguments.
    """

    def __init__(
        self,
        *,
        config: AdaptiveRequestTerminationConfig,
        tier_resolver: TierResolver | None = None,
    ) -> None:
        self._config = config
        # Short opaque per-request tag for correlating log lines without leaking content.
        self._request_tag = uuid4().hex[:12]
        self._own_declared_tier: str | None = None
        # Catalog mode: read the tier from the shared batch decision so a declaration scheduled
        # after its sibling `run_research_batch` cannot leave the first batch unbudgeted.
        self._resolver = tier_resolver
        self._phase: str = "active"
        self._exhaustion_reason: str | None = None
        self._batch_call_count = 0
        self._total_query_count = 0
        self._model_turn_count = 0
        self._query_signature_counts: dict[str, int] = {}

    # --- introspection helpers (used by the fallback and by tests) ---------------------------

    @property
    def _declared_tier(self) -> str | None:
        """The tier this guard budgets against: the shared resolver's, else its own capture."""
        if self._resolver is not None:
            return self._resolver.tier
        return self._own_declared_tier

    @_declared_tier.setter
    def _declared_tier(self, tier: str | None) -> None:
        """Set the locally-cached tier. Ignored in catalog mode, where the resolver is the source."""
        self._own_declared_tier = tier

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def exhaustion_reason(self) -> str | None:
        return self._exhaustion_reason

    def _mark_finalizing(self, reason: str) -> None:
        if self._phase == "active":
            self._phase = "finalizing"
        self._exhaustion_reason = reason

    def _log_block(self, reason: str, *, signature: str | None = None, budget: int | None = None) -> None:
        logger.warning(
            "Orchestrator loop guard blocked research batch | request=%s tier=%s phase=%s reason=%s "
            "batches=%d queries=%d turns=%d limit=%s signature=%s",
            self._request_tag,
            self._declared_tier,
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
            tool_call_id=tool_call.get("id", "orchestrator-loop-guard"),
            name=tool_call.get("name", _RUN_RESEARCH_BATCH_TOOL),
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

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Withdraw research and think tools once the request is finalizing."""
        if not self._config.enabled or self._phase == "active":
            return tools
        hidden = {_RUN_RESEARCH_BATCH_TOOL, _THINK_TOOL}
        return [tool for tool in tools if _request_tool_name(tool) not in hidden]

    def _maybe_force_finalize_on_turns(self) -> None:
        budgets = self._config.budgets_for_tier(self._declared_tier)
        if budgets is None:
            return
        if self._model_turn_count > budgets.max_orchestrator_turns and self._phase == "active":
            self._mark_finalizing("orchestrator turn budget")
            self._log_block("orchestrator_turn_budget", budget=budgets.max_orchestrator_turns)

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
        """Capture the tier and enforce request-wide batch/query/duplicate budgets."""
        tool_call = getattr(request, "tool_call", None)
        if not self._config.enabled or not isinstance(tool_call, dict):
            return await handler(request)

        name = tool_call.get("name")
        if name == _DECLARE_EFFORT_TIER_TOOL:
            args = tool_call.get("args")
            if isinstance(args, dict) and args.get("tier") and self._resolver is None:
                self._own_declared_tier = args["tier"]
                logger.debug(
                    "OrchestratorLoopGuardMiddleware: request=%s declared tier=%s",
                    self._request_tag,
                    self._own_declared_tier,
                )
            return await handler(request)

        if name != _RUN_RESEARCH_BATCH_TOOL:
            return await handler(request)

        if self._resolver is not None:
            decision = self._resolver.decide(getattr(request, "state", None))
            if decision.error:
                # The batch is invalid and ComplexityRouterMiddleware (inner) will reject this
                # call. Pass through without reserving budget so a rejected batch does not
                # consume the request's research allowance.
                return await handler(request)
            self._resolver.commit(decision)

        budgets = self._config.budgets_for_tier(self._declared_tier)
        if budgets is None:
            # single_shot / direct / meta / pre-declaration: this guard does not bound them.
            return await handler(request)

        # If we are already finalizing, no further research may be requested.
        if self._phase != "active":
            self._log_block("already_finalizing")
            return self._blocked_result(
                tool_call,
                "Source research is closed: the request has reached its research budget and is finalizing. "
                "Do not call run_research_batch again. Use get_verified_sources and submit_final_report "
                "to write your final answer from the notes already gathered; represent any missing "
                "components as explicit gaps.",
            )

        queries = self._extract_queries(tool_call)
        incoming = len(queries)

        # --- Count and check BEFORE awaiting the handler so concurrent batch calls in one turn
        # share one hard ceiling (no await between the checks and the increments). ---
        if self._batch_call_count + 1 > budgets.max_batch_calls:
            self._mark_finalizing("research batch-call budget")
            self._log_block("batch_call_budget", budget=budgets.max_batch_calls)
            return self._blocked_result(
                tool_call,
                f"Research batch budget reached ({self._batch_call_count}/{budgets.max_batch_calls} calls). "
                "No further research batches will run. Call get_verified_sources and submit_final_report to "
                "finalize from the evidence already gathered; record unsupported requirements as gaps.",
            )

        if incoming and self._total_query_count + incoming > budgets.max_total_research_queries:
            self._mark_finalizing("total delegated-query budget")
            self._log_block("total_query_budget", budget=budgets.max_total_research_queries)
            remaining = budgets.max_total_research_queries - self._total_query_count
            return self._blocked_result(
                tool_call,
                f"Delegated-query budget reached: this batch of {incoming} would exceed the remaining "
                f"{max(remaining, 0)} of {budgets.max_total_research_queries} queries for this request. The "
                "batch was not run. Call get_verified_sources and submit_final_report to finalize from the "
                "evidence already gathered; record unsupported requirements as gaps.",
            )

        signatures = [_canonical_research_query_signature(q) for q in queries]
        for signature in signatures:
            if self._query_signature_counts.get(signature, 0) >= self._config.max_identical_research_queries:
                self._mark_finalizing("repeated delegated-query signature")
                self._log_block("duplicate_query", signature=signature)
                return self._blocked_result(
                    tool_call,
                    "Duplicate research query blocked: this request has already researched an identical query. "
                    "Retrying the same query will not surface new evidence. Call get_verified_sources and "
                    "submit_final_report to finalize; if a required period or component is unavailable in the "
                    "configured sources, state it as an explicit evidence gap instead of searching again.",
                )

        # Reserve the budget atomically (still before the first await).
        self._batch_call_count += 1
        self._total_query_count += incoming
        for signature in signatures:
            self._query_signature_counts[signature] = self._query_signature_counts.get(signature, 0) + 1
        logger.info(
            "Orchestrator loop guard: request=%s tier=%s batch=%d/%d queries=%d/%d turns=%d",
            self._request_tag,
            self._declared_tier,
            self._batch_call_count,
            budgets.max_batch_calls,
            self._total_query_count,
            budgets.max_total_research_queries,
            self._model_turn_count,
        )
        return await handler(request)
