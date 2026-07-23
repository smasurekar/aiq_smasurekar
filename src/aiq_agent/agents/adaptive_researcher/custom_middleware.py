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
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

# Reuse the tool-name reader from deep_researcher so tool shapes are handled identically.
from aiq_agent.agents.deep_researcher.custom_middleware import _request_tool_name
from aiq_agent.agents.deep_researcher.researcher_context import CURRENT_RESEARCHER_GUARD_STATE

from .models import AdaptiveRequestTerminationConfig
from .models import ResearcherLoopGuardConfig
from .tiers import tier_ceiling

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
_DELEGATION_TOOLS = ("task", "write_todos")
_RUN_RESEARCH_BATCH_TOOL = "run_research_batch"
_DECLARE_EFFORT_TIER_TOOL = "declare_effort_tier"


def hidden_tools_for_ceiling(ceiling: str, *, allow_delegation: bool = False) -> set[str]:
    """Return the tool names to hide from the orchestrator for a given enabled-tiers ceiling.

    - ceiling below ``deep``  -> hide ``advanced_web_search_tool`` (deep-only heavy retrieval).
    - ceiling below ``standard`` (i.e. shallow-only: ``single_shot`` / ``direct``) -> also hide
      ``task`` and ``write_todos`` so the orchestrator cannot delegate or plan, unless
      ``allow_delegation`` preserves them for a parent-report delta request.

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
    return hidden


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

    The declared tier is captured via ``awrap_tool_call`` by intercepting the
    ``declare_effort_tier`` execution rather than by scanning ``request.messages``. The
    messages-scan approach is unreliable because framework middleware earlier in the stack
    (e.g. SummarizationMiddleware) may have transformed or omitted prior AIMessages by the
    time this middleware's ``awrap_model_call`` runs. Each middleware instance is created per
    request (in ``build_adaptive_research_graph`` called from ``AdaptiveResearcherAgent.run``),
    so caching the tier on ``self`` is safe with concurrent requests.

    When a ``prompt_renderer`` is supplied (``dynamic_orchestrator_sections=True``), the
    middleware also performs a dynamic *prompt* swap: the graph is built with a minimal "router"
    system prompt, and once a tier is declared this middleware replaces the system message on
    every subsequent model call with ``prompt_renderer(tier)`` — the prompt trimmed to just that
    tier's sections. Renders are memoized per tier so the swapped prompt is byte-stable across a
    run's model calls (KV-cache friendly) and each tier renders at most once. Re-declaring a
    higher tier (escalation) simply renders and caches that tier's larger prompt.

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
    ) -> None:
        """Compute hidden tools, preserving the citation-safe delta writer path when needed."""
        self._hidden_tool_names = hidden_tools_for_ceiling(
            tier_ceiling(enabled_tiers),
            allow_delegation=allow_delegation,
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

    async def awrap_tool_call(self, request, handler):
        """Cache the declared tier and enforce the single_shot search budget.

        Two responsibilities, both keyed on the tool name read from ``request.tool_call``:

        1. When ``declare_effort_tier`` fires, cache the chosen tier for the tool-swap / prompt-
           swap logic (unchanged behavior).
        2. On the ``single_loop_single_shot`` ``single_shot`` path, count each direct source-tool
           call. The search still executes; but once the count reaches ``_search_budget`` we
           *append* a corrective nudge to the returned result so the model is told, in-context,
           to stop searching and finalize. The hard guarantee (withdrawing the search tools) is
           applied separately in ``_filter_tools`` on the next model call — this nudge only
           explains why the tools vanished, and it preserves the retrieved evidence by appending
           rather than overwriting.
        """
        tool_call = getattr(request, "tool_call", None)
        name = None
        if tool_call is not None:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            if name == _DECLARE_EFFORT_TIER_TOOL:
                args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
                if isinstance(args, dict) and args.get("tier"):
                    self._declared_tier = args["tier"]
                    logger.debug("ComplexityRouterMiddleware: declared tier = %s", self._declared_tier)

        # Count this call up-front (before running it) only when it is a budgeted single_shot
        # search, so the post-call threshold check and the next _filter_tools both see it.
        is_budgeted_search = (
            self._single_loop_single_shot
            and self._declared_tier == "single_shot"
            and name is not None
            and name in self._direct_source_tool_names
        )
        if is_budgeted_search:
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

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Return the tool list filtered by ceiling rules and the single-shot swap when active."""
        if self._single_loop_single_shot and self._direct_source_tool_names:
            if self._declared_tier == "single_shot":
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
        renderer is configured and a tier has been declared, swaps the system message to that
        tier's trimmed prompt — mirroring ``TodoSuppressionMiddleware._clean_request`` in
        deep_researcher (one overrides dict, a freshly built ``SystemMessage``). Before a tier is
        declared (turn 1) ``_declared_tier`` is None, so the baked-in router prompt is left intact.
        """
        overrides: dict[str, object] = {"tools": self._filter_tools(request.tools)}
        if self._prompt_renderer is not None and self._declared_tier is not None:
            tier = self._declared_tier
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

    def __init__(self, *, config: AdaptiveRequestTerminationConfig) -> None:
        self._config = config
        # Short opaque per-request tag for correlating log lines without leaking content.
        self._request_tag = uuid4().hex[:12]
        self._declared_tier: str | None = None
        self._phase: str = "active"
        self._exhaustion_reason: str | None = None
        self._batch_call_count = 0
        self._total_query_count = 0
        self._model_turn_count = 0
        self._query_signature_counts: dict[str, int] = {}

    # --- introspection helpers (used by the fallback and by tests) ---------------------------

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
            if isinstance(args, dict) and args.get("tier"):
                self._declared_tier = args["tier"]
                logger.debug(
                    "OrchestratorLoopGuardMiddleware: request=%s declared tier=%s",
                    self._request_tag,
                    self._declared_tier,
                )
            return await handler(request)

        if name != _RUN_RESEARCH_BATCH_TOOL:
            return await handler(request)

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
