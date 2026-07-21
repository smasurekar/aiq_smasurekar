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

import logging

from langchain.agents.middleware import AgentMiddleware

# Reuse the tool-name reader from deep_researcher so tool shapes are handled identically.
from aiq_agent.agents.deep_researcher.custom_middleware import _request_tool_name

from .tiers import tier_ceiling

logger = logging.getLogger(__name__)

_THINK_TOOL = "think"
_DEFAULT_MAX_CONSECUTIVE_THINKS = 3

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
    """

    def __init__(
        self,
        *,
        enabled_tiers: list[str] | None,
        allow_delegation: bool = False,
        direct_source_tools: list[object] | None = None,
        single_loop_single_shot: bool = False,
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
        # Populated by awrap_tool_call the moment declare_effort_tier executes.
        self._declared_tier: str | None = None

    async def awrap_tool_call(self, request, handler):
        """Intercept ``declare_effort_tier`` to cache the declared tier for tool-swap logic."""
        tool_call = getattr(request, "tool_call", None)
        if tool_call is not None:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            if name == _DECLARE_EFFORT_TIER_TOOL:
                args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
                if isinstance(args, dict) and args.get("tier"):
                    self._declared_tier = args["tier"]
                    logger.debug("ComplexityRouterMiddleware: declared tier = %s", self._declared_tier)
        return await handler(request)

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Return the tool list filtered by ceiling rules and the single-shot swap when active."""
        if self._single_loop_single_shot and self._direct_source_tool_names:
            if self._declared_tier == "single_shot":
                # Collapse path: expose source tools, remove run_research_batch
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

    def wrap_model_call(self, request, handler):
        """Hide or swap tools before a synchronous model call."""
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(self, request, handler):
        """Hide or swap tools before an asynchronous model call."""
        return await handler(request.override(tools=self._filter_tools(request.tools)))


class ConsecutiveThinkGuardMiddleware(AgentMiddleware):
    """Break infinite think-loops by injecting a corrective nudge after N consecutive think calls.

    When the LLM calls ``think`` repeatedly with no intervening action tool, its context
    never changes and it loops indefinitely. This middleware detects that pattern and overwrites
    the ``think`` result with a warning that instructs the model to call a real tool next.

    Each instance is created per-request (inside ``build_adaptive_research_graph``), so
    ``self._consecutive_think_count`` is safe under concurrent requests.
    """

    def __init__(self, *, max_consecutive_thinks: int = _DEFAULT_MAX_CONSECUTIVE_THINKS) -> None:
        self._max = max_consecutive_thinks
        self._consecutive_think_count = 0

    async def awrap_tool_call(self, request, handler):
        """Track consecutive think calls and inject a corrective message when the threshold is hit."""
        tool_call = getattr(request, "tool_call", None)
        name = None
        if tool_call is not None:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)

        if name == _THINK_TOOL:
            self._consecutive_think_count += 1
        else:
            self._consecutive_think_count = 0

        result = await handler(request)

        if name == _THINK_TOOL and self._consecutive_think_count >= self._max:
            warning = (
                f"Thought recorded. WARNING: You have called 'think' "
                f"{self._consecutive_think_count} times in a row without taking action. "
                "You MUST now call a real tool (e.g., run_research_batch, a search/retrieval "
                "tool, or submit_final_report) instead of thinking again."
            )
            logger.warning(
                "ConsecutiveThinkGuardMiddleware: %d consecutive think calls — injecting corrective nudge",
                self._consecutive_think_count,
            )
            try:
                result = result.model_copy(update={"content": warning})
            except Exception:
                pass
        return result
