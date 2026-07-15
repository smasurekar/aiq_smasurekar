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
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware

# Reuse the tool-name reader from deep_researcher so tool shapes are handled identically.
from aiq_agent.agents.deep_researcher.custom_middleware import _request_tool_name

from .tiers import tier_ceiling

logger = logging.getLogger(__name__)

# Tools considered "heavier" than a given effort ceiling. When the deep-most enabled tier is
# below these thresholds, the corresponding tools are hidden from the orchestrator's model
# requests. These are the same knobs the POC's per-tier exposure table describes (§4.7 C).
_ADVANCED_WEB_SEARCH_TOOL = "advanced_web_search_tool"
_DELEGATION_TOOLS = ("task", "write_todos")


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

    Wired only when ``enforce_tier_tools=True``. With the default (off) it is never attached,
    so the agent behaves exactly as the prompt-driven Layer-A design intends. The ceiling comes
    from static ``enabled_tiers``; a parent-report request may preserve delegation tools because
    its citation-safe writer workflow is mandatory. This does not alter the system-prompt prefix.
    """

    def __init__(self, *, enabled_tiers: list[str] | None, allow_delegation: bool = False) -> None:
        """Compute hidden tools, preserving the citation-safe delta writer path when needed."""
        self._hidden_tool_names = hidden_tools_for_ceiling(
            tier_ceiling(enabled_tiers),
            allow_delegation=allow_delegation,
        )

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Return the tool list with ceiling-disallowed tools removed."""
        if not self._hidden_tool_names:
            return tools
        return [tool for tool in tools if _request_tool_name(tool) not in self._hidden_tool_names]

    def wrap_model_call(self, request, handler):
        """Hide ceiling-disallowed tools before a synchronous model call."""
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(self, request, handler):
        """Hide ceiling-disallowed tools before an asynchronous model call."""
        return await handler(request.override(tools=self._filter_tools(request.tools)))
