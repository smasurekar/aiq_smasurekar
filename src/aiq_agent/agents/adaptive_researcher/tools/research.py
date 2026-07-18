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

"""Adaptive batched research tool.

Mirrors ``deep_researcher.tools.research.build_research_batch_tool`` but types the tool's
``queries`` argument as ``AdaptiveResearchQuery`` so the orchestrator can attach a per-query
``depth`` hint. The heavy lifting (concurrent worker execution, source registration, note
persistence) is reused verbatim from the deep researcher — those helpers are typed to the
``ResearchQuery`` base and accept ``AdaptiveResearchQuery`` instances unchanged, and
``format_research_request`` serializes the whole query (including ``depth``) into the
researcher's task message.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any
from typing import cast

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from aiq_agent.agents.deep_researcher.tools.research import _persist_research_notes
from aiq_agent.agents.deep_researcher.tools.research import _run_research_queries

from ..models import AdaptiveResearchQuery

logger = logging.getLogger(__name__)

_NO_TOOL_RUNTIME = cast(ToolRuntime, None)
_QUERY_LOG_MAX_LENGTH = 80


def _log_research_depths(queries: list[AdaptiveResearchQuery]) -> None:
    """Emit an observability signal for the per-query depth guidance driving this batch."""
    depths = Counter(query.depth for query in queries)
    logger.info(
        "Research batch dispatched | %d queries | depth low=%d medium=%d high=%d",
        len(queries),
        depths["low"],
        depths["medium"],
        depths["high"],
    )
    for query in queries:
        preview = query.query if len(query.query) <= _QUERY_LOG_MAX_LENGTH else f"{query.query[:_QUERY_LOG_MAX_LENGTH]}…"
        logger.info("  research query | depth=%-6s | %s", query.depth, preview)


def build_adaptive_research_batch_tool(
    *,
    researcher_runnable: Any,
    callbacks: list[Any],
    max_research_concurrency: int,
    backend: Any | None = None,
    source_registry_middleware: Any | None = None,
) -> BaseTool:
    """Build the orchestrator-only ``run_research_batch`` tool typed to ``AdaptiveResearchQuery``."""

    @tool
    async def run_research_batch(
        queries: list[AdaptiveResearchQuery],
        runtime: ToolRuntime = _NO_TOOL_RUNTIME,
    ) -> str:
        """Run planned research queries in parallel and return ResearchNotes JSON.

        Set each query's ``depth`` to steer the researcher's effort: ``low`` for a single quick
        lookup, ``medium`` for a few corroborating searches, ``high`` for iterative multi-hop.
        """
        if not queries:
            return "[]"

        if len(queries) > max_research_concurrency:
            raise ValueError(
                f"run_research_batch accepts at most {max_research_concurrency} curated queries. "
                f"Received {len(queries)}. Rank, merge, or drop lower-priority queries and call again."
            )
        _log_research_depths(queries)
        successful_queries, notes, errors = await _run_research_queries(
            queries=queries,
            researcher_runnable=researcher_runnable,
            runtime=runtime,
            callbacks=callbacks,
            max_concurrency=max_research_concurrency,
        )
        if source_registry_middleware is not None:
            source_registry_middleware.register_research_note_sources(notes)
        _persist_research_notes(backend=backend, queries=successful_queries, notes=notes)

        if errors:
            retained_detail = ""
            if notes:
                retained_actions = []
                if source_registry_middleware is not None:
                    retained_actions.append("registered")
                if backend is not None:
                    retained_actions.append("persisted under /shared/")
                retained_text = " and ".join(retained_actions) if retained_actions else "retained"
                retained_detail = (
                    f" {len(notes)} successful researcher worker(s) were {retained_text}; "
                    "resubmit only the failed queries."
                )
            raise RuntimeError(
                f"run_research_batch failed for {len(errors)} of {len(queries)} researcher worker(s). "
                f"Errors: {'; '.join(errors)}.{retained_detail}"
            )

        return json.dumps(
            [note.model_dump(mode="json", exclude_none=True) for note in notes],
            indent=2,
            ensure_ascii=False,
        )

    return run_research_batch
