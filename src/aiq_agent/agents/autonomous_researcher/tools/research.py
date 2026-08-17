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

"""Batched research fan-out tool for the autonomous orchestrator.

This mirrors ``deep_researcher.tools.research.build_research_batch_tool`` but types the
``queries`` argument as ``AutonomousResearchQuery`` so each query can carry a per-query ``depth``
hint (``low`` / ``medium`` / ``high``), which ``ResearcherLoopGuardMiddleware`` budgets against.
``depth`` is a capability knob, not an effort level: it says how much *sequential, multi-hop*
iteration one query is worth, independently of how many queries are fanned out.

Everything expensive — concurrent worker execution, per-job resource accounting, note
persistence, source registration — is reused verbatim from the deep researcher's private
helpers, so the three agents cannot drift on evidence handling.

The tool description is load-bearing: in a description-driven architecture it is what
differentiates this path from a direct source-tool call and from ``task(researcher-agent)``.
See ``prompts/orchestrator.j2`` ("Deciding what to do") for the matching prompt guidance: this
tool is the path for unknowns that do *not* depend on each other.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from typing import Any
from typing import cast

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from aiq_agent.agents.deep_researcher.resource_limits import DeepResearchResourceLimits
from aiq_agent.agents.deep_researcher.resource_limits import StateBudgetLedger
from aiq_agent.agents.deep_researcher.tools.research import _persist_research_notes
from aiq_agent.agents.deep_researcher.tools.research import _research_note_files
from aiq_agent.agents.deep_researcher.tools.research import _run_research_queries

from ..models import AutonomousResearchQuery

logger = logging.getLogger(__name__)

_NO_TOOL_RUNTIME = cast(ToolRuntime, None)
_QUERY_LOG_MAX_LENGTH = 80

# Routing text, not documentation. This description is what makes the independent-unknowns path of
# the orchestrator prompt's "Deciding what to do" section reachable, so it must stay in step with
# prompts/orchestrator.j2. Three claims here are load-bearing and were added deliberately:
# a one-query batch is legitimate (so a single unknown goes through a worker rather than the
# orchestrator searching directly into its own long-lived context); `high` depth is priced as
# expensive and capped at one per request (it was declared on 42% of queries for no measurable F1
# return); and a
# second batch is scoped to consuming a resolved prerequisite (two-batch runs were the worst
# scoring bucket in the eval). See
# misc/autonomous_researcher/autonomous-orchestrator-prompt-redesign-plan.md §D3, §D6, §D7.
_RESEARCH_BATCH_DESCRIPTION = """Run one or more independent research questions in parallel isolated contexts.

This is the normal way to research. A batch of ONE query is valid and is the right call for a single
self-contained fact — prefer it over searching yourself, because a worker's search trail is digested
before it reaches you instead of accumulating in your context.

Each query runs as its own worker, so nothing one worker learns can inform another. If one question
cannot be written until another is answered, that is a prerequisite chain: use
`task(subagent_type="researcher-agent", ...)` for the whole chain instead of fanning out.

Issue ONE batch per request as the default. A second batch is for consuming a prerequisite you have
now resolved — not for re-asking a question that came back thin.

Each `ResearchQuery` needs: `query` (full standalone context — workers cannot see your conversation),
`preferred_tools` (exact source-tool names), `target_components`, a `rationale`, and a `depth`:
  - `low`    — one quick self-contained lookup (the default choice);
  - `medium` — a few corroborating searches;
  - `high`   — iterative multi-hop, where each result informs the next search. Expensive: at most one
               per request, and only for a genuine chain.

Returns a JSON array of `ResearchNotes` and persists each note as a JSON file under `/shared/`;
every source the workers cited is added to the verified-source set for `get_verified_sources`."""


def _log_research_depths(queries: list[AutonomousResearchQuery]) -> None:
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
        preview = (
            query.query if len(query.query) <= _QUERY_LOG_MAX_LENGTH else f"{query.query[:_QUERY_LOG_MAX_LENGTH]}…"
        )
        logger.info("  research query | depth=%-6s | %s", query.depth, preview)


def build_autonomous_research_batch_tool(
    *,
    researcher_runnable: Any,
    callbacks: list[Any],
    max_research_concurrency: int,
    resource_limits: DeepResearchResourceLimits | None = None,
    backend: Any | None = None,
    state_budget: StateBudgetLedger | None = None,
    source_registry_middleware: Any | None = None,
) -> BaseTool:
    """Build the orchestrator-only ``run_research_batch`` tool typed to ``AutonomousResearchQuery``."""
    limits = resource_limits or DeepResearchResourceLimits()
    state_budget = state_budget or StateBudgetLedger(limits=limits, files={}, sandbox_enabled=True)
    ledger_lock = asyncio.Lock()
    consumed_queries = 0
    consumed_query_chars = 0
    persisted_note_count = 0
    persisted_note_bytes = 0

    @tool
    async def run_research_batch(
        queries: list[AutonomousResearchQuery],
        runtime: ToolRuntime = _NO_TOOL_RUNTIME,
    ) -> str:
        """Run planned research queries in parallel and return ResearchNotes JSON."""
        nonlocal consumed_queries
        nonlocal consumed_query_chars
        nonlocal persisted_note_bytes
        nonlocal persisted_note_count

        if not queries:
            return "[]"

        if len(queries) > max_research_concurrency:
            raise ValueError(
                f"run_research_batch accepts at most {max_research_concurrency} curated queries. "
                f"Received {len(queries)}. Rank, merge, or drop lower-priority queries and call again."
            )
        batch_query_chars = sum(
            len(query.query) + sum(len(subquery) for subquery in query.subqueries) for query in queries
        )
        async with ledger_lock:
            if consumed_queries + len(queries) > limits.max_research_queries:
                raise ValueError(f"run_research_batch exceeds the {limits.max_research_queries}-query per-job limit")
            if consumed_query_chars + batch_query_chars > limits.max_total_query_chars:
                raise ValueError(
                    f"run_research_batch exceeds the {limits.max_total_query_chars}-character aggregate query limit"
                )
            consumed_queries += len(queries)
            consumed_query_chars += batch_query_chars

        _log_research_depths(queries)
        successful_queries, notes, errors = await _run_research_queries(
            queries=queries,
            researcher_runnable=researcher_runnable,
            runtime=runtime,
            callbacks=callbacks,
            max_concurrency=max_research_concurrency,
        )
        note_files = _research_note_files(successful_queries, notes)
        batch_note_bytes = sum(len(content) for _, content in note_files)
        oversized_notes = [path for path, content in note_files if len(content) > limits.max_research_note_bytes]
        if oversized_notes:
            raise ValueError(f"ResearchNotes exceeds the {limits.max_research_note_bytes}-byte per-note limit")
        async with ledger_lock:
            # Each accepted ResearchQuery can yield at most one ResearchNotes file, so reusing the
            # consumed-query ceiling bounds the job-wide note count too.
            if persisted_note_count + len(note_files) > limits.max_research_queries:
                raise ValueError(f"ResearchNotes exceeds the {limits.max_research_queries}-note per-job limit")
            if persisted_note_bytes + batch_note_bytes > limits.max_total_research_note_bytes:
                raise ValueError(
                    f"ResearchNotes exceeds the {limits.max_total_research_note_bytes}-byte aggregate per-job limit"
                )
            persisted_note_count += len(note_files)
            persisted_note_bytes += batch_note_bytes

        try:
            _persist_research_notes(backend=backend, note_files=note_files, state_budget=state_budget)
        except Exception:
            async with ledger_lock:
                persisted_note_count -= len(note_files)
                persisted_note_bytes -= batch_note_bytes
            raise
        if source_registry_middleware is not None:
            source_registry_middleware.register_research_note_sources(notes)

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

    run_research_batch.description = _RESEARCH_BATCH_DESCRIPTION
    return run_research_batch
