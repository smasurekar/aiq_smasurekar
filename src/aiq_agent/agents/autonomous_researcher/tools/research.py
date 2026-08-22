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
from aiq_agent.common.logging_utils import log_content_metadata

from ..models import AutonomousResearchQuery

logger = logging.getLogger(__name__)

_NO_TOOL_RUNTIME = cast(ToolRuntime, None)
_QUERY_LOG_MAX_LENGTH = 80

# Routing text, not documentation. This description is what makes the independent-unknowns path of
# the orchestrator prompt's "Deciding what to do" section reachable, so it must stay in step with
# prompts/orchestrator.j2. Two claims here are load-bearing and were added deliberately:
# a one-query batch is legitimate (so a single unknown goes through a worker rather than the
# orchestrator searching directly into its own long-lived context), and `high` depth is priced as
# expensive (it was declared on 42% of queries for no measurable F1 return). See
# misc/autonomous_researcher/autonomous-orchestrator-prompt-redesign-plan.md §D3, §D6, §D7.
#
# NO BUDGET COUNTS HERE. A tool description is model input exactly like the system prompt, so a
# hard-coded "ONE batch per request" or "at most one high-depth query" drifts from max_batch_calls
# and max_high_depth_queries the same way the deleted `# Budgets` prompt section drifted — this
# file simply hides the drift better. Ceilings belong to
# AutonomousOrchestratorLoopGuardMiddleware, which states them when it blocks. The single
# exception is the per-call query cap below, which is interpolated from the same
# max_research_concurrency this tool validates against and therefore cannot drift.
# test_bound_tool_descriptions_state_no_budget_counts fails the build if a count returns.
#
# Scope boundary (2026-08-18): this description owns the DELEGATION CONTRACT — what one
# ResearchQuery must contain and what the call returns. It deliberately does NOT carry loop
# control: keeping a query ledger, recovering from a thin or failed pass, and deciding whether to
# run another pass are orchestrator behavior across turns, and live in the prompt's
# "# The Research Loop" section. Adding either concern to the other file reintroduces the
# duplication that section split was meant to remove. See
# misc/autonomous_researcher/autonomous-researcher-review-feedback-analysis.md §1.
_RESEARCH_BATCH_DESCRIPTION = """Run one or more independent research questions in parallel isolated contexts.

Send 1-{max_research_concurrency} queries in one call; more than that is rejected outright.

This is the normal way to research. A batch of ONE query is valid and is the right call for a single
self-contained fact — prefer it over searching yourself, because a worker's search trail is digested
before it reaches you instead of accumulating in your context.

Each query runs as its own worker, so nothing one worker learns can inform another. If one question
cannot be written until another is answered, that is a prerequisite chain: {chain_route}

Prefer a single well-formed batch. A follow-up batch is for consuming a prerequisite this one
resolves — not for re-asking a question that came back thin.

Each `ResearchQuery` needs: `query` (full standalone context — workers cannot see your conversation,
and a plan component id such as `latest_price_anchor` means nothing to them: spell the topic out),
`preferred_tools` (exact source-tool names), `target_components`, a `rationale`, and a `depth`:
  - `low`    — one quick self-contained lookup (the default choice);
  - `medium` — a few corroborating searches;
  - `high`   — iterative multi-hop, where each result informs the next search. Expensive: reserve
               it for a genuine chain, never for a question one search answers.

Returns a JSON array of `ResearchNotes` and persists each note as a JSON file under `/shared/`, each
carrying its own `evidence_judgment`; every source the workers cited is added to the verified-source
set for `get_verified_sources`."""


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
    # The preview above covers `query` only, truncated. `target_components`, `rationale`,
    # `subqueries` and `preferred_tools` are the rest of a worker's entire instruction, and
    # a worker that cannot satisfy its output contract is usually diagnosable from them.
    # Digest-only by default; AIQ_LOG_PAYLOADS prints the batch verbatim.
    logger.info(
        "  research batch payload | %s",
        log_content_metadata(json.dumps([query.model_dump(mode="json") for query in queries], ensure_ascii=False)),
    )


def build_autonomous_research_batch_tool(
    *,
    researcher_runnable: Any,
    callbacks: list[Any],
    max_research_concurrency: int,
    resource_limits: DeepResearchResourceLimits | None = None,
    backend: Any | None = None,
    state_budget: StateBudgetLedger | None = None,
    source_registry_middleware: Any | None = None,
    researcher_subagent_enabled: bool = True,
) -> BaseTool:
    """Build the orchestrator-only ``run_research_batch`` tool typed to ``AutonomousResearchQuery``.

    Args:
        researcher_subagent_enabled: Whether ``task(subagent_type="researcher-agent")`` is offered
            alongside this tool. The description hands prerequisite chains to that subagent when it
            exists; when it does not, chains have to be resolved as successive batches instead, and
            pointing at an absent subagent would cost the orchestrator a turn for nothing.
    """
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

    run_research_batch.description = _RESEARCH_BATCH_DESCRIPTION.format(
        max_research_concurrency=max_research_concurrency,
        chain_route=(
            'use\n`task(subagent_type="researcher-agent", ...)` for the whole chain instead of fanning out.'
            if researcher_subagent_enabled
            else "batch the link you\ncan write now, then send a follow-up batch that consumes its answer."
        ),
    )
    return run_research_batch
