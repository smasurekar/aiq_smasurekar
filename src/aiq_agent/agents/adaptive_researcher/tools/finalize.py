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

"""Finalize tools for the adaptive research orchestrator.

``declare_effort_tier``
    Called by the orchestrator as its very first tool call after deciding the effort
    level. Logs the tier immediately — before any research or delegation begins — and
    persists the choice to ``/shared/effort_tier.json`` so it is readable by
    ``AdaptiveResearcherAgent.run()`` even on paths that do not call
    ``submit_final_report`` (e.g. the deep / writer-agent path).

``submit_final_report``
    On the shallow / direct / meta paths the orchestrator writes the answer inline
    instead of delegating to writer-agent, so no ``/shared/output.md`` exists. Rather
    than relax the salvage heuristic (which risks accepting a short acknowledgment or a
    re-plan as the report), the orchestrator emits a *positive* finalize signal by
    calling this tool. It records the report markdown to ``/shared/final_report.md`` and
    a sidecar ``/shared/final_report_meta.json`` that carries the ``researched`` and
    ``tier`` fields. ``AdaptiveResearcherAgent.run()`` reads both from the run's
    ``/shared/`` filesystem.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

FINAL_REPORT_PATH = "/shared/final_report.md"
FINAL_REPORT_META_PATH = "/shared/final_report_meta.json"
EFFORT_TIER_PATH = "/shared/effort_tier.json"


def build_declare_effort_tier_tool(*, backend: Any | None = None) -> BaseTool:
    """Build the ``declare_effort_tier`` observability tool.

    The orchestrator calls this as its very first tool call after deciding the effort
    level. The tool logs the tier immediately (before any research or delegation) and
    persists ``/shared/effort_tier.json`` via the shared backend so ``run()`` can read
    the tier even on the deep / writer-agent path (which never calls
    ``submit_final_report``).
    """

    @tool
    def declare_effort_tier(tier: str) -> str:
        """Record the selected effort tier at the start of the run.

        Call this as your very first tool call, immediately after deciding the effort
        level and before any research, planning, or delegation. This is a backend
        observability signal — do not mention it to the user.

        Args:
            tier: The effort tier you chose for this run. Must be one of the enabled
                tier names (e.g. "direct", "single_shot", "standard", "deep", "meta").
        """
        tier = (tier or "").strip()
        if not tier:
            raise ValueError("declare_effort_tier requires a non-empty 'tier' argument.")
        if backend is not None:
            files = [(EFFORT_TIER_PATH, json.dumps({"tier": tier}).encode("utf-8"))]
            responses = backend.upload_files(files)
            errors = [f"{r.path}: {r.error}" for r in responses if getattr(r, "error", None)]
            if errors:
                raise RuntimeError(f"failed to record effort tier: {'; '.join(errors)}")
        logger.info("Effort tier selected  : %s", tier)
        return "Tier recorded."

    return declare_effort_tier


def build_submit_final_report_tool(*, backend: Any | None = None) -> BaseTool:
    """Build the orchestrator-only ``submit_final_report`` tool.

    The tool persists to the shared backend the same way ``run_research_batch`` persists
    research notes (``backend.upload_files``), so the files surface in the graph result's
    ``files`` mapping and are readable by ``AdaptiveResearcherAgent.run()``.
    """

    @tool
    def submit_final_report(markdown: str, researched: bool = True, tier: str | None = None) -> str:
        """Record the final answer as the authoritative report and finish the run.

        Call this once, at the end of a shallow / single-shot / direct / meta run, instead of
        delegating to writer-agent. Do NOT call it on the deep path (there the writer-agent
        writes /shared/output.md and you return its completion marker).

        Args:
            markdown: The complete final answer as Markdown. For researched answers this must be
                a cited report using numeric citations ([1], [2], ...) drawn from
                get_verified_sources.
            researched: True when this answer is backed by run_research_batch results (citations
                must verify). False ONLY for a deliberate no-research answer (meta / chit-chat,
                a transparent capability limitation, or a trivial timeless question answered
                from your own knowledge) — this skips citation verification for that answer.
            tier: The effort tier chosen for this run (e.g. "direct", "single_shot", "standard",
                "meta"). Pass it for observability — it is recorded in the run metadata and
                surfaced in structured logs. Optional; omit only when the tier is genuinely
                unknown.
        """
        text = (markdown or "").strip()
        if not text:
            raise ValueError(
                "submit_final_report requires a non-empty 'markdown' argument containing the final answer."
            )
        if backend is not None:
            files = [
                (FINAL_REPORT_PATH, text.encode("utf-8")),
                (FINAL_REPORT_META_PATH, json.dumps({"researched": bool(researched), "tier": tier}).encode("utf-8")),
            ]
            responses = backend.upload_files(files)
            errors = [f"{r.path}: {r.error}" for r in responses if getattr(r, "error", None)]
            if errors:
                raise RuntimeError(f"failed to record final report: {'; '.join(errors)}")
        logger.info(
            "Finalized report | tier=%s | researched=%s | length=%d chars",
            tier or "unknown",
            bool(researched),
            len(text),
        )
        return "Recorded final report."

    return submit_final_report
