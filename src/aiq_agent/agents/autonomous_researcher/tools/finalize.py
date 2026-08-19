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

"""Inline finalize tool for the autonomous research orchestrator.

The autonomous agent has exactly two valid exits and this tool owns one of them:

* **writer exit** — the orchestrator delegated to ``writer-agent``, which committed
  ``/shared/output.md`` through the upstream ``FinalReportCommitMiddleware``. The orchestrator
  then returns only the writer's completion marker and never calls this tool.
* **inline exit** — the orchestrator wrote the answer itself. There is no ``/shared/output.md``,
  so it emits a *positive* finalize signal by calling ``submit_final_report``, which records the
  markdown to ``/shared/final_report.md`` plus a sidecar ``/shared/final_report_meta.json``
  carrying the ``researched`` flag.

Both exits are recorded on one :class:`AutonomousFinalReportCommitTracker`, so the finalization
guard can accept either without forcing writer delegation. See ``custom_middleware`` for the
tracker and the guard.

Differences from ``adaptive_researcher.tools.finalize``, which this is adapted from:

* no ``tier=`` argument and no tier metadata — the autonomous agent has no effort tiers;
* no ``declare_effort_tier`` tool at all;
* ``researched`` is defined over *any* research path (a direct source-tool call, a
  ``task(researcher-agent)`` delegation, or a ``run_research_batch`` fan-out), not only batch
  research, because the autonomous orchestrator holds all three.
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


def commit_final_report(
    *,
    backend: Any | None,
    tracker: Any | None,
    markdown: str,
    researched: bool,
    source: str = "orchestrator",
) -> str:
    """Persist ``markdown`` as the run's authoritative inline report and record the commit.

    Extracted from ``submit_final_report`` because there are now two ways the inline exit is
    reached: the orchestrator calling the tool, and ``ShallowFinalizationMiddleware`` committing a
    completed shallow-researcher report without an orchestrator turn. Both must write the same two
    files, fail the same way on an upload error, and record the commit in the same order — so
    there is exactly one implementation of that contract.

    Args:
        backend: The run's shared filesystem backend, or ``None`` to skip persistence.
        tracker: The run's :class:`AutonomousFinalReportCommitTracker`, or ``None``.
        markdown: The complete final answer. Must be non-empty after stripping.
        researched: Whether any research backs the answer; drives citation verification downstream.
        source: Log-only label naming which path produced the report.

    Returns:
        The stripped report text that was committed.

    Raises:
        ValueError: If ``markdown`` is empty.
        RuntimeError: If the backend rejected either write.
    """
    text = (markdown or "").strip()
    if not text:
        raise ValueError("a final report requires non-empty markdown containing the final answer.")
    if backend is not None:
        files = [
            (FINAL_REPORT_PATH, text.encode("utf-8")),
            (FINAL_REPORT_META_PATH, json.dumps({"researched": bool(researched)}).encode("utf-8")),
        ]
        responses = backend.upload_files(files)
        errors = [f"{r.path}: {r.error}" for r in responses if getattr(r, "error", None)]
        if errors:
            raise RuntimeError(f"failed to record final report: {'; '.join(errors)}")
    if tracker is not None:
        # Commit the inline side of the dual-exit contract only after the write succeeded, so a
        # failed upload cannot satisfy the finalization guard with an unreadable report.
        tracker.record_inline(text)
    # Metadata only: the report body is never logged (see common.logging_utils).
    logger.info(
        "Finalized report inline | source=%s | researched=%s | length=%d chars",
        source,
        bool(researched),
        len(text),
    )
    return text


def build_submit_final_report_tool(
    *,
    backend: Any | None = None,
    tracker: Any | None = None,
) -> BaseTool:
    """Build the orchestrator-only ``submit_final_report`` tool.

    Args:
        backend: The run's shared filesystem backend. The tool persists through
            ``backend.upload_files`` — the same path ``run_research_batch`` uses for research
            notes — so the files surface in the graph result's ``files`` mapping and are readable
            by ``AutonomousResearcherAgent.run()``.
        tracker: The run's :class:`AutonomousFinalReportCommitTracker`. Recording the inline
            commit here is what lets ``AutonomousFinalizationMiddleware`` accept a run that never
            delegated to the writer.
    """

    # return_direct=True ends the ReAct loop the moment this tool executes, so the framework does
    # not spend an extra, discarded model turn just to emit a terminating (no-tool-call)
    # AIMessage. That trailing turn is pure waste here: the authoritative answer is loaded from
    # /shared/final_report.md by AutonomousResearcherAgent.run(), which then overwrites the last
    # message with the post-processed markdown. LangChain's create_agent routes to its exit node
    # when every client-side tool call in a turn is return_direct; submit_final_report is always
    # called as a lone tool call on the finalize step, so that condition holds.
    @tool(return_direct=True)
    def submit_final_report(markdown: str, researched: bool = True) -> str:
        """Record the final answer as the authoritative report and finish the run.

        Call this exactly once, when you wrote the answer yourself. Do NOT call it after
        delegating to `writer-agent` — there the writer owns `/shared/output.md` and you return
        only its completion marker.

        Args:
            markdown: The complete final answer as Markdown. For researched answers this must be
                a cited report using numeric citations ([1], [2], ...) drawn from
                get_verified_sources.
            researched: True when the answer is backed by ANY research this run — a source tool
                you called directly, a `task(subagent_type="researcher-agent")` delegation, or a
                `run_research_batch` fan-out. Citations must then verify. False ONLY for a
                deliberate no-research answer (chit-chat, a transparent capability limitation, or
                a trivial timeless fact answered from your own knowledge); that skips citation
                verification for this answer. An honest "I could not verify this" after research
                was attempted stays True.
        """
        if not (markdown or "").strip():
            raise ValueError(
                "submit_final_report requires a non-empty 'markdown' argument containing the final answer."
            )
        commit_final_report(
            backend=backend,
            tracker=tracker,
            markdown=markdown,
            researched=researched,
            source="orchestrator",
        )
        return "Recorded final report."

    return submit_final_report
