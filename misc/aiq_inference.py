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

"""Run an AI-Q research query the way the web UI does — and stream the result.

Unlike a poll-and-fetch smoke test, this mirrors the browser client
(``frontends/ui/src/adapters/api/deep-research-client.ts``): it submits an async
job and then consumes the Server-Sent Events (SSE) stream as the source of
truth, reconnecting with ``Last-Event-ID`` if the connection drops. This is why
the UI "just works" where a naive poller is fragile:

  * The stream delivers live progress (workflow / tool / citation / output
    events) and the terminal ``job.status``, so a keep-alive drop mid-run is
    recoverable instead of fatal.
  * ``adaptive_researcher`` (this script's default, matching the
    ``config_adaptive_frag.yml`` workflow) selects an effort tier per request and
    reliably calls retrieval tools, so it does not fail with
    ``EmptySourceRegistryError`` the way ``shallow_researcher`` can when a model
    answers a well-known question from memory without searching. The agent type
    submitted must exist as a function in the server's loaded config; the default
    deep-research configs expose ``deep_researcher`` instead.

Protocol (verified against ``frontends/aiq_api/src/aiq_api/routes/jobs.py`` and
the UI client):

    Submit     POST /v1/jobs/async/submit      {agent_type, input, data_sources?}
    Stream     GET  /v1/jobs/async/job/{id}/stream[/{after_id}]   (SSE)
    Report     GET  /v1/jobs/async/job/{id}/report                (final text)

SSE events consumed: stream.start, stream.mode, job.status, job.heartbeat,
workflow.start/end, llm.start/chunk/end, tool.start/end, artifact.update
(todo | citation_source | citation_use | output | file).

Token usage from each ``llm.end`` event is accumulated and presented with the
elapsed wall time in a run summary after the final report. Both LangChain's
``input_tokens`` / ``output_tokens`` names and the provider-style
``prompt_tokens`` / ``completion_tokens`` names are supported.

Auth: the server defaults to ``REQUIRE_AUTH=false`` and treats localhost as an
internal caller, so no token is required. For an auth-enabled deployment pass
``--token`` (or set ``AIQ_TOKEN``); it is sent as ``Authorization: Bearer``.

Usage:

    python aiq_inference.py "How has Apple's total net sales changed over time?"
    python aiq_inference.py "..." --server-url http://localhost:8100
    python aiq_inference.py "..." --agent shallow_researcher          # knowledge-base only (config_shallow_frag.yml)
    AIQ_AGENT=shallow_researcher python aiq_inference.py "..."        # same, via env var (no flag needed)
    python aiq_inference.py "..." --data-sources web_search,knowledge_layer
    python aiq_inference.py "..." --quiet          # only the final report
    python aiq_inference.py "..." --raw            # dump raw SSE events

The ``requests`` and ``rich`` libraries are required. Exit code is 0 on a
completed job, non-zero on failure or timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests
from rich.console import Console
from rich.markdown import Markdown

DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_AGENT = "adaptive_researcher"
DEFAULT_TIMEOUT = 1200  # seconds, whole-job wall clock
_TERMINAL_OK = {"success", "completed"}
_TERMINAL_BAD = {"failure", "failed", "interrupted", "cancelled"}
# Marks a headless caller so the agent skips the interactive clarifier
# (auth/utils.is_headless_request checks this exact header).
_HEADLESS_HEADER = {"X-AIQ-Mode": "headless"}
_MAX_RECONNECTS = 8


class _Style:
    """ANSI styling that degrades to plain text when output is not a terminal.

    Honors the NO_COLOR convention (https://no-color.org): when ``NO_COLOR`` is
    set or stdout is redirected, every style becomes a no-op so captured logs
    stay clean and greppable.
    """

    _CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36"}

    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def paint(self, name: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{self._CODES[name]}m{text}\033[0m"


STYLE = _Style()
_WIDTH = 72


def _rule(char: str = "─") -> str:
    return char * _WIDTH


def _banner(title: str, meta: dict[str, str]) -> None:
    print(STYLE.paint("cyan", _rule("═")))
    print(f"  {STYLE.paint('bold', title)}")
    print(STYLE.paint("cyan", _rule("═")))
    for key, value in meta.items():
        print(f"  {STYLE.paint('dim', f'{key}:'.ljust(14))}{value}")
    print()


def _event(icon_color: str, icon: str, label: str, detail: str = "") -> None:
    """Print a single live progress line, styled and aligned."""
    head = STYLE.paint(icon_color, f"{icon} {label}")
    line = f"  {head}"
    if detail:
        line += STYLE.paint("dim", f"  {detail}")
    print(line, flush=True)


def _truncate(text: str, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_duration(seconds: float) -> str:
    """Render an elapsed duration as a compact, human-readable string.

    Examples: ``4.2s``, ``1m 07s``, ``1h 02m 05s``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _render_markdown(text: str) -> None:
    """Render Markdown in a terminal while preserving it when redirected."""
    text = text.strip()
    if not sys.stdout.isatty():
        print(text)
        return

    console = Console(file=sys.stdout, no_color=os.environ.get("NO_COLOR") is not None, highlight=False)
    console.print(Markdown(text))


class InferenceError(RuntimeError):
    """Raised when the job ends in a non-successful terminal state."""


class AIQClient:
    """Submits an async research job and streams it to completion."""

    def __init__(self, server_url: str, token: str | None, quiet: bool, raw: bool) -> None:
        self.base = server_url.rstrip("/")
        self.quiet = quiet
        self.raw = raw
        self.session = requests.Session()
        self.session.headers.update(_HEADLESS_HEADER)
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

        # Collected as the stream unfolds.
        self.status: str = ""
        self.error: str | None = None
        self.referenced: list[tuple[str, str]] = []  # (url, title) discovered
        self.cited: list[tuple[str, str]] = []  # (url, title) actually used
        self.outputs: list[str] = []  # 'output' artifact bodies (report text)
        self.llm_calls = 0
        self.usage_reported_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0

    # -- REST helpers ------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def submit(self, agent_type: str, query: str, data_sources: list[str] | None) -> str:
        """POST the job; retry transient network/5xx errors with backoff."""
        body: dict[str, Any] = {"agent_type": agent_type, "input": query}
        if data_sources:
            body["data_sources"] = data_sources

        last_exc: Exception | None = None
        for attempt in range(1, 5):
            try:
                resp = self.session.post(self._url("/v1/jobs/async/submit"), json=body, timeout=60)
            except requests.RequestException as exc:
                last_exc = exc
            else:
                if resp.status_code < 500:
                    if not resp.ok:
                        raise InferenceError(f"submit failed: HTTP {resp.status_code}: {_truncate(resp.text, 300)}")
                    job_id = str(resp.json().get("job_id", ""))
                    if not job_id:
                        raise InferenceError("submit returned no job_id")
                    return job_id
                last_exc = InferenceError(f"HTTP {resp.status_code}: {_truncate(resp.text, 200)}")
            time.sleep(min(2**attempt, 10))
        raise InferenceError(f"submit failed after retries: {last_exc}")

    def fetch_report(self, job_id: str) -> str:
        try:
            resp = self.session.get(self._url(f"/v1/jobs/async/job/{job_id}/report"), timeout=60)
            resp.raise_for_status()
            return str(resp.json().get("report") or "")
        except requests.RequestException:
            return ""

    # -- SSE streaming -----------------------------------------------------

    def stream(self, job_id: str, deadline: float) -> None:
        """Consume the SSE stream to a terminal status, reconnecting on drops."""
        last_event_id: str | None = None
        reconnects = 0
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"job {job_id} did not finish before the {DEFAULT_TIMEOUT}s deadline")

            path = f"/v1/jobs/async/job/{job_id}/stream"
            if last_event_id:
                path += f"/{last_event_id}"
            try:
                with self.session.get(
                    self._url(path),
                    headers={"Accept": "text/event-stream"},
                    stream=True,
                    timeout=(10, 65),  # (connect, read); read>heartbeat interval
                ) as resp:
                    resp.raise_for_status()
                    for event, data, event_id in _iter_sse(resp):
                        if event_id:
                            last_event_id = event_id
                        self._handle(event, data)
                        if self.status in _TERMINAL_OK or self.status in _TERMINAL_BAD:
                            return
            except (requests.RequestException, ConnectionError) as exc:
                # A drop before a terminal status is recoverable — the UI's
                # EventSource does exactly this. Resume from last_event_id.
                reconnects += 1
                if reconnects > _MAX_RECONNECTS:
                    raise InferenceError(f"stream lost after {reconnects} reconnect attempts: {exc}") from exc
                if not self.quiet:
                    _event("yellow", "⟳", "reconnecting", f"attempt {reconnects}/{_MAX_RECONNECTS}")
                time.sleep(min(2**reconnects, 8))

    def _handle(self, event: str, data: Any) -> None:
        if self.raw:
            blob = data if isinstance(data, str) else json.dumps(data)
            print(STYLE.paint("dim", f"  «{event}» {_truncate(blob, 160)}"))

        if event == "stream.start":
            if not self.quiet:
                _event("cyan", "▸", "stream connected", str(data.get("job_id", "")) if isinstance(data, dict) else "")
            return
        if event == "stream.mode":
            mode = data.get("mode") if isinstance(data, dict) else None
            if mode and not self.quiet:
                _event("dim", "·", "stream mode", str(mode))
            return

        payload = data.get("data", data) if isinstance(data, dict) else {}

        if event == "job.status":
            self.status = str(payload.get("status", "")).lower()
            self.error = payload.get("error")
            if not self.quiet:
                color = "green" if self.status in _TERMINAL_OK else "yellow"
                _event(color, "◆", f"status: {self.status}", self.error or "")
        elif event == "job.heartbeat":
            if not self.quiet:
                up = payload.get("uptime_seconds", "")
                _event("dim", "♥", "heartbeat", f"{up}s" if up != "" else "")
        elif event == "workflow.start":
            if not self.quiet:
                name = data.get("name") if isinstance(data, dict) else ""
                _event("blue", "⚙", f"workflow: {name}", _truncate(payload.get("input", ""), 80))
        elif event == "workflow.end":
            if not self.quiet:
                name = data.get("name") if isinstance(data, dict) else ""
                _event("blue", "✓", f"workflow done: {name}")
        elif event == "tool.start":
            if not self.quiet:
                name = data.get("name") if isinstance(data, dict) else ""
                _event("cyan", "🔎", f"tool: {name}", _truncate(json.dumps(payload.get("input", "")), 80))
        elif event == "tool.end":
            if not self.quiet:
                name = data.get("name") if isinstance(data, dict) else ""
                _event("cyan", "↩", f"tool done: {name}", _truncate(payload.get("output", ""), 80))
        elif event == "artifact.update":
            self._handle_artifact(data)
        elif event == "llm.end":
            self._handle_llm_usage(data)
        # llm.start/chunk are intentionally not rendered line-by-line to keep
        # progress readable; the final report is assembled below.

    def _handle_llm_usage(self, data: Any) -> None:
        """Accumulate token usage attached to an ``llm.end`` event."""
        self.llm_calls += 1
        if not isinstance(data, dict):
            return

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            return
        usage = metadata.get("usage")
        if not isinstance(usage, dict):
            return

        input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
        cached_tokens = _usage_int(usage, "cached_tokens")
        reasoning_tokens = _usage_int(usage, "reasoning_tokens")

        # Newer LangChain/provider responses place these values in detail maps.
        input_details = usage.get("input_token_details") or usage.get("prompt_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = cached_tokens or _usage_int(input_details, "cache_read", "cached_tokens")
        output_details = usage.get("output_token_details") or usage.get("completion_tokens_details")
        if isinstance(output_details, dict):
            reasoning_tokens = reasoning_tokens or _usage_int(output_details, "reasoning", "reasoning_tokens")

        self.usage_reported_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        self.reasoning_tokens += reasoning_tokens

    def usage_summary(self) -> dict[str, int]:
        """Return aggregate usage for the complete research query."""
        return {
            "llm_calls": self.llm_calls,
            "usage_reported_calls": self.usage_reported_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def _handle_artifact(self, data: Any) -> None:
        art = data.get("data", data) if isinstance(data, dict) else {}
        atype = art.get("type")
        if atype == "citation_source":
            self.referenced.append((art.get("url", ""), str(art.get("content", ""))))
        elif atype == "citation_use":
            self.cited.append((art.get("url", ""), str(art.get("content", ""))))
            if not self.quiet:
                _event("green", "❝", "cited source", _truncate(art.get("url", ""), 80))
        elif atype == "output":
            content = art.get("content")
            if isinstance(content, str) and content.strip():
                self.outputs.append(content)
        elif atype == "todo" and not self.quiet:
            todos = art.get("content") or []
            done = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
            _event("dim", "☑", "plan", f"{done}/{len(todos)} steps complete")

    # -- final report ------------------------------------------------------

    def final_report(self, job_id: str) -> str:
        # Prefer the stream's assembled output; fall back to the report endpoint.
        text = "\n\n".join(self.outputs).strip()
        return text or self.fetch_report(job_id)


def _iter_sse(response: requests.Response):
    """Yield (event, data, id) tuples from an SSE response.

    Implements the minimal SSE framing the backend emits: ``event:``, ``data:``
    (repeatable, newline-joined), ``id:`` fields, dispatched on a blank line.
    ``data`` is JSON-decoded when possible, else returned as a string.
    """
    event = "message"
    data_lines: list[str] = []
    event_id: str | None = None
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\r")
        if line == "":  # dispatch
            if data_lines:
                blob = "\n".join(data_lines)
                try:
                    parsed: Any = json.loads(blob)
                except ValueError:
                    parsed = blob
                yield event, parsed, event_id
            event, data_lines, event_id = "message", [], None
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
        elif field == "id":
            event_id = value


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    """Return the first non-negative integer token count found in ``usage``."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return 0


def _print_run_summary(job_id: str, elapsed_seconds: float, usage: dict[str, int]) -> None:
    """Print a compact, aligned summary of query performance and token usage."""
    title = " RUN SUMMARY "
    side = (_WIDTH - len(title)) // 2
    print(STYLE.paint("cyan", "─" * side + title + "─" * (_WIDTH - side - len(title))))

    rows = [
        ("Status", STYLE.paint("green", "Completed")),
        ("Job ID", job_id),
        ("Elapsed time", _format_duration(elapsed_seconds)),
        ("LLM calls", f"{usage['llm_calls']:,}"),
    ]
    if usage["usage_reported_calls"]:
        rows.extend(
            [
                ("Input tokens", f"{usage['input_tokens']:,}"),
                ("Output tokens", f"{usage['output_tokens']:,}"),
                ("Total tokens", f"{usage['total_tokens']:,}"),
            ]
        )
        if usage["cached_tokens"]:
            rows.append(("Cached tokens", f"{usage['cached_tokens']:,}"))
        if usage["reasoning_tokens"]:
            rows.append(("Reasoning tokens", f"{usage['reasoning_tokens']:,}"))
        if usage["usage_reported_calls"] < usage["llm_calls"]:
            coverage = f"{usage['usage_reported_calls']:,} of {usage['llm_calls']:,} LLM calls"
            rows.append(("Usage coverage", STYLE.paint("yellow", coverage)))
    else:
        rows.append(("Token usage", STYLE.paint("yellow", "Not reported by the model")))

    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {STYLE.paint('dim', f'{label}:'.ljust(label_width + 1))}  {value}")
    print(STYLE.paint("cyan", _rule()))


def run(args: argparse.Namespace) -> int:
    server_url = args.server_url
    data_sources = [p.strip() for p in args.data_sources.split(",") if p.strip()]

    if not args.quiet:
        _banner(
            "AI-Q Inference",
            {
                "Server": server_url,
                "Agent": args.agent,
                "Data sources": ", ".join(data_sources) if data_sources else "agent default (all)",
                "Query": _truncate(args.query, 60),
            },
        )

    client = AIQClient(server_url, args.token, args.quiet, args.raw)
    started = time.time()
    try:
        job_id = client.submit(args.agent, args.query, data_sources)
    except InferenceError as exc:
        print(STYLE.paint("red", f"  ✘ {exc}"))
        return 2

    if not args.quiet:
        _event("bold", "●", "job submitted", job_id)

    deadline = time.time() + args.timeout
    try:
        client.stream(job_id, deadline)
    except (InferenceError, TimeoutError) as exc:
        print(STYLE.paint("red", f"\n  ✘ {exc}"))
        return 3

    if client.status not in _TERMINAL_OK:
        detail = client.error or "no detail provided"
        print(STYLE.paint("red", f"\n  ✘ job ended '{client.status}': {detail}"))
        if "EmptySourceRegistry" in detail or "no sources" in detail.lower():
            print(
                STYLE.paint(
                    "dim",
                    "    (the agent captured no sources — retry, ask a retrieval-specific\n"
                    "     question, or use --agent deep_researcher for guaranteed tool use)",
                )
            )
        return 4

    report = client.final_report(job_id)

    # -- render final report --------------------------------------------------
    title = " RESEARCH REPORT "
    side = (_WIDTH - len(title)) // 2
    print()
    print(STYLE.paint("cyan", "─" * side + title + "─" * (_WIDTH - side - len(title))))
    print()
    if report:
        _render_markdown(report)
    else:
        print(STYLE.paint("dim", "(job succeeded but returned no report text)"))
    print()
    print(STYLE.paint("cyan", _rule()))

    if client.cited:
        print(f"  {STYLE.paint('bold', 'Cited sources')} ({len(client.cited)})")
        for url, _title in client.cited:
            print(STYLE.paint("dim", f"    • {url}"))
    elif client.referenced:
        print(STYLE.paint("dim", f"  {len(client.referenced)} source(s) referenced, none marked cited"))
    print()
    if not args.quiet:
        _print_run_summary(job_id, time.time() - started, client.usage_summary())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="The research question to run.")
    parser.add_argument(
        "--server-url",
        default=os.environ.get("AIQ_SERVER_URL", DEFAULT_SERVER_URL),
        help=f"AI-Q base URL (default: {DEFAULT_SERVER_URL} or $AIQ_SERVER_URL).",
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("AIQ_AGENT", DEFAULT_AGENT),
        help=(
            f"Agent type to submit (default: {DEFAULT_AGENT} or $AIQ_AGENT). "
            "Use 'shallow_researcher' for a knowledge-base-only run against config_shallow_frag.yml."
        ),
    )
    parser.add_argument(
        "--data-sources",
        default="",
        help="Comma-separated data source IDs (e.g. web_search,knowledge_layer). Omit to use the agent's defaults.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AIQ_TOKEN"),
        help="Bearer token for auth-enabled deployments (default: $AIQ_TOKEN). Omit when REQUIRE_AUTH=false.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Whole-job wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress live progress; print only the final report.")
    parser.add_argument("--raw", action="store_true", help="Also dump each raw SSE event (debugging).")
    args = parser.parse_args()

    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        print(STYLE.paint("yellow", "\n  interrupted by user"))
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
