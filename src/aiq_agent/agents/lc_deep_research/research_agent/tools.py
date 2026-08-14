"""Research Tools.

This module provides search and content processing utilities for the research agent,
using Tavily for URL discovery and fetching full webpage content.
"""
# Ported from the LangChain DeepAgents deep_research example
# (deepagents/examples/deep_research/research_agent/tools.py). Everything the model sees -- the tool
# docstrings, which ARE the LLM-facing tool descriptions, and every success-path return string -- is
# byte-identical on purpose. Do not "improve" this file: it is the accuracy baseline this example
# exists to measure.
#
# Two deviations from upstream, both on paths that would otherwise crash rather than degrade:
#   1. TavilyClient is constructed lazily (see below), because AI-Q imports every registered plugin
#      at startup and upstream's module-scope construction raises without TAVILY_API_KEY.
#   2. tavily_search catches API errors and returns them as text, matching the error handling
#      upstream already applies to fetch_webpage_content.

import itertools
import json
import logging
import os
import re

import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

# Upstream builds `tavily_client = TavilyClient()` at module scope, which raises when
# TAVILY_API_KEY is unset. AI-Q imports every registered plugin at startup regardless of which
# agent the active config selects, so an import-time raise here would break unrelated configs --
# and CLAUDE.md requires missing-secret paths to degrade gracefully rather than crash. Deferring
# construction to the first search keeps behaviour identical from the first call onward.
_tavily_client: TavilyClient | None = None

logger = logging.getLogger(__name__)

_TOOL_IO_LOG_ENV = "AIQ_LC_DEEP_RESEARCH_LOG_TOOL_IO"
_TOOL_IO_PREVIEW_CHARS = 4000
_tool_call_ids = itertools.count(1)
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(?:nvapi|tvly(?:-prod)?|sk)-[a-z0-9_-]{8,}\b"), "[REDACTED]"),
    (
        re.compile(r"(?i)(api[_-]?key|access[_-]?token|password)(\s*[:=]\s*)([^\s,;]+)"),
        r"\1\2[REDACTED]",
    ),
)


def _tool_io_logging_enabled() -> bool:
    """Return whether bounded tool-I/O diagnostics were explicitly enabled."""
    return os.getenv(_TOOL_IO_LOG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _redact_diagnostic_text(value: str) -> str:
    """Redact common credential shapes before diagnostic text reaches application logs."""
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _log_tool_io(event: str, *, call_id: int, **fields: object) -> None:
    """Log bounded, correlated tool I/O without changing anything returned to the model."""
    if not _tool_io_logging_enabled():
        return

    bounded_fields: dict[str, object] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            redacted = _redact_diagnostic_text(value)
            bounded_fields[key] = redacted[:_TOOL_IO_PREVIEW_CHARS]
            if len(redacted) > _TOOL_IO_PREVIEW_CHARS:
                bounded_fields[f"{key}_truncated"] = True
                bounded_fields[f"{key}_chars"] = len(redacted)
        else:
            bounded_fields[key] = value

    logger.info(
        "LC deep research tool I/O: %s",
        json.dumps({"event": event, "call_id": call_id, **bounded_fields}, ensure_ascii=False, default=str),
    )


def _get_tavily_client() -> TavilyClient:
    """Return the process-wide Tavily client, constructing it on first use."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    call_id = next(_tool_call_ids)
    _log_tool_io(
        "search_start",
        call_id=call_id,
        query=query,
        query_chars=len(query),
        max_results=max_results,
        topic=topic,
    )

    # Use Tavily to discover URLs.
    #
    # Upstream calls .search() bare, so any API rejection propagates out of the tool, out of the
    # graph, and fails the whole request. Observed on Nemotron Ultra: the model appended terms to a
    # query across retries (231 -> 298 -> 535 -> 1206 -> 2116 chars) until Tavily returned
    # "Query is too long. Max query length is 1500 characters." and the run died with no answer.
    #
    # Returning the error as a string instead is the same idiom upstream already applies to
    # fetch_webpage_content below: the model sees what went wrong and can retry a shorter query.
    # The success path is byte-for-byte unchanged.
    try:
        search_results = _get_tavily_client().search(
            query,
            max_results=max_results,
            topic=topic,
        )
    except Exception as e:
        # Truncate the echoed query: a pathological one can be thousands of characters, and
        # replaying it in full both bloats context and invites the model to repeat it verbatim.
        error = f"Error searching for '{query[:200]}': {str(e)}"
        _log_tool_io("search_error", call_id=call_id, response=error, response_chars=len(error))
        return error

    # Fetch full content for each URL
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]

        _log_tool_io("search_result", call_id=call_id, title=title, url=url)

        # Fetch webpage content
        content = fetch_webpage_content(url)
        _log_tool_io(
            "fetch_result",
            call_id=call_id,
            url=url,
            content=content,
            content_chars=len(content),
            is_error=content.startswith("Error fetching content from "),
        )

        result_text = f"""## {title}
**URL:** {url}

{content}

---
"""
        result_texts.append(result_text)

    # Format final response
    response = f"""🔍 Found {len(result_texts)} result(s) for '{query}':

{chr(10).join(result_texts)}"""

    _log_tool_io("search_response", call_id=call_id, response=response, response_chars=len(response))

    return response


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
