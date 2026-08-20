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

"""NAT registration for ``web_page_fetch`` - opens web pages by URL and returns their text.

Every other package under ``sources/`` is a *search* API: you give it keywords and it returns
ranked snippets. None of them can open a URL you already hold, which is why the autonomous
researcher performed zero page opens across the 90-task DSQA evaluation and scored 0.12 on the
questions that name their source (see ``misc/autonomous_researcher/fetch-url-tool-plan.md``).

**Backend.** Iteration 1 uses Tavily Extract exclusively. It is a maintained extraction service,
it runs no egress from the AI-Q host, and in smoke testing it handled every case measured - a
395-page 31.7MB PDF in 2.1s, HTML articles, and index pages - except dense numeric tables, which it
flattens (row-to-value association is lost). A local HTTP + PDF-table path recovers that one shape
and is deferred to iteration 2 rather than carried now.

Three Tavily behaviours are guarded below because each is silent rather than loud:

* its ``query`` parameter is never forwarded - passing it collapsed a 1.07M-character extract to
  1,025 characters that did not contain the requested table;
* a soft 404 (the site's own "not found" page served with HTTP 200) comes back as a *success*;
* error payloads make ``langchain_tavily`` raise ``AttributeError`` rather than return a value.
"""

import asyncio
import logging
import os
import re
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .formatting import FetchedPage
from .formatting import compact
from .formatting import parse_fetched_pages
from .formatting import render_page_section
from .formatting import render_result
from .formatting import render_skipped_section
from .formatting import select_window

logger = logging.getLogger(__name__)

# Warn once per process, not once per configured instance - a config may declare several fetch
# tools and duplicate warnings obscure the log.
_missing_key_warned = False

# The citation parser registry is process-global and append-only, while @register_function bodies
# run once per configured instance. Without this guard a config with two fetch tools, or a test
# that builds the same config twice, would stack duplicate parsers for one name.
_registered_parsers: set[str] = set()

# Populated at registration time; read by the routing probe so it tests the shipped description.
_DESCRIPTION_FOR_PROBE: str | None = None

# Phrases that indicate a site served its "not found" page with a 200. Deliberately phrase-level:
# matching a bare "404" would flag any page that happens to discuss HTTP status codes.
_SOFT_404_MARKERS = (
    "page not found",
    "404 not found",
    "page cannot be found",
    "page could not be found",
    "page you requested could not be found",
    "page doesn't exist",
    "page does not exist",
    "no longer exists",
    "error 404",
)

# Only pages shorter than this are eligible for the soft-404 check. A real article that mentions
# "page not found" in its body is long; a site's 404 template is short.
_SOFT_404_MAX_CHARS = 6000


class FetchUrlInput(BaseModel):
    """Model-facing argument schema.

    Passed explicitly to ``FunctionInfo.from_fn`` rather than inferred from the signature, so these
    field descriptions are guaranteed to reach the model. They are the second line of defence
    against the failure this tool exists to prevent: a search query arriving here instead of at
    ``web_search_tool``.
    """

    urls: list[str] = Field(
        ...,
        description=(
            "Exact, complete URLs to open, for example "
            "['https://www.iea.nl/sites/default/files/ICILS_2023_report.pdf']. NOT search "
            "keywords - every item must start with http:// or https://. Pass several URLs to open "
            "them in one call."
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "Optional. What you are looking for on these pages, for example 'table 2.2 computer "
            "literacy by country'. Used ONLY to choose which part of a long page to show you. It "
            "does not search the web and does not change which pages are opened."
        ),
    )
    start_line: int = Field(
        default=0,
        ge=0,
        description=(
            "Optional. Line number to resume from when continuing a page that was truncated; the "
            "truncation note tells you which line to pass. Applies to every URL in this call."
        ),
    )


class WebPageFetchToolConfig(FunctionBaseConfig, name="web_page_fetch"):
    """Tool that opens web pages by URL and returns their extracted text.

    Requires a TAVILY_API_KEY environment variable or api_key config.
    """

    max_urls_per_call: int = Field(default=4, ge=1, description="Maximum URLs accepted in one call")
    max_chars_per_page: int = Field(
        default=10000,
        ge=500,
        description=(
            "Maximum characters shown per page. This is a prompt-context budget, not a download "
            "limit: pages are extracted in full and then windowed, so a large document costs the "
            "same context as a small one."
        ),
    )
    max_chars_per_call: int = Field(
        default=24000,
        ge=500,
        description="Maximum characters shown across all URLs in one call, spent in request order.",
    )
    extract_depth: str = Field(
        default="advanced",
        description=(
            "Tavily extraction depth: 'basic' or 'advanced'. 'advanced' retains tables and embedded "
            "elements and is the default; 'basic' is faster on simple pages."
        ),
    )
    timeout_seconds: int = Field(default=30, ge=1, description="Per-call timeout for the extraction request")
    api_key: SecretStr | None = Field(default=None, description="The API key for the Tavily service")


# Referenced inside _validate_url's message; kept as a module constant so the wording stays in one
# place if the configured tool name ever changes.
_TOOL_NAME_HINT = "This tool"


def _validate_url(candidate: str) -> tuple[str, str]:
    """Return ``(normalized_url, "")`` or ``("", reason)`` for one requested URL.

    The not-a-URL reason *redirects* rather than merely refusing. A model that reached for this tool
    with keywords has the right intent and the wrong tool, so naming the right one converts a wasted
    call into a corrected one.
    """
    text = (candidate or "").strip()
    if not text:
        return "", "Empty URL. Pass a complete address starting with http:// or https://."

    parsed = urlparse(text)
    # Scheme is checked before netloc: `file:///etc/passwd` parses to an *empty* netloc, so a
    # netloc-first check would report it as "not a URL" and send the model back to search instead
    # of telling it the scheme is refused.
    if parsed.scheme and parsed.scheme not in ("http", "https", "www"):
        return "", (
            f"Only http:// and https:// URLs can be opened (got '{parsed.scheme}://'). This tool reads web pages only."
        )
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "", (
            f'"{text[:120]}" is not a URL. {_TOOL_NAME_HINT} opens pages you already have the '
            "address for; use web_search_tool to find one first, then pass the URL it returns."
        )
    return text, ""


def _looks_like_soft_404(text: str) -> bool:
    """Return whether an extracted page is probably a "not found" page served with HTTP 200.

    Tavily reports these as successes, so without this check the agent would read a site's 404
    template as though it were the requested document - and, worse, cite it.
    """
    if not text or len(text) > _SOFT_404_MAX_CHARS:
        return False
    lowered = re.sub(r"\s+", " ", text.lower())
    return any(marker in lowered for marker in _SOFT_404_MARKERS)


def _page_from_result(requested: str, result: dict) -> FetchedPage:
    """Build a :class:`FetchedPage` from one Tavily result entry."""
    text = compact((result.get("raw_content") or "").strip())
    final_url = (result.get("url") or requested).strip()
    title = (result.get("title") or "").strip()

    if not text:
        return FetchedPage(
            url=requested,
            status="failed",
            reason="Could not read this page: the extractor returned no content for it.",
        )
    if _looks_like_soft_404(text):
        return FetchedPage(
            url=requested,
            final_url=final_url,
            title=title,
            text=text,
            status="suspect",
            reason=(
                "[Caution: this looks like a 'page not found' page rather than the document you "
                "asked for. Do not cite it. Search for the current location of this page instead.]"
            ),
        )
    return FetchedPage(url=requested, final_url=final_url, title=title, text=text, status="ok")


def _normalize(url: str) -> str:
    """Return a comparison key tolerant of the trailing-slash differences Tavily introduces."""
    return url.strip().rstrip("/").lower()


async def _extract(tool, urls: list[str], *, extract_depth: str, timeout_seconds: int) -> tuple[list[dict], str]:
    """Call Tavily Extract and return ``(results, error)``.

    ``query`` is deliberately absent from the payload. Forwarding it made Tavily return a
    relevance-selected fragment instead of the document - measured at 1,025 characters from a
    1.07M-character report, not containing the requested table. Windowing is done locally instead,
    over the full extract, where a miss is visible and recoverable via ``start_line``.
    """
    try:
        payload = {"urls": urls, "extract_depth": extract_depth}
        response = await asyncio.wait_for(tool.ainvoke(payload), timeout=timeout_seconds)
    except TimeoutError:
        return [], f"the extraction service did not respond within {timeout_seconds} seconds"
    except AttributeError:
        # langchain_tavily assumes a dict response and calls .get() on it; an API-level error comes
        # back as a bare string, so the wrapper raises instead of surfacing the message.
        return [], "the extraction service returned an error response"
    except Exception as exc:  # noqa: BLE001 - a tool must return a message, never raise
        logger.warning("web_page_fetch extraction failed: %s", type(exc).__name__)
        return [], f"the extraction request failed ({type(exc).__name__})"

    if isinstance(response, str):
        return [], "the extraction service returned an error response"
    if not isinstance(response, dict) or response.get("error"):
        return [], "the extraction service returned an error response"
    results = response.get("results")
    return (results if isinstance(results, list) else []), ""


@register_function(config_type=WebPageFetchToolConfig)
async def web_page_fetch(tool_config: WebPageFetchToolConfig, builder: Builder):
    """Register the web page fetch tool with NAT."""
    from langchain_tavily import TavilyExtract

    if not os.environ.get("TAVILY_API_KEY") and tool_config.api_key:
        os.environ["TAVILY_API_KEY"] = tool_config.api_key.get_secret_value()

    tool_name = tool_config.type
    try:
        tool_name = tool_config.name or tool_config.type
    except AttributeError:  # pragma: no cover - older NAT config shapes
        pass

    if not os.environ.get("TAVILY_API_KEY"):
        global _missing_key_warned
        if not _missing_key_warned:
            logger.warning(
                "TAVILY_API_KEY not found. The page fetch tool will be registered but will "
                "return an error when called. To enable: set TAVILY_API_KEY in your environment, "
                ".env file, or specify api_key in your workflow config."
            )
            _missing_key_warned = True

        async def _fetch_url_stub(urls: list[str], query: str | None = None, start_line: int = 0) -> str:
            """Page fetch tool (unavailable - missing TAVILY_API_KEY)."""
            return (
                "Error: Page fetching is unavailable because TAVILY_API_KEY is not set.\n"
                "To enable this tool:\n"
                "1. Get an API key from https://tavily.com/\n"
                "2. Set the API key in your environment or in your .env file\n"
                "3. Restart the application"
            )

        yield FunctionInfo.from_fn(
            _fetch_url_stub,
            input_schema=FetchUrlInput,
            description=_fetch_url_stub.__doc__,
        )
        return

    # Replace the citation registry's generic URL scraper for this tool. Without it, every outbound
    # link on a fetched page would be registered as a source the agent never read.
    if tool_name not in _registered_parsers:
        try:
            from aiq_agent.common.citation_verification import register_source_parser

            register_source_parser(lambda name, _t=tool_name.lower(): name == _t, parse_fetched_pages)
            _registered_parsers.add(tool_name)
        except ImportError:  # pragma: no cover - package used outside an AI-Q install
            logger.debug("aiq_agent not importable; skipping source-parser registration")

    extractor = TavilyExtract(extract_depth=tool_config.extract_depth)

    async def _fetch_url(urls: list[str], query: str | None = None, start_line: int = 0) -> str:
        """Opens web pages you already have the URL for and returns their full text.

        This is a READER, not a FINDER. Give it exact URLs and it returns the complete page
        content - full tables, full lists, figures and footnotes - not a snippet.

        USE THIS TOOL WHEN:
        - The question names or supplies a specific source ("according to JD Power", "in World Bank
          Open Data", "table 2.2 of the ICILS 2023 report", or a literal https://... in the question).
        - A search result looks right and you need the actual numbers, rows, dates, or wording from
          it. Search snippets are truncated and routinely omit the exact cell you need.
        - You must read a table, a filing, a timetable, a database page, or a PDF.
        - You are about to run the same search a third time. Open the best URL you already have.

        DO NOT USE THIS TOOL WHEN:
        - You do not have a URL yet. It cannot discover pages. Call web_search_tool first, then open
          the URL it returns.
        - You would be passing keywords, a question, or a site name. Those are search inputs, and
          this tool will reject them.

        HOW IT DIFFERS FROM THE SEARCH TOOLS:
        | | web_search_tool / advanced_web_search_tool | fetch_url_tool |
        | Input | keywords or a question | exact URLs (https://...) |
        | Answers | "which pages exist about X?" | "what does THIS page actually say?" |
        | Returns | ranked, truncated snippets | the full page, tables included |
        | Finds new pages | yes | no |

        The normal loop is: search -> pick the URL -> fetch -> read. A search tells you where the
        answer lives; only a fetch tells you what it is.

        EXAMPLES:
            # The question supplies a URL - open it, do not search around it.
            urls=["https://pmc.ncbi.nlm.nih.gov/articles/PMC9506306/"]

            # Search first, then open the result you need the numbers from.
            web_search_tool("ICILS 2023 international report chapter 2")
              -> returns https://www.iea.nl/sites/default/files/ICILS_2023_report.pdf
            urls=["https://www.iea.nl/sites/default/files/ICILS_2023_report.pdf"],
            query="table 2.2 availability of CIL and CT subjects"

            # Compare three official pages in one call.
            urls=["https://www.fdic.gov/a", "https://www.fdic.gov/b", "https://www.fdic.gov/c"]

            # WRONG - this is a search, not a fetch. Use web_search_tool instead.
            urls=["maple syrup production by state 2017"]

        Args:
            urls (list[str]): Exact, complete URLs to open. Not search terms.
            query (str | None): Optional. What you are looking for on the page; used only to choose
                which part of an over-long page to show.
            start_line (int): Optional. Line to resume from when continuing a truncated page.

        Returns:
            str: One section per URL - the resolved URL, the page title, and the page content with
                line numbers, or a per-URL reason it could not be read.
        """
        requested = list(urls or [])
        if not requested:
            return (
                "Error: No URLs provided. Pass one or more exact URLs, for example "
                "urls=['https://www.example.gov/report']. To find a URL first, use web_search_tool."
            )
        if len(requested) > tool_config.max_urls_per_call:
            return (
                f"Error: {tool_name} accepts at most {tool_config.max_urls_per_call} URLs per call; "
                f"received {len(requested)}. Split them across calls, most important first."
            )

        valid: list[str] = []
        pages: dict[str, FetchedPage] = {}
        for candidate in requested:
            normalized, reason = _validate_url(candidate)
            if reason:
                pages[candidate] = FetchedPage(url=candidate, status="failed", reason=reason)
            else:
                valid.append(normalized)

        if valid:
            results, error = await _extract(
                extractor,
                valid,
                extract_depth=tool_config.extract_depth,
                timeout_seconds=tool_config.timeout_seconds,
            )
            by_url = {_normalize(str(item.get("url", ""))): item for item in results if isinstance(item, dict)}
            for url in valid:
                match = by_url.get(_normalize(url))
                if match is not None:
                    pages[url] = _page_from_result(url, match)
                elif error:
                    pages[url] = FetchedPage(
                        url=url,
                        status="failed",
                        reason=f"Could not read this page: {error}.",
                    )
                else:
                    pages[url] = FetchedPage(
                        url=url,
                        status="failed",
                        reason=(
                            "Could not read this page: the extractor returned nothing for this URL. "
                            "It may be unreachable, blocked, or not a readable document."
                        ),
                    )

        # Render in the order the model asked for, so the output maps onto its own request.
        sections: list[str] = []
        remaining = tool_config.max_chars_per_call
        any_success = False
        for candidate in requested:
            page = pages.get(candidate) or pages.get((candidate or "").strip())
            if page is None:  # pragma: no cover - every requested URL gets a page above
                continue
            if page.status == "failed":
                sections.append(render_page_section(page, None, tool_name=tool_name))
                continue

            any_success = True
            budget = min(tool_config.max_chars_per_page, remaining)
            if budget <= 0:
                sections.append(render_skipped_section(page, max_chars_per_call=tool_config.max_chars_per_call))
                continue
            window = select_window(
                page.text,
                max_chars=budget,
                query=query or "",
                start_line=start_line,
            )
            remaining -= window.shown_chars
            sections.append(render_page_section(page, window, tool_name=tool_name))

        if not any_success:
            # Leading with "Error:" is load-bearing: AI-Q's source-tool wrapper uses that prefix to
            # classify a result as a non-citable failure, which keeps a wholly-failed fetch out of
            # the citation registry and feeds the source-tool circuit breaker.
            reasons = "\n".join(f"- {p.url}: {p.reason}" for p in (pages.get(c) for c in requested) if p is not None)
            return f"Error: none of the requested URLs could be read.\n{reasons}"

        return render_result(sections)

    # Exported for the routing probe (tests/aiq_agent/.../test_routing_probe.py), which must
    # exercise the description that actually ships rather than a hand-copied one.
    global _DESCRIPTION_FOR_PROBE
    _DESCRIPTION_FOR_PROBE = _fetch_url.__doc__

    yield FunctionInfo.from_fn(
        _fetch_url,
        input_schema=FetchUrlInput,
        description=_fetch_url.__doc__,
    )
