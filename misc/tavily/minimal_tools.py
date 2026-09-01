#!/usr/bin/env python3
"""Call the AI-Q Tavily tools directly, without an agent or a workflow config.

Registers each NAT function by hand and invokes its ``single_fn``, so the output is exactly the
string an agent would see.

Usage (TAVILY_API_KEY must be set, e.g. via ``dotenv -f deploy/.env run``):

    uv run python misc/tavily/minimal_tools.py search "ICILS 2023 report"
    uv run python misc/tavily/minimal_tools.py fetch https://www.iea.nl/ --query "computer literacy"
"""

import argparse
import asyncio

from tavily_web_search.register import TavilyWebSearchToolConfig
from tavily_web_search.register import tavily_web_search
from web_page_fetch.register import FetchUrlInput
from web_page_fetch.register import WebPageFetchToolConfig
from web_page_fetch.register import web_page_fetch


async def search(question: str, max_results: int, advanced: bool) -> str:
    """Run web_search_tool once and return its rendered documents."""
    config = TavilyWebSearchToolConfig(max_results=max_results, advanced_search=advanced, max_content_length=1000)
    async with tavily_web_search(config, None) as info:
        return await info.single_fn(question)


async def fetch(urls: list[str], query: str | None, start_line: int) -> str:
    """Run fetch_url_tool once and return its rendered page sections."""
    config = WebPageFetchToolConfig(max_urls_per_call=4, max_chars_per_page=10000, max_chars_per_call=24000)
    async with web_page_fetch(config, None) as info:
        return await info.single_fn(FetchUrlInput(urls=urls, query=query, start_line=start_line))


def main() -> None:
    """Parse arguments and print the selected tool's output."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    search_cmd = commands.add_parser("search", help="call web_search_tool (tavily_web_search)")
    search_cmd.add_argument("question")
    search_cmd.add_argument("--max-results", type=int, default=5)
    search_cmd.add_argument("--advanced", action="store_true")

    fetch_cmd = commands.add_parser("fetch", help="call fetch_url_tool (web_page_fetch)")
    fetch_cmd.add_argument("urls", nargs="+")
    fetch_cmd.add_argument("--query", default=None, help="selects which window of a long page is shown")
    fetch_cmd.add_argument("--start-line", type=int, default=0)

    args = parser.parse_args()
    if args.command == "search":
        print(asyncio.run(search(args.question, args.max_results, args.advanced)))
    else:
        print(asyncio.run(fetch(args.urls, args.query, args.start_line)))


if __name__ == "__main__":
    main()
