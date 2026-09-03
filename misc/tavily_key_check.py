#!/usr/bin/env python3
"""Validate a Tavily API key and display its current usage and quota."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

API_BASE_URL = "https://api.tavily.com"
TIMEOUT_SECONDS = 30


def tavily_request(api_key: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send an authenticated request to Tavily and return its JSON response."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{API_BASE_URL}{endpoint}",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        response_body = error.read().decode(errors="replace")
        try:
            details = json.loads(response_body)
        except json.JSONDecodeError:
            details = response_body

        if error.code == 401:
            reason = "The API key is missing, invalid, or revoked."
        elif error.code == 429:
            retry_after = error.headers.get("Retry-After", "the indicated delay")
            reason = f"Rate limit reached; retry after {retry_after} seconds."
        else:
            reason = f"Tavily returned HTTP {error.code}."
        raise RuntimeError(f"{reason}\nResponse: {json.dumps(details, indent=2)}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Tavily: {error.reason}") from error


def number(value: Any) -> int | float:
    """Return a numeric API value, using zero for absent/non-numeric values."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def print_usage(usage: dict[str, Any]) -> None:
    """Print a friendly usage summary followed by the complete API response."""
    key = usage.get("key", {})
    account = usage.get("account", {})

    key_used = number(key.get("usage"))
    key_limit = number(key.get("limit"))
    plan_used = number(account.get("plan_usage"))
    plan_limit = number(account.get("plan_limit"))
    paygo_used = number(account.get("paygo_usage"))
    paygo_limit = number(account.get("paygo_limit"))

    print("Tavily API key: VALID")
    print(f"Plan: {account.get('current_plan', 'unknown')}")
    print()
    print("Key quota")
    print(f"  Used:      {key_used}")
    print(f"  Limit:     {key_limit}")
    print(f"  Remaining: {max(key_limit - key_used, 0)}")
    print()
    print("Account quota")
    print(f"  Plan used:      {plan_used}")
    print(f"  Plan limit:     {plan_limit}")
    print(f"  Plan remaining: {max(plan_limit - plan_used, 0)}")
    print(f"  PAYGO used:     {paygo_used}")
    print(f"  PAYGO limit:    {paygo_limit}")

    usage_fields = ("search_usage", "extract_usage", "crawl_usage", "map_usage", "research_usage")
    print()
    print("Usage by operation")
    for field in usage_fields:
        label = field.removesuffix("_usage").capitalize()
        print(f"  {label:<9} key={number(key.get(field)):<8} account={number(account.get(field))}")

    print()
    print("Complete /usage response")
    print(json.dumps(usage, indent=2, sort_keys=True))


def run_search_test(api_key: str) -> None:
    """Run a minimal basic search. This currently consumes one Tavily credit."""
    result = tavily_request(
        api_key,
        "/search",
        {
            "query": "What is Tavily?",
            "search_depth": "basic",
            "max_results": 1,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_usage": True,
        },
    )
    print()
    print("Live search test: PASSED")
    print(f"  Request ID: {result.get('request_id', 'not returned')}")
    print(f"  Credits:    {result.get('usage', {}).get('credits', 'not returned')}")
    print(f"  Results:    {len(result.get('results', []))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-search",
        action="store_true",
        help="also run a real basic search (normally costs one credit)",
    )
    args = parser.parse_args()

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        print("TAVILY_API_KEY is not set.", file=sys.stderr)
        print("Set it with: export TAVILY_API_KEY='tvly-...'", file=sys.stderr)
        return 2

    try:
        usage = tavily_request(api_key, "/usage")
        print_usage(usage)
        if args.test_search:
            run_search_test(api_key)
    except RuntimeError as error:
        print(f"Tavily check FAILED: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
