#!/usr/bin/env python3
"""Round three: does ``include_raw_content`` on /search inherit the Extract risk profile?

Rounds one and two (``security_probe.py``, ``security_probe2.py``) characterised the *Extract*
endpoint. This run asks the narrower question the review actually needs answered: if we stop
short of a fetch tool and instead just flip ``include_raw_content`` on the existing Search call,
what changes about the injection surface?

  T6  Identity and volume. How does search ``raw_content`` compare to the reranked ``content``
      chunk Search returns today, and to Extract's ``raw_content`` for the same URL? Compared by
      SHA-1, not just by length, so "similar" and "the same bytes" are distinguishable.
  T7  Carriers. Do the invisible-markup channels measured in T2 (HTML comments, display:none,
      non-rendered ``title=`` attributes) survive into search raw_content?
  T8  Filtering. Does Search apply any content safety to raw_content that Extract did not?
      Queried against pages whose entire subject matter is prompt injection.
  T9  Format. ``include_raw_content`` accepts true / "markdown" / "text". Does the choice change
      how much attacker-controlled text reaches the model? Run over several queries because a
      single query's result set drifts between calls and a one-shot comparison reads as signal.

The URL-choice question is deliberately *not* a probe: on Search the URL set is chosen by
Tavily's ranker from a natural-language query, never by the model. That structural difference
is the whole point of the comparison and needs no measurement.

Usage:
    dotenv -f deploy/.env run uv run python misc/tavily/security_probe3.py
"""

import hashlib
import html
import json
import os
import re
import sys

import httpx

API = "https://api.tavily.com"
KEY = os.environ.get("TAVILY_API_KEY", "")
# Wikipedia and most CDNs 403 a bare client; without this the direct-fetch baseline collapses.
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

# Same page as T1/T2 so the numbers are directly comparable to the Extract findings.
PAGE_HIDDEN = "https://en.wikipedia.org/wiki/Prompt_injection"

# Queries chosen to surface pages that are *about* prompt injection, i.e. the strongest
# adversarial-text material a benign public query can pull into the result set.
INJECTION_QUERIES = [
    "prompt injection payload ignore previous instructions examples github",
    "jailbreak prompts DAN do anything now collection",
]

# Marker families that indicate raw adversarial instruction text reached the caller verbatim.
MARKERS = {
    "ignore previous/above": re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    "injected imperative": re.compile(r"\b(you\s+are\s+now|from\s+now\s+on|disregard)\b", re.I),
    "DAN / do anything now": re.compile(r"\bdo\s+anything\s+now\b|\bDAN\b", re.I),
    "system-prompt exfil": re.compile(r"(reveal|print|repeat)\s+(your\s+)?(system\s+)?prompt", re.I),
}


def post(path: str, payload: dict, timeout: float = 90.0) -> dict:
    """POST to the Tavily API and return the decoded body, or a synthetic error dict."""
    try:
        r = httpx.post(f"{API}{path}", json=payload, headers={"Authorization": f"Bearer {KEY}"}, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - probe reports transport failures as data
        return {"_transport_error": f"{type(exc).__name__}: {exc}"}
    try:
        body = r.json()
    except ValueError:
        body = {"_non_json_body": r.text[:400]}
    body["_http_status"] = r.status_code
    return body


def search(query: str, *, raw: object = None, max_results: int = 3, depth: str = "advanced") -> dict:
    """Call POST /search, optionally requesting raw_content in the given format."""
    payload = {"query": query, "max_results": max_results, "search_depth": depth}
    if raw is not None:
        payload["include_raw_content"] = raw
    return post("/search", payload)


def extract(urls: list[str], depth: str = "advanced") -> dict:
    """Call POST /extract for the given URLs, for volume comparison against search raw_content."""
    return post("/extract", {"urls": urls, "extract_depth": depth})


def banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def hidden_strings(page_html: str) -> dict[str, list[str]]:
    """Pull text a browser would not render: comments, hidden elements, title attributes.

    Conservative and regex-based on purpose: we only need representative samples long enough to
    search for in the API output, not a faithful parse.
    """
    comments = [html.unescape(c).strip() for c in re.findall(r"<!--(.*?)-->", page_html, re.S)]
    # Looser than round two's paired-tag pattern on purpose: round two required a matching
    # closing tag and >40 chars, which silently dropped Wikipedia's 34-char display:none
    # short-description and produced a false "0 survived".
    hidden_pattern = r"<[^>]*(?:display:\s*none|aria-hidden=\"true\"|\shidden[\s>])[^>]*>(.*?)<"
    hidden = [html.unescape(re.sub(r"<[^>]+>", " ", m)).strip() for m in re.findall(hidden_pattern, page_html, re.S)]
    titles = [html.unescape(t).strip() for t in re.findall(r'\stitle="([^"]{25,})"', page_html)]

    def keep(items: list[str]) -> list[str]:
        """Drop fragments too short to search for reliably, and cap the sample size."""
        return [x for x in items if len(x) >= 25][:40]

    return {
        "HTML comments": keep(comments),
        "display:none / aria-hidden": keep(hidden),
        "title= attributes": keep(titles),
    }


def carriers_report(page_html: str, output: str) -> None:
    """Report, per carrier family, how many hidden candidates survived into the output text."""
    flat = re.sub(r"\s+", " ", output)
    for label, samples in hidden_strings(page_html).items():
        if not samples:
            print(f"  {label:<28} 0 candidates on page")
            continue
        survivors = [s for s in samples if re.sub(r"\s+", " ", s)[:60] in flat]
        print(f"  {label:<28} {len(samples):>3} candidates -> {len(survivors):>3} present in output")
        for s in survivors[:2]:
            print(f"      SURVIVED -> {s[:110]!r}")


def t6_identity() -> None:
    """T6a: is search raw_content the same bytes as extract raw_content for one pinned URL?

    Pinned to a page the ranker reliably returns for its own title, because comparing across two
    independent search calls otherwise trips over result-set drift.
    """
    banner("T6a  Is search raw_content byte-identical to extract raw_content?")
    res = search("Prompt injection wikipedia", raw=True, max_results=5)
    hit = next((r for r in (res.get("results") or []) if "wiki/Prompt_injection" in (r.get("url") or "")), None)
    if hit is None:
        print(f"  target page not in results: {[r.get('url') for r in (res.get('results') or [])]}")
        return
    sraw = hit.get("raw_content") or ""
    ext = extract([PAGE_HIDDEN]).get("results") or []
    eraw = (ext[0].get("raw_content") or "") if ext else ""
    for label, text in (("search  raw_content", sraw), ("extract raw_content", eraw)):
        print(f"  {label}: {len(text):>8,} chars  sha1 {hashlib.sha1(text.encode()).hexdigest()[:16]}")
    print(f"  identical: {sraw == eraw}")


def t6_volume() -> None:
    """T6b: search content chunk vs search raw_content vs extract raw_content, same URLs."""
    banner("T6b  Volume: search 'content' chunk vs search raw_content vs extract raw_content")
    q = "prompt injection attack against language models"
    plain = search(q, max_results=3)
    withraw = search(q, raw=True, max_results=3)
    if not (plain.get("results") and withraw.get("results")):
        print(f"  search failed: {json.dumps(plain)[:200]} / {json.dumps(withraw)[:200]}")
        return
    print(f"  result fields without include_raw_content: {sorted(plain['results'][0])}")
    print(f"  result fields with    include_raw_content: {sorted(withraw['results'][0])}")

    by_url = {r.get("url"): r for r in withraw["results"]}
    for hit in plain["results"][:3]:
        url = hit.get("url")
        chunk = hit.get("content") or ""
        raw = (by_url.get(url, {}).get("raw_content") or "") if url in by_url else ""
        ext_results = extract([url]).get("results") or []
        full = (ext_results[0].get("raw_content") or "") if ext_results else ""
        print(f"\n  {url}")
        print(f"    search 'content' chunk   : {len(chunk):>8,} chars")
        if url not in by_url:
            print("    search 'raw_content'     : URL not in the raw-enabled result set (ranker drift)")
        else:
            ratio = f"{len(raw) / len(chunk):.0f}x chunk" if chunk and raw else "n/a"
            print(f"    search 'raw_content'     : {len(raw):>8,} chars   ({ratio})")
        cover = f"{len(raw) / len(full):.0%} of extract" if raw and full else "n/a"
        print(f"    extract 'raw_content'    : {len(full):>8,} chars   (search raw = {cover})")


def t7_carriers() -> None:
    """T7: do invisible-markup carriers survive into search raw_content, as they do on Extract?"""
    banner("T7  Hidden-markup carriers in search raw_content (same page as T2)")
    try:
        page = httpx.get(PAGE_HIDDEN, headers=UA, timeout=60.0, follow_redirects=True).text
    except Exception as exc:  # noqa: BLE001
        print(f"  baseline fetch failed: {type(exc).__name__}: {exc}")
        return
    print(f"  source HTML: {len(page):,} chars")

    # Query the page by title so the ranker is very likely to return it as a top hit.
    res = search("Prompt injection wikipedia", raw=True, max_results=5)
    hit = next((r for r in (res.get("results") or []) if "wiki/Prompt_injection" in (r.get("url") or "")), None)
    if hit is None:
        print(f"  target page not in results: {[r.get('url') for r in (res.get('results') or [])]}")
        return
    raw = hit.get("raw_content") or ""
    print(f"  search raw_content: {len(raw):,} chars\n")
    carriers_report(page, raw)


def t8_filtering() -> None:
    """T8: does Search apply content filtering to raw_content that Extract did not?"""
    banner("T8  Adversarial-text passthrough in search raw_content")
    for q in INJECTION_QUERIES:
        res = search(q, raw=True, max_results=3)
        results = res.get("results") or []
        print(f"\n  query: {q}")
        print(f"  http {res.get('_http_status')} | envelope fields: {sorted(k for k in res if not k.startswith('_'))}")
        if not results:
            print(f"    no results: {json.dumps(res)[:200]}")
            continue
        for r in results:
            raw = r.get("raw_content") or ""
            found = [name for name, rx in MARKERS.items() if rx.search(raw)]
            print(f"    {len(raw):>7,} chars  {r.get('url')}")
            print(f"             markers: {', '.join(found) if found else 'none'}")


# Three unrelated queries: an adversarial-corpus one, a vendor-docs one, and a benign report
# lookup, so the format comparison is not read off a single result set.
FORMAT_QUERIES = [
    "prompt injection attack against language models",
    "NVIDIA NeMo guardrails documentation",
    "ICILS 2023 international computer literacy report findings",
]


def t9_formats() -> None:
    """T9: compare no-raw / true / 'markdown' / 'text' for coverage and volume, across queries."""
    banner("T9  include_raw_content format variants (per-query, 5 results each)")
    for q in FORMAT_QUERIES:
        print(f"\n  query: {q}")
        for variant in (None, True, "markdown", "text"):
            res = search(q, raw=variant, max_results=5)
            results = res.get("results") or []
            if not results:
                print(f"    {json.dumps(variant):<12} http {res.get('_http_status')} -> {json.dumps(res)[:140]}")
                continue
            nonnull = sum(1 for r in results if r.get("raw_content"))
            raw_chars = sum(len(r.get("raw_content") or "") for r in results)
            chunk_chars = sum(len(r.get("content") or "") for r in results)
            mult = f"{raw_chars / chunk_chars:.0f}x chunks" if chunk_chars and raw_chars else "-"
            print(
                f"    {json.dumps(variant):<12} raw {nonnull}/{len(results)} results"
                f" | raw {raw_chars:>8,} chars | chunks {chunk_chars:>6,} chars | {mult}"
            )


def main() -> None:
    """Run every probe in order."""
    if not KEY:
        sys.exit("TAVILY_API_KEY is not set. Run under: dotenv -f deploy/.env run ...")
    print(f"Tavily search raw_content probe | key prefix {KEY[:6]}... | endpoint {API}")
    t6_identity()
    t6_volume()
    t7_carriers()
    t8_filtering()
    t9_formats()
    print("\ndone")


if __name__ == "__main__":
    main()
