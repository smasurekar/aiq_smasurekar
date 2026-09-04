#!/usr/bin/env python3
"""Round two of the Tavily Extract security probe: corrected baselines and a sharper SSRF test.

Round one (``security_probe.py``) established that Extract refuses literal private-IP and
non-HTTP targets, and that it returns 14-25x more text than a Search content chunk. It could
not answer T1/T2 because the direct-fetch baseline was blocked by user-agent filtering, and
T3's corpus URL 404'd. This run fixes both and adds:

  T4b  Does URL validation resolve DNS, or only pattern-match? ``localtest.me`` is a public
       hostname that resolves to loopback. If Extract accepts it, the round-one refusals are
       string filtering rather than egress control, and the SSRF finding is much weaker.

Usage:
    dotenv -f deploy/.env run uv run python misc/tavily/security_probe2.py
"""

import html
import json
import os
import re
import sys

import httpx

API = "https://api.tavily.com"
KEY = os.environ.get("TAVILY_API_KEY", "")
# Wikipedia and most CDNs 403 a bare client; the round-one baseline collapsed to 126 chars without this.
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

PAGE_HIDDEN = "https://en.wikipedia.org/wiki/Prompt_injection"
INJECTION_CORPORA = [
    "https://raw.githubusercontent.com/greshake/llm-security/main/README.md",
    "https://raw.githubusercontent.com/TakSec/Prompt-Injection-Everywhere/main/README.md",
    "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/README.md",
]
# Public hostnames that resolve into loopback / private space. Distinguishes pattern-matching
# from resolution-aware egress control.
REBIND_TARGETS = [
    "http://localtest.me/",
    "http://127.0.0.1.nip.io/",
    "http://10.0.0.1.nip.io/",
    "http://spoofed.burpcollaborator.net/",
]


def post(path: str, payload: dict, timeout: float = 60.0) -> dict:
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


def extract(urls: list[str], depth: str = "advanced") -> dict:
    """Call POST /extract for the given URLs at the given extraction depth."""
    return post("/extract", {"urls": urls, "extract_depth": depth})


def norm(text: str) -> str:
    """Collapse whitespace and case so substring checks survive markdown reflowing."""
    return re.sub(r"\s+", " ", text or "").lower()


def banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def t1_t2() -> None:
    """T1/T2: raw-content fidelity, and whether non-rendered markup survives extraction."""
    banner("T1/T2  Raw-content fidelity and hidden-markup pass-through")
    print(f"page: {PAGE_HIDDEN}")
    raw_html = httpx.get(PAGE_HIDDEN, headers=UA, timeout=30, follow_redirects=True).text

    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    stripped = re.sub(r"<!--.*?-->", " ", stripped, flags=re.S)
    visible = norm(html.unescape(re.sub(r"<[^>]+>", " ", stripped)))

    res = extract([PAGE_HIDDEN])
    results = res.get("results") or []
    if not results:
        print(f"  no results: {json.dumps(res)[:300]}")
        return
    content = results[0].get("raw_content") or ""
    visible_ratio = len(content) / max(len(visible), 1)
    print(f"  source HTML              : {len(raw_html):>9,} chars")
    print(f"  direct visible text      : {len(visible):>9,} chars")
    print(f"  Tavily raw_content       : {len(content):>9,} chars  ({visible_ratio:.0%} of visible text)")

    sentences = [s.strip() for s in re.split(r"(?<=\.) ", visible) if 80 < len(s.strip()) < 200]
    sample = sentences[10:30]
    verbatim = sum(1 for s in sample if norm(s) in norm(content))
    print(f"  verbatim reproduction    : {verbatim}/{len(sample)} sampled page sentences appear unchanged")
    print("  --- sample of returned text ---")
    print("  " + content[1000:1500].replace("\n", "\n  "))

    # T2: does anything a browser would not render make it into the returned text?
    comments = [html.unescape(c).strip() for c in re.findall(r"<!--(.*?)-->", raw_html, re.S) if len(c.strip()) > 40]
    hid = re.findall(
        r'<(\w+)[^>]*(?:display\s*:\s*none|aria-hidden="true"|\shidden[ =>])[^>]*>(.*?)</\1>',
        raw_html,
        re.S | re.I,
    )
    hidden_el = [html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", f))).strip() for _, f in hid]
    hidden_el = [t for t in hidden_el if len(t) > 40]

    for kind, items in (("html_comment", comments), ("hidden_element", hidden_el)):
        leaked = [t for t in items if norm(t)[:60] and norm(t)[:60] in norm(content)]
        print(f"  {kind:<16}: {len(items):>2} candidates on page, {len(leaked)} present in extractor output")
        for t in leaked[:3]:
            print(f"      LEAKED -> {t[:110]!r}")


def t3() -> None:
    """T3: is adversarial instruction text filtered, or returned verbatim?"""
    banner("T3  Adversarial instruction text pass-through")
    for url in INJECTION_CORPORA:
        res = extract([url])
        results = res.get("results") or []
        if not results:
            print(f"\n  {url}\n    no results: {json.dumps(res.get('failed_results'))[:200]}")
            continue
        content = results[0].get("raw_content") or ""
        n = norm(content)
        markers = {
            "ignore previous/above": bool(re.search(r"ignore (all )?(previous|above|prior)", n)),
            "'you are now'": "you are now" in n,
            "'do anything now'/DAN": "do anything now" in n or re.search(r"\bdan\b", n) is not None,
            "'system prompt'": "system prompt" in n,
            "'jailbreak'": "jailbreak" in n,
            "injected imperative": bool(re.search(r"(new instructions?|disregard|override)", n)),
        }
        print(f"\n  {url}")
        print(f"    {len(content):,} chars returned, HTTP {res.get('_http_status')}")
        print(f"    markers found: {[k for k, v in markers.items() if v]}")
        snippet = content[:400].replace("\n", " ")
        print(f"    verbatim head: {snippet[:300]!r}")


def t4b() -> None:
    """T4b: is Extract's URL validation resolution-aware, or only pattern-matching?"""
    banner("T4b  Does URL validation resolve DNS, or only pattern-match the literal?")
    for target in REBIND_TARGETS:
        res = extract([target], depth="basic")
        ok = res.get("results") or []
        status = res.get("_http_status")
        if ok:
            body = (ok[0].get("raw_content") or "")[:200].replace("\n", " ")
            print(f"  ACCEPTED  {target}\n            HTTP {status} -> {body!r}")
        else:
            blob = json.dumps(res)
            kind = "format-rejected" if "Invalid URL format" in blob else "fetch-failed"
            print(f"  {kind:<9} {target}\n            HTTP {status} -> {blob[:200]}")


def main() -> None:
    """Run every round-two probe."""
    if not KEY:
        sys.exit("TAVILY_API_KEY is not set. Run under: dotenv -f deploy/.env run ...")
    print(f"Tavily security probe (round 2) | endpoint {API}")
    t1_t2()
    t3()
    t4b()
    print("\ndone")


if __name__ == "__main__":
    main()
