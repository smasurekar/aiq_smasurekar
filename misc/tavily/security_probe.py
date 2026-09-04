#!/usr/bin/env python3
"""Empirical security-compliance probe for the Tavily Extract endpoint.

Answers, with evidence rather than vendor claims, the questions raised in the AI-Q security
review of the fetch tool:

  T1  Does Extract return raw page content, or a summary / filtered rendition?
  T2  Does hidden HTML (comments, display:none, aria-hidden) survive into the returned text?
      This is the classic invisible prompt-injection carrier.
  T3  Does Extract pass adversarial instruction text through verbatim, or does it filter it?
  T4  Does Extract honor arbitrary URLs pointing at internal / metadata / non-HTTP targets?
      This is the SSRF-as-a-service question.
  T5  How much more raw text does Extract expose than Search's reranked content chunks?
      Rich Harang credited chunking as risk-reducing; this quantifies the delta.

Calls the Tavily REST API directly so the response is unmodified by the AI-Q formatting layer.

Usage:
    dotenv -f deploy/.env run uv run python misc/tavily/security_probe.py
"""

import html
import json
import os
import re
import sys

import httpx

API = "https://api.tavily.com"
KEY = os.environ.get("TAVILY_API_KEY", "")

# Public pages used as probes. Chosen because they are stable, benign, and already contain the
# structures under test (hidden markup; adversarial instruction text as *documented subject matter*).
PAGE_HIDDEN = "https://en.wikipedia.org/wiki/Prompt_injection"
PAGE_INJECTION_CORPUS = "https://raw.githubusercontent.com/greshake/llm-security/main/scenarios/manipulation/simple.md"

# Targets that a correctly-hardened extractor should refuse. These probe the vendor's own
# egress controls; nothing here is attacked, we only observe whether Tavily will dial them.
SSRF_TARGETS = [
    "http://169.254.169.254/latest/meta-data/",  # AWS/GCP instance metadata
    "http://metadata.google.internal/computeMetadata/v1/",  # GCP metadata by name
    "http://127.0.0.1:8000/",  # extractor-local loopback
    "http://[::1]:8000/",  # loopback, IPv6
    "http://10.0.0.1/",  # RFC1918 private space
    "file:///etc/passwd",  # non-HTTP scheme
]


def post(path: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST to the Tavily API and return the decoded body, or a synthetic error dict."""
    try:
        r = httpx.post(
            f"{API}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {KEY}"},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - probe reports transport failures as data
        return {"_transport_error": f"{type(exc).__name__}: {exc}"}
    try:
        body = r.json()
    except ValueError:
        body = {"_non_json_body": r.text[:500]}
    body["_http_status"] = r.status_code
    return body


def extract(urls: list[str], depth: str = "advanced") -> dict:
    """Call POST /extract for the given URLs at the given extraction depth."""
    return post("/extract", {"urls": urls, "extract_depth": depth})


def hidden_strings(page_html: str) -> dict[str, list[str]]:
    """Pull text that a browser would not render: comments and hidden-attribute elements.

    Deliberately regex-based and conservative. We only need representative samples long enough
    to search for in the extractor output, not a complete parse.
    """
    out: dict[str, list[str]] = {}
    comments = re.findall(r"<!--(.*?)-->", page_html, re.S)
    out["html_comment"] = [html.unescape(c).strip() for c in comments if len(c.strip()) > 40][:8]

    hidden = re.findall(
        r'<(\w+)[^>]*(?:style="[^"]*display\s*:\s*none|aria-hidden="true"|hidden=)[^>]*>(.*?)</\1>',
        page_html,
        re.S | re.I,
    )
    texts = [re.sub(r"<[^>]+>", " ", frag) for _, frag in hidden]
    texts = [html.unescape(re.sub(r"\s+", " ", t)).strip() for t in texts]
    out["hidden_element"] = [t for t in texts if len(t) > 40][:8]
    return out


def norm(text: str) -> str:
    """Collapse whitespace so substring checks survive markdown reflowing."""
    return re.sub(r"\s+", " ", text or "").lower()


def banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def t1_t2_raw_and_hidden() -> None:
    """T1/T2: compare Extract output against the page's own HTML, including hidden markup."""
    banner("T1/T2  Raw-content fidelity and hidden-markup pass-through")
    print(f"page: {PAGE_HIDDEN}")

    raw_html = httpx.get(PAGE_HIDDEN, timeout=30, follow_redirects=True).text
    body_text = norm(re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I))
    body_text = norm(re.sub(r"<[^>]+>", " ", body_text))

    res = extract([PAGE_HIDDEN])
    results = res.get("results") or []
    if not results:
        print(f"  no results; response: {json.dumps(res)[:400]}")
        return
    content = results[0].get("raw_content") or ""
    print(f"  direct HTML visible text : {len(body_text):>8,} chars")
    print(f"  Tavily raw_content       : {len(content):>8,} chars")
    print(f"  ratio                    : {len(content) / max(len(body_text), 1):.1%}")
    print(f"  response keys            : {sorted(k for k in results[0])}")
    print(f"  top-level keys           : {sorted(k for k in res)}")

    # Verbatim check: do long sentences from the page appear unchanged in the extraction?
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s", body_text) if 80 < len(s.strip()) < 200]
    sample = sentences[5:15]
    verbatim = sum(1 for s in sample if norm(s) in norm(content))
    print(f"  verbatim sentence match  : {verbatim}/{len(sample)} sampled sentences reproduced exactly")

    hid = hidden_strings(raw_html)
    for kind, items in hid.items():
        if not items:
            print(f"  {kind:<24} : none found on this page")
            continue
        leaked = [t for t in items if norm(t)[:60] in norm(content)]
        print(f"  {kind:<24} : {len(items)} candidates, {len(leaked)} present in extractor output")
        for t in leaked[:3]:
            print(f"      LEAKED -> {t[:110]}")


def t3_injection_passthrough() -> None:
    """T3: check whether adversarial instruction text is filtered or returned verbatim."""
    banner("T3  Adversarial instruction text pass-through")
    print(f"page: {PAGE_INJECTION_CORPUS}")

    res = extract([PAGE_INJECTION_CORPUS])
    results = res.get("results") or []
    failed = res.get("failed_results") or []
    if not results:
        print(f"  no results (failed_results={json.dumps(failed)[:300]})")
        print("  probe page unreachable; T3 relies on the Wikipedia probe below instead")
    else:
        content = results[0].get("raw_content") or ""
        print(f"  returned {len(content):,} chars, HTTP {res.get('_http_status')}")
        markers = [
            "ignore",
            "instruction",
            "you are",
            "system",
            "assistant",
            "prompt",
        ]
        hits = [m for m in markers if m in norm(content)]
        print(f"  imperative/injection markers present: {hits}")
        print("  --- first 600 chars, verbatim ---")
        print("  " + content[:600].replace("\n", "\n  "))

    # Any moderation signal in the envelope at all?
    envelope = {k: v for k, v in res.items() if k not in ("results", "failed_results")}
    print(f"  response envelope (non-content fields): {json.dumps(envelope)[:300]}")


def t4_ssrf() -> None:
    """T4: observe whether Tavily will dial internal, metadata, and non-HTTP targets."""
    banner("T4  SSRF surface: does Extract dial internal / metadata / non-HTTP targets?")
    for target in SSRF_TARGETS:
        res = extract([target], depth="basic")
        ok = res.get("results") or []
        bad = res.get("failed_results") or []
        status = res.get("_http_status")
        if ok:
            body = (ok[0].get("raw_content") or "")[:160].replace("\n", " ")
            print(f"  REACHED  {target}\n           HTTP {status} -> {body!r}")
        elif bad:
            err = bad[0].get("error") if isinstance(bad[0], dict) else bad[0]
            print(f"  refused  {target}\n           HTTP {status} -> {str(err)[:140]}")
        else:
            print(f"  refused  {target}\n           HTTP {status} -> {json.dumps(res)[:160]}")


def t5_search_vs_extract() -> None:
    """T5: quantify how much more raw text Extract exposes than Search's content chunks."""
    banner("T5  Search content chunks vs Extract raw_content (same page)")
    q = "prompt injection attack against language models"
    s = post("/search", {"query": q, "max_results": 3, "search_depth": "advanced"})
    hits = s.get("results") or []
    if not hits:
        print(f"  search returned nothing: {json.dumps(s)[:300]}")
        return
    print(f"  search result fields: {sorted(hits[0])}")
    for h in hits[:3]:
        url = h.get("url")
        chunk = h.get("content") or ""
        raw_in_search = h.get("raw_content")
        e = extract([url])
        er = e.get("results") or []
        full = (er[0].get("raw_content") or "") if er else ""
        ratio = f"{len(full) / len(chunk):.0f}x" if chunk and full else "n/a"
        raw_content_size = "null (not requested)" if raw_in_search is None else f"{len(raw_in_search):,} chars"
        print(f"\n  {url}")
        print(f"    search 'content' chunk : {len(chunk):>8,} chars")
        print(f"    search 'raw_content'   : {raw_content_size}")
        print(f"    extract 'raw_content'  : {len(full):>8,} chars   ({ratio} the chunk)")


def main() -> None:
    """Run every probe in order."""
    if not KEY:
        sys.exit("TAVILY_API_KEY is not set. Run under: dotenv -f deploy/.env run ...")
    print(f"Tavily security probe | endpoint {API}")
    t1_t2_raw_and_hidden()
    t3_injection_passthrough()
    t4_ssrf()
    t5_search_vs_extract()
    print("\ndone")


if __name__ == "__main__":
    main()
