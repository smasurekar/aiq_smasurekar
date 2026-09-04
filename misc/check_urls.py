"""
Check which URLs the agent attempted to fetch (by embedding them in search queries)
actually respond with a successful HTTP status code vs 404/403/etc., AND verify that
the server actually returns the expected file type (PDF, CSV, etc.) rather than an
HTML error/login page masquerading as a 200.

Usage:
    python misc/check_urls.py \
        --records /path/to/records.jsonl \
        --output  misc/url_check_results.json \
        [--workers 20] [--timeout 15]
"""

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

URL_RE = re.compile(r'https?://[^\s"\'\\<>)\]]+')

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; research-url-checker/1.0; +https://github.com/nvidia)"),
    "Accept": "*/*",
}

# Expected MIME prefixes for each file extension
EXPECTED_MIME = {
    "pdf": ["application/pdf"],
    "csv": ["text/csv", "text/plain", "application/octet-stream", "application/csv"],
    "xlsx": [
        "application/vnd.openxmlformats-officedocument.spreadsheetml",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ],
    "ods": ["application/vnd.oasis.opendocument.spreadsheet", "application/octet-stream"],
    "json": ["application/json", "text/json"],
    "asp": [],  # ASP pages can return any content-type — skip check
    "html": ["text/html"],
    "htm": ["text/html"],
    "unknown": [],  # no expectation
}

# MIME prefixes that signal an HTML gate/login wall even on a 200
HTML_GATE_MIME = ["text/html", "text/xhtml", "application/xhtml"]


def _is_html_gate(content_type_header: str, expected_ext: str) -> bool:
    """Return True when the server returns HTML for a non-HTML resource (login wall)."""
    if expected_ext in ("html", "htm", "unknown", "asp"):
        return False
    ct_lower = content_type_header.lower()
    return any(ct_lower.startswith(h) for h in HTML_GATE_MIME)


def _content_type_matches(content_type_header: str, expected_ext: str) -> bool:
    """Return True when the response MIME matches what we expect for the extension."""
    expected = EXPECTED_MIME.get(expected_ext, [])
    if not expected:
        return True  # no expectation — treat as match
    ct_lower = content_type_header.lower()
    return any(ct_lower.startswith(e) for e in expected)


def extract_urls(records_path: str) -> dict[str, list[str]]:
    """Return {url: [sample_ids]} from all web-search tool.start events."""
    url_to_samples: dict[str, list[str]] = {}

    with open(records_path) as f:
        for line in f:
            d = json.loads(line)
            sample_id = d.get("sample_id", "unknown")
            raw = d.get("raw", {})
            if not isinstance(raw, dict):
                continue
            events = raw.get("events", [])
            if not isinstance(events, list):
                continue

            for e in events:
                if not isinstance(e, dict):
                    continue
                if e.get("type") != "tool.start":
                    continue
                if e.get("name") not in (
                    "advanced_web_search_tool",
                    "tavily_search",
                    "web_search_tool",
                ):
                    continue
                data = e.get("data", {})
                if not isinstance(data, dict):
                    continue
                inp = data.get("input", {})
                if not isinstance(inp, dict):
                    continue
                queries = inp.get("queries", [])
                if isinstance(queries, str):
                    queries = [queries]
                if not isinstance(queries, list):
                    continue

                for q in queries:
                    if not isinstance(q, str):
                        continue
                    for m in URL_RE.findall(q):
                        url = m.rstrip(".,;)")
                        if url not in url_to_samples:
                            url_to_samples[url] = []
                        if sample_id not in url_to_samples[url]:
                            url_to_samples[url].append(sample_id)

    return url_to_samples


def _guess_ext(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for ext in (".pdf", ".csv", ".xlsx", ".ods", ".json", ".asp", ".html", ".htm"):
        if path.endswith(ext):
            return ext.lstrip(".")
    return "unknown"


def check_url(url: str, timeout: int = 15) -> dict:
    """
    HEAD (for status + Content-Type), then GET fallback if HEAD fails at network level.
    Returns a result dict with:
      - http_status, reachable, redirect_url, error
      - server_content_type   : Content-Type header value (or None)
      - expected_ext          : file extension guessed from URL
      - content_type_match    : True when MIME matches the extension
      - html_gate             : True when a non-HTML resource returns HTML (login wall)
      - effectively_reachable : reachable AND correct content type AND not an HTML gate
    """
    expected_ext = _guess_ext(url)
    result = {
        "url": url,
        "expected_ext": expected_ext,
        "http_status": None,
        "reachable": False,
        "redirect_url": None,
        "server_content_type": None,
        "content_type_match": None,
        "html_gate": False,
        "effectively_reachable": False,
        "error": None,
    }

    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, headers=HEADERS, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result["http_status"] = resp.status
                result["reachable"] = resp.status < 400
                final = resp.geturl()
                if final != url:
                    result["redirect_url"] = final

                ct = resp.headers.get("Content-Type", "") or ""
                result["server_content_type"] = ct
                result["content_type_match"] = _content_type_matches(ct, expected_ext)
                result["html_gate"] = _is_html_gate(ct, expected_ext)
                result["effectively_reachable"] = (
                    result["reachable"] and result["content_type_match"] and not result["html_gate"]
                )
                return result

        except urllib.error.HTTPError as exc:
            result["http_status"] = exc.code
            result["reachable"] = False
            result["effectively_reachable"] = False
            result["error"] = str(exc)
            # HEAD returned a definitive HTTP error — no point trying GET
            return result

        except urllib.error.URLError as exc:
            result["error"] = str(exc.reason)
            if method == "GET":
                return result
            # Try GET next for network-level HEAD failures

        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            return result

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=(
            "/home/smasurekar/Desktop/Swapnil/gitlab_repos/rag/results/aiq/"
            "deepsearchqa-adaptive-all-post-token-optim-full/records.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/smasurekar/Desktop/Swapnil/gitlab_repos/rag/results/aiq/"
            "deepsearchqa-adaptive-all-post-token-optim-full/url_check_results.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    print(f"Extracting URLs from {args.records} ...", flush=True)
    url_to_samples = extract_urls(args.records)
    urls = list(url_to_samples.keys())
    print(f"Found {len(urls)} unique URLs — checking reachability + content type ...", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_url, url, args.timeout): url for url in urls}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            r["sample_ids"] = url_to_samples[r["url"]]
            results.append(r)

            if not r["reachable"]:
                mark = "FAIL"
                detail = str(r["http_status"] or r["error"] or "?")
            elif r["html_gate"]:
                mark = "GATE"
                detail = "200 but HTML gate"
            elif not r["content_type_match"]:
                mark = "MISMATCH"
                detail = f"200 but got {(r['server_content_type'] or '?')[:40]}"
            else:
                mark = "OK"
                detail = str(r["http_status"])

            print(f"  [{done}/{len(urls)}] {mark:<8} {detail:<30}  {r['url'][:80]}", flush=True)

    # ---- summary buckets ----
    truly_ok = [r for r in results if r["effectively_reachable"]]
    html_gates = [r for r in results if r["reachable"] and r["html_gate"]]
    mime_mismatch = [r for r in results if r["reachable"] and not r["html_gate"] and not r["content_type_match"]]
    unreachable = [r for r in results if not r["reachable"]]

    by_status: dict[str, int] = {}
    for r in results:
        key = str(r["http_status"] or r["error"] or "unknown")
        by_status[key] = by_status.get(key, 0) + 1

    by_ext: dict[str, dict] = {}
    for r in results:
        ext = r["expected_ext"]
        if ext not in by_ext:
            by_ext[ext] = {"truly_ok": 0, "html_gate": 0, "mime_mismatch": 0, "unreachable": 0}
        if r["effectively_reachable"]:
            by_ext[ext]["truly_ok"] += 1
        elif r["html_gate"]:
            by_ext[ext]["html_gate"] += 1
        elif r["reachable"]:
            by_ext[ext]["mime_mismatch"] += 1
        else:
            by_ext[ext]["unreachable"] += 1

    total = len(urls)
    summary = {
        "total_urls": total,
        "truly_ok": len(truly_ok),
        "html_gate_200": len(html_gates),
        "mime_mismatch_200": len(mime_mismatch),
        "unreachable": len(unreachable),
        "effectively_reachable_pct": round(100 * len(truly_ok) / total, 1) if total else 0,
        "by_http_status": {k: v for k, v in sorted(by_status.items())},
        "by_file_extension": by_ext,
        "detail": {
            "truly_ok": [
                {
                    "url": r["url"],
                    "status": r["http_status"],
                    "content_type": r["server_content_type"],
                    "samples": r["sample_ids"],
                }
                for r in sorted(truly_ok, key=lambda x: x["url"])
            ],
            "html_gate_200": [
                {
                    "url": r["url"],
                    "content_type": r["server_content_type"],
                    "redirect": r["redirect_url"],
                    "samples": r["sample_ids"],
                }
                for r in html_gates
            ],
            "mime_mismatch_200": [
                {
                    "url": r["url"],
                    "expected_ext": r["expected_ext"],
                    "content_type": r["server_content_type"],
                    "samples": r["sample_ids"],
                }
                for r in mime_mismatch
            ],
            "unreachable": [
                {"url": r["url"], "status": r["http_status"], "error": r["error"], "samples": r["sample_ids"]}
                for r in sorted(unreachable, key=lambda x: str(x["http_status"]))
            ],
        },
        "all_results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Total unique URLs checked       : {total}")
    print(f"Truly OK (right content type)   : {len(truly_ok)}  ({summary['effectively_reachable_pct']}%)")
    print(f"200 but HTML gate (login wall)  : {len(html_gates)}")
    print(f"200 but wrong MIME type         : {len(mime_mismatch)}")
    print(f"Unreachable (4xx/5xx/DNS/etc.)  : {len(unreachable)}")
    print("\nBy HTTP status:")
    for status, count in sorted(by_status.items()):
        print(f"  {status:>10}  {count} URLs")
    print("\nBy file extension  (ok / gate / mismatch / unreachable):")
    for ext, info in sorted(by_ext.items()):
        print(
            f"  {ext:>10}  {info['truly_ok']} ok / {info['html_gate']} gate / "
            f"{info['mime_mismatch']} mismatch / {info['unreachable']} unreachable"
        )
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
