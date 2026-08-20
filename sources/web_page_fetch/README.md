# Web Page Fetch

A NeMo Agent Toolkit (NAT) tool that **opens web pages by URL** and returns their extracted text.

Every other package under `sources/` is a *search* API: you give it keywords and it returns ranked
snippets. This one is the counterpart — you give it a URL you already have, and it returns what that
page actually says, including tables.

## Why it exists

On the 90-task DeepSearch QA evaluation the autonomous researcher performed **zero page opens**, and
scored 0.12 fully-correct on the 16 questions that name their source (a JD Power table, an ICILS
report table, a USDA census PDF) against a reference agent's 0.81. Those questions need a document
read; a search snippet cannot deliver one. See
`misc/autonomous_researcher/fetch-url-tool-plan.md`.

## Backend

Iteration 1 uses **Tavily Extract** exclusively — a maintained extraction service that runs no
network egress from the AI-Q host. In smoke testing it handled a 395-page / 31.7 MB PDF in 2.1 s,
HTML articles, and index pages.

**Known limitation:** it flattens *dense numeric* tables — row-to-value association is lost, so a
statistical table like the USDA agricultural census reads as a list of labels followed by a list of
numbers. A local HTTP + PDF-table path recovers that shape and is deferred to a later iteration.

## Configuration

```yaml
functions:
  fetch_url_tool:
    _type: web_page_fetch
    max_urls_per_call: 4        # URLs accepted in one call
    max_chars_per_page: 10000   # prompt-context budget per page
    max_chars_per_call: 24000   # prompt-context budget per call, spent in request order
    extract_depth: advanced     # 'advanced' retains tables; 'basic' is faster
    timeout_seconds: 30
```

Budgets are in **characters, not bytes**. Pages are extracted in full and then windowed, so a large
document costs the same prompt context as a small one.

Requires `TAVILY_API_KEY` in the environment or `api_key` in config. When it is missing the tool
still registers, and returns a clear error string rather than failing at import.

## Arguments

| Argument | Purpose |
| :-- | :-- |
| `urls` | Exact, complete URLs to open. Not search terms — non-URLs are rejected with a pointer to the search tool. |
| `query` | Optional. Selects *which window* of a long page to show. It does not search the web and does not change which pages are opened. |
| `start_line` | Optional. Resumes a page that was truncated; the truncation note states the line to pass. |

## Output

One delimited section per URL, so a partially-successful call is still useful:

```
<fetched_page url="https://..." title="..." status="ok">
573 | ... page content, line-numbered ...
[Showing lines 573-574 of 2378 (1,732 of 1,073,693 characters). To read further, call
 fetch_url_tool again with start_line=575, or pass a narrower `query`.]
</fetched_page>
```

`status` is `ok`, `suspect` (probable soft 404 — shown with a caution, not citable), `failed`, or
`skipped` (the per-call budget was spent by earlier URLs).

## Citations

The package registers a source parser scoped to its own configured tool name, so **only the pages
actually fetched** enter AI-Q's citation registry. Without it the registry's generic fallback would
register every outbound link on a fetched page as a citable source the agent never read.

## Tests

```bash
uv pip install -e ./sources/web_page_fetch
uv run pytest sources/web_page_fetch/tests
```
