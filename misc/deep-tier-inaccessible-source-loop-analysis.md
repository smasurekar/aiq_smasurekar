# Deep-Tier Timeouts: Inaccessible-Source Loop & Structured-Data Spiral

Analysis and recommended fixes for the two dominant deep-tier failure modes observed in the
DeepSearch QA full run.

- **Eval run:** `results/aiq/deepsearchqa-adaptive-all-post-token-optim-full`
- **Config:** `configs/config_adaptive_frag.yml`
- **Sample size:** 198 records, of which 45 classified `deep`
- **Status:** analysis complete; fixes proposed, not yet implemented

> **Concrete examples:** 15 verbatim failure cases (exact questions, reference answers, and the
> URLs the agent tried to open) are in
> [`deep-tier-failure-examples.md`](./deep-tier-failure-examples.md) — 5 `standard`-tier PDF
> failures, 5 `deep`-tier timeouts, and 5 `deep`-tier fetch-blocked cases. That doc is the
> stakeholder-facing companion to this one.

---

## 1. Observed impact

| Deep-tier group | n | Avg accuracy | Avg latency |
| :-- | --: | --: | --: |
| Completed (no timeout) | 22 | **0.68** | 711 s |
| Timed out at 1200 s | 23 | **0.09** | 1325 s |
| All deep | 45 | 0.36 | 1025 s |

Deep-tier reasoning quality is **not** the problem. When a deep run finishes, it is the most
accurate tier in the whole eval (0.68). Accuracy collapses to 0.09 only when the wall-clock
deadline fires. 23 of 45 deep runs (51%) died this way.

---

## 2. What actually happens (evidence)

### 2.1 The termination budgets never fire

`configs/config_adaptive_frag.yml` gives the `deep` tier a generous envelope:

```yaml
deep:
  max_batch_calls: 6
  max_total_research_queries: 24
  max_orchestrator_turns: 100
```

Measured consumption at the moment the 1200 s deadline fired:

| Metric | Budget | Timeout runs (avg) | Runs that reached the budget |
| :-- | --: | --: | --: |
| `run_research_batch` calls | 6 | 2.4 | **1 / 23** |
| Delegated research queries | 24 | 0.2 | **0 / 23** |

The loop guards in `custom_middleware.py` are working as designed and are simply never reached.
**Wall clock is the only binding constraint.** Tuning `max_batch_calls` or
`max_total_research_queries` down would not have changed these runs; tuning them up is
meaningless. This is the single most important finding — it rules out "the guards are broken"
as the explanation.

### 2.2 Throughput is identical; the work required is not

| Group | Outer search rounds | Latency | Seconds per round |
| :-- | --: | --: | --: |
| Timed out | 36.9 | 1200 s | **32.5 s** |
| Completed | 20.5 | 711 s | **34.7 s** |

The agent is not stalling, deadlocking, or spinning on a hung call — the per-round pace is the
same in both groups. Timed-out questions simply **need ~1.8× more search rounds than fit inside
1200 s**. The loop is a *throughput* problem, not a *liveness* problem.

### 2.3 Repeat detection is defeated by rewording

The researcher guard blocks identical source calls
(`models/loop_guard.py:43`, `max_identical_source_calls: 2`), keyed on a SHA-256 of the tool name
plus canonicalized JSON args (`custom_middleware.py:302`). That is an **exact** match.

What the model actually emits (`dsqa_id_31`, three consecutive rounds at 44.5 s / 57.2 s / 64.9 s):

```
Our World in Data total emigrants by country 1990 2024 dataset
Our World in Data migrant stock emigrants CSV download 1990 2024
Our World in Data migrant stock emigrants CSV download
```

Semantically one query, lexically three. Three different hashes, so the guard never fires. The
same pattern drives `dsqa_id_194` (NCES table), `dsqa_id_155` (Stats NZ API), `dsqa_id_127`
(CTBUH ranking). The orchestrator-level dedup
(`_canonical_research_query_signature`, `custom_middleware.py:493`) normalizes case and
whitespace but is still exact-match on the normalized string, so it has the same blind spot.

### 2.4 64% of repeated queries are URLs fed to a search engine

Of 107 distinct queries repeated ≥3× across the 23 timeouts, **69 (64%) are URL-, filename-, or
`site:`-shaped**, e.g.:

```
https://ourworldindata.org/grapher/migrant-stock-emigrants.csv
https://api.data.stats.govt.nz/rest/data/STATSNZ,CEN23_HAD_020,1.0/2023.01+02+...
https://nces.ed.gov/programs/digest/d23/tables/dt23_226.60.asp
dataportal.orr.gov.uk/media passenger-performance-2021-22-q3 Table 3133.ods
```

The model has correctly identified the exact artifact holding the answer and is trying to open
it. But there is **no tool in the deployment that can fetch a URL.** A repo-wide search for a
fetch/download/scrape capability across `src/aiq_agent/agents/deep_researcher/` and `sources/`
returns nothing usable. The only two web tools are the same Tavily search function:

```yaml
web_search_tool:              # configs/config_adaptive_frag.yml:70
  _type: tavily_web_search
  max_results: 5
  max_content_length: 1000

advanced_web_search_tool:     # configs/config_adaptive_frag.yml:76
  _type: tavily_web_search
  max_results: 2
  advanced_search: true
```

Passing a `.csv` / `.ods` / `.asp` URL to a search engine returns *results about that page*,
never its bytes. The action can never succeed, the model gets no signal saying so, and it
reformulates indefinitely. That is the entire mechanism of both reported failure modes.

### 2.5 Retrieval yield is starved, which multiplies the round count

`advanced_web_search_tool` — the deep-only tool — returns **2 results**. `web_search_tool`
truncates each result to **1000 characters**. Deep questions are overwhelmingly multi-constraint
table lookups ("all EU states meeting X and Y", "top 10 stations by ridership"). A 1000-character
snippet from 2 results essentially never contains a full table, so each round yields one or two
cells and the model must search again. This is the direct driver of the 1.8× round-count gap in
§2.2: low yield per round → more rounds → deadline.

---

## 3. Root cause

The two reported symptoms are one defect with three compounding layers:

1. **Capability gap (primary).** The agent can find where data lives but cannot open it. No
   fetch/download tool exists. Every attempt at a structured artifact is unsatisfiable.
2. **No failure signal.** A search for a URL returns a normal, non-error result set. Nothing tells
   the model "this action class cannot work," so retrying looks rational.
3. **Guard blind spot.** Exact-hash dedup does not catch the reworded retries that layers 1 and 2
   produce, so nothing stops the churn until the wall clock does.

Low retrieval yield (§2.5) then ensures even *reachable* questions run out of time.

---

## 4. Recommended fixes

Ordered by expected accuracy gain per unit of effort. Fixes 1–3 are config-only and independent
of each other; fix 4 is the real remedy.

### Fix 1 — Raise `workflow_timeout_seconds` to 2400 (immediate, config-only)

`configs/config_adaptive_frag.yml`:

```yaml
request_termination:
  workflow_timeout_seconds: 2400        # was 1200
  fallback_finalizer_timeout_seconds: 120
```

Constraint enforced at `models/request_termination.py:124`:
`fallback_finalizer_timeout_seconds` must be **strictly less than** `workflow_timeout_seconds`.

Also confirm the eval client waits at least as long — `run_aiq_eval.py` `--timeout` (already 2400
on the eval machine). These are two independent timeouts: the eval-side one governs how long the
harness waits, the config one governs when the agent aborts itself. The 1200 s figure in the
generated answers comes from the config value, interpolated at `agent.py:579`.

*Expected effect:* at 32.5 s/round, 2400 s buys ~74 rounds vs the 36.9 the timeouts reached.
This alone should convert a large share of the 23 timeouts. It treats the symptom, not the cause —
it does not stop wasted work on genuinely unfetchable artifacts, and it roughly doubles worst-case
cost per query. Do it first because it is one line and unblocks the eval, but do not stop here.

### Fix 2 — Increase retrieval yield per round (config-only)

```yaml
web_search_tool:
  _type: tavily_web_search
  max_results: 5
  max_content_length: 4000      # was 1000

advanced_web_search_tool:
  _type: tavily_web_search
  max_results: 5                # was 2
  advanced_search: true
```

Attacks the round-count multiplier directly rather than the deadline. Fewer rounds to the same
evidence means more questions finish inside *any* deadline. Raises input tokens per round, but
deep tier already averages 1.27 M input tokens — dominated by round count, not per-round payload,
so total tokens may well fall. Worth A/B-ing against fix 1 alone to see which lever actually pays.

### Fix 3 — Make repeat detection semantic, not exact-hash

In `custom_middleware.py`, replace the exact SHA comparison with near-duplicate detection so
reworded retries collapse to one signature. Minimal viable version, no new dependency:

- Normalize as today (NFKC, casefold, collapse whitespace).
- Reduce to a token *set*, drop stopwords and generic research filler
  (`download`, `dataset`, `csv`, `full`, `table`, `official`, `site`, year literals).
- Treat two queries as identical when Jaccard similarity ≥ ~0.8.
- Additionally, normalize any URL-shaped query to its host + path, so the same URL with
  different query strings or `site:` prefixes collapses to one signature.

Apply at both layers — `_canonical_source_signature` (`:302`) and
`_canonical_research_query_signature` (`:493`).

Also close the scope gap: `OrchestratorLoopGuardMiddleware` only inspects `run_research_batch`
(`custom_middleware.py:665`). Deep runs also research via the `task` tool (present in 23/23
timeouts), whose sub-agent searches are outside this guard. Either bound `task` the same way or
document why it is exempt.

*Caveat:* the 0.8 threshold is a guess and must be tuned against the eval — too aggressive will
block legitimate query refinement, which is a normal and productive deep-research behavior. Ship
this behind a config flag and measure.

### Fix 4 — Add a URL-fetch tool (the actual fix)

This is the only change that makes the failing questions *answerable* rather than merely
cheaper to fail. Add a NAT function under `sources/` — e.g. `sources/web_fetch/` — registered with
`@register_function` and a `FunctionBaseConfig` schema, per the conventions in
`.agents/skills/` and `CLAUDE.md`.

Capabilities, in priority order against the observed failures:

| Capability | Observed need |
| :-- | :-- |
| Fetch a URL and return text | Every URL-shaped repeat (69 queries) |
| Parse CSV/TSV → rows, with column/row slicing | `ourworldindata.org/.../*.csv`, `dsqa_id_31`, `dsqa_id_178` |
| Parse HTML `<table>` → structured rows | `nces.ed.gov/...asp` (`dsqa_id_194`), CTBUH (`dsqa_id_127`) |
| Parse XLSX/ODS | ORR `Table 3133.ods` (`dsqa_id_68`) |
| Return JSON from a REST endpoint | Stats NZ API (`dsqa_id_155`), `data.ny.gov` (`dsqa_id_8`) |
| PDF text extraction | ocindex.net profile PDFs (`dsqa_id_0`) |

Design requirements:

- **Explicit, actionable failure.** On 403/404/paywall/unparseable, return a clear error stating
  the artifact is unreachable and instructing the model to record an evidence gap rather than
  retry. This is what removes layer 2 of the root cause — the absent failure signal — and it
  matters as much as the fetch capability itself.
- **Bounded output.** Cap returned bytes and support pagination/row-slicing; a 50 k-row CSV must
  not enter the context whole. Note this interacts with the §2.5 yield problem — the cap has to be
  generous enough to actually carry a table.
- **Route URL-shaped queries here.** Detect URL/filename-shaped input in the search tools and
  redirect to fetch, so the model's existing (correct) instinct stops being silently wasted.
- **Security.** Honor the repo's existing rules — no secrets in logs or errors, `SecretStr` for any
  key, graceful degradation when unconfigured. Add SSRF protection: block private/link-local
  address ranges and cap redirects. Consider an allowlist if these agents ever run against
  untrusted input.

The `dsqa_id_100` and `dsqa_id_105` runs (190 and 169 `read_file` calls) suggest a partial
download path already exists somewhere in the sandbox layer and is being used ineffectively —
worth confirming before building new, in case this is better framed as fixing an existing
capability than adding one.

---

## 5. Suggested sequencing

1. **Fix 1** — one line, unblocks the eval, re-run to get a clean deep-tier baseline.
2. **Fix 2** — one config block; re-run and compare round counts against step 1 to see whether
   yield or deadline was the binding lever.
3. **Fix 4** — the real remedy; scope from the capability table above.
4. **Fix 3** — most valuable *after* fix 4, when a genuine "this is unfetchable" signal exists and
   the guard can act on it instead of guessing from lexical similarity.

Re-run `deepsearchqa` after each step and track deep-tier timeout count and mean accuracy. Given
completed deep runs already score 0.68, eliminating the timeouts is worth roughly **+0.13 overall
E2E accuracy** (0.33 → ~0.46) on its own, before any gain from newly-answerable questions.

---

## 6. Open questions

- Should the fetch tool be deep-tier-only, or exposed to `standard` too? `standard` shows the same
  wrong-source-data failures (55% of its failures) and would likely benefit.
- Is there an existing sandbox download path (implied by the heavy `read_file` counts) that should
  be fixed rather than duplicated?
- What is the acceptable cost ceiling? Fix 1 roughly doubles worst-case latency and tokens per
  deep query.

---

## 7. Fetch tool design — error and success response specification

### 7.1 URL reachability findings

Before designing the fetch tool, the 160 unique URLs the agent embedded in search queries across
all 198 eval records were checked against live HTTP. Results:

| Outcome | Count | % |
| :-- | --: | --: |
| Truly OK (200 + correct Content-Type) | 88 | 55% |
| 200 but wrong MIME type (ORR `.ods` → `application/oleobject`) | 2 | 1% |
| Unreachable (4xx / 5xx / DNS / timeout) | 70 | **44%** |

**The "fetch tool fixes everything" framing overstates the case.** 29 of the 70 unreachable URLs
are 404 — the files are gone or were never at those paths. A fetch tool returns the same error the
web does; it does not make missing content available. It is still necessary (it stops the retry
loop by providing an explicit failure signal), but it fixes ~4–5 of the 15 documented failure
cases outright, not all of them. See `url_reachability_report.md` in the eval results folder for
the per-case verdict.

---

### 7.2 Error response format (unreachable artifacts)

The most important design requirement: the error must be a **plain-text string returned as the
tool result**, not a Python exception. Exceptions are logged, not shown to the model, so they
provide no signal and the retry loop continues.

**Template:**
```
FETCH_ERROR: <status>
URL: <url>
Expected: <file type>

<One sentence stating what went wrong.> Do not retry this URL or reformulate
the same request. Record an evidence gap in your findings and continue with
other sources.
```

**Error taxonomy:**

| Condition | `FETCH_ERROR` label | Additional note to include |
| :-- | :-- | :-- |
| 404 Not Found | `404 Not Found` | "File does not exist at this path." |
| 403 Forbidden | `403 Forbidden` | "Access is restricted; fetch cannot bypass this." |
| 401 Unauthorized | `401 Unauthorized` | "API key required; configure credentials or skip." |
| 302 → HTML login | `302 Login Wall` | "Redirects to a login page; content is behind auth." |
| 405 Method Not Allowed | `405 Method Not Allowed` | "Server blocks automated access." |
| Timeout | `Timeout` | "Server did not respond within N seconds." |
| DNS failure | `DNS Failure` | "Hostname does not exist." |
| 200 but HTML returned | `200 HTML Gate` | "Server returned HTML instead of expected file — likely a login or error page." |
| PDF no text layer | `PDF Unreadable` | "PDF contains only scanned images; no text layer found." |
| PDF encrypted | `PDF Encrypted` | "PDF is password-protected." |

The "do not retry" instruction and "record an evidence gap" suggestion are what convert a
loop-inducing dead end into a terminating path. They must appear in every error variant.

---

### 7.3 Success response format (artifact retrieved)

**Template:**
```
FETCH_OK: <Content-Type>
URL: <url>
Pages: <total>  |  Returned: <range>  |  Characters: <n>  |  Has TOC: <Yes/No (page N)>  |  Tables detected: <n>
[TRUNCATED: showing pages X–Y of Z. Call again with page_range="Z+1-..." to continue.]

--- BEGIN CONTENT ---
<extracted text or table rows>
--- END CONTENT ---
```

Key fields:
- **Total page count** — always present so the agent knows the document's extent on the first call.
- **Has TOC** — flags whether a table of contents was found and on which page, enabling targeted
  navigation without a second call.
- **Tables detected** — signals when `extract_tables=True` would be more useful than prose.
- **Truncation notice** — explicit, with the exact `page_range` to use for continuation.

---

### 7.4 PDF-specific handling

**Recommended library stack:**

| Layer | Library | Reason |
| :-- | :-- | :-- |
| Text extraction (primary) | `pdfminer.six` | Best layout-aware reading order for multi-column docs and tables |
| Text extraction (fallback) | `pypdf` | Faster; handles edge cases `pdfminer` misses |
| Table extraction (opt-in) | `pdfplumber` or `camelot` | Serializes tables as rows × cols rather than garbled prose |
| OCR (opt-in, expensive) | `pytesseract` + `pdf2image` | Image-only PDFs; off by default, configurable |

**Extraction quality signal** — flag degraded output when the text-to-noise ratio is low
(common with two-column layouts, footnote interleaving, scanned PDFs that passed text-layer
detection):
```
[WARNING: extraction quality may be low — complex layout detected (tables/columns)]
```

---

### 7.5 Three ranked approaches for large PDFs

When a PDF exceeds the context budget, the tool and agent together need a strategy for reading
only the relevant slice. Three approaches, ranked by accuracy-to-complexity ratio:

---

#### Rank 1 — Metadata header on every response + `page_range` parameter *(recommended default)*

Every fetch response, regardless of `page_range`, prepends:
```
Pages: 47  |  Has TOC: Yes (page 1)  |  Tables detected: 4  |  Text layer: Yes
```

The agent reads the metadata on the first call, identifies the right pages from the TOC or
section titles, and issues a second call with a precise `page_range`:

```
# Call 1 — get metadata and TOC
fetch_url(url, page_range="1-2")
→ "Pages: 47 | Has TOC: Yes (page 1) | ..."

# Call 2 — jump directly to the relevant section
fetch_url(url, page_range="31-33")
→ table with exact data needed
```

**Why best:** The agent does the reasoning (reading the TOC); the tool does the fetching. Clean
separation. Works for ~80% of government and statistical PDFs. Low implementation complexity —
just prepend metadata and honour `page_range`.

**Weakness:** Requires 2 round-trips when a TOC exists; falls back to blind pagination when it
does not.

---

#### Rank 2 — Query-scored page selection *(fallback for unstructured documents)*

Pass the agent's research question into the fetch call. The tool extracts text page-by-page,
scores each page against the query (TF-IDF or lightweight embedding), and returns only the
top-K most relevant pages:

```
fetch_url(url, query="alcohol-impaired fatality rate per 100M VMT by state 2022")
→ "Query matched: pages 8, 9, 31 — returning those"
```

**Why second:** Best option when there is no TOC and the answer could be anywhere in a long
document (e.g. academic PDFs, B4 case). One call, no navigation step needed.

**Weakness:** Tool does reasoning the agent should be doing, and does it blind to context the
agent already has. Table-heavy pages score low on text similarity even when they contain the
answer. Adds `sklearn` or an embedding model as a dependency.

---

#### Rank 3 — Raw paginated fetch *(simplest, lowest accuracy)*

Return the first N characters with an explicit continuation signal:
```
FETCH_OK: application/pdf  |  Pages: 1–10 of 47
To retrieve more, call with page_range="11-20"
```

**Why third:** Simplest to implement but forces the agent to page through blindly if the answer
is not in the first chunk. For a 47-page NHTSA report where the relevant table is on page 31,
this costs 3–4 extra tool calls before landing on it.

**Use as the fallback** when Rank 1 finds no TOC and Rank 2 is not configured.

---

#### Summary

| Rank | Approach | Calls to answer | Works without TOC | Complexity |
| :-- | :-- | :-- | :-- | :-- |
| 1 | Metadata header + `page_range` | 2 (1 if lucky) | Partially | Low |
| 2 | Query-scored page selection | 1 | Yes | Medium |
| 3 | Raw paginated fetch | 1–N | Yes (slowly) | Lowest |

**Practical implementation:** ship Rank 1 as the default (`page_range` always supported, metadata
always prepended). Add Rank 2 as an opt-in `query=` parameter. Rank 3 is the automatic fallback
when neither TOC nor query is available.
