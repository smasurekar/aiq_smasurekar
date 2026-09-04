# Fetch tool port plan — `web_page_fetch` onto `dev/smasurekar/fetch-tool`

**Source of truth:** commit `0099c80` ("Add fetch url tool in AIQ") on
`dev/smasurekar/aiq-auto-agent-shallow-subagent`.

**Target:** `dev/smasurekar/fetch-tool` (branched from `develop`).

**Goal:** `fetch_url_tool` — a tool that *opens* a URL and returns its text — available to the
shallow researcher, the deep researcher, and the chat researcher, as production code destined
for a PR against `develop`.

---

## 0. Executive summary

The tool package itself ports cleanly: it depends only on `nat.*` and on
`aiq_agent.common.citation_verification.{register_source_parser, SourceEntry}`, and both of
those exist on `develop` with identical signatures
(`src/aiq_agent/common/citation_verification.py:59,534`).

**What changes versus the reference commit is the wiring, not the tool.** The autonomous
researcher branch wired the tool into one agent's explicit `tools:` list and edited two Jinja
prompts. `develop` has a different, better mechanism: `data_source_registry` with tool
auto-inheritance. Every agent in scope resolves its tool set through it:

| Agent | Wiring | Anchor |
| :-- | :-- | :-- |
| shallow researcher | `config.tools` empty → `get_all_tool_refs()`; `exclude_tools: [advanced_web_search_tool]` | `agents/shallow_researcher/register.py:74-86` |
| deep researcher | same inheritance; `exclude_tools: [web_search_tool]` | `agents/deep_researcher/register.py:197-208` |
| chat researcher (intent classifier) | same inheritance; feeds `tools_info` name+description | `agents/chat_researcher/register.py:222-240` |

So **two lines of YAML per config** put the tool in front of all three agents, with **zero
changes to agent Python and zero changes to any `.j2` prompt**. That is the whole
integration. The rest of this plan is packaging, trimming, and validation.

Total footprint: 1 new package (~5 files), ~6 lines across 4 existing wiring files, 2–9 config
edits depending on scope, 2 doc touches.

---

## 1. Port the package — `sources/web_page_fetch/`

Copy from `0099c80` and then trim (§2):

```
sources/web_page_fetch/
├── pyproject.toml          # verbatim; deps pydantic + langchain-tavily (already in the lock)
├── README.md               # trim (see §2)
├── src/
│   ├── __init__.py         # verbatim
│   ├── register.py         # trim (see §2)
│   └── formatting.py       # trim (see §2)
└── tests/
    ├── conftest.py         # verbatim — fake `langchain_tavily`, no network
    ├── test_formatting.py  # verbatim
    └── test_register.py    # verbatim minus autonomous-researcher-specific cases
```

Retrieve with:

```bash
git checkout 0099c80 -- sources/web_page_fetch
rm -rf sources/web_page_fetch/web_page_fetch.egg-info sources/web_page_fetch/**/__pycache__
```

> Note: `sources/web_page_fetch/` currently exists in the working tree as **stale build
> artifacts only** (`egg-info/`, `__pycache__/`) — `git ls-files` returns nothing. Delete them
> first so they do not land in the commit.

### Design points worth preserving as-is

These are load-bearing and should survive the trim:

1. **`query` is never forwarded to Tavily Extract** (`_extract`). Forwarding it collapsed a
   1.07M-character extract to 1,025 characters. Windowing is done locally over the full
   extract, where a miss is recoverable via `start_line`.
2. **Soft-404 detection** (`_looks_like_soft_404`). Tavily returns a site's own "not found"
   page with HTTP 200 as a *success*; without this the agent would cite a 404 page. Status
   `suspect` is shown to the model but is not citable.
3. **`AttributeError` guard** in `_extract`. `langchain_tavily` calls `.get()` on a bare string
   error payload and raises instead of returning.
4. **The tool-scoped citation parser** (`parse_fetched_pages`, registered via
   `register_source_parser`). The registry's generic fallback
   (`citation_verification._parse_generic_urls`, line 707) extracts *every* URL in a tool
   result. A fetched page carries hundreds of outbound links; without a scoped parser one page
   open would register hundreds of never-read pages as citable sources. **This is the single
   most important integration detail in the port.**
5. **The `"Error: ..."` prefix** on a wholly-failed call. `is_non_citable_status_output`
   (`citation_verification.py:632`, regex `^error(?:\s*[:=]|\s+[45]\d{2}\b)`) classifies it as
   non-evidence, which keeps it out of the citation registry and feeds the deep researcher's
   source-tool circuit breaker (`source_tool_batching.py:55`).
6. **`FetchUrlInput` passed explicitly to `FunctionInfo.from_fn`** rather than inferred, so the
   field descriptions are guaranteed to reach the model.

---

## 2. Trim for production

The reference implementation is a POC and reads like one. Concrete cuts — this is the
"don't be verbose" pass:

| Location | Action | Reason |
| :-- | :-- | :-- |
| `register.py` module docstring | Drop the DSQA-eval / Codex-comparison narrative and the `misc/autonomous_researcher/fetch-url-tool-plan.md` reference. Keep the three Tavily guard rationales. | Eval anecdotes from a POC branch do not belong in shipped source; the guard rationale does. |
| `register.py` `_DESCRIPTION_FOR_PROBE` global + its assignment | **Delete.** | Existed only for the autonomous researcher's `test_routing_probe.py`, which does not exist on `develop`. Dead code. |
| `register.py` `_TOOL_NAME_HINT` constant | Inline into `_validate_url`. | Single use; indirection buys nothing. |
| `register.py` `_fetch_url` docstring | Tighten from ~50 to ~25 lines. **Keep:** the READER-not-FINDER contract, the USE / DO-NOT-USE lists, the search-vs-fetch contrast, two examples (one URL-in-question, one search→fetch), and the Args/Returns block. **Cut:** the markdown comparison table (redundant with the two lists) and the third example. | This docstring *is* the tool's prompt (§4) — it must stay strong, but it does not need to be an essay. |
| `formatting.py` module docstring | Compress to the three responsibilities; keep the `_parse_generic_urls` rationale (§1.4). Drop the ICILS character counts. | Same rule. |
| `README.md` | Cut to purpose, config knobs, and a usage snippet. | 80 lines of POC findings. |
| Comments citing "the 90-task DSQA evaluation", "Codex", per-run scores | Remove throughout. | Not reproducible context for a reviewer on `develop`. |

Everything else — the inline rationale comments explaining *why* a guard exists — stays. Per
repo convention, changed code carries clear docstrings and rationale comments; the cut is
aimed at POC narrative, not at explanation.

---

## 3. Wire it — config only

### 3.1 The edit

Two additions per config. In the `data_sources` registry, extend the `web_search` source:

```yaml
  data_sources:
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        description: "Search the web for real-time information, and open specific pages."
        tools:
          - web_search_tool
          - advanced_web_search_tool
          - fetch_url_tool
```

and declare the function:

```yaml
  # The only web tool that OPENS a page rather than searching for one. Budgets are in
  # characters of prompt context, not bytes downloaded: pages are extracted in full and then
  # windowed, so a 30 MB PDF costs the same context as a short article.
  fetch_url_tool:
    _type: web_page_fetch
    max_urls_per_call: 4
    max_chars_per_page: 10000
    max_chars_per_call: 24000
    extract_depth: advanced
```

### 3.2 Why this reaches all three agents with no code change

- **Shallow:** inherits all registry tools, excludes only `advanced_web_search_tool` → gets
  `fetch_url_tool`.
- **Deep:** inherits all registry tools, excludes only `web_search_tool` → gets
  `fetch_url_tool`.
- **Chat:** `intent_classifier` inherits all registry tools and builds `tools_info` from
  name + description (`chat_researcher/register.py:237`), so depth routing sees the new
  capability. The chat researcher then delegates to shallow or deep, which both have it.
- **Per-request source filtering** (`filter_tools_by_sources`) keeps working: a user who turns
  Web Search off loses the fetch tool too, which is the correct coupling — no web egress when
  web search is disabled.

### 3.3 Scope of config edits

**Tier 1 (required):**
- `configs/config_cli_default.yml` — the default CLI profile.
- `configs/config_web_frag.yml` — the main web profile (registry at line 117).

**Tier 2 (consistency; mechanical, same two-block edit):**
`config_web_default_guardrails.yml`, `config_web_default_llamaindex.yml`,
`config_web_azure_ai_search.yml`, `config_web_opensearch.yml`, `config_openshell.yml`,
`config_domain_routing_and_skills.yml`, `config_frontier_models.yml`.

**Explicitly excluded — `configs/config_mcp.yml`.** It is the frozen public MCP profile and is
pinned field-by-field by `mcp/tests/test_config_and_packaging.py:73-99`, which asserts the
exact `functions` key set and the exact `web_search` tools list. The MCP image
(`mcp/Dockerfile`) also does not install `web_page_fetch`. Leaving it alone keeps `mcp/tests`
green with no edits and no `mcp/uv.lock` churn.

**Recommendation:** ship Tier 1 + Tier 2 in one commit. The edit is identical and mechanical,
and a half-applied registry is the kind of inconsistency that generates review churn later.

---

## 4. Prompts — no `.j2` changes

The tool description is the only prompt surface, which is what was asked for. Verified render
paths:

| Consumer | Renders | Anchor |
| :-- | :-- | :-- |
| Shallow researcher | `{% for tool in tools %}- **{{ tool.name }}**: {{ tool.description }}` | `shallow_researcher/prompts/researcher.j2:84-86` |
| Deep researcher worker | same loop under "You can ONLY use the tools provided below" | `deep_researcher/prompts/researcher.j2:99-100` |
| Deep source router | `runtime_source_tools()` carries `{"name", "description"}` into the router catalog | `deep_researcher/tools/source_routing.py:130-162` |
| Chat intent classifier | `tools_info = [{"name", "description"}]` | `chat_researcher/register.py:237` |

Because `fetch_url_tool` is registered under the `web_search` source id,
`get_source_id_for_tool` resolves it, `runtime_source_tools` includes it, and the source router
can name it in a `ResearchQuery.preferred_tools`. All of that is automatic.

### One flagged risk, deliberately not fixed in v1

`shallow_researcher/prompts/researcher.j2:18-24` — "**Query Rewriting (before calling search
tools)** … rewrite the user's question into a search-friendly query" — could induce a model to
rewrite a URL before passing it to the fetch tool.

Three defences already exist: the `FetchUrlInput.urls` field description ("NOT search keywords
— every item must start with http:// or https://"), the `_fetch_url` docstring's DO-NOT-USE
list, and `_validate_url`, which rejects a non-URL with a message that *names* `web_search_tool`
as the right tool — converting a wasted call into a corrected one rather than a dead end.

**Plan:** ship without the prompt edit and measure. If evals show URL mangling, the minimal
follow-up is a single clause on line 18 — `(this applies to search tools only; pass URLs to
fetch_url_tool exactly as they appear)` — not a new prompt section. This is the only `.j2` edit
that could ever be justified, and it is not justified yet.

Likewise, `researcher.j2:28` ("Max 2 calls per tool") bounds shallow to 2 fetch calls × 4 URLs
= 8 page opens. That is a sensible shallow-tier ceiling; leave it.

---

## 5. Deep researcher compatibility — verified, no code change

The fetch tool has three model-facing fields (`urls`, `query`, `start_line`), so it is **not**
single-string batchable. Traced through `adapt_source_tools_for_research`
(`deep_researcher/tools/source_tool_batching.py:344-390`):

1. `candidate.name in source_tool_names` → **true** (it is registry-mapped), so it is treated
   as a source tool.
2. `_single_string_input_field(candidate)` → **`None`** (3 fields, line 228-239).
3. → `_make_throttled_source_tool` (line 307). This **preserves the original schema** and
   applies the per-job source-call budget, the concurrency limiter, and the circuit breaker.

This is exactly the intended degradation path for a non-batchable source tool. The tool does
its own batching internally via `urls: list[str]`, so one call = one budget unit for up to 4
pages — strictly cheaper than 4 searches.

**Budgets:** `resource_limits.max_source_tool_calls` defaults to 100
(`resource_limits.py:29`), and `config_cli_default.yml` sets
`max_researcher_model_calls: 100`. Headroom is ample. **Do not raise any budget in this PR.**
(The reference commit bumped `max_direct_source_calls` 5→8, but that knob belongs to the
autonomous researcher's `request_termination` block, which does not exist on `develop`.)

---

## 6. Packaging — this is where the reference commit does not transfer

`develop`'s `pyproject.toml` has **no `runtime-tools` extra**; the reference branch did. And
`develop`'s `deploy/Dockerfile` copies `sources/` wholesale (line 62), so there is **no
per-package metadata `COPY` block** to update. Adapted list:

| File | Edit |
| :-- | :-- |
| `pyproject.toml` `[dependency-groups].dev` (line ~220, beside `"tavily-web-search"`) | add `"web-page-fetch",` |
| `pyproject.toml` `[tool.uv.sources]` (line ~270) | add `web-page-fetch = { workspace = true }` |
| `pyproject.toml` `[tool.uv.workspace].members` | **no change** — already `sources/*` |
| `deploy/Dockerfile:86` | add `&& uv pip install --no-deps -e ./sources/web_page_fetch \` |
| `scripts/setup.sh:118` | add `"${UV_BIN}" pip install -e ./sources/web_page_fetch` |
| `uv.lock` | regenerate: `uv lock` |
| `mcp/Dockerfile`, `mcp/uv.lock` | **no change** — see §3.3 |

### The Dockerfile line is the one that matters

`COPY sources/ ./sources/` is a wildcard, so the source tree reaches the image whether or not
it is installed — but NAT discovers tools through `nat.plugins` **entry points**, which exist
only for *installed* distributions. Skip the install line and the build succeeds, then every
config using `_type: web_page_fetch` fails at startup with:

```
ValueError: Invalid configuration: functions: Input tag 'web_page_fetch' found using
discriminator() does not match any of the expected tags: ...
```

No dependency change is needed: `langchain-tavily` is already resolved in `uv.lock` for
`tavily_web_search`.

---

## 7. Tests

**Port** `sources/web_page_fetch/tests/` (picked up automatically — `testpaths` includes
`sources/**/tests`, `pyproject.toml:116`). All three files run against a fake
`langchain_tavily`; nothing touches the network.

**Drop** from the reference commit:
- `tests/aiq_agent/agents/autonomous_researcher/test_factory.py` and `test_routing_probe.py` —
  that agent's tool wiring does not exist on `develop`.
- The `_DESCRIPTION_FOR_PROBE` assertions (§2).

**Keep and adapt** the reference's `tests/deploy/test_runtime_image_packages.py`, but reduce it
to the narrow invariant that actually applies here — `develop`'s Dockerfile has only one
hand-maintained list, so the sprawling version is over-engineered:

```python
def test_web_page_fetch_is_installed_in_the_runtime_image_and_dev_setup():
    """The Dockerfile's COPY sources/ is a wildcard, so a missing install line still builds —
    and then every config using `_type: web_page_fetch` fails at startup."""
    assert "./sources/web_page_fetch" in (REPO_ROOT / "deploy" / "Dockerfile").read_text()
    assert "./sources/web_page_fetch" in (REPO_ROOT / "scripts" / "setup.sh").read_text()
```

**No change needed** to `tests/test_config_defaults.py` (it only asserts the Lightning LLM
output cap) or to `mcp/tests/` (§3.3).

**Add one integration assertion** — cheap and it guards the §3.2 claim that is the whole point
of the PR: in `tests/aiq_agent/agents/`, assert that with `configs/config_cli_default.yml`'s
registry, `fetch_url_tool` survives both `exclude_tools` filters (shallow's and deep's). A
config-level YAML assertion is sufficient; no builder needed.

---

## 8. Docs

- `docs/source/extending/adding-a-tool.md` — add a row to the "Existing Tool Reference" table
  (line ~465):
  `| Web Page Fetch | \`web_page_fetch\` | \`sources/web_page_fetch\` | \`TAVILY_API_KEY\` |`
- `docs/source/customization/tools-and-sources.md` — note that the `web_search` source now
  carries a reader alongside the two searchers, and that toggling the source off disables both.

The reference commit also expanded `adding-a-tool.md` with a "Shipping it in the runtime
container" section and a matching `.agents/skills/aiq-add-tool/SKILL.md` gotcha. That guidance
is genuinely useful and generalizes, **but it is a separate concern from this tool.** Ship it
as a follow-up doc PR rather than widening this diff; if it goes in here, note in the PR body
that it is documentation-only.

No new secret to document — `TAVILY_API_KEY` is already required by `tavily_web_search`.

---

## 9. Validation

```bash
uv sync --group dev
uv pip install -e ./sources/web_page_fetch

uv run ruff check .
uv run ruff format --check .

uv run pytest sources/web_page_fetch/tests -q      # narrowest first
uv run pytest tests/ -q                            # registry/config wiring
uv run --project mcp --extra dev pytest mcp/tests  # must stay green untouched (§3.3)

# startup smoke — proves the entry point resolves and the registry accepts the tool
nat validate --config_file configs/config_cli_default.yml || \
  nat serve --config_file configs/config_cli_default.yml --port 8000
```

Then a live check of each tier, confirming the tool is offered and that a URL survives intact:

```bash
./scripts/start_cli.sh
# shallow: "Summarize https://www.iea.nl/sites/default/files/ICILS_2023_report.pdf"
# deep:    a question naming a specific report/table, to exercise source-router preferred_tools
```

**Container check** (the §6 failure mode is invisible outside the image):

```bash
docker build -f deploy/Dockerfile -t aiq-fetch-check .
docker run --rm aiq-fetch-check /app/.venv/bin/python -c \
  "from importlib.metadata import entry_points; \
   assert any(e.name == 'web_page_fetch' for e in entry_points(group='nat.plugins'))"
```

---

## 10. Commit / PR shape

One commit, DCO-signed. Suggested scoping if the reviewer prefers smaller units:

1. `feat: add web_page_fetch source package` — package + tests + packaging (§1, §2, §6, §7).
2. `feat: offer fetch_url_tool to shallow, deep, and chat research` — config edits + docs
   (§3, §8).

PR body must carry the §9 evidence (commands run and results), per repo PR hygiene rules.

### What this PR explicitly does not do

State these in the PR body so reviewers do not read them as oversights:

- No `.j2` prompt changes (§4) — with the one carve-out named as a measured follow-up.
- No budget or resource-limit changes (§5).
- No `configs/config_mcp.yml` / MCP image changes (§3.3).
- No local HTTP/PDF-table fallback backend. Tavily Extract flattens dense numeric tables and
  loses row-to-value association. This is a known, bounded gap; a second backend is a
  measured follow-up, not iteration-1 scope.
