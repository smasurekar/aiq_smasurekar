# Adaptive Researcher — Token Reduction: `single_shot` Collapse + `standard` Width/Depth Tuning

> **Disclaimer**: This document is the outcome of a discussion between a human developer and a Claude agent. It is a recommendation only — not a finalized implementation. Actual implementation requires further design, code review, and validation to get the details right.

---

# Problem

For single_shot/standard queries the `adaptive_researcher` spends far more tokens than the older single-loop `shallow_researcher`, for comparable answers.

Measured on the same account/setup:

| Query | Agent | LLM calls | Input tokens |
| :-- | :-- | --: | --: |
| "Hey" (meta) | adaptive | 2 | 23,706 |
| "How has Apple's total net sales changed over time?" (standard) | adaptive | 10 | 114,638 |
| (comparable shallow queries, avg) | shallow | — | ~30,115 |

The gap is **~4x on a standard query** and is almost entirely **architectural**, not prompt-size.

### Root cause: two nested LLM loops instead of one

`shallow_researcher` is a single ReAct loop — one LLM role (`RESEARCHER`), one 60-line prompt, search tools called **directly**, free-form output (regex citation post-processing), hard cap of ~5 tool iterations + 1 synthesis turn (`src/aiq_agent/agents/shallow_researcher/agent.py`).

`adaptive_researcher` splits the same work across **two independent LLM loops**, and that split is what multiplies tokens. Even on the "cheap" `standard`-inline branch (planner/writer **not** invoked), the 10 calls for the Apple query break down as:

| Loop | ~Calls | Re-sent on every call in that loop |
| :-- | --: | :-- |
| **Orchestrator** (`orchestrator.j2`) | 4 | 198-line prompt + full nested `AdaptiveResearchQuery`/`run_research_batch` schema + growing history (`declare_effort_tier` → `run_research_batch` → `get_verified_sources` → `submit_final_report`) |
| **Researcher subagent** (`researcher.j2`) | 5–6 | 114-line prompt + all real search-tool schemas + `ResearchNotes` structured-output schema + **all accumulated search results** (`think` + 3 search rounds + synthesis) |

### Why the tokens explode — ranked by impact

1. **Bulky research payload is paid for twice.** Web-search results (each returns thousands of tokens) accumulate inside the researcher loop and are re-sent on every one of its turns; then the *entire* `ResearchNotes` JSON (38 sources for the Apple query) is returned up to the orchestrator via `run_research_batch` and lives in *its* context through `get_verified_sources` + `submit_final_report`. The same evidence occupies two contexts. Shallow keeps it in one.
2. **Context re-transmission is the dominant term.** Each ReAct call re-sends the full history; adaptive has *two* loops each doing this, and the researcher does 3 sequential search rounds, so by its last turn it re-sends ~9 bulky result sets.
3. **Heavier per-call base in each loop.** Every orchestrator call carries the nested query schema; every researcher call carries the `ResearchNotes` output schema. Shallow uses free-form text + regex (`verify_citations`, `sanitize_report`) — zero structured-output schema tokens.
4. **More roundtrips** (10 vs ~5–6), each re-sending its loop's growing prefix.

The extra cost is not pure waste — it buys isolation, structured self-judged notes (`evidence_judgment`), and corroboration across more sources. That is the deep-research bargain. The goal here is to **stop paying it on tiers that don't need it**.

---

# Recommended Solution

**Collapse `single_shot` to a single-loop execution path (orchestrator calls the real search tools directly). Leave `standard` on the two-loop orchestrator→researcher architecture and reduce its cost via width/depth only. Reserve the full pipeline for `deep` (and the mandatory delta writer pipeline).**

The orchestrator already declares its tier up front via `declare_effort_tier` before any research. Use that signal to pick the execution shape:

- **`direct` / meta** → single turn, no tools (already cheap; pair with the greeting heuristic from the meta discussion).
- **`single_shot`** → the orchestrator calls the **real search tools directly in its own loop** (shallow-style), instead of delegating to a separate researcher subagent via `run_research_batch`. This eliminates the second loop, the duplicated notes, and the per-call `ResearchNotes` schema.
- **`standard`** → **unchanged architecture** (keep `run_research_batch` and the researcher subagent). Reduce cost by tightening **width** (fewer fanned-out queries) and **depth** (bias queries to `low`/`medium`) — never by collapsing the loop. See "Why `standard` stays two-loop" below.
- **`deep` / delta** → unchanged. Full orchestrator → planner → researcher-subagent → writer pipeline. Fidelity preserved where it matters.

For `single_shot` this attacks driver #3 and #4 directly — no second-loop fixed overhead (114-line researcher prompt + `ResearchNotes` schema per worker call), and the orchestrator's own calls no longer carry the `run_research_batch` schema.

### Why `standard` stays two-loop

The subagent split is not overhead in `standard`'s regime — it is a **context-isolation + distillation** mechanism whose value scales with `query_count × depth`:

- What crosses the loop boundary is the **distilled `ResearchNotes`, not the raw search dumps.** Each researcher worker holds its own bulky raw results in an *isolated* context and returns only compact notes to the orchestrator.
- `single_shot` (1–3 queries, ~1 search each) has tiny raw-result volume, so collapsing the loop just sheds the second-loop overhead with almost no downside.
- `standard` (3–5 queries, up to ~3 searches each = 9–15 searches) has **large** raw-result volume. Collapsing the loop would force all of it into the orchestrator's *single* context, re-transmitted across every synthesis turn *and* carried through `submit_final_report` — the opposite of the saving. It would also lose `run_research_batch`'s **concurrent fan-out** (workers run in parallel), hurting latency exactly where fan-out is widest.

So single-loop **inverts the benefit** for `standard`. Keep the architecture; turn the width/depth knobs that already exist (`max_research_concurrency`, the per-query `depth` field, and the `depth: low` plan/skill skip already landed).

### Why this over the alternatives

- **vs. collapsing `standard` too (the earlier draft of this plan)**: rejected — see above; it regresses `standard` by de-isolating its larger raw-result volume and dropping parallel fan-out.
- **vs. the existing `budget_hint` recommendation**: `budget_hint` makes the researcher subagent's *internal* loop cheaper (skip plan read, one tool call), but still keeps the two-loop split and still returns full `ResearchNotes` up to the orchestrator. It is complementary to (and largely already realized by) the `depth` knob; single-loop routing removes the subagent entirely for `single_shot`, capturing the larger saving there.
- **vs. just tuning `depth`/fan-out**: this *is* the recommended lever for `standard`; for `single_shot` it leaves the second-loop fixed overhead on the table, which single-loop removes.
- **vs. compressing prompts/schemas**: marginal — prompt text is a small fraction of the 114k; re-sent evidence dominates.

### Honest tradeoffs

- This is a **design change**, not a tweak: it reintroduces a shallow-style direct-tool path into an agent whose stated design is "one graph, one uniform researcher subagent." Per `CLAUDE.md`, worth a design discussion before building. (Scoping it to `single_shot` keeps the blast radius small — `standard`/`deep` are untouched structurally.)
- `single_shot` answers lose the researcher's structured `evidence_judgment` and note persistence. Acceptable for a single bounded factual lookup; `standard` keeps them since its architecture is unchanged.
- Citation handling differs between the two paths (structured `get_verified_sources` vs. shallow's regex capture). The `single_shot` single-loop path must reuse the shallow agent's `tool_node_with_source_capture` + `verify_citations` so citations stay correct.

---

# Implementation details

> Scope note: keep `deep_researcher` untouched. All changes live under `src/aiq_agent/agents/adaptive_researcher/`, mirroring how the adaptive agent already subclasses/aliases deep-researcher pieces.

### 1. Decide the routing seam

The single-loop change targets **`single_shot` only**. `standard` keeps `run_research_batch` and is addressed by prompt-level width/depth tuning (§1b). The orchestrator emits `declare_effort_tier(tier=...)` as its first tool call (`tools/finalize.py`, `build_declare_effort_tier_tool`). Two viable seams for the `single_shot` collapse:

- **(A) Prompt-branch within the single orchestrator loop (lowest-risk, recommended first).** Keep one loop, but for `single_shot` give the orchestrator the **real source tools directly** (as the shallow agent does) instead of `run_research_batch`. The orchestrator then searches and writes inline — no subagent. Gate which tools are bound using the tier ceiling machinery already present in `custom_middleware.py` (`ComplexityRouterMiddleware` / `hidden_tools_for_ceiling`), extended to *swap* `run_research_batch` for direct source tools when the declared tier is `single_shot`, rather than only hide.
- **(B) Separate execution graph per tier.** A post-`declare_effort_tier` router dispatches `single_shot` to a shallow-style graph and everything else to the current graph. Cleaner separation, but heavier: a second graph build and a hard classifier seam (chicken-and-egg — you only know the tier after the orchestrator's first call unless you add an upstream classifier). Prefer (A) unless evals show it's insufficient.

### 1b. `standard` cost reduction — prompt-only, no architecture change

Reduce `standard`'s tokens by tightening the fan-out the orchestrator authors, not by changing the loop:

- **Width**: in `orchestrator.j2`'s `standard` section, bias toward the low end of the ~3–5 query range (e.g. 2–3) unless the question is genuinely multi-part. `max_research_concurrency` is the hard ceiling; this is the soft guidance.
- **Depth**: instruct the orchestrator to default `standard` queries to `depth: low`/`medium`, reserving `high` for a real multi-hop sub-question. This leans on the per-query `depth` field (`models/subagent_contracts.py:35`) and the `depth: low` plan/skill-skip already landed in `researcher.j2`.
- No code, schema, or tool-wiring change for `standard`.

### 2. Reuse shallow_researcher's direct-loop primitives (`single_shot` only)

For the `single_shot` single-loop path, reuse (do not reimplement):

- `shallow_researcher/agent.py` — the 2-node ReAct graph (`agent -> tools -> agent`), `bind_tools(..., parallel_tool_calls=True)`, `max_tool_iterations` cap + forced synthesis turn.
- `tool_node_with_source_capture` and the citation helpers `verify_citations` / `sanitize_report` / `_append_minimal_citation` — so the inline path produces correct citations without the structured `ResearchNotes`/`get_verified_sources` contract.
- `shallow_researcher/prompts/researcher.j2` (60 lines) as the template for the `single_shot` system prompt, adapted to the adaptive agent's finalize convention (`submit_final_report`).

### 3. Tool wiring (`single_shot` only)

- `single_shot`: bind the actual data-source search tools (`web_search_tool`, `tavily_search`, `knowledge_search`, etc.) directly to the single-loop LLM — the same set `register.py` already resolves (`get_all_tool_refs()` → `filter_tools_by_sources`), minus `advanced_web_search_tool` (already excluded in config).
- Do **not** bind `run_research_batch` on `single_shot`. Do **not** bind the heavy subagent `task`/`write_todos` delegation tools (`hidden_tools_for_ceiling` already hides these below `standard`).
- `standard` / `deep`: unchanged tool set (keep `run_research_batch`).

### 4. Config

- Add an optional `single_shot_researcher_llm: LLMRef | None` to `AdaptiveResearchAgentConfig` (mirrors the existing per-role `*_llm` fields, `register.py:70-74`). If unset, reuse the orchestrator LLM. Lets operators point the `single_shot` loop at a smaller/cheaper model.
- No new required config; behavior is opt-in and defaults to current behavior if the collapse is disabled by a flag (e.g. `single_loop_single_shot: bool = False` during rollout).

### 5. Files likely touched

| File | Change |
| :-- | :-- |
| `agents/adaptive_researcher/factory.py` | Build/select the `single_shot` single-loop path; swap tool binding when the declared tier is `single_shot` |
| `agents/adaptive_researcher/custom_middleware.py` | Extend `hidden_tools_for_ceiling` to swap `run_research_batch` ↔ direct source tools for `single_shot` |
| `agents/adaptive_researcher/prompts/orchestrator.j2` | `single_shot` section: instruct direct-tool search + inline `submit_final_report` (drop the `run_research_batch` delegation wording). `standard` section: tighten width/depth guidance (§1b) — no structural change |
| `agents/adaptive_researcher/register.py` | Optional `single_shot_researcher_llm` field + `single_loop_single_shot` rollout flag |
| `agents/adaptive_researcher/agent.py` | Thread the flag/LLM through to the factory |
| `tests/aiq_agent/agents/adaptive_researcher/` | Cover: `single_shot` binds direct tools (no `run_research_batch`); `standard`/`deep` unchanged (still delegate); citations correct on the `single_shot` inline path |

### 6. Validation

- **Token/quality A/B**: rerun the probe queries and a `single_shot` factual query; record `LLM calls` and `Input tokens` before/after. Targets: a `single_shot` query drops toward shallow's ~30k band (removal of the second loop); a `standard` query drops modestly from ~114k via tighter width/depth — *not* to the shallow band, since its architecture is intentionally preserved. Watch for answer-quality regression on both.
- **`standard` isolation check**: confirm a `standard` query still delegates via `run_research_batch` (researcher subagent runs) — the width/depth tuning must not accidentally collapse it.
- **Eval harness**: run `freshqa` (has both `config_shallow_research_only.yml` and full-workflow configs) and a `deepsearch_qa` slice to confirm `standard` answers don't lose coverage/citation accuracy under tighter width/depth.
- **Deep-tier regression**: confirm a `deep` query still runs the full planner → research → writer pipeline unchanged (no tokens saved there is expected and correct).
- **Local caveat**: this environment cannot build/run the agent (no compiler; `uv run`/pytest fail on `annoy`). Validation must run where the backend is deployable; verify statically with `py_compile` + targeted reads before pushing.

### 7. Rollout

1. Land the `single_shot` collapse behind `single_loop_single_shot=False` (no behavior change by default). The `standard` width/depth prompt tuning can ship independently (it is pure prompt guidance, no flag).
2. Enable in a dev config (`config_adaptive_frag.yml`), gather A/B numbers.
3. Flip default once evals confirm `single_shot` parity + token drop, and that `standard` still delegates and keeps its coverage.
