# single_shot: cut the redundant LLM calls (target: 4 calls per run)

**Status:** Investigated, not yet implemented. Ready to hand to another agent.
**Scope:** `adaptive_researcher` agent, `single_loop_single_shot` `single_shot` path. Fix A also
benefits `direct` / `standard`-inline / meta (all finish via `submit_final_report`).
**Est. win:** ~84k → ~50k input tokens; 6 LLM calls → 4.

This doc covers **two independent fixes** that together bring single_shot to the ideal call
sequence. They can be implemented and shipped separately.

- **Fix A — collapse call 4 + 5:** auto-provide the verified-sources whitelist so the model
  writes the cited answer in one turn instead of a separate `get_verified_sources` round-trip.
- **Fix B — remove call 6:** `return_direct=True` on `submit_final_report` so the ReAct loop
  ends at the finalize tool instead of taking one more discarded model turn.

---

## Context: current 6-call trace

After adding the single_shot **search-budget cap** (see
[adaptive-researcher-token-reduction-plan.md](adaptive-researcher-token-reduction-plan.md)),
the query "What was Apple sale number for 2022" runs in **6 LLM calls / 84,097 input tokens**
(down from 9 calls / ~150k). The 6 calls are:

| # | prompt toks | completion | Action |
| - | ----------- | ---------- | ------ |
| 1 | 11,891 | 93   | Router turn → `declare_effort_tier(single_shot)` |
| 2 | 11,668 | 63   | Prompt swapped to single_shot; **search #1** (`knowledge_search`) |
| 3 | 13,535 | 91   | Sees search-1; **search #2** |
| 4 | 15,137 | 65   | Budget hit **(2/2)** → tools withdrawn + nudge; `get_verified_sources`  ← **Fix A** |
| 5 | 15,396 | 1,053 | Writes the full cited report → `submit_final_report(...)`               ← **Fix A** |
| 6 | 16,470 | 146  | **Terminal turn** after `submit_final_report` returned "Recorded final report." ← **Fix B** |

Sum of prompt tokens = 84,097 (matches the run summary exactly).

### Target after both fixes for single_shot tier only

| # | Action |
| - | ------ |
| 1 | tier selection (`declare_effort_tier`) |
| 2 | search 1 |
| 3 | search 2 → result carries **budget nudge + verified-sources whitelist** (Fix A) |
| 4 | write cited report + `submit_final_report` → `return_direct` ends the loop (Fix B) |

---

## Fix A — collapse call 4 (`get_verified_sources`) into call 5

### Why call 4 is redundant

The source registry is **auto-populated from every search**, not by `get_verified_sources`.
`SourceRegistryMiddleware.awrap_tool_call` captures citation keys from each tool result
(`src/aiq_agent/agents/deep_researcher/custom_middleware.py:477`; the run log shows
`[CitationRegistry] Captured 3 source(s) from knowledge_search`). `get_verified_sources` is
just a **read** of that already-built registry —
`SourceRegistryMiddleware.get_source_list_text(mode="compact")`
(`deep_researcher/custom_middleware.py:671`). Spending a full LLM round-trip only to display a
list the system already holds is the redundancy.

### The change

When the single_shot search budget is hit, have `ComplexityRouterMiddleware` **append the
verified-sources registry text to the same budget nudge** it already injects into the last
search result. Then the turn after the final search (call 4) has both the evidence *and* the
whitelist, so the model writes the cited report and calls `submit_final_report` in that one
turn — no standalone `get_verified_sources` call.

Touch points:
- `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py`
  - Pass a `source_registry_middleware` handle into `ComplexityRouterMiddleware.__init__`.
  - In `awrap_tool_call`, when the budget-reached branch fires, build the nudge as
    `_SINGLE_SHOT_BUDGET_NUDGE` **plus** `source_registry_middleware.get_source_list_text("compact")`
    (guard for `None` / empty registry). Keep appending to the preserved result content.
- `src/aiq_agent/agents/adaptive_researcher/factory.py`
  - `source_registry_middleware` is already in scope in `build_adaptive_research_graph` and is
    passed to sibling middleware — pass it into the `ComplexityRouterMiddleware(...)` block too.
- `src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2` (single_shot direct-tool path)
  - Drop the explicit "3. Call `get_verified_sources`" step. Replace with: once you have
    finished searching (or the search budget is reached), the verified sources are provided to
    you in the tool result — write the cited answer directly and `submit_final_report`.

### Edge case — voluntary early stop (under budget)

If the model finishes after only 1 search (below the budget), the budget-hit injection does not
fire. Keep `get_verified_sources` available as a fallback tool for that path (it stays in
`helper_tools`, always exposed). With budget = 2 the cap almost always triggers, so the common
case is covered; the fallback just preserves correctness for the rare 1-search finish.

### Correctness note

The injected text is the same registry that downstream `verify_citations` checks against
(`agent.py` post-processing), so citations still verify. Confirm in the e2e run.

---

## Fix B — remove call 6 (`return_direct` on `submit_final_report`)

### Why call 6 is redundant

`submit_final_report` is a plain `@tool` (no `return_direct`) —
`src/aiq_agent/agents/adaptive_researcher/tools/finalize.py:97-98`. In the deepagents /
LangGraph ReAct loop, **every** tool call is followed by another model invocation to decide
"call another tool, or stop?". So after `submit_final_report` returns
`"Recorded final report."`, the loop invokes the model **one more time** (call #6); it emits an
AIMessage with no tool calls, which is how the loop terminates.

**Call #6's output is discarded.** The report the user sees comes from
`/shared/final_report.md` (written by `submit_final_report` at call #5), read back in
`AdaptiveResearcherAgent.run()` via `_resolve_output_file_markdown`
(`agent.py:408`, checks `FINAL_REPORT_PATH`), then call #6's message is **overwritten** by
`_replace_last_message_content` (`agent.py:373`). So the ~16.5k-input / 146-output turn exists
purely as the loop's termination mechanism and contributes nothing to the answer.

Log evidence (job `44c9c368-...`):
```
callbacks:221 [Tokens] prompt=15396, completion=1053   # call #5 -> submit_final_report
callbacks:239 [Tool Result] content='Recorded final report.' name='submit_final_report'
callbacks:221 [Tokens] prompt=16470, completion=146     # call #6 -> discarded terminal turn
agent:512     Final answer length : 939 characters       # sourced from the file, not call #6
```

### The change

Set `return_direct=True` on `submit_final_report` (`build_submit_final_report_tool` in
`finalize.py`) so the loop routes to the exit node immediately after the tool executes — no
trailing model call. Keep the docstring/signature. `declare_effort_tier` must stay a normal tool
(it is the FIRST call and the run must continue).

**This IS honored by the installed stack** (verified — not the old prebuilt behavior that
ignored `return_direct`). `create_deep_agent` builds on LangChain's `create_agent`
(`langchain/agents/factory.py`), which wires a `return_direct` exit route:
- `factory.py:1507-1511` — adds an `exit_node` destination when any tool has `return_direct=True`.
- `factory.py:1819-1825` — after the ToolNode runs, if the executed tool calls are
  `return_direct`, route to the exit node instead of back to the model.

The repo already relies on this attribute existing (`source_tool_batching.py:176` reads
`original_tool.return_direct`).

### Caveat (from `factory.py:1825`)

The exit fires only when **all** tools executed in that turn are `return_direct`. That holds on
the finalize step (the model calls `submit_final_report` as a lone tool call). If it were ever
batched with a non-`return_direct` tool in the same turn, the exit would not trigger — not a
concern for the finalize path, but worth knowing.

---

## Verification (both fixes)

Local env cannot run pytest (`annoy`/compiler missing — verify via `py_compile` + ast). Then:

1. `uv run pytest tests/aiq_agent/agents/adaptive_researcher/ -q`
   - Fix A: add a `test_custom_middleware.py` case asserting the budget nudge includes the
     rendered verified-sources text when the registry is non-empty (mock
     `source_registry_middleware.get_source_list_text`).
   - Fix B: add a `test_finalize.py` case asserting `submit_final_report` has `return_direct is True`.
2. `uv run ruff check . && uv run ruff format --check .`
3. End-to-end: re-run
   `python3 misc/aiq_inference.py --server-url http://localhost:8000 "What was Apple sale number for 2022"`
   and confirm:
   - **4 LLM calls** (both `get_verified_sources` and the terminal turn gone), input tokens ~50k;
   - the report still renders, cites `[1]`/`[2]`, and passes citation verification;
   - `docker logs aiq-agent` shows: budget-reached line, then the report written at the
     search-2 follow-up turn, then `submit_final_report`, then the run ends with **no** further
     `[Tokens]` line and **no** `get_verified_sources` tool start.
4. Sanity-check the other inline finishers still work: a `direct`/meta query (e.g. "hi") and a
   `standard`-inline query — each must still end cleanly and return its report from the file.
5. Sanity-check the under-budget path (Fix A edge case): a query the model answers after a
   single search — confirm it still finalizes correctly (via the `get_verified_sources` fallback).

## Watch-outs

- `run()` post-processing (citation verify, sanitize, artifact harvest) runs **after** `ainvoke`
  and reads from `result.files`, so it is unaffected by ending the loop earlier — but confirm in
  the e2e run that citations still verify.
- The **writer-agent (deep/standard-writer) path does NOT call `submit_final_report`** (it
  returns the completion marker and the runtime loads `/shared/output.md`), so neither fix
  touches that path.
- Fixes A and B are independent — if one causes trouble in eval, ship the other alone. B is the
  lower-risk one-liner; A touches the middleware/prompt and the citation flow, so verify
  citations carefully.
