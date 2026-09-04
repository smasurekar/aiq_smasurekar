# Lightweight Researcher via `budget_hint` — Recommendation

> **Disclaimer**: This document is the outcome of a discussion between a human developer and a Claude agent. It is a recommendation only — not a concrete implementation plan. Actual implementation requires further design, code review, and validation to get the details right.

---

## Problem

For `single_shot` tier, the researcher sub-agent runs the same heavy agentic loop as `deep`: reads `/shared/plan.json`, checks skills, iterates over tool calls, and assesses coverage — all unnecessary for a quick single-query lookup.

## Recommendation: Add `budget_hint` to `ResearchQuery`

**One schema change, one prompt change, zero new runnables or tools.**

### What to change

1. **`ResearchQuery` schema** — add an optional field:
   ```
   budget_hint: "minimal" | "standard"  (default: "standard")
   ```

2. **`researcher.j2`** — add a branch at the top of the protocol:
   - If `budget_hint == "minimal"`: skip `/shared/plan.json` read, skip skill check, one source-tool call max, return `ResearchNotes` immediately.
   - Otherwise: run the full existing protocol.

3. **`orchestrator.j2` single_shot section** — instruct the orchestrator to set `budget_hint: "minimal"` on every `ResearchQuery` it forms for `single_shot`.

### Optional enhancement

Add a `single_shot_researcher_llm` config key. If set, build a second researcher runnable using a smaller/faster model only for `minimal` budget queries. If not set, reuse the same runnable — no infra change needed.

## Why this over a separate `search.j2`

- No new runnables, no new batch tools, no orchestrator tool-choice complexity.
- The orchestrator stays simple with one `run_research_batch` tool.
- The lighter behavior is data-driven (the `ResearchQuery` field) rather than structural.
- Opt-in: `standard`/`deep` queries are completely unaffected.
