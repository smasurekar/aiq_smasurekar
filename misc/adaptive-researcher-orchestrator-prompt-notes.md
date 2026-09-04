<!--  =================== CLAUDE NOTES START =================== -->

# Adaptive Researcher — Orchestrator Prompt: Design Notes

> **Scope.** How the `adaptive_researcher` orchestrator prompt
> (`src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2`) was designed and
> revised, why each decision was made, and how the prompt is wired to code. Companion to the
> POC plan in [`AI-Q-3.0-Unified-Research-POC-Plan.md`](./AI-Q-3.0-Unified-Research-POC-Plan.md)
> (Approach 1.0). Reflects the state after the correctness and simplicity review.

---

## 1. What this prompt has to do

The adaptive researcher is **one** `create_deep_agent` orchestrator built once with all roles
present (source-router, planner, researcher, writer). There is **no upstream classifier** and
**no per-request graph rebuild** — adaptivity is achieved *in the prompt*: the orchestrator
self-selects how much effort to spend per request and self-limits its planning, fan-out, and
tool use. So the orchestrator prompt carries the entire routing brain:

1. **Effort self-assessment** — pick an effort level (`direct` / `single_shot` / `standard` /
   `deep`) from the query, with no separate LLM call.
2. **Per-level procedure** — exactly what to do at each level (skip planner, fan-out width,
   write inline vs delegate to writer-agent).
3. **Finalization** — how to end the run so the runtime captures the report and citation
   verification behaves correctly.
4. **Meta / no-research handling** — answer chit-chat directly without tripping the
   empty-source-registry safeguard, even when the `direct` factual tier is disabled.
5. **Delta / parent-report handling** — run a mandatory planned writer pipeline so preserved
   citations stay valid, even when normal effort tiers are shallow-only.
6. **Tier configurability** — only describe the tiers an operator enabled via `enabled_tiers`.

Everything else in the package (finalize tool, no-research safeguard, tier clamp, middleware)
exists to *support* what this prompt instructs. The one deliberate runtime exception is delta
safety: Layer-B enforcement preserves delegation tools for a request that carries parent-report
context, because an inline rewrite cannot safely preserve parent citations.

---

## 2. Design principles

### 2.1 Soft, prompt-driven effort (not code-gated)
Per POC §4 and the `deepagents` facts, planning (`write_todos`) and delegation (`task`) are
optional, prompt-driven middleware — a model can simply choose not to call them. So "skip the
planner and answer inline for a simple query" is native behavior, not a special code path. The
prompt therefore *describes behaviors*; it does not rely on the runtime to gate them (aside
from the static cost ceilings and the parent-report delegation exception that live in code).

### 2.2 Strict KV-cache boundary (hard requirement)
NIM reuses the KV cache for a shared prompt prefix. Every prompt in this repo marks that seam
with `{#- === KV CACHE BOUNDARY — dynamic content below === -#}`. Discipline:

- **Above the boundary = 100% deployment-static.** Identical across every request for a given
  config, so it forms one cacheable prefix.
- **Below the boundary = 100% per-request dynamic.**

The marker is a Jinja **comment**, so it is stripped at render time. It documents the seam in
the source template; the *rendered* guarantee is that the static prefix is byte-identical
across requests. In the rendered output the first below-boundary section is `## Context`, so
"the prefix" = everything before `## Context`.

**Config-static vs per-request** (this is the subtle part):
- *Config-static* (may live above the boundary): `enabled_tiers`, `enable_source_router`,
  `execution_enabled`, `max_research_concurrency`, and the tier profiles. These vary only
  across deployments, never across requests within a deployment → one prefix per config.
- *Per-request* (must live below): `current_datetime`, `user_info`, `available_documents`,
  `clarifier_result`, `triage_hint`, `parent_report_context_available` (the *fact*), and the
  `retrieval_tools` list (varies with per-request `data_sources` filtering).

A deliberate improvement over the deep-researcher orchestrator: deep renders the
`parent_report_context_available` delta block *above* its boundary (a per-request conditional
in the static prefix, creating two prefixes). Here the delta **rule** is static (above) and
the delta **fact** ("Delta Mode State") is dynamic (below) — so there is exactly one static
prefix regardless of request type.

### 2.3 One source of truth per concern
Tier summaries are defined once in `tiers.py` and rendered once under **Effort Levels**.
Detailed writer behavior is defined once in the shared **Planned Writer Pipeline** and reused
by `deep`, `standard`-writer, escalation-to-writer, and delta runs. The prompt intentionally
does not repeat the same tier mapping in a second table; fewer independently worded mappings
mean fewer opportunities for semantic drift.

---

## 3. Prompt structure (section by section)

### Above the boundary (deployment-static)

| Section | Purpose | Notes |
|---|---|---|
| Role line | Frames the orchestrator as an adaptive router that self-selects effort. | |
| **Effort Levels** | One compact list of the *enabled* tiers with when/plan/writer/width/tools/finalize. | `{% for tier in tier_profiles %}` — filtered by `enabled_tiers`; no duplicate routing table. |
| **Choosing Effort** | Effort-selection checklist. **Delta is step 0** and bypasses normal tier selection. Includes the "you are a RESEARCH agent — don't answer from memory when unsure" caution. | The model does not announce its internal tier to the user. |
| **Delta / Parent-Report Rule** | Static rule: delta ⇒ mandatory Planned Writer Pipeline, never inline. | This is a safety workflow, not selection of a disabled tier. The *fact* renders below. |
| **Available Subagents** | Lists planner / writer (+ source-router if enabled), annotated with which levels use them. | source-router bullet gated on `enable_source_router`. |
| **Filesystem Notes** | `/shared/` semantics; sandbox paths when `execution_enabled`. | |
| **Sequential Handoffs** | One dependent subagent per turn; route→plan ordering when source router on. | |
| **Workflow** | An always-available meta/capability path, one shared Planned Writer Pipeline, and gated `direct`, `single_shot`, `standard`, and `deep` procedures. | The common pipeline owns plan→research→writer ordering. |
| **Research Loop** | `run_research_batch` batch cap, submitted-query ledger, error handling. | Uses `max_research_concurrency`. |
| **Finalize Protocol** | **Mechanism-based** finish rule (see §5). | |
| **In-loop Escalation** | Thin evidence ⇒ refine or run the complete planned writer path; planner output must be researched before writer. | |
| **Stopping and Evidence Failure** | Stop on sufficient/redundant coverage; partial evidence is disclosed; an empty registry stays a research failure. | |
| **Inline Citation Contract** + **Important** | Claim-level verified citations, exact locators, sources section, same-language, virtual paths. | Mirrors the essential writer rules without copying the entire writer prompt. |

### Below the boundary (per-request dynamic)

`## Context` (datetime, user) · `## Triage Hint` (if present) · `## Delta Mode State` (if
`parent_report_context_available`) · `## Clarification Context` (if `clarifier_result`) ·
`## User Uploaded Documents` (if any) · **`## Retrieval Tools`** (the source tools available
to `run_research_batch` this run) · `## Your Tools (callable)` (the orchestrator's own tools).

---

## 4. The tier model and per-tier paths

Four normal effort levels exist in the graph; `enabled_tiers` controls which are described
and selectable (Layer-A enforcement). Delta is a separate citation-safety override, not a
fifth tier. The normal tier profiles are defined in `tiers.py::TIER_PROFILES`:

| Level | When | Plan | Width | Who writes | How to finish |
|---|---|---|---|---|---|
| `direct` | a trivially known, time-insensitive fact | skip | **0 — no research** | inline | `submit_final_report(researched=false)` |
| `single_shot` | one bounded factual question | skip | 1–3 queries | inline | `submit_final_report(researched=true)` |
| `standard` | lightly multi-part | optional | ~3–5 queries | inline **or** writer-agent | inline→`submit_final_report(true)` **or** writer→marker |
| `deep` | comparison / trend / multi-hop / report | planner-agent | up to the cap | writer-agent | writer → `/shared/output.md` marker |

Each `## Workflow` procedure is an explicit numbered recipe gated by
`{% if "<tier>" in enabled_tiers %}`. Key details:

- **No-Research Meta / Capability Path** — always available for greetings, thanks, identity,
  small talk, or an honest explanation that the current tier set cannot research the request.
- **`direct`** — answer only a trivially known timeless fact, call no tools except the
  finalizer, and never fabricate citations.
- **`single_shot`** — author 1–3 `ResearchQuery` objects directly (no planner/router/todos).
  The prompt spells out the *required* query fields (`query`, one or more `preferred_tools`
  from the Retrieval Tools list, `target_components`, `rationale`) because `ResearchQuery`
  enforces `preferred_tools` min-length ≥ 1 — otherwise inline query construction would fail
  validation. Then `get_verified_sources` → write cited markdown → finalize.
- **`standard`** — two complete branches: author ~3–5 queries and finish inline, or run the
  full planned writer pipeline. The prompt forbids authoring plan-less queries and then
  delegating to a writer that requires `/shared/plan.json`.
- **`deep`** — delegates to the shared Planned Writer Pipeline.

### 4.1 Planned Writer Pipeline

The writer path has one ordered contract used by `deep`, `standard`-writer, escalation, and
delta: optional/advised routing → planner writes `/shared/plan.json` → orchestrator executes
the plan's still-uncovered queries → writer reads plan + notes + verified sources → writer
writes `/shared/output.md` → orchestrator returns the marker. This prevents two invalid paths:

- writer-agent without the plan it requires;
- planner-agent followed directly by writer-agent without executing the new planned queries.

For a delta request, this pipeline is mandatory even under `[single_shot]`. When Layer-B tool
enforcement is enabled, `ComplexityRouterMiddleware` preserves `task` and `write_todos` for
that request while retaining the rest of the configured ceiling.

---

## 5. The finalize mechanism (why mechanism, not tier)

The runtime resolves the final report in order: `/shared/output.md` (writer) →
`/shared/final_report.md` (the `submit_final_report` tool) → last-resort inline salvage. The
`researched` flag written alongside the inline report tells the agent whether to run citation
verification or skip it (the no-research safeguard).

The Finalize Protocol keys off **how the answer was produced**, not which tier was picked:

- **Delegated to writer-agent** (it wrote `/shared/output.md`) → return its marker; do **not**
  call `submit_final_report`.
- **Wrote the answer inline** (`direct`, `single_shot`, or `standard`-inline) →
  `submit_final_report(markdown, researched=…)` exactly once.
  - `researched=true` when backed by `run_research_batch` (citations must verify).
  - `researched=false` **only** for a `direct` no-research answer (skips verification, avoids
    the empty-registry error).

This mechanism rule is what makes the four tiers, the `standard`-inline-or-writer choice, and
**mid-run escalation** all finish coherently with one rule instead of tier-by-tier special
cases.

---

## 6. Retrieval Tools (and the small factory change)

The orchestrator does **not** hold the source tools — it reaches retrieval only through
`run_research_batch`, and its "Your Tools (callable)" list is just `think`,
`get_verified_sources`, `run_research_batch`, `submit_final_report`. But the inline
`single_shot` / `standard` paths must name real retrieval tools in
`ResearchQuery.preferred_tools`. Without a list of valid names the model would guess or
hallucinate them.

Fix: `factory.py::build_adaptive_research_graph` passes `retrieval_tools=tool_set.tools_info`
(the source tools' names/descriptions) into the orchestrator render, and the prompt renders a
`## Retrieval Tools` section **below the KV boundary** (it varies with per-request
`data_sources` filtering) instructing the model to name these in `preferred_tools` and never
call them directly.

---

## 7. Divergences from the deep-researcher orchestrator

- **No `SourceRoutingGuardMiddleware`** on the orchestrator. That guard blocks every tool call
  until source-routing writes its file — which would deadlock the shallow / single-shot path.
  `enable_source_router` therefore defaults to `False`, and even when on, routing is advisory.
- **Extra tool**: `submit_final_report` on the orchestrator surface (+ allow-listed in the
  tool-name sanitizer).
- **Cleaner KV boundary**: delta rule (static, above) split from delta fact (dynamic, below);
  `enabled_tiers`/`tier_profiles`/`retrieval_tools` added.
- **Prompt is the router**: meta-vs-research, shallow-vs-deep, and the parent-report delta
  safety decision live in this one prompt instead of an upstream `intent_classifier`.
  General `report_ask` / `report_edit` nodes remain out of the POC scope.

---

## 8. Review → revamp (what the first draft got wrong, and the fixes)

The first draft was reviewed for tier-path clarity and consistency. Findings and fixes:

| # | Problem in the first draft | Fix |
|---|---|---|
| Critical | `standard` had no distinct procedure and its finalize contradicted itself (profile said "inline or delegate", but Finalize said `submit_final_report` was only for direct/single_shot). | Gave `standard` its own procedure; switched to the **mechanism-based** finalize rule. |
| Critical | Finalize was coupled to tier, so **escalation** (single_shot → writer) had no correct finish. | Mechanism-based rule covers escalation. |
| Critical | Deep-path step numbering **skipped step 3** when the source router was off (Jinja `if/else` numbering artifact). | Branched the numbering on `enable_source_router` (1–6 with, 1–5 without). |
| High | Workflow / subagents were **not gated by `enabled_tiers`** — a deep-only or shallow-only config still described every path. | Per-tier procedures gated by `"<tier>" in enabled_tiers`; subagents annotated by level. |
| High | No single tier→path→finalize mapping. | Added the **Path per Effort Level** table. |
| Medium | Orchestrator couldn't see retrieval tool names, but inline paths must set `preferred_tools`. | Added `## Retrieval Tools` + the `factory.py` `retrieval_tools` kwarg; `single_shot` step lists the required query fields. |
| Medium | `direct` caution too weak for a research agent (hallucination/staleness risk). | Strengthened: don't answer from memory when a fact could be time-sensitive or you're unsure. |
| Medium | Delta was only a separate rule, reachable after effort selection. | Made delta **step 0** of Choosing Effort. |
| Polish | Redundant Meta rule; ambiguous "Available Tools" naming. | Folded meta into the `direct` procedure; renamed to "Your Tools (callable)". |

The correctness and simplicity review found another set of semantic gaps:

| # | Gap | Resolution |
|---|---|---|
| Critical | Delta unconditionally selected `deep`, even when `deep` was disabled; shallow Layer-B enforcement could also hide delegation. | Made delta a mandatory safety workflow outside normal tier selection and preserve delegation tools for parent-report requests. |
| Critical | `standard` allowed direct query authoring followed by writer delegation, but writer requires `/shared/plan.json`. | Split `standard` into two complete branches; only the planned branch may call writer-agent. |
| Critical | Escalation said planner→writer and could skip execution of the planner's new queries. | Centralized planner→planned research→writer ordering in the shared pipeline and made escalation reference it. |
| Medium | Inline citation guidance was weaker than writer-agent's citation contract. | Added a compact claim-level verified citation contract with exact locator and Sources rules. |
| Medium | No terminal policy distinguished partial evidence, redundant retrieval, and an empty verified registry. | Added explicit stopping, limitation, and research-failure behavior. |
| Polish | The tier table duplicated the bullets and handwritten workflows, while the model announced internal tier labels. | Removed the duplicate table and user-facing tier announcement; retained one generated summary plus one shared writer pipeline. |
| High | The recommended `[single_shot, deep]` preset removed the gated `direct` procedure and therefore removed meta/chit-chat handling. | Made meta/capability handling an always-available safety path outside normal tiers; `direct` now covers only timeless factual answers. |

---

## 9. Files touched for the prompt work

- `src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2` — the prompt (rewritten,
  then revamped).
- `src/aiq_agent/agents/adaptive_researcher/tiers.py` — `TierProfile` (+`finalize` field),
  `TIER_PROFILES`, `enabled_tier_profiles`, `clamp_to_enabled_tiers`, `tier_ceiling`.
- `src/aiq_agent/agents/adaptive_researcher/factory.py` — passes `retrieval_tools` +
  `enabled_tiers` + `tier_profiles` into the orchestrator render; guard-free orchestrator
  middleware; preserves delta delegation under Layer-B enforcement.
- `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py` — retains `task` and
  `write_todos` for the mandatory parent-report writer path while preserving the normal tier
  ceiling for other requests.
- `src/aiq_agent/agents/adaptive_researcher/tools/finalize.py` — documents transparent
  no-research capability limitations as a valid `researched=false` use.
- `tests/aiq_agent/agents/adaptive_researcher/test_factory.py`,
  `test_tiers.py` — render/KV-boundary/tier-gating/numbering/finalize tests.

---

## 10. How the prompt is validated

- **KV-boundary invariance** — render the prompt with two very different per-request inputs
  (delta on/off, docs, user, clarifier, triage hint, retrieval tools) and assert the prefix
  before `## Context` is byte-identical (`test_factory::test_kv_prefix_invariant_...`).
- **Tier gating** — only enabled normal tiers appear in Effort Levels and their procedures;
  the meta/capability path and delta writer pipeline remain available independently.
- **Writer-pipeline ordering** — planner, planned research, and writer remain ordered with and
  without source routing; tests assert that research appears between planning and writer.
- **Standard branch safety** — the rendered prompt requires a plan before writer delegation
  and forbids mixing plan-less queries with the writer branch.
- **Delta enforcement** — a shallow-only prompt still contains the mandatory writer pipeline,
  and Layer-B middleware preserves `task` / `write_todos` when parent context is mounted.
- **Finalize / retrieval-tools / escalation / citations / stopping** — each contract is asserted
  directly against the rendered text.
- **Real graph build** — `create_deep_agent` assembles the full graph with a fake chat model
  across default / `[single_shot]` / `[deep]`+source-router configs (no network needed).
- **Behavioral safeguard** — `test_agent.py` drives `agent.run()` with a mocked graph:
  `researched=false` + empty registry does **not** raise; `researched=true` + empty registry
  still raises.

All prompt-related tests live under `tests/aiq_agent/agents/adaptive_researcher/`. Validation
results should be reported from the current environment rather than assumed from the test text.


<!--  =================== CLAUDE NOTES END =================== -->




<!--  =================== CODEX NOTES START =================== -->

## Codex review notes

### Review perspective

This review treated the prompt as an executable coordination contract, not only as prose. The
main question for every instruction was: *Can the orchestrator actually perform this sequence
with the tools, files, tier configuration, and downstream subagent contracts available at that
point in the run?* The review also looked for duplicated instructions that could drift apart and
for valid configurations that produced no coherent path.

The resulting mental model is deliberately small:

| Request class | Required path | Who writes | How it finishes |
|---|---|---|---|
| Meta / chit-chat, or a transparent no-research capability limitation | No-Research Meta / Capability Path | Orchestrator inline | `submit_final_report(researched=false)` |
| Normal research request | Lowest sufficient enabled tier, with in-loop escalation if needed | Orchestrator inline or writer-agent, as the selected procedure requires | Inline finalizer or writer marker, based on who wrote |
| Parent-report delta | Planned Writer Pipeline, outside normal tier selection | writer-agent | `/shared/output.md` marker |

This separates **normal effort selection** from two correctness paths that must not disappear
when tiers are disabled: meta handling and parent-report delta safety.

### Non-negotiable invariants

1. **No upstream classifier call.** Effort selection stays inside the orchestrator's normal
   first reasoning turn. It must not add a classification tool call or a separate LLM round trip.
2. **Meta never causes research.** Greetings, thanks, identity questions, and small talk use the
   always-available no-research path, including under the recommended `[single_shot, deep]`
   preset where the `direct` factual tier is disabled.
3. **Delta never goes inline.** Mounted parent-report context always invokes the Planned Writer
   Pipeline. This is a citation-safety override, not an implicit selection of disabled `deep`.
4. **Writer implies plan.** writer-agent must never be called unless `/shared/plan.json` exists.
   This follows directly from writer-agent's required-input contract.
5. **A new plan implies new research.** Escalation must execute the planner's still-uncovered
   `ResearchQuery` objects before invoking writer-agent. `planner → writer` is not a valid
   escalation sequence; it must be `planner → planned research → writer`.
6. **Research stays behind `run_research_batch`.** The orchestrator never calls retrieval tools
   directly. Directly authored queries must use exact names from the dynamic Retrieval Tools
   list in `preferred_tools`.
7. **Finalization follows authorship, not tier labels.** Inline answers use
   `submit_final_report`; writer-agent answers use `/shared/output.md` and its completion marker.
   This remains true after escalation.
8. **No-research signaling is narrow.** `researched=false` is valid only for the meta/capability
   path or a genuinely timeless `direct` answer. Failed research must not be relabeled as direct.
9. **Citations come only from the verified registry.** Every material sourced claim in an
   inline answer needs a numeric citation mapped one-to-one to an exact verified locator.
10. **The KV-cache boundary remains strict.** Request-specific facts—including parent-context
    presence and filtered retrieval tools—stay below `## Context`; only reusable rules stay above.

### Why the final structure is simpler

- Tier metadata is rendered once under **Effort Levels**; the duplicate routing table was
  removed.
- Plan→research→writer ordering is written once in **Planned Writer Pipeline** and reused by
  `deep`, `standard`-writer, delta, and escalation.
- `standard` has two complete branches instead of independent planner/writer choices that could
  be combined incorrectly.
- Meta and delta are explicit safety paths rather than hidden exceptions inside tier prose.
- The orchestrator does not announce internal tier labels to the user; its tool path is already
  observable in traces.
- The inline citation contract contains only the essential writer-equivalent rules instead of
  copying the entire writer prompt.

### Intentional tradeoffs

- **Delta can exceed a shallow-only normal ceiling.** This is intentional: preserving parent
  citations is more important than honoring a low-cost tier for a request that cannot safely be
  completed inline. Layer-B middleware preserves only the delegation needed for this workflow.
- **Tier adherence remains soft.** Except for configured tool visibility and safety overrides,
  the model still decides how much work to perform. Evaluation must measure under- and
  over-escalation rather than assuming prompt compliance.
- **`standard` retains model discretion.** It may finish inline or choose the planned writer
  branch. The prompt makes both branches valid, but evals should determine whether the decision
  boundary is stable across supported orchestrator models.
- **`direct` is deliberately conservative.** A fact that may be stale or uncertain must move to
  a researched tier. This may add latency to some easy questions, but it protects a research
  product from confident memory-only answers.
- **An empty verified registry remains a failure.** The orchestrator may explain that it could
  not verify the request, but it must keep `researched=true` so the runtime does not mistake a
  failed retrieval attempt for a legitimate no-research response.

### Highest-value behavioral eval cases

1. **Meta with `direct` disabled:** `[single_shot, deep]` + "thanks" should call only the
   finalizer with `researched=false`.
2. **Bounded current fact:** should choose `single_shot`, run one narrow batch, cite verified
   sources, and avoid planner/writer.
3. **Light multi-part request:** should choose a valid complete `standard` branch; the writer
   branch must create `/shared/plan.json` first.
4. **Thin shallow evidence:** should issue distinct refined queries or run
   planner→planned-research→writer without resubmitting successful queries.
5. **Delta under shallow enforcement:** `[single_shot]` + `enforce_tier_tools=true` + parent
   context must still expose delegation and produce a standalone writer revision.
6. **Direct-only research request:** `[direct]` + a time-sensitive factual question should
   return a transparent capability limitation, not answer from memory.
7. **Partial evidence:** should answer only supported components, cite them, and disclose the
   missing components without repeated redundant searches.
8. **Empty verified registry:** should never downgrade to `researched=false` or fabricate a
   sources section.
9. **Uploaded-document request:** directly authored queries should prefer the exact available
   knowledge-retrieval tool name and carry enough document context for the researcher worker.
10. **Source-router on/off:** both rendered Planned Writer Pipeline variants must preserve
    contiguous route→plan→research→writer ordering.

### Scope boundaries and follow-ups

- The prompt handles parent-report **delta rewriting**, but general `report_ask` / `report_edit`
  nodes remain outside this POC.
- The clarifier remains outside the POC; `clarifier_result` is only consumed when supplied.
- `triage_hint` is an available advisory hook, but the current factory supplies an empty value;
  it should not be described as an active heuristic until code populates it.
- Conversation position is still not an effort signal unless turn context is explicitly passed
  into this agent.
- After deterministic/render tests pass, model-level eval should focus on effort selection,
  branch adherence, latency/cost by tier, citation survival, and correct stopping behavior.

### Local validation status at review time

- Strict Jinja rendering passed for all-tier, `[single_shot, deep]`, shallow-only delta, and
  direct-only configurations.
- The static prefix before `## Context` remained identical when per-request delta state changed.
- Python compilation and `git diff --check` passed for the touched implementation and tests.
- Ruff and pytest could not execute in the local environment: the executables were absent from
  the project venv, and dependency synchronization failed while building `annoy` because
  `x86_64-linux-gnu-g++` was unavailable. This is an environment limitation, not a passing test
  result; CI or a fully provisioned development environment must still run the scoped suite.

<!--  =================== CODEX NOTES END =================== -->
