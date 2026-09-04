# AI-Q 3.0 — Unified Research Orchestrator POC Plan

> **Status:** Proposed (POC / experimental)
> **Date:** 2026-07-14
> **Owner:** smasurekar
> **Goal:** Evaluate whether a single **DeepAgents-based** research orchestrator can
> replace the current binary shallow-vs-deep split by *dynamically* choosing the
> **width** (fan-out) and **depth** (planning + iteration) needed per request.

> ⚠️ **POC disclaimer (applies to the entire document).** This is an experimental
> POC. Every approach below is **subject to iterative change** based on eval
> observations, latency/cost findings, and implementation challenges. Expect
> **continuous architecture development** — the phases, agent shape, prompts, and
> even the chosen approach may be revised mid-flight. Treat this plan as a starting
> hypothesis, not a fixed spec.

> 📄 **Source-doc note.** The referenced design PDF
> (`misc/AI-Q Shallow Research POC.pdf`) was **not present on disk** when this plan
> was authored. This plan was written from (a) the written Approach 1.0 / 1.1
> description, (b) an in-depth code analysis of `deep_researcher` + `shallow_researcher`,
> and (c) an in-depth analysis of the `deepagents` library. If the PDF is restored,
> reconcile this plan against it — especially the exact wording of "Approach 2".

---

## 1. Objective & success criteria

Replace the current router-driven choice between two separate agents
(`shallow_research_agent` ⇄ `deep_research_agent`, decided by a binary
`DepthDecision` in `chat_researcher`) with **one unified orchestrator** that treats
every query as a DeepAgents workflow but **does not force** elaborate planning,
a separate writer, or a fixed depth/width. Simple queries collapse to a fast,
(near) single-shot path; complex queries expand to the full planner→research→write
pipeline — all decided inside the orchestrator, not by an upstream router.

This corresponds to **"Approach 2"** in the POC doc: *"Treat every research question
as one that goes through the main deepagents workflow but without forcing elaborate
planning, synthesis and fixed depth quotients."*

**A hard constraint on this design: no independent classifier/router LLM call.** The current
production path runs a standalone `intent_classifier` LLM call to decide shallow-vs-deep *before*
invoking the research agent. That "classify-then-invoke" topology is explicitly **not** carried
into the unified agent — the one agent decides effort (and handles meta chatter and report routing)
**inside its own loop**, with no separate classification round-trip. Subagents *within* the single
deepagents graph (planner / researcher / writer / source-router) and parallel research fan-out are
retained — they are part of the one agent, not extra upstream calls.

**POC scope.**
- **In scope:** the single unified research agent (adaptive shallow↔deep in one graph), inline
  writer-skip, the no-research direct-answer + meta path, and eval vs. baseline.
- **Out of scope (later increments):** the **clarifier** (HITL clarification) and the
  `report_ask` / `report_edit` report-follow-up nodes. The POC does not add or replace HITL.

**POC exit criteria (to standardize before coding — see Phase 0):**
- Unified path **matches or beats** the agreed baselines on the chosen eval sets
  (accuracy/quality) **without** materially regressing latency/cost on simple
  queries and **without** regressing quality on complex queries.
- Once acceptable, proceed to **regression benchmark runs** across the full harness.

---

## 2. Baseline established from code analysis

### 2.1 Current split (what we are unifying)

| | `shallow_researcher` | `deep_researcher` |
|---|---|---|
| Framework | plain LangGraph `StateGraph` (`agent ⇄ tools`) | **`deepagents.create_deep_agent`** + subagents |
| Planning | none | dedicated `planner-agent` → structured `ResearchPlan` |
| Width | single agent, sequential tool calls | parallel researcher workers via `run_research_batch` (`max_research_concurrency`=6) |
| Roles | one agent does everything | orchestrator + source-router + planner + researcher(s) + writer |
| Writer | writes its own answer inline | separate `writer-agent` → `/shared/output.md` |
| Budget | `max_tool_iterations`=5, `max_llm_turns`=10, forced synthesis at cap | concurrency/batch caps, `recursion_limit`=2000 |
| Routing | chosen **upstream** by `chat_researcher` `DepthDecision` (`Literal["shallow","deep"]`) | same |

**Key finding:** the split is a *static, binary, upstream* decision. There is **no
adaptive depth/width** today — that is exactly what the POC introduces.

### 2.2 What already exists in `deep_researcher` that we can lean on

The current deep agent is already ~80% of the mechanism we need:

- **Conditional subagents** — `build_deep_research_subagents()`
  (`deep_researcher/factory.py:409`) already *conditionally* includes the
  source-router: `if context.enable_source_router:`. We add an analogous
  `enable_planner` / adaptive gate.
- **Inline-report salvage** — `_salvage_inline_report()`
  (`deep_researcher/agent.py`) already accepts the orchestrator's inline final
  message as the report when no `/shared/output.md` exists (≥400 chars + heading).
  This is the seam for a **writer-less shallow path**.
- **Width dial** — `max_research_concurrency` is threaded into `run_research_batch`
  and rejects over-budget query batches (`tools/research.py`). Making this
  **per-request adaptive** (instead of a fixed 6) is the width knob.
- **Per-role LLM injection** — `LLMProvider` + `LLMRole`
  (`common/llm_provider.py`: `ROUTER, PLANNER, RESEARCHER, ORCHESTRATOR, REPORT_WRITER`);
  `provider.get(role)` falls back to a default. Reusable verbatim.
- **NAT registration pattern** — `DeepResearchAgentConfig(FunctionBaseConfig, name="deep_research_agent")`
  + `DeepResearchWorkflowConfig` (`register.py`). We clone this surface.

### 2.3 What `deepagents` gives us (library facts that shape the design)

- `create_deep_agent(...)` (`libs/deepagents/deepagents/graph.py`) does **not** own a
  loop — it assembles **middleware** on top of LangChain 1.x `create_agent`'s ReAct
  loop. Planning (`TodoListMiddleware`/`write_todos`) and subagents (`task` tool)
  are **middleware, and are optional / prompt-driven**. A model can *choose* not to
  plan or not to delegate — a lightweight path emerges naturally.
- Planning can be **hard-disabled** (`TodoListMiddleware` is not in the protected set)
  or **gated dynamically** via a **custom middleware** that rewrites `request.tools`
  / `request.system_message` per call. This is the idiomatic way to build the
  width/depth router **inside** one agent — **no loop forking needed**.
- **Depth budget knob:** per-invoke `recursion_limit` (`agent.invoke(state, config={"recursion_limit": N})`)
  caps LangGraph super-steps; subagents can be bounded independently.
- **Width** is prompt/config-driven (numbers formatted into the delegation prompt),
  plus the hard `run_research_batch` concurrency cap.
- **Cost:** each `SubAgent` can carry its **own `model`** — cheap model for the
  shallow path, frontier model for planning/orchestration.
- **Critical for Approach 1.1:** the system prompt is **re-assembled and re-sent on
  every model call**. There is no built-in "send once on turn 1." On NIM the fix is
  **server-side KV-cache reuse** (the `KV CACHE BOUNDARY` discipline) and/or **filesystem
  offload** — **not** Anthropic `cache_control`, which is a no-op on this endpoint. (See §5.)

---

## 3. Recommended name for the unified agent

The existing agents are named by *behavior* (`shallow_researcher`, `deep_researcher`).
The unifying property is **adaptivity**, so:

- **Recommended:** `adaptive_researcher`
  - config `_type`: **`adaptive_research_agent`**
  - workflow `_type`: **`adaptive_research_workflow`**
  - classes: `AdaptiveResearchAgentConfig`, `AdaptiveResearchWorkflowConfig`, `AdaptiveResearcherAgent`
  - folder: `src/aiq_agent/agents/adaptive_researcher/`

**Alternatives** (pick per team preference): `unified_researcher` (matches the POC
brief literally), `research_agent` (implies it is *the* researcher going forward),
`dynamic_researcher`, `elastic_researcher`. Recommendation: **`adaptive_researcher`** —
it names the differentiating capability and reads well next to the deprecated
shallow/deep names.

> The rest of this document uses `adaptive_researcher` as the working name.

---

## 4. Approach 1.0 — Unified orchestrator with adaptive planner/no-planner path

**Thesis:** one `create_deep_agent` orchestrator, built **once with all roles present**,
whose **prompt** tells it how much effort to spend per query. The agent self-assesses the
request and self-limits its planning, fan-out, and tool use to respect a **soft budget** —
so adaptivity is achieved **in-loop, by the model**, with **no per-request graph rebuild
and no build-time subagent gating**. Built as a **clone of `deep_researcher`** with
rewritten prompts.

> **Design note (why soft, not hard).** An earlier draft gated the planner/writer
> subagents and `recursion_limit` per request. That is unworkable: `agent.run()` compiles
> the whole graph *before* the first LLM call, so nothing the LLM decides mid-run can add
> or remove a compiled subagent, and mid-run "escalation" across the planner/writer
> boundary would need a full graph rebuild. Making the budget **soft (prompt-driven)**
> sidesteps this entirely — all roles exist at build time, and the model simply *chooses*
> whether to call them. deepagents already makes planning (`write_todos`) and subagents
> (`task`) optional and prompt-driven, so "skip the planner on a simple query" is native
> behavior, not a special path.

### 4.1 New package layout

```
src/aiq_agent/agents/adaptive_researcher/
├── register.py            # AdaptiveResearchAgentConfig / AdaptiveResearchWorkflowConfig
├── agent.py               # AdaptiveResearcherAgent (clone of DeepResearcherAgent + salvage-first)
├── factory.py             # build_adaptive_research_graph (+ adaptive subagent gating)
├── deepagents_runtime.py  # reuse deep_researcher runtime (or import it)
├── custom_middleware.py   # + ComplexityRouterMiddleware (new)
├── models/
│   ├── state.py           # AdaptiveResearchAgentState (adds complexity/budget fields)
│   └── subagent_contracts.py  # reuse ResearchPlan / ResearchQuery / ResearchNotes / …
└── prompts/
    ├── orchestrator.j2    # REWRITTEN: embeds the path-selection workflow
    ├── planner.j2         # reuse / trim
    ├── researcher.j2      # reuse; add "single-shot + write-inline" mode
    ├── writer.j2          # reuse (used only on deep path)
    └── source_router.j2   # reuse (optional)
```

> Prefer **importing** shared pieces from `deep_researcher` (runtime, tools,
> middleware, contracts) over copy-paste, to keep the POC small and reduce drift.
> Copy only what needs POC-specific edits (prompts, factory gating, agent salvage
> order). Keep the change scoped to the new package (per repo guidance).

### 4.2 The path decision — soft, prompt-driven effort (one static graph)

The graph is built **once**, with all roles present (source-router, planner, researcher,
writer). Adaptivity comes entirely from the **prompt**, not from reshaping the graph.

**Layer A (primary): effort guidance in the orchestrator prompt.** Rewrite
`orchestrator.j2` so the orchestrator's normal first reasoning step is to **self-assess
how much effort the query needs** and then **act within that effort level** — choosing
whether to call the planner, how many queries to fan out, which tools to prefer, and
whether to write inline or delegate to the writer. This is *behavioral guidance the model
honors through its tool use*, not a code-enforced gate. Because deepagents subagents/todos
are optional and prompt-driven, "skip the planner and write inline for a simple query" is
just the model choosing not to call those tools. (deepagents' own `deep_research` example
works this way — width/iteration numbers are formatted into the prompt and the model
self-limits.) The self-assessment folds into the orchestrator's **existing** first turn —
it is **not** a separate classifier call, so it adds no extra LLM round-trip.

*Optional soft signal (no separate LLM call).* To reduce variance, a one-line hint may be
**templated into** the prompt (e.g. *"This looks like a simple factual query"*) sourced from a
**trivial, non-LLM keyword heuristic** (a string/regex match on the raw query). It must **not**
be an independent LLM classifier: the whole point of the unified design is to *remove* the
upstream `intent_classifier` — a standalone LLM call that classifies depth *before* the agent
runs. The heuristic seeds the model's judgment without adding any LLM round-trip and without
hard-gating anything.

**No upstream LLM classifier — its responsibilities move *inside* the one agent.** Today
`chat_researcher` runs a standalone `intent_classifier` LLM call (`intent_classifier.py:128-131`)
that, before the deep agent is ever invoked, decides shallow-vs-deep **and** meta-vs-research
**and** report routing in a single classification call. The unified agent **deletes that
pre-step**; every one of those jobs is handled in the orchestrator's own first turn, with **no
extra LLM call**:

- **Meta / chit-chat** ("hi", "who are you", "thanks") — the orchestrator recognizes these from
  the query and **answers conversationally itself, running no research** (no `run_research_batch`,
  no subagents, no citations). This is native ReAct behavior — a research agent simply need not
  search — and requires the no-research safeguard in §4.4.5 so it can't trip the empty-registry
  error.
- **Research routing (shallow vs deep)** — the same in-turn effort self-assessment (§4.3), not a
  separate classifier.
- **Report follow-ups** — gated off `parent_report_context` **already present in state** (a
  non-LLM state check, not a classifier); when present, route to full (writer) effort so parent
  citations stay valid (see the delta caveat in §4.4.6).

No independent classification/router LLM call is introduced anywhere in the path.

**Layer B (optional enforcement, only if eval shows drift): `ComplexityRouterMiddleware`.**
If the model repeatedly ignores its effort guidance, add a custom middleware that, per
model call, hides heavier tools (e.g. `advanced_web_search_tool`, `task`, `write_todos`)
by rewriting `request.tools` via `request.override(...)` — the same mechanism the existing
`TodoSuppressionMiddleware`/tool-visibility middleware already use. Note this is **in-loop
tool visibility**, *not* build-time subagent gating; the subagents still exist, they are
just not advertised on that turn. Introduce this only where measured drift justifies it.

**Cheap static ceilings (the only hard part — not per-tier).** Independent of the soft
budget, keep the existing runaway-cost backstops: `run_research_batch`'s per-call query
cap, a generous static `recursion_limit`, and (optionally) a middleware counter on total
`run_research_batch` calls. These need no per-request decision and no rebuild — they exist
so a misbehaving run cannot blow up cost, while the soft budget shapes normal behavior.

### 4.3 Effort guidance (the soft budget) — replaces the fixed depth quotient

Today, "depth" and "width" are **static config constants** (`max_research_concurrency`=6,
`recursion_limit`=2000, planner/writer always on). The POC replaces the *rigidity* — not
with per-request code gates, but with **effort guidance injected into the orchestrator
prompt** that the model honors through how it plans and uses tools. The "budget" is a
description of *how to behave*, and the mechanism of enforcement is the model's own tool
use — backed by a few static ceilings so cost can never run away.

#### 4.3.1 What the guidance describes (soft, not code-enforced)

The prompt describes an **effort level** the model should self-select and then respect. It
is the same conceptual knob-set as before, but expressed as *instructions*, not fields the
runtime checks:

- **Plan or not** — call `planner-agent` only for multi-faceted work; skip it for factual
  lookups.
- **Width** — how many queries to send to `run_research_batch` (1–3 for simple, up to ~6
  for broad), self-limited by the model.
- **Writer or inline** — delegate final synthesis to `writer-agent` for reports; write the
  answer inline for simple asks.
- **Tool preference** — prefer `web_search_tool` over the heavier `advanced_web_search_tool`
  unless depth is needed; use `knowledge_retrieval` when documents are present.

(For eval, the prompt can also ask the model to state the effort level it chose and why —
a free-text `rationale` that surfaces in traces for mis-routing analysis. It is
observational only; nothing keys off it.)

#### 4.3.2 What signals the model self-assesses from

The model reads these from its context (query + templated hints) and chooses an effort
level accordingly:

- **Query intent / shape** — factual lookup / definition / single-doc summary → low effort;
  comparison / trend / "comprehensive report" / multi-hop → high effort. These are the same
  cues the existing depth router keys on, now read by the agent itself.
- **Clarifier result** (`state.clarifier_result`) — a well-scoped clarified question implies
  less planning than a broad, ambiguous brief. *(The clarifier itself is out of POC scope — §1;
  this signal applies only if a clarifier is reintroduced later, and `clarifier_result` is simply
  absent in the POC.)*
- **Selected data sources** (`state.data_sources`) — few sources + a factual question ⇒
  narrow; many heterogeneous sources + a comparison ⇒ wider fan-out.
- **Available documents** — uploaded docs bias toward `knowledge_retrieval` + a single-shot
  summary over broad web fan-out.
- **Explicit user directives** — "quick"/"brief"/"just tell me" vs. "deep dive"/"full
  report"/"exhaustive" are strong, cheap steers.

> Conversation position ("first turn vs follow-up") is a useful signal but is **not
> currently available** to the agent as wired — the agent receives only a query string
> (+ `clarifier_result`); turn history/`checkpoint_db` live in `chat_researcher`. Using it
> would require passing turn context down (tracked as an open item in §8).

#### 4.3.3 How the guidance is applied (no extra LLM call)

1. **In-prompt self-assessment.** The rewritten `orchestrator.j2` instructs the model, as
   part of its **normal first reasoning turn**, to judge the effort level and then act. No
   separate `set_research_budget` tool call, no `response_format` pre-step, no dedicated
   classifier turn — so nothing is added to the critical path of a simple query.
2. **Optional templated hint (non-LLM only).** A cheap hint from a **keyword/regex heuristic**
   may be interpolated into the prompt to reduce variance — e.g. *"Assessed as: simple factual
   query."* Advisory; the model can override it. It is **not** an independent classifier LLM call
   — the upstream `intent_classifier` is removed (§4.2), not reused.
3. **Static ceilings do the hard bounding** (§4.3.4) — not per-request logic.

#### 4.3.4 What stays hard: static ceilings (the only enforced part)

These are the runaway-cost backstops, applied once at build time and independent of the
effort level. They do **not** vary per request and require no pre-flight decision:

- **Per-call query cap** — `run_research_batch` already rejects batches larger than
  `max_research_concurrency` (`tools/research.py`). Keep it as the width ceiling.
- **`recursion_limit`** — a single generous static value via
  `agent.with_config({"recursion_limit": …})` (the seam that currently hard-codes `2000`).
  Simple queries finish well within it; it only bites on pathological loops.
- **Optional total-call counter** — a middleware that hard-stops after N `run_research_batch`
  calls, if eval shows the model over-searching.

> The one place a **hard, per-tier** clamp is still applied is operator-driven tier
> disabling via config (§4.8) — an allow-list enforced by hiding the disabled effort
> level's tools (Layer B middleware), not by rebuilding the graph.

#### 4.3.5 Effort profiles (what the prompt tells the model to do)

Descriptions the prompt carries, **not** code presets. The model picks one and self-limits:

| Effort | When | Planner | Writer | Width (self-limited) | Tool preference |
|---|---|---|---|---|---|
| `direct` | a truly timeless fact where retrieval adds nothing | skip | inline | **0 queries — no research** | none (answers directly) |
| `single_shot` (shallow) | one bounded factual question | skip | inline | 1–3 queries | basic web / knowledge |
| `standard` | lightly multi-part | optional | optional | ~3–5 queries | + advanced web if needed |
| `deep` | comparison / trend / report | use | delegate | up to the cap | full set incl. advanced web |

All four normal tiers are present in the one graph; the labels are shorthand for behaviors
the prompt describes, so they are as revisable as prompt text (start with 2 — §"Are 4
required?").

> **No-research safety paths.** `direct` handles a trivial timeless fact where research genuinely
> adds nothing. Meta/chit-chat bypasses normal tier selection and remains available even when
> `direct` is disabled (important for the recommended `[single_shot, deep]` preset). Both call
> **no** `run_research_batch`. These paths leave the source registry empty, which the current code
> treats as a hard failure (`EmptySourceRegistryError`, `agent.py:328`) — so it needs the explicit
> no-research safeguard in §4.4.5 to skip citation verification for that answer instead of erroring.

#### 4.3.6 Adaptation is in-loop and free (no rebuild, no re-invoke)

Because every role exists in the one graph, adjusting effort mid-run is just the model
using more or fewer of the tools it already has:

- **Escalation.** If a shallow pass yields thin evidence (few/low-confidence
  `ResearchNotes`, empty source registry), the model **calls `run_research_batch` again**
  and/or **invokes `planner-agent`** — in the *same* loop, no graph rebuild — bounded by the
  static ceilings. The prompt tells it when to do this. (This is why the soft model matters:
  hard per-request gating would have compiled those subagents *out*, making in-loop
  escalation impossible.)
- **De-escalation / early stop.** If coverage is already sufficient, the model stops
  searching and synthesizes — spending less than the ceiling allows.

Net: **effort is chosen by the model from the query and revised in-loop from evidence, held
inside static ceilings** — replacing the fixed depth quotient with soft, prompt-driven,
per-request behavior.

### 4.4 Shallow behavior = writer-less, (near) single-shot — by the model's choice

The shallow effort levels (`direct` / `single_shot`) are where the biggest latency win
lives. Under the soft model there is **no separate shallow path in code** — the same one
graph runs, and the model simply *chooses not to call* the source-router, planner, and
writer, writing the answer inline. This subsection is grounded in the actual
deep_researcher control flow: what the model skips, what must stay, and the small code
changes needed to make inline writing reliable.

#### 4.4.1 What the deep effort does (the baseline the model trims)

Per `orchestrator.j2`, a full deep run is a chain of **sequential subagent round-trips**:
`write_todos` → (`source-router-agent`) → `planner-agent` (returns `ResearchPlan`,
persisted to `/shared/plan.json`) → `run_research_batch` (parallel researchers return
`ResearchNotes`, persisted to `/shared/research_note_*.json`) → **`writer-agent`**
(reads plan + notes + verified sources, writes `/shared/output.md`, returns only the
marker `Wrote /shared/output.md`) → orchestrator returns the marker. `agent.py`'s
`_extract_final_markdown()` then reads `/shared/output.md` as the authoritative report.

Each handoff is a full LLM turn on the critical path. For a simple factual query most of
them are pure overhead — so the prompt tells the model to skip them.

#### 4.4.2 What "shallow" means here — and the latency math

At `direct` / `single_shot` effort, the orchestrator prompt (§4.2/§4.3) tells the model to:

- **Not call `source-router-agent`** (it is advisory; unnecessary for a narrow query).
- **Not call `planner-agent`** — form the (1–3) `run_research_batch` queries directly from
  the user request instead of delegating and reading `/shared/plan.json`.
- **Not call `write_todos`** — trivial work needs no todo list (native deepagents optionality).
- **Write the answer inline** instead of delegating to `writer-agent`.

The subagents still *exist* in the graph — the model just doesn't invoke them. Net: a
shallow run collapses from ~5 sequential subagent turns to essentially **one retrieval
batch + one inline synthesis** — the primary simple-query latency win.

#### 4.4.3 What MUST stay: retrieval routes through `run_research_batch`

This is a hard correctness constraint the code imposes, unchanged by the soft model. **The
orchestrator must not call source tools directly.** Two reasons, from the code:

1. The orchestrator only holds `helper_tools + run_research_batch` (source tools are
   deliberately excluded — `factory.py:552`; the researcher holds the source tools);
   a direct source-tool call is rejected by the runtime.
2. `run_research_batch` is what calls `register_research_note_sources(notes)` on the
   `SourceRegistryMiddleware`. Citation verification runs **only if**
   `source_registry_middleware.has_sources()` is true (`agent.py:303`); if enabled but
   empty, the run **raises `EmptySourceRegistryError`** (`agent.py:336`).

So even single-shot keeps `run_research_batch` as the retrieval seam — just with a small,
orchestrator-authored query set and no planner in front. The researcher still returns
structured `ResearchNotes` (via `response_format=ResearchNotes`), keeping source
registration, `/shared/` note persistence, and citation integrity intact.

#### 4.4.4 Who writes the report

Because the writer subagent still exists but is not called, the **orchestrator writes the
report inline**: after `run_research_batch` returns, it calls `get_verified_sources` (its
helper tool — the same verified-source list the writer would read) and emits the final
cited report. See the one-line framing in §"who writes the report":

> The orchestrator writes the report inline from the researchers' structured notes,
> dropping the separate writer-agent (**Option A, recommended** — keeps citations intact);
> alternatively the whole graph is bypassed for a single researcher loop on the simplest
> queries (**Option B**, max latency win), or the researcher emits the markdown itself
> (**Option C**, but loses automatic source capture).

Ship **Option A** first. Consider **Option B** (a true graph bypass, ≈ today's
`shallow_researcher` folded in) only if eval shows the orchestrator's own turn is a
material share of simple-query latency — note it re-plumbs the `SourceRegistryMiddleware` /
artifact wiring that `build_deep_research_graph` sets up, so its "same post-processing" is
not free. **Option C** is a fallback only if you specifically want the cheap researcher
model doing prose.

#### 4.4.5 Concrete code changes (small — prompt + a reliable finalize signal)

No build-time subagent gating and no per-request rebuild are needed. The changes are:

1. **Orchestrator prompt** — the rewritten `orchestrator.j2` (§4.2) carries the effort
   guidance: at shallow effort, skip router/planner/todos, build `run_research_batch`
   queries directly, call `get_verified_sources`, and write the final cited Markdown report.
2. **Add a positive finalize signal — do NOT just relax the salvage heuristic.** The safe
   way to capture an inline report is a dedicated **`submit_final_report(markdown)` tool**
   the orchestrator calls when done. `_extract_final_markdown()` then resolves in order:
   `/shared/output.md` (deep) → `submit_final_report` payload (inline) → `_salvage_inline_report`
   (last-resort). This gives a reliable "**this** is the report" marker and avoids the risk
   of relaxing `_salvage_inline_report` to "any last message" — which would wrongly accept a
   short acknowledgment, a re-plan, or refusal as the report (the current ≥400-char +
   heading gate at `agent.py:236-242` exists precisely to reject such chatter, and
   `_extract_final_markdown` takes the *last* message).
3. **Relax the hard error only for the inline case** — the
   `raise ValueError("writer-agent did not produce a final Markdown answer")` at
   `agent.py:300` must accept a `submit_final_report` payload as a valid final answer.
4. **Guard the no-research (`direct`) and meta path so it can't crash.** When the orchestrator
   answers a trivial factual query — or a meta/chit-chat message (§4.2) — from its own knowledge
   with **no** `run_research_batch` call, the source registry is legitimately empty, and today
   that path raises `EmptySourceRegistryError` (`agent.py:328`: the
   `elif self.enable_citation_verification` branch fires when the registry has no sources). The
   direct-answer path must therefore carry an explicit **"no research performed" signal** — e.g.
   `submit_final_report(markdown, researched=False)` or a state marker — so `run()` **skips
   citation verification and the empty-registry raise** for that answer. Crucially, **keep** the
   raise for the *intended-to-research-but-found-nothing* case (that is a real failure that should
   still surface or escalate), so the signal must come from the model's deliberate "answer
   directly" choice, not merely from an empty registry.
5. **Static ceilings unchanged** — keep the generous static `recursion_limit`; shallow
   runs simply use far fewer super-steps within it (no per-request shrink needed).

#### 4.4.6 Post-processing is preserved (important — closes a listed risk)

`_extract_final_markdown()` feeds the *same* `final_message` into the rest of `agent.run()`
regardless of source, so the inline report still flows through **citation verification**
(`verify_citations` against the registry populated by `run_research_batch`),
**`sanitize_report`**, artifact harvest, and `emit_final_report`. Writer-skip therefore
does **not** weaken the citation/sanitization guarantees — closing the "writer-skip
correctness" item in §8.

> **Parent-report / delta mode caveat.** `deep_researcher` supports delta rewrites via
> `parent_report_context` (`_seed_parent_sources`, `writer.j2` `parent_report_context_available`).
> A report follow-up is exactly the kind of request a model might treat as "shallow," but
> the inline path bypasses the parent-context seeding that keeps preserved citations valid.
> The prompt must **route delta/parent-report requests to the full (writer) effort** even
> when they look small (tracked in §8).

#### 4.4.7 Adaptation / escalation is in-loop (no rebuild)

If a shallow attempt yields thin evidence (empty/low-confidence `ResearchNotes`, empty
source registry), the model simply **does more in the same loop** — another
`run_research_batch`, or now invoking `planner-agent` — because every role already exists
in the one graph. The prompt tells it when to escalate; the static ceilings (§4.3.4) bound
it. This is the soft-model equivalent of today's shallow→deep escalation, but internal and
requiring **no graph rebuild or re-invoke** — the specific thing hard per-request gating
would have made impossible.

### 4.5 Deep path = current behavior (with a stretch goal)

- The `standard`/`deep` tiers keep the current planner→research→writer pipeline
  essentially unchanged.
- **Stretch goal — Ralph-style outer loop:** deepagents ships a
  `examples/ralph_mode/` pattern (fresh-context outer iterations using the
  filesystem/state as memory). For very deep requests, wrap the orchestrator in an
  outer loop that re-enters with accumulated `/shared/` notes until a completion
  gate is met. **Explicitly out of scope for the first eval pass** — land only if
  baseline results justify it (cost/latency risk is high).

### 4.6 Wiring & workflow integration

- Register `AdaptiveResearchAgentConfig` (`name="adaptive_research_agent"`) +
  `AdaptiveResearchWorkflowConfig` (`name="adaptive_research_workflow"`), export in
  `agents/__init__.py`.
- **Integration with `chat_deepresearcher_agent` — the chat layer becomes a thin host.**
  The unified agent **replaces the `intent_classifier` node entirely**: meta-vs-research,
  shallow-vs-deep, and report-follow-up routing all move *inside* the one agent (§4.2), so
  there is **no standalone classification LLM call** in front of it.
  1. **New config file** (`configs/config_adaptive_poc.yml`) that points the chat workflow at
     the unified agent and **removes the `intent_classifier` pre-step** (the unified agent *is*
     the router now). *Recommended for clean A/B.*
  2. Keep the existing graph but make it always select the unified agent. Simplest diff; leaves
     the classifier node as dead code.
- **Out of POC scope (see §1):** the `clarifier` (HITL) and `report_ask` / `report_edit` nodes.
  The POC targets the single-agent research path; clarification and report-editing follow-ups are
  a later increment.
- **Kept (not LLM calls, orthogonal to the agent):** the LangGraph checkpointer / `checkpoint_db`
  multi-turn conversation memory and session-scoped citation registry — these are host-layer
  persistence, not classifier calls, and need no change.
- Reuse tools/data-source inheritance exactly as deep_researcher
  (`get_all_tool_refs()` / explicit `tools` + `exclude_tools`, per-request
  `filter_tools_by_sources`, per-user MCP). No data-source changes needed.

### 4.7 Tools integrated with the adaptive agent (existing tools only)

> **No new external tools.** The adaptive agent inherits the **exact same** tool
> surface as `deep_researcher` — the same `data_source_registry` retrieval tools and
> the same internally-built orchestration/helper tools. The POC's novelty is *when
> and how many* of these are used per tier, **not** any new integration.

**A. External data-source / retrieval tools** (registered under `sources/`, inherited
via the `data_source_registry`; toggled per request by `data_sources` +
`filter_tools_by_sources`):

| Tool (config name) | `_type` / package | Purpose |
|---|---|---|
| `web_search_tool` | `tavily_web_search` | Basic web search |
| `advanced_web_search_tool` | `tavily_web_search` | Deeper web search (excluded on shallow tiers) |
| `exa_web_search` | `exa_web_search` | Neural web search |
| `duckduckgo_news_search` | `duckduckgo_news_search` | News search |
| `paper_search_tool` | `paper_search` / `google_scholar_paper_search` | Academic paper search (optional, key-gated) |
| `polymarket_search` | `polymarket_prediction_market` | Prediction-market data (optional) |
| `knowledge_retrieval` | `knowledge_layer` | Uploaded-document / KB retrieval (optional) |

These are **only ever called through `run_research_batch`** by the researcher workers
(the orchestrator does not call source tools directly — same rule as deep_researcher).
Optional/key-gated tools degrade gracefully when their secret is absent (per repo
security rules) — no change for the POC.

**B. Internal orchestration / helper tools** (built by `factory.py`, not external
integrations — reused as-is):

| Tool | Built by | Held by | Role |
|---|---|---|---|
| `run_research_batch` | `tools/research.py` | orchestrator | Fan out `ResearchQuery[]` to concurrent researchers (the **width dial**) |
| `get_verified_sources` | `tools/source_registry.py` | orchestrator + researcher + writer (helper) | Return citation-verified sources captured during the run |
| `think` | helper set | all agents | Scratchpad reasoning |
| `lookup_source_catalog` | `tools/source_routing.py` | source-router (deep tiers only) | Advisory domain/source routing |
| `execute` (sandbox shell) | `FilesystemMiddleware` + sandbox | researcher/writer | Optional code execution — only when a sandbox is configured |
| deepagents built-ins: `write_todos`, `task`, filesystem (`ls/read_file/write_file/edit_file/glob/grep`) | deepagents middleware | orchestrator/subagents | Planning + delegation + `/shared/` scratchpad |

**C. Per-tier tool exposure** (the adaptive part — same tools, different availability):

| Tier | Source tools (via `run_research_batch`) | `write_todos`/`task` planning | `lookup_source_catalog` | Writer / `execute` |
|---|---|---|---|---|
| `direct` / `single_shot` | web/exa/ddg/knowledge (basic), narrow set | suppressed | off | writer skipped; `execute` off |
| `standard` | broader set | light | optional | writer optional |
| `deep` | full inherited set incl. `advanced_web_search_tool` | full | on (if `enable_source_router`) | writer on; `execute` if sandbox set |

Tier-based exposure is enforced by the orchestrator prompt (Layer A) and, if needed,
the `ComplexityRouterMiddleware` rewriting `request.tools` (Layer B, §4.2).

> **Parent-report safety exception.** A delta rewrite is not a normal tier selection. It always
> retains planner/writer delegation—even under a shallow-only preset—because the inline path
> cannot safely preserve parent citations. The configured ceiling still governs normal requests.
> Meta/chit-chat is the other non-tier safety path: it always remains no-research so disabling
> `direct` does not turn a greeting into a retrieval job.

### 4.8 Hard-disabling tiers via config

The tiers are *presets the router selects between* (§4.3.5), so operators can restrict
which ones the adaptive agent is allowed to use — without code changes — via a config
allow-list. A disabled tier can **never execute**, regardless of what the LLM triage
picks. Parent-report delta work is the deliberate exception: it runs the mandatory planned
writer safety workflow without treating `deep` as an enabled normal tier.

#### 4.8.1 Why it's a small change

Effort is **soft guidance the model self-selects** (§4.3) — there is no LLM "budget-emission"
step to intercept. Hard-disabling therefore works purely as a **post-hoc clamp on the effort/tier
the model's behavior maps to**: whatever tier a run resolves to is snapped into the configured
allow-list *before* any tier-specific tool exposure is applied. It is one small deterministic
function plus the Layer-B tool-visibility middleware; nothing else in the flow changes.

#### 4.8.2 Implementation

1. **Config field** on `AdaptiveResearchAgentConfig` (`register.py`) — same booleans /
   list-field style as the existing `enable_source_router` / `enable_citation_verification`:

   ```python
   enabled_tiers: list[Literal["direct", "single_shot", "standard", "deep"]] = Field(
       default=["direct", "single_shot", "standard", "deep"],
       min_length=1,
       description="Tiers the adaptive router may select. The emitted tier is clamped to "
                   "the nearest enabled tier, so disabled tiers never execute.",
   )
   ```

2. **Clamp** applied to the tier the run resolves to, before tier-specific tool exposure:

   ```python
   _TIER_ORDER = ["direct", "single_shot", "standard", "deep"]

   def clamp_to_enabled_tiers(tier: str, enabled: list[str]) -> str:
       if tier in enabled:
           return tier
       idx = _TIER_ORDER.index(tier)
       # snap to nearest enabled tier by rank; ties -> the deeper one (safer for quality)
       return min(enabled, key=lambda t: (abs(_TIER_ORDER.index(t) - idx), -_TIER_ORDER.index(t)))
   ```

3. **Two enforcement points** (belt-and-suspenders): the clamp fixes the effective tier
   before anything runs, and the `ComplexityRouterMiddleware` (§4.2 Layer B) then only
   exposes that tier's tool/subagent surface. **Escalation** (§4.3.6) is likewise bounded
   by the allow-list — it can never escalate into a disabled tier. For a request with mounted
   parent-report context, middleware preserves `task` / `write_todos` solely for the mandatory
   citation-safe writer workflow.

**Behavior:** disabling `deep` caps every request at the highest remaining tier;
disabling `single_shot`/`direct` floors simple queries up to the lowest remaining tier. These
statements apply to normal effort selection; delta safety bypasses tier selection.

#### 4.8.3 YAML (existing AIQ config style)

Drop-in on the adaptive agent block — same inline-field + comment convention as
`deep_research_agent`:

```yaml
  adaptive_research_agent:
    _type: adaptive_research_agent
    enable_citation_verification: true
    orchestrator_llm: nemotron_super_llm
    source_router_llm: nemotron_super_llm
    researcher_llm: nemotron_super_llm
    planner_llm: nemotron_super_llm
    writer_llm: nemotron_super_llm
    # Tiers the adaptive router may select for normal requests.
    # Parent-report delta rewrites always retain the planned writer safety workflow.
    # Omit to allow all four (default).
    enabled_tiers:
      - single_shot
      - standard
      - deep
    exclude_tools:
      - web_search_tool
```

Common presets, purely in config (no code change):

```yaml
    enabled_tiers: [deep]                 # deep-only — behaves like today's deep_research_agent
    enabled_tiers: [single_shot]          # shallow-only normal lane; delta still uses writer safety
    enabled_tiers: [single_shot, deep]    # 2-tier POC default (recommended starting point)
```

The last preset is the **"start with 2 tiers"** recommendation — and because it is
config, 2-tier vs 4-tier can be A/B'd across eval runs without touching code.

---

## 5. Approach 1.1 — Seed the orchestrator with initial retrieval

**Idea:** run a cheap initial retrieval from the raw query and feed the top chunks
into the orchestrator's triage so it routes/plans better on the first turn.

### 5.1 The core constraint (confirmed in library analysis)

> In the DeepAgents / LangChain ReAct loop the **system prompt is re-assembled and
> re-sent on every model call.** There is **no built-in "inject on turn 1 only."**
> So naively hardcoding chunks into `orchestrator.j2` means they are resent every
> orchestrator turn — token cost every step, and low added value after turn 1.

### 5.2 How caching actually works here — NIM KV-cache reuse, *not* Anthropic `cache_control`

> ⚠️ **Correction (from library + repo analysis).** deepagents' bundled prompt-caching middleware
> is **Anthropic/Bedrock/Fireworks-specific and a no-op on a NIM / OpenAI-compatible endpoint**:
> `AnthropicPromptCachingMiddleware` short-circuits unless the model is `ChatAnthropic`
> (`langchain_anthropic/.../prompt_caching.py:101-110`), and deepagents constructs it with
> `unsupported_model_behavior="ignore"`. So a `cache_control`-tagged prefix does **nothing** on
> Nemotron via `integrate.api.nvidia.com`. (An earlier draft recommended exactly this — it is wrong
> for our endpoint.)

The repo's real caching model is **server-side KV-cache reuse (a.k.a. automatic prefix caching)** —
which is exactly why *every* prompt in this codebase carries a `{#- === KV CACHE BOUNDARY === -#}`
marker (static content above, dynamic below). NIM enables it with `NIM_ENABLE_KV_CACHE_REUSE=1`; its
backends (vLLM / TRT-LLM / SGLang) hash KV blocks so any later prompt sharing a prefix skips
recomputing it. The `tokenomics` module already meters it — `cached_tokens`,
`cached_input_per_1m_tokens`, and the profiler's `prompt_caching_prefixes` /
`token_uniqueness_forecast` options (`docs/source/profiling/index.md`).

Two consequences for a *seed*:

- The **static system prompt** (instructions above the KV boundary) is shared across requests and is
  reused automatically. The **seed chunks are unique per query**, so they get **no cross-request
  reuse**.
- Within a *single* run's multiple orchestrator turns the seed is a shared prefix (reused turn-to-turn
  if kept above the boundary), but that saves **prefill compute/latency**, not necessarily **billed
  tokens** — a token discount applies only if the endpoint reports `cached_tokens` *and* pricing sets
  `cached_input_per_1m_tokens`. On the hosted preview endpoint, assume neither.

**Handling the "inject once" challenge (corrected ranking):**

1. **Filesystem offload (recommended — endpoint-agnostic).** Write the seed chunks to
   `/shared/seed_context.md` and let the orchestrator `read_file` them on its first turn
   (deepagents-idiomatic for large context). They never sit in the recurring system prompt, so there
   is **zero per-turn resend cost** regardless of whether KV reuse is enabled — the robust default on
   NIM.
2. **Keep the seed above the KV-cache boundary + rely on NIM KV-cache reuse (complementary,
   self-hosted).** If the seed must live in-context, place it in the static prefix and run with
   `NIM_ENABLE_KV_CACHE_REUSE=1`. This removes the per-turn *recompute* cost but not necessarily the
   *billed-token* cost on a hosted endpoint. **Do not** use the Anthropic `cache_control` middleware —
   it is inert on NIM.
3. **Inject-then-evict via a custom middleware.** Add chunks to the request only on turn 1, then drop
   them via the eviction/overflow helpers. Truly stops resending, but the model loses that context on
   later turns — most complexity/failure surface; use only if (1)/(2) are insufficient.

### 5.3 The follow-up-query challenge

- For **follow-up turns** in a conversation, seeding from the raw latest query is
  often wrong (the query may be a refinement referencing prior turns). **Gate the
  seed retrieval:** only run it on the **first research turn of a session** (detected
  from the checkpointed history / absence of a prior report), skipping it for follow-ups,
  report deltas, and meta-chatter. The gate is a **non-LLM state check** — not the removed
  `intent_classifier` — consistent with §4.2's no-upstream-classifier rule.
- This gating is cheap and lives in the same in-turn triage as the effort decision.

### 5.4 Recommendation for 1.1

Treat **Approach 1.1 as a second-iteration experiment**, layered on a working 1.0.
Start with option (1) **filesystem-offloaded seed** **behind a config flag**
(`enable_seed_retrieval: false` by default) so it can be A/B'd against 1.0 without
touching the core path. Use the profiler's `prompt_caching_prefixes` /
`token_uniqueness_forecast` to measure whether the seed actually helps routing and how it
affects cache behavior before investing in (2)/(3).

---

## 6. Evaluation strategy

Reuse the existing NAT eval harness (`frontends/benchmarks/`) — no new framework.

### 6.1 Standardize & baseline (Phase 0, before any code)

1. **Pick the eval sets** from existing harnesses: `freshqa` (fast factual —
   shallow-leaning), `deepsearch_qa`, `deepresearch_bench` (deep/report-leaning).
   Cover **both** simple and complex queries so adaptivity is actually exercised.
2. **Freeze metrics:** answer accuracy/quality per harness, **plus** latency and
   **LLM-call / token cost** per query (the whole point is cost/latency on simple
   queries). Capture per-tier breakdown once the unified agent exists.
3. **Run baselines on `develop`** (current shallow + deep + router) and record the
   agreed numbers. These are the comparison bar.

### 6.2 POC eval loop (Phase 3–4)

- Run the same sets against `config_adaptive_poc.yml`.
- Compare vs baseline: **quality ≥ baseline**, **latency/cost not worse on simple
  queries**, **quality not worse on complex queries**.
- Slice results by tier the orchestrator chose (mis-routing analysis: did a complex
  query get `single_shot`? did a trivial query go `deep`?). Mis-routing is the #1
  risk — track it explicitly.

### 6.3 Regression phase (Phase 5, after exit criteria met)

- Broaden to the **full benchmark suite** for regression testing.
- Add unit/graph tests mirroring the deep_researcher test layout under
  `tests/aiq_agent/agents/adaptive_researcher/` (subagent gating, budget selection,
  writer-skip salvage path).

---

## 7. Phased execution plan

| Phase | Deliverable | Notes |
|---|---|---|
| **0. Standardize & baseline** | Chosen eval sets + frozen metrics + baseline numbers on `develop` | Blocking prerequisite; no code yet |
| **1. Scaffold** | `adaptive_researcher/` package registered, imports deep_researcher pieces, runs the **deep path only** (parity with current deep agent) | Prove the clone works end-to-end |
| **2. Adaptive path** | Soft effort guidance in `orchestrator.j2` (absorbs meta-chatter + shallow/deep + report routing — no `intent_classifier`) + `submit_final_report` finalize tool + inline writer-skip + no-research/meta safeguard + static ceilings | Core of Approach 1.0 (one static graph) |
| **3. Enforcement (opt.)** | `ComplexityRouterMiddleware` hiding heavier tools per call (in-loop tool visibility) | Only if eval shows the model ignores its effort guidance |
| **4. Eval & compare** | Unified-vs-baseline results + per-tier / mis-routing analysis | Decide pass/iterate |
| **5. Regression** | Full-suite regression runs + tests | After exit criteria met |
| **Stretch** | Approach 1.1 seed retrieval (filesystem-offloaded / KV-reuse, flag-gated); Ralph-style deep loop | Layer on a working 1.0 |

---

## 8. Risks & open questions

- **Mis-routing / effort adherence** (biggest risk): the soft budget is advisory, so the
  model can over-spend on a simple query or under-research a complex one. This now also covers
  the **meta-vs-research** split the removed `intent_classifier` used to make: a genuine research
  query mistaken for chit-chat (answered with no research) or trivial chatter that triggers a full
  run. No hard per-query guarantee — mitigate with the static ceilings, in-loop escalation, optional
  Layer-B tool hiding, and effort-sliced eval (track meta-misroute explicitly).
- **No-research / meta path must not crash:** the `direct` and meta answers legitimately produce an
  empty source registry, which today raises `EmptySourceRegistryError` (`agent.py:328`). Requires the
  explicit "no research performed" signal (§4.4.5) to skip verification for those answers — while
  still erroring on the *tried-to-research-but-found-nothing* case.
- **Delta / parent-report requests routed shallow:** a report follow-up may look "simple"
  but the inline path bypasses `parent_report_context` seeding that keeps preserved
  citations valid. The prompt must force full (writer) effort for delta/parent-report runs.
- **Clarifier & report follow-ups are out of POC scope (§1):** the POC removes the upstream
  `intent_classifier` and does **not** add HITL clarification or `report_ask`/`report_edit`. If a
  clarifier is reintroduced later, the "first-turn vs follow-up" effort signal (and `clarifier_result`)
  becomes available; for the POC it is absent and effort is judged from the query + state only.
- **Summarization middleware = an accepted internal LLM call:** `create_deep_agent` adds a
  `SummarizationMiddleware` that fires a bounded summarizer call when context crosses a threshold
  (`deepagents graph.py`, `summarization.py`). This is an *internal* backstop, not the upstream
  classifier pattern being removed, and is **accepted** for the POC — noted so it is not mistaken for
  a stray extra call in traces.
- **Prompt sprawl:** the orchestrator prompt now carries effort guidance + coordination.
  Keep the KV-cache boundary discipline the current prompts already use; the static graph
  helps (stable system prompt across queries → cross-request KV-cache reuse, §5.2).
- **Writer-skip correctness:** citation verification / sanitization (`verify_citations`,
  `sanitize_report`) must run on the inline report — preserved because `_extract_final_markdown`
  feeds the same `final_message` downstream (§4.4.6); the `submit_final_report` finalize
  tool avoids accepting chatter as the report.
- **Cost of seed retrieval (1.1)** on follow-ups and its interaction with KV-cache reuse /
  filesystem offload (§5.2 — the Anthropic `cache_control` path is inert on NIM).
- **Config surface:** decide which knobs are config ceilings (max width/depth, `enabled_tiers`)
  vs. left to the soft prompt, and whether to keep shallow/deep agents registered for fallback.
- **Reuse vs fork:** how much of `deep_researcher` to import vs copy — lean toward
  import to avoid drift during the POC.

---

## 9. Summary

- **Approach 1.0 is well-supported by the existing code.** `deep_researcher` already
  uses `create_deep_agent`, already gates subagents conditionally, already has an
  inline-report salvage path, and already exposes a width dial. The POC is mostly
  (a) **in-turn effort self-assessment** in a rewritten orchestrator prompt — replacing the
  separate `intent_classifier` LLM call and also absorbing meta-chatter + report routing,
  (b) making the **planner and writer optional** by the model's choice, and (c) a reliable
  inline finalize path (`submit_final_report`) plus the no-research/meta safeguard.
  Recommended agent name: **`adaptive_researcher`**.
- **Approach 1.1** is feasible but constrained by the resend-every-call nature of the
  loop; on NIM the clean answer is **filesystem offload** (or the KV-cache-reuse boundary
  discipline) — **not** Anthropic `cache_control`, which is inert on NIM — **gated to
  first research turns only**, behind a config flag; a *second-iteration* experiment.
- This is a **POC**: approaches, prompts, and architecture will change iteratively
  from eval evidence; expect continuous development.





------------------ Shortened Version of the Plan ------------------

> One point per section, key technical details preserved. Full detail above.

- **Disclaimers.** POC/experimental — approaches, prompts, and architecture change
  iteratively from eval evidence (continuous development). Source PDF
  `misc/AI-Q Shallow Research POC.pdf` was missing; plan built from the written
  Approach 1.0/1.1 + code analysis of `deep_researcher`/`shallow_researcher` +
  `deepagents`.

- **§1 Objective.** Replace the upstream binary `DepthDecision` (shallow ⇄ deep) with
  **one DeepAgents orchestrator** that decides width/depth *itself* per query (= POC
  "Approach 2"). **Hard constraint: no independent classifier LLM call** — the
  `intent_classifier` pre-step is removed; the agent handles meta/depth/report routing in
  its own turn (subagents *inside* the one graph are fine). **Clarifier + `report_ask`/
  `report_edit` are out of POC scope.** Exit criteria: quality ≥ baseline, no latency/cost
  regression on simple queries, no quality regression on complex queries → then regression runs.

- **§2 Baseline.** Today's split is static/binary/upstream with no adaptivity;
  `deep_researcher` is ~80% ready — it already uses `create_deep_agent`, conditionally
  gates subagents (`if context.enable_source_router`, `factory.py:409`), has inline
  salvage (`_salvage_inline_report`), a width dial (`max_research_concurrency`→
  `run_research_batch`), per-role LLMs (`LLMProvider`/`LLMRole`). deepagents planning
  (`write_todos`) and subagents (`task`) are **optional, prompt-driven middleware**;
  depth = per-invoke `recursion_limit`; **system prompt is resent every call**.

- **§3 Name.** Recommend **`adaptive_researcher`** — config `_type`
  `adaptive_research_agent`, workflow `adaptive_research_workflow`, folder
  `src/aiq_agent/agents/adaptive_researcher/`. (alts: `unified_researcher`,
  `dynamic_researcher`).

- **§4 Approach 1.0** — one `create_deep_agent` orchestrator, built **once with all roles
  present**, that self-selects effort **via prompt (soft budget)**. Adaptivity is in-loop
  and LLM-driven; **no per-request graph rebuild, no build-time subagent gating** (an
  earlier hard-gating draft was unworkable — the graph compiles before the first LLM call,
  so nothing mid-run can add/remove a subagent). Clone of `deep_researcher`, rewritten
  prompts:
  - **§4.1 Package.** New `src/aiq_agent/agents/adaptive_researcher/` (register / agent /
    factory / runtime / middleware / models / prompts). **Import** shared pieces from
    `deep_researcher`; copy only what needs POC edits (prompts, finalize tool, extraction
    order) — keep the change scoped.
  - **§4.2 Path decision (soft, prompt-driven; one static graph).** *Layer A (primary):*
    `orchestrator.j2` instructs the model, in its **normal first turn** (no separate
    classifier call), to self-assess effort and self-limit planning/fan-out/tool use;
    skipping planner/writer/todos is native deepagents optionality. Optional templated hint from
    a **non-LLM keyword heuristic only** (never a classifier LLM call) to cut variance. **No
    upstream `intent_classifier`:** its jobs move inside the first turn — **meta/chit-chat answered
    directly with no research**, shallow/deep chosen in-turn, report follow-ups gated off
    `parent_report_context` state. *Layer B (only if eval shows drift):* `ComplexityRouterMiddleware`
    hides heavier tools per call via `request.override(...)` — **in-loop tool visibility, not
    build-time gating.** *Static ceilings* (hard, not per-tier): `run_research_batch` cap, generous
    `recursion_limit`, optional call counter — runaway-cost backstops.
  - **§4.3 Effort guidance = the soft budget (replaces fixed depth quotient).** Not
    code-enforced fields — **behavioral guidance in the prompt** the model honors via tool
    use (plan-or-not, width, writer-or-inline, tool preference). Model self-assesses from
    query shape / `data_sources` / docs / explicit "quick"/"deep" (clarifier + conversation-position
    signals **out of POC scope** — §1/§8). Applied in the model's first reasoning turn
    (no extra LLM call); optional templated hint. **Only hard part = static ceilings**
    (§4.3.4). Effort profiles `direct/single_shot/standard/deep` are prompt descriptions,
    all available in the one graph (start with 2). **Adaptation is in-loop & free:** thin
    evidence → model calls `run_research_batch`/`planner-agent` again in the same loop
    (no rebuild); enough coverage → stop early.
  - **§4.4 Shallow behavior (writer-less, near single-shot) — biggest latency win, by the
    model's choice.** Same graph; model skips router/planner/todos/writer and writes inline,
    collapsing ~5 turns to **one batch + one inline synthesis**.
    - **Must stay:** retrieval via **`run_research_batch`** (orchestrator holds no source
      tools — `factory.py:552`; it registers sources into `SourceRegistryMiddleware` that
      citation verification needs, else `EmptySourceRegistryError` at `agent.py:336`).
      Researcher still returns structured `ResearchNotes`.
    - **Who writes:** orchestrator writes inline from the notes (Option A, recommended,
      citations intact); Option B = graph bypass for a single researcher loop (max latency
      win); Option C = researcher emits markdown (loses source capture).
    - **Code changes (small):** effort guidance in `orchestrator.j2`; **add a positive
      `submit_final_report(markdown, researched)` finalize tool** (extraction order:
      `/shared/output.md` → finalize payload → `_salvage_inline_report`) — do **not** just relax
      the ≥400-char/heading gate (`agent.py:236-242`), which would accept chatter/acknowledgments as
      the report; relax the `agent.py:300` hard error for the finalize case; **+ no-research/meta
      safeguard**: `direct`/meta answers skip citation verification (else empty registry →
      `EmptySourceRegistryError` at `agent.py:328`), while a genuine tried-but-found-nothing run
      still errors; static `recursion_limit`.
    - **Preserved:** citation verification + `sanitize_report` + artifact harvest run on the
      inline report unchanged (closes §8 writer-skip risk). **Delta/parent-report requests
      must route to full writer effort** (inline path bypasses parent-context seeding — §8).
  - **§4.5 Deep path.** `standard`/`deep` keep the current planner→research→writer
    pipeline unchanged; **Ralph-style outer loop = stretch, out of first eval pass.**
  - **§4.6 Wiring.** Register `AdaptiveResearchAgentConfig`/`AdaptiveResearchWorkflowConfig`,
    export in `agents/__init__.py`; integrate via a new `config_adaptive_poc.yml` that
    points the chat workflow at the unified agent and **removes the `intent_classifier` pre-step**
    (unified agent *is* the router — meta/depth/report routing absorbed); **clarifier +
    `report_ask`/`report_edit` out of POC scope**; checkpointer/multi-turn memory kept (not an LLM
    call); tool/data-source inheritance unchanged.

- **§4.7 Tools (existing only, no new external tools).** Inherits the exact
  `deep_researcher` surface: **retrieval** (`web_search_tool` &
  `advanced_web_search_tool` = tavily, `exa_web_search`, `duckduckgo_news_search`,
  `paper_search_tool`, `polymarket_search`, `knowledge_retrieval`), called only via
  `run_research_batch`; **helpers** (`run_research_batch`, `get_verified_sources`,
  `think`, `lookup_source_catalog`, sandbox `execute`) + deepagents built-ins
  (`write_todos`, `task`, filesystem). Adaptivity = **same tools, tier-gated exposure**
  (shallow hides `advanced_web_search_tool`/planning/writer; deep exposes all).

- **§4.8 Hard-disabling tiers (config, no code change).** Config allow-list
  `enabled_tiers` on `AdaptiveResearchAgentConfig` (same style as `enable_source_router`);
  a **4th clamp** at the config-ceiling seam (§4.3.3) snaps the emitted tier into the
  allow-list, so disabled tiers **never execute for normal effort selection** (escalation is
  bounded by it too). Parent-report delta rewrites bypass normal selection and retain only the
  delegation needed for their citation-safe writer workflow. Enforced both at the clamp and by
  `ComplexityRouterMiddleware` tool exposure. YAML
  presets: `enabled_tiers: [deep]` (deep-only ≈ today's deep agent), `[single_shot]`
  (shallow-only fast lane), **`[single_shot, deep]` (2-tier POC default)** — 2-tier vs
  4-tier A/B'd purely in config.

- **§5 Approach 1.1 (seed retrieval).** Feed cheap initial-retrieval chunks into triage.
  Constraint: prompt is resent every call, so no free "turn-1-only." **Correction:** deepagents'
  Anthropic `cache_control` middleware is a **no-op on NIM**; the repo's real mechanism is
  **server-side KV-cache reuse** (`NIM_ENABLE_KV_CACHE_REUSE=1`, the `KV CACHE BOUNDARY` discipline;
  metered by `tokenomics`). Fix (ranked): **(1) filesystem offload** to `/shared/` + `read_file`
  (endpoint-agnostic, zero resend); (2) keep seed above the KV boundary + rely on KV-cache reuse
  (self-hosted; saves prefill, not necessarily billed tokens); (3) inject-then-evict. **Gate to
  first research turn** (non-LLM state check) behind `enable_seed_retrieval: false`.
  Second-iteration experiment.

- **§6 Evaluation.** Reuse NAT harness (`frontends/benchmarks/`: `freshqa`,
  `deepsearch_qa`, `deepresearch_bench`) covering simple+complex; freeze metrics =
  quality **+ latency + token/LLM-call cost**; baseline on `develop`, then compare
  unified with **per-tier mis-routing analysis**; broaden to full suite for regression.

- **§7 Phases.** 0 baseline → 1 scaffold (deep-parity) → 2 adaptive path (soft effort guidance
  absorbing meta/depth/report routing + `submit_final_report` + writer-less inline + no-research
  safeguard) → 3 optional enforcement middleware → 4 eval/compare → 5 regression + tests;
  stretch = 1.1 seed + Ralph loop.

- **§8 Risks.** Biggest = **mis-routing** (wrong effort, incl. **meta-vs-research** misclassification
  now that the classifier is gone); **no-research/meta path must skip the empty-registry raise**
  (`agent.py:328`); delta/parent-report must route to writer effort; **clarifier + report follow-ups
  out of scope**; **summarization middleware = accepted internal LLM call**; prompt sprawl / KV-cache
  boundary; writer-skip must still run `verify_citations`/`sanitize_report`; seed cost + KV-reuse on
  follow-ups; config ceilings (`enabled_tiers`); reuse-vs-fork drift (prefer import).

- **§9 Summary.** Approach 1.0 is well-supported and mostly prompt work on a `deep_researcher`
  clone: in-turn effort self-assessment (replacing the `intent_classifier` and absorbing
  meta/report routing), optional planner/writer by the model's choice, and a reliable inline
  finalize path + no-research safeguard; Approach 1.1 is a **filesystem-offloaded / KV-reuse**
  (not `cache_control`), first-turn-gated, flag-guarded follow-on. POC = expect iterative change.

------------------------------------------------------------------
