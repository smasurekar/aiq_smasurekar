# Plan: Skip the tier-selection turn (catalog-mode orchestrator)

Status: **implemented** on branch `dev/smasurekar/aiq-adaptive-skip-tier` — see §10 for what was
built, the one deviation from this plan, and the validation results. Sections 1–9 remain the
design rationale.
Review status: amended for deterministic batch resolution, the return-direct terminal exception,
flag-off compatibility, hard parallel budgets, and unchanged Harbor-side integration.

Goal: remove the dedicated tier-selection LLM call. Put every *enabled* tier's workflow
into the orchestrator system prompt up front, and let the orchestrator pick a tier **and
take its first real action in the same assistant turn**. The terminal `direct` and meta paths
are the deliberate exception: their first action is a lone `submit_final_report` call so the
existing `return_direct` fast path still terminates without a trailing model call.

---

## 1. What the extra call actually is today

With `dynamic_orchestrator_sections: true` (`factory.py:405-415`):

| Turn | Prompt | Model output | Cost |
| :-- | :-- | :-- | :-- |
| 1 | `router` preset — intro + effort catalog + selection + finalize (`tiers.py:190`) | `declare_effort_tier(tier=X)` only | 1 full LLM call, zero research progress |
| 2 | swapped to `SECTION_PRESETS[X]` by `ComplexityRouterMiddleware._model_overrides` (`custom_middleware.py:338-353`) | first real action | — |

So the router turn is pure routing overhead. On a `direct` or `single_shot` run — where the
whole request is 2–4 orchestrator calls — that is **25–50% of all orchestrator LLM calls**
and one full network round-trip of latency.

Two things make the router turn *load-bearing* today, and both must be re-engineered:

1. **The prompt swap** — the trimmed per-tier prompt can only be selected once the tier is
   known, and `awrap_model_call` runs *before* the model speaks.
2. **The tool swap** — `_filter_tools` (`custom_middleware.py:306-336`) keys the exposed tool
   set on `self._declared_tier`, which is `None` on turn 1. Under
   `single_loop_single_shot` that means turn 1 deliberately hides the direct source tools.

The design below removes the dependency on #1 for turn 1 and replaces #2 with a *union*
turn-1 tool set plus a post-hoc mismatch guard.

### Reference: how deepagents does this

`libs/deepagents/deepagents/middleware/subagents.py:593-598` builds
`TASK_TOOL_DESCRIPTION.format(available_agents=...)` — every subagent's name + description is
inlined into the single `task` tool description, and the model **selects and dispatches in one
turn**. There is no "declare which subagent you will use" pre-turn. `skills.py` does the same
with progressive disclosure: short cards in the prompt, full body loaded only on demand.

Our tiers are the same shape as their subagents: a small catalog of named procedures. The
plan below is the tier-flavoured version of that pattern.

---

## 2. Recommended architecture — "catalog mode"

Four pieces:

**(a) Catalog prompt.** A new prompt mode `catalog` renders the *union* of the enabled tiers'
sections plus the effort catalog/selection blocks. Not "the full prompt": a shallow-only
config (`enabled_tiers: [direct, single_shot]`) still renders a small prompt, because the
union is computed from `SECTION_PRESETS`, not hardcoded.

**(b) Co-declared tier, except for terminal paths.** On research paths, the prompt instructs the
model to emit `declare_effort_tier(tier=X)` **in the same tool-call batch** as the first real
action, not before it. On `direct` and meta paths, it emits only
`submit_final_report(..., tier="direct"|"meta")`: `submit_final_report(return_direct=True)` exits
immediately only when every client-side call in that turn is return-direct, and the current code
therefore requires the finalizer to be a lone call (`tools/finalize.py:97-104`). The finalizer's
existing `tier` argument and `agent.py._read_tier` fallback preserve observability without a
separate declaration.

**(c) One deterministic batch resolver.** Every middleware that needs the tier consumes the same
run-scoped resolver, but the resolver does **not** mutate once per sibling action. It inspects the
entire current AIMessage tool-call batch, computes one immutable `BatchTierDecision`, caches it by
batch identity, and gives every wrapper the same result regardless of `ToolNode` scheduling order.
Resolution priority is:

1. exactly one valid `declare_effort_tier` in the current batch;
2. the cached accepted tier from an earlier turn;
3. inference from the **complete current action batch** when the model omitted declaration.

An explicit declaration must name an enabled normal tier. `meta` is accepted only with the
no-research terminal finalizer, and `delta` is never catalog-routed. Conflicting declarations,
disabled/unknown tiers, downgrades, mixed incompatible actions, and batches with no enabled
compatible tier produce deterministic corrective `ToolMessage(status="error")` results. Inferred
tiers are persisted only after the associated action is accepted.

**(d) Mode-aware action compatibility.** Inference and mismatch handling use the configured
single-shot execution mode, enabled tiers, and current workflow phase — not one global tool→tier
map:

| First substantive action | Compatible inferred tier |
| :-- | :-- |
| lone `submit_final_report(researched=false, tier="meta")` | `meta` |
| lone `submit_final_report(researched=false, tier="direct")` | `direct` when enabled |
| direct source tool | `single_shot`, only under `single_loop_single_shot` |
| `task(subagent_type="shallow-researcher")` | `single_shot`, only under `single_shot_shallow_subagent` |
| `run_research_batch` in the ordinary delegated mode | lowest enabled of `standard`/`deep`; fall back to `single_shot` **only** when neither is enabled (see §9.1) |
| `run_research_batch` under either single-shot fast lane | lowest enabled of `standard`/`deep`; never `single_shot` |
| `task(planner-agent)` or `task(source-router-agent)` | `tier_ceiling(enabled_tiers)` among `standard`/`deep` |
| `write_todos` | an enabled `standard`/`deep` tier, resolved with the other batch actions; by itself it is not enough to infer safely |
| `task(writer-agent)` | never a valid first action; require an already resolved compatible tier plus the expected plan/research state |

Inference is a *safety net*, never the happy path — log it at WARNING so eval runs surface how
often the model skips the declaration. Step 3 is necessary for models that do not reliably emit
parallel tool calls (see `misc/adaptive-researcher-glm-intermediate-response-analysis.md`).

### Why not the alternatives

- **Drop `declare_effort_tier` entirely and infer always.** Tempting (zero prompt burden), but
  `task(planner-agent)` cannot distinguish `standard`-writer from `deep`, and
  `OrchestratorLoopGuardMiddleware.budgets_for_tier` (`custom_middleware.py:897`) plus all tier
  observability (`agent.py:283-310`) depend on that distinction. Keep the explicit signal,
  infer only on failure.
- **Add a `tier` argument to `run_research_batch` / rely on tool arguments everywhere.** This is
  used intentionally for the terminal finalizer exception, but it cannot cover the deepagents
  built-in `task`, which is the first action on deep and shallow-subagent paths. The batch resolver
  is still required.
- **Cheap classifier model for turn 1.** Still a round trip, plus a second failure mode. Worse.

---

## 3. Code changes

### 3.1 `tiers.py` — add dynamic catalog section resolution

```python
def sections_for_catalog(enabled: list[str]) -> dict[str, bool]:
    """Union of every enabled tier's sections + the selection blocks (turn-1 catalog prompt)."""
```

- Union `SECTION_PRESETS[t]` over `normalize_enabled_tiers(enabled)`.
- Force-on: `effort_catalog`, `effort_selection`.
- Keep `delta_rule` off: a request with parent-report context is detected at graph build time and
  receives the forced `delta` prompt instead of the normal catalog.
- Keep `escalation` on iff `len(enabled) > 1`.
- Do **not** register one static `"catalog"` value in `SECTION_PRESETS`: the catalog depends on the
  enabled subset, while `SECTION_PRESETS` values are static frozensets. Handle `mode == "catalog"`
  explicitly in `_render_orchestrator` and call `sections_for_catalog(enabled_tiers)`.
- **Delete the `"router"` preset** (`tiers.py:190`) and every reference to it (decision §8.3).
  Catalog replaces it outright; there is no in-config rollback.

Determinism matters: return a fully populated map in `SECTION_FLAGS` order. Test all 15 non-empty
subsets of the four normal tiers so a given enabled set always renders a byte-identical prefix.

### 3.2 `prompts/orchestrator.j2` — rewrite the selection contract

Add a render variable such as `catalog_mode` and change the `effort_selection` contract only when
it is true. This preserves the flag-off full-prompt behavior. In catalog mode replace the "very
first tool call" instruction with:

> Decide the effort level as the first step of your normal reasoning, then **act immediately**.
> In your **first** assistant turn emit two tool calls together: `declare_effort_tier(tier=<name>)`
> **and** the first action of that level's procedure (e.g. `run_research_batch`, `task`, a source
> tool). Do not spend a turn on the declaration alone. If your runtime cannot emit two calls at
> once, skip the declaration and start the procedure — never stall.

Immediately follow that with the terminal exception:

> For `direct` and the No-Research Meta / Capability Path, do **not** call
> `declare_effort_tier`. Call `submit_final_report(..., tier="direct"|"meta")` as the only tool
> call so it terminates the run immediately.

Also:

- Add a compact, conditionally rendered "Workflow index" above `## Workflow` mapping each enabled
  tier to its first action. It must reflect `single_loop_single_shot`,
  `single_shot_shallow_subagent`, and `enable_source_router`; do not hard-code one single-shot or
  planned-pipeline entry point.
- The `## Workflow` section already renders one `### <tier>` block per entry in `enabled_tiers`
  and describes the Planned Writer Pipeline exactly once — that is already the right shape for a
  catalog. No structural change needed; just verify `_orchestrator_render_kwargs` passes the
  **full** enabled set in catalog mode (`factory.py:398` currently collapses to `[mode]`).
- Keep every citation/finalization rule unchanged.

### 3.3 `custom_middleware.py` — batch resolver, catalog tools, compatibility guards

**Extract a shared batch resolver.** Today three middlewares each cache the tier independently
(`ComplexityRouterMiddleware.awrap_tool_call:270`, `OrchestratorLoopGuardMiddleware:883`,
`SingleShotShallowDelegationMiddleware:450`). Introduce one run-scoped object and an immutable
per-batch result:

```python
@dataclass(frozen=True)
class BatchTierDecision:
    tier: str | None
    source: Literal["declared", "cached", "inferred", "unresolved"]
    error: str | None

class TierResolver:
    """Resolve each complete tool batch once and cache the run's accepted tier."""
    def resolve_batch(self, state) -> BatchTierDecision: ...
    def commit_decision(self, decision) -> None: ...
    @property
    def tier(self) -> str | None: ...
```

`resolve_batch` reads every sibling call from the current AIMessage before any action policy is
applied, validates declarations and action compatibility, and memoizes the result by batch/message
identity. Every middleware invokes it for itself, so correctness does not depend on which wrapper
runs first. Resolver state updates happen before the first `await`, matching the existing atomic
budget-reservation pattern.

Construct the resolver once in `build_adaptive_research_graph` and pass it to all three
middlewares. After the whole batch validates, the first wrapper calls the idempotent
`commit_decision` before executing any accepted action; later siblings observe the already
committed decision. When a tier was inferred, commit persists it to `/shared/effort_tier.json`
(the declaration tool already writes explicit tiers — `tools/finalize.py:78`). Explicit
declaration always wins, and the inference path must never race a second conflicting write.
`agent.py._read_tier` remains the only post-run consumer, so the resolver does not need to be
plumbed into `agent.py`.

The resolver owns these validations:

- declaration is exactly one enabled tier (or the terminal `meta` exception);
- a later declaration is an enabled upward escalation, never a downgrade;
- every substantive sibling action is compatible with the resolved tier and execution mode;
- a lone ambiguous helper such as `write_todos`, `think`, `read_file`, or
  `get_verified_sources` does not invent a tier;
- conflicting action shapes in one batch are rejected consistently for all siblings;
- `writer-agent` cannot be inferred as the first action and requires the expected persisted
  workflow state before execution.

**Catalog turn-1 union tool set.** `_filter_tools` gains a catalog-only "tier unknown" branch:

| Mode | Catalog turn 1 (tier unknown) | Turn ≥2 |
| :-- | :-- | :-- |
| default / `enforce_tier_tools` | ceiling-based hiding only | unchanged |
| `single_loop_single_shot` | `run_research_batch` **+** direct source tools | unchanged tier-keyed swap |
| `single_shot_shallow_subagent` | `task` + `run_research_batch` (already exposed today) | unchanged |

Cost: one turn of extra tool schemas under `single_loop_single_shot` only, bounded by the
configured source count. Gate the unknown-tier union on catalog mode; when
`dynamic_orchestrator_sections` is false, retain today's exposure and declaration-first behavior.

**Compatibility/mismatch guard.** Use one explicit compatibility matrix, not the blanket
"absorb everywhere else" rule:

- Block a declared tier plus an action that its configured path forbids, such as
  `single_shot + run_research_batch` under either fast lane, or `standard/deep + direct source
  tool` under `single_loop_single_shot`.
- A declared `direct` plus a research action may become an implicit upward escalation only when
  at least one enabled research tier is compatible with that action. Promote to the **lowest
  enabled compatible** tier, persist it, log at INFO, and execute.
- For any other mismatch, promote only when the implied tier is enabled, higher than the cached
  tier, and the whole batch is compatible. Otherwise block with a corrective error.
- Never promote into a disabled tier, infer `single_shot` from `run_research_batch` under a fast
  lane, accept a downgrade, or use a first-call `writer-agent` task as evidence of a valid tier.

Return corrective `ToolMessage(status="error")` values through the existing `_blocked` pattern
(`custom_middleware.py:419-431`) so rejected calls stay inside the agent loop.

**Same-turn budget enforcement.** A catalog first turn may contain several parallel direct
source calls. Resolve `single_shot` before counting, reserve each budget slot before the first
`await`, and block any call beyond `single_shot_search_budget`; counting after the fact still
allows a parallel batch to overshoot the advertised hard cap. Apply the same batch decision to
the request-wide `run_research_batch` guard so a declaration scheduled after its sibling cannot
leave the first batch unbounded.

**Prompt swap from turn 2.** `_model_overrides` reads `resolver.tier` rather than its own cache.
On turn 1 the baked-in catalog prompt stands; after an accepted declared or inferred tier, the
resolved tier's trimmed prompt is swapped in exactly as today. Long `deep` runs keep the existing
per-tier token savings.

### 3.4 `factory.py` — mode selection (no new config field)

Keep the existing `dynamic_orchestrator_sections` field. Its name remains accurate — it controls
whether the initial catalog and later per-tier prompt swap are used — and preserving it avoids an
unnecessary config migration. No new enum, deprecation shim, or validator is needed.

| `dynamic_orchestrator_sections` | Before | After |
| :-- | :-- | :-- |
| `false` | full untrimmed prompt, no swap | unchanged declaration-first behavior and tool exposure |
| `true` | `router` prompt → per-tier swap after declaration | **`catalog` prompt** → per-tier swap from turn 2 |

The build-time branch (`factory.py:405-415`) stays three-way; only the last arm changes from
`_render_orchestrator("router")` to `_render_orchestrator("catalog")`. Delta runs still force
the `delta` prompt with no swap.

Inside `_render_orchestrator`, handle `mode == "catalog"` before the static
`SECTION_PRESETS` lookup, pass the full `enabled_tiers`, and call `sections_for_catalog`.
Per-tier modes still receive `[mode]`; delta still receives the full enabled set and its forced
delta sections. Pass `catalog_mode=dynamic_sections_active` through
`_orchestrator_render_kwargs` so the shared template preserves the flag-off contract.

Construct the shared resolver only for catalog mode and pass the same instance to the three
tier-aware middlewares. The normal config files already set `dynamic_orchestrator_sections:
true`, so they inherit catalog mode without any YAML key change. Update field descriptions and
config comments, but do not rename the field.

### 3.5 `tools/finalize.py` — reword the declaration tool

`declare_effort_tier`'s description is prompt surface for the model, so it must match the selected
mode. Add a build-time `catalog_mode` description variant:

- catalog mode: call it together with the first **non-terminal research action**;
- flag-off mode: retain today's declaration-first description;
- both modes: direct/meta use the lone finalizer with its `tier` argument and never co-batch the
  declaration with `submit_final_report`.

Tool behavior remains unchanged. Do not make `declare_effort_tier` return-direct: research paths
must continue after it.

---

## 4. Edge cases to get right

| Case | Handling |
| :-- | :-- |
| Model emits declaration alone anyway | Harmless — identical to today, just one wasted turn. No error. |
| `direct` / meta | Call `submit_final_report(..., tier=...)` alone; infer/read the tier from finalizer metadata and preserve `return_direct`. |
| Model emits a compatible research action with no declaration | Batch inference fallback + WARNING log + persist after acceptance. |
| Invalid, disabled, or two conflicting declarations | Reject the whole substantive batch consistently; never execute sibling research from an invalid decision. |
| Several incompatible first actions | Resolve the full batch before any wrapper runs and reject every incompatible sibling deterministically. |
| Only an ambiguous helper is called | Leave the tier unresolved and return a corrective message; do not guess from `think`, file reads, `write_todos`, or source-registry inspection alone. |
| Escalation mid-run | Accept only a higher enabled compatible tier, persist it, then swap to its memoized prompt. Reject downgrades. |
| Parent-report delta | Never routed. `is_delta` forces the full delta prompt at build time and no swap (`factory.py:412`). Catalog mode must preserve this exactly. |
| `single_shot` parallel search batch exceeds budget | Reserve slots before awaiting handlers and block excess calls; the hard cap must not overshoot on turn 1. |
| Turn-1 parallel `task` calls | Existing in-flight coalescing in the shallow adapter remains the at-most-once guard (`custom_middleware.py:308-317`). |
| `task(writer-agent)` appears first | Block: writer requires the planned workflow state and cannot establish a tier by itself. |
| `dynamic_orchestrator_sections: false` | Preserve the current full prompt, declaration-first contract, and pre-declaration tool exposure. |
| Request reaches `max_orchestrator_turns` | Test the boundary explicitly: removing the router call gives the same numeric limit one more productive turn than the baseline. Document this as intended or adjust counting deliberately. |

---

## 5. Token / latency accounting

Per request, versus the pre-change `router` behavior (the baseline is a prior commit — see §7):

- **Expected saved:** one full orchestrator LLM call when the model co-dispatches a research
  action or takes the lone direct/meta finalizer path. A model that still declares alone saves
  nothing on that request; report compliance by path rather than assuming exactly one.
- **Added:** turn 1 carries the catalog prompt instead of the router prompt — roughly the
  delta between the union of enabled tier sections and the router preset. For a shallow-only
  config that delta is small; for all four tiers it is the largest.
- **Unchanged:** turns 2+ still use the trimmed per-tier prompt.
- **Budget effect:** the same `max_orchestrator_turns` now permits one additional productive
  model turn on requests that previously spent turn 1 routing. Include boundary cases in the
  comparison so this does not masquerade as an unexplained behavior change.

Net expectation: clear win on compliant `direct` / `single_shot` requests (fewest turns, so the
fixed round trip dominates); roughly neutral on `deep` (one call saved out of many, one larger
turn-1 prompt). Verify total prompt tokens, completion tokens, latency, and quality — a lower raw
call count alone is not a win.

KV cache: turn 1 already invalidated on swap under router mode, so catalog mode does not make
cache behaviour worse. Keep `sections_for_catalog` deterministic so repeated runs of the same
config share a byte-identical prefix.

---

## 6. Implementation checklist

1. `tiers.py` — add dynamic `sections_for_catalog`; do not add a static catalog preset; delete
   the router preset and references. Test all 15 non-empty enabled-tier subsets.
2. `orchestrator.j2` — add the `catalog_mode` conditional contract, mode-aware workflow index,
   and lone direct/meta finalizer exception. Preserve citation/finalization rules.
3. `custom_middleware.py` — add `BatchTierDecision` / `TierResolver`; rewire the three
   middlewares; add catalog-only union exposure, the compatibility matrix, deterministic batch
   rejection, inferred-tier persistence, and same-turn budget reservation.
4. `factory.py` — render catalog dynamically with the full enabled set; preserve delta and
   flag-off modes; construct and inject one resolver; pass `catalog_mode` to the prompt and
   declaration-tool builder.
5. `tools/finalize.py` — add mode-aware declaration-tool description only. Preserve the
   finalizer's lone-call `return_direct` contract.
6. `agent.py` / `register.py` — update descriptions and comments only; keep the existing
   file-based `_read_tier` path and `dynamic_orchestrator_sections` config name.
7. Configs — no new keys or files. Update comments and confirm `config_adaptive_frag.yml`,
   `config_adaptive_shallow_subagent.yml`, and their FreshQA variants load unchanged.
8. Docs — update the adaptive-researcher page under `docs/source/`; remove router-mode references,
   document the direct/meta exception, resolver fallback, and flag-off behavior.

## 7. Validation

Validate in layers; if the local `uv` environment cannot build `annoy`, run the same commands in
the repository's builder image rather than reducing coverage.

1. **Static and prompt rendering**
   - `python -m py_compile` on every touched Python module.
   - `uv run ruff check` on the touched package and tests.
   - Render through the real factory/context path (Jinja `StrictUndefined`), not raw Jinja alone.
   - Snapshot all 15 non-empty enabled-tier catalogs; verify deterministic `SECTION_FLAGS` order,
     full enabled workflows, mode-aware first-action index, and static instructions before dynamic
     context for KV-cache stability.
   - Snapshot `dynamic_orchestrator_sections: false` and prove its prompt contract and initial
     tools remain unchanged. Snapshot delta and prove catalog is never rendered.

2. **Unit tests** (`tests/aiq_agent/agents/adaptive_researcher/`)
   - `test_tiers.py`: dynamic catalog union, determinism, escalation, no delta section, all subsets.
   - `test_factory.py`: catalog/flag-off/delta selection, full enabled set, `catalog_mode` variable,
     mode-aware declaration-tool description, one resolver instance shared by all consumers.
   - `test_custom_middleware.py`: batch memoization and precedence; declaration scheduled before
     and after siblings; invalid/disabled/conflicting declarations; downgrades; mixed actions;
     ambiguous helpers; every execution-mode inference row; no compatible enabled tier; direct
     upward escalation; inferred persistence only after acceptance; first-call writer rejection.
   - Budget tests: declaration scheduled last, same-turn source calls counted, and a parallel batch
     larger than `single_shot_search_budget` executes exactly the allowed number and blocks excess.
   - Loop-guard tests: the first same-turn `run_research_batch` receives standard/deep budgets and
     the changed productive-turn boundary is explicit.
   - Finalizer tests: catalog-mode direct/meta emits only `submit_final_report`, remains
     return-direct, records the tier through final-report metadata, and causes no trailing model
     call; flag-off retains its declaration-first sequence and lone finalizer turn.
   - Retarget every router-specific assertion in `test_tiers.py`, `test_factory.py`, and
     `test_custom_middleware.py`.

3. **Config and runtime smoke**
   - Load both `configs/config_adaptive_frag.yml` and
     `configs/config_adaptive_shallow_subagent.yml` with strict tool validation.
   - Smoke each config against the candidate build; verify non-empty final output, persisted tier,
     no intermediate narration accepted as final, and no Jinja errors.
   - FreshQA `smoke10` remains a useful AI-Q-local regression check; the 500 set is supplementary
     after the Harbor integration gate below.

4. **Harbor DeepSearchQA integration — no Harbor repository changes**
   - Do not edit anything under `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals`.
   - The existing Harbor YAML points at the AI-Q checkout's
     `configs/config_adaptive_frag.yml`, requires the existing function/source names, runs
     `adaptive_research_workflow`, and writes `/workspace/answer.txt`; keep that public contract
     unchanged.
   - Build the candidate AI-Q `deploy/Dockerfile` `builder` target locally and load/retag it to the
     image tag already referenced by the existing Harbor YAML. The checked-in Harbor build helper
     intentionally rejects a different revision, so invoke `docker buildx build` directly; do not
     modify that helper or bump Harbor pins.
   - Record the actual AI-Q commit SHA beside the eval results because the unchanged Harbor YAML's
     literal `AIQ_REVISION` metadata will still show its existing pin.
   - Run the existing `configs/deepsearchqa_adaptive_frag.yaml` unchanged on a small fixed subset,
     then run `configs/deepsearchqa_adaptive_shallow_subagent.yaml` unchanged because both paths
     consume the shared resolver. Require strict preflight success, zero failed trials, non-empty
     `answer.txt`, valid trajectories, and no unresolved-tier terminal research runs.
   - For A/B, sequentially build the pre-change and candidate AI-Q commits under the same referenced
     local image tag, keep Harbor code/config/dataset/model settings identical, retain separate job
     directories, and record the actual AI-Q SHA for each arm. No Harbor pin bump is required.
   - Use the paired full DeepSearchQA run for quality. Compare paired accuracy outcomes, top-level
     orchestrator LLM calls by path, total/cached tokens, wall-clock latency, tier distribution,
     declaration-only rate, inference-warning rate, and budget/mismatch blocks. Do not use the
     shallow-subagent trajectory's `unknown` attribution as an orchestrator-call metric; use the
     AI-Q logs or unambiguous top-level spans for that path.

**Ship gate:** strict Harbor preflight and output contract pass with no Harbor changes; no
invalid/disabled/unresolved research tier reaches terminal output; direct/meta retains the lone
finalizer fast path; paired DeepSearchQA quality is within the pre-agreed statistical tolerance;
total tokens and latency do not regress; and the expected call reduction is reported by path
(with the declaration-only and inferred-tier rates), not assumed to be exactly one everywhere.

---

## 8. Decisions (settled)

1. **Terminal exception** → `direct` and meta call only `submit_final_report(..., tier=...)`; never
   co-batch the declaration with a return-direct finalizer.
2. **Resolution unit** → resolve and memoize the complete tool batch once; never infer by the
   scheduling order of individual wrappers.
3. **Inference target** → choose the lowest enabled tier compatible with the configured action
   path, with two safety overrides: planner/source-router entry into the writer pipeline maps to
   the enabled ceiling among `standard`/`deep` for conservative budgets, and `run_research_batch`
   never infers `single_shot` while an enabled `standard`/`deep` exists (§9.1 — `single_shot`
   makes `OrchestratorLoopGuardMiddleware` inert). `writer-agent` cannot be a first-action
   inference signal.
4. **Mismatch handling** → use the explicit compatibility matrix. Promote only upward into an
   enabled compatible tier; otherwise block. No blanket absorption rule.
5. **`router` mode** → remove it in the same change. The evaluation baseline comes from the
   pre-change AI-Q commit or stored results, not a runtime config switch.
6. **Config surface** → keep `dynamic_orchestrator_sections`. `true` means catalog → per-tier
   swap; `false` preserves the current full-prompt declaration-first behavior.
7. **Delta** → parent-report context is detected before normal routing and always receives the
   forced delta prompt; `delta_rule` is not part of a normal catalog.
8. **Harbor** → no Harbor source, config, build-helper, or pin changes. Rebuild the AI-Q image
   locally under the already referenced tag and record the real AI-Q SHA with each result set.

---

## 9. Caveats

Everything below is a known, accepted sharp edge of this design. Each entry states the hazard,
what it costs when it bites, and the mitigation actually implemented (or deliberately not).

### 9.1 Inferring `single_shot` would silently disable the request-wide guard

**Hazard.** `AdaptiveRequestTerminationConfig.budgets_for_tier` returns `None` for `single_shot`,
`direct`, `meta`, and the pre-declaration state (`models/request_termination.py:139-143`). `None`
does **not** mean "apply default budgets" — it makes `OrchestratorLoopGuardMiddleware` completely
inert for the request:

- no `max_batch_calls` ceiling on `run_research_batch`;
- no `max_total_research_queries` ceiling;
- no duplicate-query blocking;
- and no turn ceiling either, because `_maybe_force_finalize_on_turns` returns early when
  `budgets is None` (`custom_middleware.py:852-858`, `894-900`).

Only `recursion_limit` remains. So a tier inference that lands on `single_shot` removes every
request-scoped research bound — and inference fires exactly when the model is already misbehaving
by skipping its declaration, which is the worst moment to drop the brakes.

**Why the naive rule hits it.** In the ordinary delegated mode (neither fast lane), `single_shot`
legitimately *does* research through `run_research_batch` (`orchestrator.j2:182-187`), so a
"lowest enabled compatible tier" rule would resolve `run_research_batch` → `single_shot` and go
unguarded. The error is lopsided: inferring `standard` when the truth was `single_shot` costs a
slightly larger prompt with the guard still on, while inferring `single_shot` when the truth was
`standard`/`deep` costs an unbounded run.

**Mitigation (implemented).** `run_research_batch` never infers `single_shot` while an enabled
`standard` or `deep` exists; it resolves to the lowest enabled of those two. `single_shot` is used
only when neither is enabled, i.e. when it is the sole research tier available and the guard would
have been inert regardless. This deviates from the original §2(d)/§8.3 wording, which has been
amended to match.

**Residual exposure.** Low. Every adaptive config in the repo enables a fast lane
(`config_adaptive_frag.yml`, `config_adaptive_frag_sandbox.yml`,
`config_adaptive_shallow_subagent.yml`, and the four FreshQA variants), and under a fast lane
`single_shot` is reached via direct source tools or the shallow sub-agent — never
`run_research_batch`. The ordinary-mode path is the *code* default
(`single_loop_single_shot` and `single_shot_shallow_subagent` both `default=False`,
`register.py:126,161`) but no shipped config uses it. Treat this as hardening for a config shape
that does not exist yet, not a live defect.

### 9.2 Co-batching the declaration with the finalizer would cost the call it saves

**Hazard.** `submit_final_report` is `@tool(return_direct=True)` (`tools/finalize.py:104`), and
LangChain routes to the exit node only when **every** client-side tool call in the turn is
return-direct (`langchain/agents/factory.py:1819-1825`). Emitting `declare_effort_tier` alongside
`submit_final_report` therefore breaks the fast exit and forces a trailing model call — exactly
cancelling the call this design removes, on the very tiers (`direct`, meta) where the saving is
proportionally largest.

**Mitigation (implemented).** `direct` and meta never declare. They emit a lone
`submit_final_report(..., tier=...)`, and observability comes from the finalizer's `tier`
argument through `/shared/final_report_meta.json` and the existing `agent.py:310` fallback
(`_extract(EFFORT_TIER_PATH) or _extract(FINAL_REPORT_META_PATH)`).

**Residual exposure.** The model can still ignore the instruction and co-batch. That degrades to
the pre-change cost (one extra turn) and is not an error — it is deliberately not blocked, since
blocking a finalize would be worse than paying one turn. The declaration-only and co-batch rates
are reported by the eval so drift is visible.

### 9.3 Turn budgets now buy one more productive turn

Removing the routing turn does not change `max_orchestrator_turns`; it changes what the budget is
spent on. `_model_turn_count` increments on every model call including the old router turn, so the
same numeric limit now permits one additional *productive* turn. This is intended, but it means a
before/after comparison at the turn boundary is not apples-to-apples — a run that previously
tripped the turn budget may now complete. Boundary cases are covered explicitly in the tests so
this does not surface later as an unexplained behaviour change.

### 9.4 Batch memoization depends on message identity

The resolver memoizes one decision per tool-call batch, keyed on the current `AIMessage`. Messages
carry an `id`, but it is not guaranteed non-`None` for every provider, so the key falls back to
`id(message)` (object identity). Two consequences: a message with no `id` that is re-instantiated
between wrapper invocations would be re-resolved rather than reusing the memo (correct, just not
cached), and the cache is bounded per run and never keyed on message *content*. Upstream
middleware that rewrites history (e.g. `SummarizationMiddleware`) cannot corrupt the decision,
because only the last message is ever inspected — the same constraint the existing
`_declared_tiers_in_current_tool_batch` documents (`custom_middleware.py:137-149`).

### 9.5 Catalog turn 1 exposes a wider tool set

Under `single_loop_single_shot`, catalog turn 1 exposes `run_research_batch` **and** the direct
source tools, because the tier that decides between them is not yet known. This costs one turn of
extra tool schemas, bounded by the configured source count, and it means the model *can* pick the
wrong one. The compatibility matrix is the correctness boundary that replaces the old turn-1
hiding: a declared `single_shot` calling `run_research_batch` (or a declared `standard`/`deep`
calling a source tool) is blocked with a corrective error rather than executed. The wider exposure
is gated on catalog mode only — with `dynamic_orchestrator_sections: false`, turn-1 exposure and
the declaration-first contract are byte-for-byte unchanged.

### 9.6 Harbor: rebuilding under the pinned tag destroys the baseline image

The Harbor config pins `AIQ_RUNTIME_IMAGE: aiq-harbor:c075362751ce` — a **local tag**, not a
registry digest — with `force_build: false`
(`configs/deepsearchqa_adaptive_frag.yaml:35`). That is what makes "rebuild locally, no pin bump"
work. Three consequences worth planning around:

1. **Rebuilding overwrites the baseline.** Once you build the candidate under that tag, the
   pre-change image is gone. Before the first candidate build, preserve it:
   `docker tag aiq-harbor:c075362751ce aiq-harbor:baseline-c075362751ce`. Without that, every A/B
   arm switch costs a full rebuild and the earlier baseline results are not reproducible.
2. **The recorded revision will lie.** `AIQ_REVISION: c075362751ce...` stays literal in the
   unchanged Harbor YAML while the image contains different code. Record the real AI-Q SHA
   alongside each result set; do not trust the YAML metadata in the write-up.
3. **The config is read live from the host.** `config_file` points at the AI-Q checkout's
   `configs/config_adaptive_frag.yml` on disk, so the working tree must stay on the commit that
   matches the running arm for the whole run. Checking out the candidate branch while an arm-A run
   is in flight silently mixes pre-change code with post-change config.

### 9.7 Inference is a fallback, not a contract

The whole inference layer exists for models that will not reliably emit parallel tool calls (see
`misc/adaptive-researcher-glm-intermediate-response-analysis.md`). It is logged at WARNING and
reported as a rate by the eval. If that rate is not near zero on the target model, the right
response is to fix the prompt or reconsider catalog mode for that model — not to grow the
inference table. A large inference table is a signal that the co-declaration contract is not
landing.

### 9.8 Existing configs change behaviour on this branch

`dynamic_orchestrator_sections: true` now means catalog mode rather than router mode, and every
adaptive config in the repo already sets it. So `config_adaptive_frag.yml`,
`config_adaptive_frag_sandbox.yml`, `config_adaptive_shallow_subagent.yml`, and the four FreshQA
variants all switch modes with no YAML edit. That is the intended migration, but it means a run
against any of them is no longer comparable to the stored pre-change results without accounting
for the mode change — see §7 for where the baseline comes from.

---

## 10. Implementation summary (as built)

What actually landed on `dev/smasurekar/aiq-adaptive-skip-tier`. Nine files changed, one added;
no new config key.

### 10.1 The change in one paragraph

The orchestrator no longer spends an LLM call choosing its effort tier. Turn 1 now renders the
**catalog** prompt — the union of every *enabled* tier's sections — and the model emits
`declare_effort_tier(tier=X)` in the **same tool-call batch** as that tier's first action. Because
the tier now arrives mid-batch rather than a turn early, a single run-scoped `TierResolver`
resolves each batch once and every tier-aware middleware reads that one decision. The terminal
`direct` and meta paths are the exception: they emit a lone `submit_final_report(..., tier=...)`
with no declaration, preserving the finalizer's `return_direct` fast exit (§9.2). From turn 2 the
resolved tier's trimmed prompt is swapped in exactly as before, so the per-tier token savings are
unchanged.

### 10.2 File by file

| File | What changed |
| :-- | :-- |
| `tiers.py` | Added `sections_for_catalog(enabled)` — derives the turn-1 union from `SECTION_PRESETS` (forces the selection blocks on, `delta_rule` off, `escalation` only when >1 tier is enabled). Deleted the `router` preset. |
| `custom_middleware.py` | Added `BatchTierDecision` + `TierResolver` (batch memoization, declaration validation, mode-aware inference, compatibility matrix, inferred-tier persistence). Rewired `ComplexityRouterMiddleware`, `OrchestratorLoopGuardMiddleware`, and `SingleShotShallowDelegationMiddleware` to read that one resolver. Added the catalog-only turn-1 union tool branch and changed the single-shot search budget to reserve-before-await. |
| `factory.py` | Build-time prompt switches from `_render_orchestrator("router")` to `("catalog")`; `catalog` renders with the **full** enabled set. Constructs one `TierResolver` per request (catalog mode only) and injects the same instance into all three middlewares. |
| `prompts/orchestrator.j2` | New `catalog_mode` variable gates the selection contract ("decide and act in the same turn" vs. the legacy "very first tool call"), adds a mode-aware **First action by level** index, and states the lone direct/meta finalizer exception. |
| `tools/finalize.py` | `build_declare_effort_tier_tool(..., catalog_mode=...)` selects between two descriptions. Behaviour unchanged. |
| `register.py`, `agent.py` | Field/arg descriptions updated to describe catalog mode. **No new or renamed config key.** |
| `test_catalog_mode.py` *(new)* | 47 tests: resolution precedence, batch-order independence, memoization, declaration validation, every inference row, compatibility matrix, turn-1 exposure, parallel budget. |
| `test_tiers.py`, `test_factory.py` | Router-specific assertions retargeted to catalog; added catalog union/determinism tests and a "flag-off is unchanged" pair. |

### 10.3 Behaviour matrix

| `dynamic_orchestrator_sections` | Turn 1 prompt | Tier signal | Turn-1 tools | Resolver |
| :-- | :-- | :-- | :-- | :-- |
| `false` (default) | full untrimmed prompt | declaration-first, own turn | unchanged | not built |
| `true` | catalog (enabled-tier union) | co-declared with the first action, or inferred | union of the reachable paths | one, shared |
| `true` + parent report | forced `delta` prompt, no swap | unchanged | unchanged | not built |

### 10.4 Deviation from the plan as written

One rule was hardened during implementation, and §2(d)/§8.3 were amended to match:
`run_research_batch` **never** infers `single_shot` while an enabled `standard`/`deep` exists,
because `budgets_for_tier` returns `None` for `single_shot` and that makes the request-wide loop
guard entirely inert. Full rationale in §9.1.

Writing the tests then surfaced a second, related hole in that same rule: with a fast lane active
and only `[direct, single_shot]` enabled, the fallback still inferred `single_shot` for
`run_research_batch` — a tier that under a fast lane cannot run that tool at all. `_implied_tier`
now rejects the call with a message naming the lane's real research path instead.

### 10.5 Validation performed

- `tests/aiq_agent/agents/`: **1001 passed, 8 failed**. The failing set is byte-identical to the
  same suite run on the pre-change tree, i.e. all 8 are pre-existing on this branch and unrelated
  (7 are an `AdaptiveRequestTerminationConfig` validator rejecting the test's own budget fixture;
  1 is a factory tool-list assertion). No new failures introduced.
- `test_catalog_mode.py`: 47/47. `test_shallow_subagent.py`: 29/29.
- All 15 non-empty enabled-tier subsets render deterministically through Jinja, each containing
  every enabled tier's workflow block. Catalog with all four tiers is ~17.1k chars; the per-tier
  prompts swapped in from turn 2 are 3.8k–10.9k, so the post-turn-1 savings hold.
- `config_adaptive_shallow_subagent.yml`'s exact flag combination was exercised end-to-end: one
  shared resolver across three middlewares, a shallow-aware first-action index, `task` reachable
  on turn 1, `declare(single_shot) + task(shallow-researcher)` resolving in one batch, forced
  shallow delegation preserved, `single_shot + run_research_batch` blocked, and the lone `direct`
  finalizer resolving with no declaration.
- `ruff check` and `ruff format` clean on every touched file. One pre-existing `E501` remains in
  the untouched `tools/research.py`.

### 10.6 Not done here

Still open, deliberately: the docs page under `docs/source/` has not been updated to drop
router-mode references, and no eval has been run — the Harbor A/B in §7.4 and the ship gate are
the remaining gate before this is trustworthy beyond unit level.
