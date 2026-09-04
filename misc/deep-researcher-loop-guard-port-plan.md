# Porting `ResearcherLoopGuardMiddleware` to the Deep Researcher

**Target branch:** `dev/smasurekar/research-guard` (currently identical to `develop`)
**Source branch:** `dev/smasurekar/aiq-adaptive-shallow-subagent`
**Scope:** per-researcher circuit breaker for `deep_research_agent` only. No adaptive
tiering, no `declare_effort_tier`, no `OrchestratorLoopGuardMiddleware`,
no `ComplexityRouterMiddleware`, no shallow sub-agent.

## Plan summary

Iteration 1 adds one request-safe `ResearcherLoopGuardMiddleware` to each deep-researcher
worker. A per-invocation `ContextVar` tracks a flat hard ceiling of 10 model-issued source-tool
invocations, repeated identical tool arguments, and consecutive `think` calls; the think logic
is folded into the same middleware so `enabled: false` disables the complete circuit breaker.
When a limit is reached, the guard preserves the last useful source result, withdraws source
tools and `think`, and directs the worker to return evidence-backed `ResearchNotes` with explicit
`ResearchGap` entries. Existing soft search guidance remains unchanged, while new
`StrictUndefined`-safe prompt instructions explain the hard ceiling and graceful exit path.

The guard is wired only into researcher workers, before `ToolRetryMiddleware`, with isolated
state for concurrent queries. The same change also gives the researcher subgraph an explicit,
budget-derived recursion limit instead of inheriting the orchestrator's limit of 2000; this is
kept in a separate commit because it is an independent behavior change. Configuration is added
to the Python schema but not duplicated into shipped YAML profiles until evals calibrate the
default ceiling. Depth-aware budgets, request-wide/orchestrator guards, guards for other agent
roles, and a turn budget that withdraws every remaining tool are deferred to later work.

---

## 1. What the feature is

`ResearcherLoopGuardMiddleware` bounds **one researcher sub-agent invocation** — i.e. one
`ResearchQuery` executed inside `run_research_batch`. It is a deterministic circuit breaker
that runs *underneath* the model, so it does not depend on the model obeying prompt guidance.

Five behaviours, all in `awrap_tool_call` / `wrap_model_call`:

1. **Source-call budget — one single limit for every research query.** Counts every call to a
   *source* tool (the names in `DeepResearchToolSet.source_tool_names`). Helper tools (`think`,
   `get_verified_sources`) and filesystem tools are never counted. **Iteration 1 applies one
   flat limit, `max_source_calls_per_query`, to every `ResearchQuery` identically.** Upstream
   keys this budget off a per-query `low`/`medium`/`high` depth hint; that is deliberately
   **out of scope here** — see §2.
2. **Identical-request blocking.** Each `(tool_name, canonical args)` pair is SHA-256 hashed
   (`_canonical_source_signature`) and counted. Past `max_identical_source_calls`, the call is
   **not executed** — a `ToolMessage(status="error")` is returned instead.
3. **Concurrency safety.** The counter is incremented *before* `await handler(request)`, so
   several source calls dispatched in one assistant turn share a single hard ceiling rather
   than all passing the check and then all executing.
4. **Graceful termination, not a crash.** When a limit is reached:
   - the *last allowed* result is preserved and a nudge is **appended** (never overwritten), so
     the evidence from that final search survives (`_append_nudge`);
   - `wrap_model_call` / `awrap_model_call` withdraw the source tools **and** `think` from every
     later model call (`_filter_tools`), so the model physically cannot search again;
   - the nudge and the blocked-call message both instruct the researcher to return
     `ResearchNotes` now, recording unsupported `target_components` as explicit `ResearchGap`
     entries instead of guessing.

5. **Consecutive-think blocking.** Counts *uninterrupted* `think` calls (any other tool resets
   the counter) and, at `max_consecutive_thinks`, rewrites the think result into a corrective
   warning and sets `state.think_blocked`, which withdraws `think` from the next model call.
   This closes the second loop shape. The alternating
   `think → same_search → think → same_search` loop is caught by the *signature* rule (2), not
   this one — there is an explicit test for it.

> **Divergence from the source branch — the think guard is merged in, not ported as a
> separate class.** Upstream splits (5) into a standalone `ConsecutiveThinkGuardMiddleware`.
> This port folds it into `ResearcherLoopGuardMiddleware`. See §3.3a for the full rationale;
> in short, upstream's split class has no model-call hook and no `_filter_tools` of its own —
> it only *writes* `state.think_blocked`, which `ResearcherLoopGuardMiddleware` then *reads* to
> do the actual enforcement. The configuration (`max_consecutive_thinks`) and the state
> (`ResearcherRunGuardState`) were already unified upstream; only the class boundary was not.

### Source-call accounting semantics — what the limit actually counts

`ResearcherLoopGuardMiddleware` counts **model-issued source-tool invocations**, not individual
provider calls. A non-batched invocation normally maps to one concrete provider call, but a
batch-capable invocation carrying several queries still consumes **one** unit of the
researcher's budget, even though `adapt_source_tools_for_research` expands it into several
concrete calls.

Worked example: one `web_search_tool` invocation containing four queries counts as **one**
researcher-loop unit and **four** concrete source calls. With
`max_source_calls_per_query: 3` and `max_source_tool_batch_size: 4`, a single researcher could
therefore initiate up to twelve concrete calls — bounded not by this guard but by the job-wide
limits below.

This is intentional: the loop guard bounds *model search iterations* and repeated tool-use
loops, not retrieval volume. The four limits in play, and what each one actually bounds:

| Limit | Scope | Bounds |
| :-- | :-- | :-- |
| `researcher_loop_guard.max_source_calls_per_query` | one researcher invocation | model-issued search **iterations** |
| `max_source_tool_batch_size` | one tool invocation | concrete **inputs** in one logical call (`register.py:114`) |
| `max_concurrent_source_tool_calls` | whole job | concrete calls executing **concurrently** (`register.py:109`) |
| `resource_limits.max_source_tool_calls` | whole job | aggregate concrete **attempts and batch items** across all researchers (`resource_limits.py:275`) |

Two consequences worth stating in the config field description so operators are not surprised:

- **The repeated-request rule is also per logical invocation.** Its signature covers the tool
  name plus the complete canonical argument object, so re-sending an identical *batch* is caught,
  but two different batches that happen to share some individual queries have different
  signatures and are **not** deduplicated item by item.
- **Framework retries cost one unit, not several.** Because the guard sits outside
  `ToolRetryMiddleware` (§3.5a), a request that `ToolRetryMiddleware` retries three times still
  consumes a single researcher-loop unit. Retries performed internally by a provider SDK are
  likely invisible to both this counter and the job-wide one.

### State isolation

All mutable state lives in a **`ContextVar`**, not on the middleware instance:

```python
# src/aiq_agent/agents/deep_researcher/researcher_context.py  (new file)

@dataclass
class ResearcherRunGuardState:
    invocation_id: str
    source_call_count: int = 0
    source_signature_counts: dict[str, int] = field(default_factory=dict)
    exhausted: bool = False
    exhaustion_reason: str | None = None
    consecutive_think_count: int = 0
    think_blocked: bool = False

CURRENT_RESEARCHER_GUARD_STATE: ContextVar[ResearcherRunGuardState | None] = ContextVar(..., default=None)
```

No `depth` field, no `ResearchDepth` type, no `normalize_research_depth` — iteration 1 has no
depth concept anywhere (§2). Upstream's `researcher_context.py` carries all three; drop them.

This matters because `DeepResearcherAgent.__init__` builds `self.middleware_set` **once** and
reuses those middleware instances for every run (`agent.py:142`). A counter on `self` would
leak across concurrent researcher workers and across requests. The `ContextVar` is set and
reset per invocation in `_run_research_query`, so N concurrent researchers each get their own
budget, and no state survives the invocation. This mirrors the existing job-wide
`activate_source_tool_budget(...)` ContextVar in `agent.py:304` — the two are complementary:
that one bounds the *whole job*, this one bounds *one researcher*.

### Termination semantics — partially addressed in iteration 1

> **Status: two of three fixes are in scope.** The analysis below explains why an exhausted
> researcher can keep looping. Iteration 1 addresses it with **(a)** a prompt instruction that
> makes graceful termination the obvious action once tools are withdrawn (§3.6) and **(b)** an
> explicit researcher `recursion_limit` so a non-cooperative model fails in ~20 turns instead of
> ~1000 (§3.4, §3.5c). The third fix — a turn budget that withdraws *all* tools for a graceful
> stop — remains **deferred** in **§9**, because (a) and (b) should make it unnecessary.
>
> The honest limit of the prompt fix: it raises the floor, it does not impose a ceiling. The
> prompt already carries four separate "stop" instructions that a looping model has ignored.
> `recursion_limit` is what bounds the non-cooperative case; §9 is what would make that bound
> *graceful* rather than a crash.

**The guard never terminates the sub-agent. It only withdraws tools.** Understanding the exit
path matters, because in the pathological case the current design produces the *inverse* of the
feature's stated goal.

*Intended exit.* After exhaustion the source tools and `think` are gone. The researcher is
`create_agent(..., response_format=ResearchNotes)`, so the loop ends when the model emits the
structured response; with nothing worth calling, a cooperative model does that immediately.
`StructuredResponseTextFallbackMiddleware` (`custom_middleware.py:105`) recovers the near-miss
where the model returns schema-valid JSON as prose. This is the common case and it works.

*Why it can still loop.* Exhaustion withdraws source tools and `think` — **nothing else**. The
researcher still holds `get_verified_sources`, the whole `FILESYSTEM_TOOL_NAMES` set
(`read_file`, `ls`, `grep`, `glob`, `edit_file`, `write_file`, plus `execute` when sandboxed),
and any skills tools. None are counted, and the guard has **no turn bound**. A model that keeps
trying can loop `ls → read_file → grep` indefinitely — and each of those resets
`consecutive_think_count`, so the think branch never fires either.

*The hard backstop is weaker than it looks.* `factory.py:669` sets `recursion_limit: 2000` on
the **orchestrator**. `researcher_invoke_config` (`tools/research.py:67–75`) builds the child
config as `dict(runtime.config)` and pops only `run_id` and `configurable`, so `recursion_limit`
survives; and it propagates by design — it is in `CONFIG_KEYS`
(`langchain_core/runnables/config.py:129`) and is inherited through `var_child_runnable_config`
in `ensure_config` even when not copied explicitly. **The researcher therefore runs at 2000
steps ≈ ~1000 model turns, not LangGraph's default of 25.** Contrast
`shallow_researcher/agent.py:358`, which sets `(max_llm_turns * 2) + 10` explicitly; the deep
researcher's sub-agent has no equivalent.

*What happens when it blows.* `GraphRecursionError` → caught at `tools/research.py:93` →
re-raised as `RuntimeError("researcher worker failed for query ...")` →
`asyncio.gather(return_exceptions=True)` (line 167) makes it an error string for that query while
siblings' notes survive → but `run_research_batch` still **raises** at line 293 even though
other workers succeeded → and that tool error is retryable by
`ToolRetryMiddleware(max_retries=3)` in the orchestrator stack (`factory.py:274`), re-running the
whole batch including its successful workers. What finally stops the retries is the
`consumed_queries` ledger (`research.py:236`) tripping the 20-query per-job cap. The ultimate
backstop, `asyncio.timeout(resource_limits.max_execution_seconds)` (`agent.py:309`), kills the
**entire job**, not the sub-agent.

*The gap.* A stuck researcher burns ~1000 turns and then yields a **failed query** — the partial
evidence it did gather is discarded, because the exception path produces no `ResearchNotes` at
all. That is the opposite of "return evidence-backed ResearchNotes with explicit ResearchGaps."

*Scope note.* Two candidate fixes exist (an explicit researcher `recursion_limit`, and a turn
budget that withdraws all tools for a graceful exit). Both are **deferred** — see §9 for the
sketches, the open questions, and why neither is safe to bundle into iteration 1.

### Config shape

```python
# src/aiq_agent/agents/deep_researcher/models/loop_guard.py  (new file)
class ResearcherLoopGuardConfig(BaseModel):            # extra="forbid", frozen=True
    enabled: bool = True
    max_source_calls_per_query: int = 10               # ge=1 — the single flat limit
    max_identical_source_calls: int = 2                # ge=1
    max_consecutive_thinks: int = 3                    # ge=1
```

One flat model, one scalar limit. Upstream's `ResearcherSourceCallBudgets` (the `low`/`medium`/
`high` nested model, its `low <= medium <= high` validator, and `for_depth`) is **not ported** —
see §2.

The field is named `max_source_calls_per_query`, not `max_source_calls`, to keep it visually
distinct from the existing job-wide `resource_limits.max_source_tool_calls`. The two are
complementary: the job-wide one caps a whole run, this one caps a single researcher worker.

No `max_researcher_turns` field: the turn budget stays deferred (§9, fix B). The researcher's
`recursion_limit` **is** in scope, but it is *derived* from `max_source_calls_per_query`
(§3.5c) rather than configured separately — one knob, not two.

---

## 2. Scope decision: iteration 1 has NO depth hint — one flat limit per query

**Decided, not open.** Iteration 1 implements `ResearcherLoopGuardMiddleware` with a **single
source-call limit applied identically to every `ResearchQuery`**. There is no depth hint
anywhere: not on `ResearchQuery`, not in the guard config, not in the guard state, not in the
prompts. `ResearchQuery` is **not touched at all** in iteration 1.

Concretely, the following upstream pieces are **not ported**:

| Upstream artefact | Iteration 1 |
| :-- | :-- |
| `AdaptiveResearchQuery.depth` field | not ported — `ResearchQuery` unchanged |
| `AdaptiveResearchPlan` (planner `response_format` swap) | not ported |
| `ResearcherSourceCallBudgets` (low/medium/high + `for_depth`) | replaced by one `int` |
| `ResearchDepth` type alias | not ported |
| `normalize_research_depth()` | not ported |
| `ResearcherRunGuardState.depth` field | not ported |
| Retyped `run_research_batch` `queries` argument | not ported |
| Depth sections in `orchestrator.j2` / `planner.j2` | not ported |
| Three-way budget table in `researcher.j2` | replaced by one number |
| `depth: low` skips plan read / skill check | not ported |

What this buys: zero structured-output schema changes, so planner and orchestrator LLM
behaviour and every existing eval baseline are untouched. The change is purely a circuit
breaker.

### Iteration 2 (only after the guard has landed): adding the depth hint

Sequencing is deliberate — **the depth hint is implemented only once
`ResearcherLoopGuardMiddleware` is already on this branch**, because depth is a way to *vary* a
budget that must exist first. When that happens, **do it by subclassing, not by editing the base
schema.**
`ResearchPlan` is the planner subagent's `response_format`, and both it and `ResearchQuery`
inherit `_StrictContract`. Adding `depth` to the base `ResearchQuery` therefore changes the
planner's structured-output JSON schema on **every** deep-research run — new enum property in
the response schema for every provider, invalidated prompt-prefix caching, and any golden plan
fixtures broken. That blast radius is exactly what the source branch avoided. Its actual shape:

1. Subclass, don't edit: `AdaptiveResearchQuery(ResearchQuery)` adds the one field;
   `AdaptiveResearchPlan(ResearchPlan)` overrides only `queries` to narrow the element type
   (`adaptive/models/subagent_contracts.py:32-53`).
2. Swap the planner's `response_format` for that subclass at graph-build time, leaving the
   shared spec builder untouched (`adaptive/factory.py:418-424`).
3. Retype **only** the batch tool's `queries` argument to the subclass — this is what makes
   `depth` appear in the tool schema the orchestrator sees (`adaptive/tools/research.py:77`).
4. Leave the shared execution path alone: `format_research_request` already serializes the whole
   query model, so `depth` reaches the worker as text with no change.

Mirror that structure rather than touching `deep_researcher/models/subagent_contracts.py`. At
that point `max_source_calls_per_query` grows into the three-way budget model (or, cleaner, gains
a sibling `source_call_budgets` and is deprecated), the guard state regains a `depth` field, and
`_run_research_query` regains `normalize_research_depth(getattr(query, "depth", None))` — that
defensive `getattr` is exactly what lets a depth-less `ResearchQuery` keep working, so iteration
1 and iteration 2 compose cleanly.

The prompt work (a per-query depth section in `orchestrator.j2`, a bullet in `planner.j2`, the
three-way budget table in `researcher.j2`) is the larger part and needs its own eval pass.

**Two behaviours to decide on deliberately in iteration 2**, because upstream's `depth: low`
does more than shrink a call budget — it also makes the researcher *skip the
`/shared/plan.json` read* and downgrades the mandatory-skill check from MUST to MAY
(`adaptive/prompts/researcher.j2:7,11`). Those trade correctness for latency: skipping the plan
read loses `answer_strategy`, out-of-scope boundaries, component definitions and the declared
language; skipping the skill check lets a query bypass a skill meant to be the controlling
procedure. Port the budget without them unless there is evidence they are needed.

---

## 3. Files to change

> **Branch topology caution.** `dev/smasurekar/aiq-adaptive-shallow-subagent` is **not** a
> descendant of `develop` — the merge base is `e7abd3d` ("feat: isolate and attest OpenShell
> jobs (#298)"). `git diff develop...<branch>` is therefore merge-base relative and mixes the
> guard with unrelated drift in both directions (for example, `develop` later added
> `Annotated`/`StringConstraints` to `ResearchQuery`; the branch does not have them). This is
> low risk for this port — §3.1/§3.2 are new files and §3.3 is append-only — but the one shared
> file being modified, `tools/research.py` (§3.4), was checked line-by-line against current
> `HEAD` rather than trusted from the diff. Re-verify by reading, not by applying a patch.

### 3.1 New: `src/aiq_agent/agents/deep_researcher/researcher_context.py`

Adapted from the source branch (it already lives at this exact path there —
`git show dev/smasurekar/aiq-adaptive-shallow-subagent:src/aiq_agent/agents/deep_researcher/researcher_context.py`),
**not** ported verbatim. Keep `ResearcherRunGuardState` (minus its `depth` field) and
`CURRENT_RESEARCHER_GUARD_STATE`. **Delete** `ResearchDepth` and `normalize_research_depth` —
both exist only to serve the depth hint, which is out of scope (§2). Roughly 25 lines after the
cuts. It imports nothing from `adaptive_researcher`.

### 3.2 New: `src/aiq_agent/agents/deep_researcher/models/loop_guard.py`

**Do not port `adaptive_researcher/models/loop_guard.py` as-is.** Write the flat single-limit
version instead:

```python
class ResearcherLoopGuardConfig(BaseModel):
    """Hard limits for one deep researcher worker invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # False disables the ENTIRE circuit breaker: source budget, repeated-request blocking,
    # and consecutive-think protection. There is no partial-disable mode, and no tool is
    # ever withdrawn while this is False.
    enabled: bool = True
    # Counts model-issued source-tool INVOCATIONS, not concrete provider calls: one
    # batch-capable invocation carrying N queries costs one unit. Bounds search iterations,
    # not retrieval volume — resource_limits.max_source_tool_calls bounds that.
    max_source_calls_per_query: int = Field(default=10, ge=1)
    # Blocks a repeat of the same tool name + identical canonical arguments. Argument
    # key ORDER is canonicalized; case and whitespace are NOT — "AI research" and
    # "ai research" are distinct requests.
    max_identical_source_calls: int = Field(default=2, ge=1)
    max_consecutive_thinks: int = Field(default=3, ge=1)
```

**The `enabled` contract is all-or-nothing, by construction.** Because the think handling is
merged into the same class (§3.3a), a single `if not self._config.enabled` guard at the top of
`awrap_tool_call` and `_filter_tools` disables every behaviour at once. This is the property the
split design could not offer — there, `enabled: false` silently left think-warning injection
running. Write the disabled-path test per behaviour (§5) so this stays true.

Drop `ResearcherSourceCallBudgets` entirely — the nested model, its
`low <= medium <= high` validator, `for_depth`, and the `ResearchDepth` import all exist only
for the depth hint. The file ends up with no import from `researcher_context.py` at all, which
also removes the import-ordering concern noted in §3.3.

Export `ResearcherLoopGuardConfig` from `deep_researcher/models/__init__.py` following the
existing single-line import style there.

### 3.3 Modify: `src/aiq_agent/agents/deep_researcher/custom_middleware.py`

Append a **single** `ResearcherLoopGuardMiddleware` that absorbs the think handling, plus
`_canonical_source_signature`, `_RESEARCHER_BUDGET_NUDGE`, and the `_THINK_TOOL` constant —
i.e. the substance of lines **524–712** of `adaptive_researcher/custom_middleware.py`,
restructured per §3.3a. Nothing else from that file.

Put them here rather than in a new module because `_request_tool_name` (the tool-shape reader
the guard depends on) already lives in this file at line 780 — the source branch imports it
from here. Move the two classes *after* `_request_tool_name`'s definition, or keep them at the
end of the file.

New imports needed at the top of the file: `hashlib`, `json`, and

```python
from .researcher_context import CURRENT_RESEARCHER_GUARD_STATE
from .models.loop_guard import ResearcherLoopGuardConfig   # or via .models
```

`unicodedata` / `_normalize_text` / `_canonical_research_query_signature` are **not** needed —
those belong to `OrchestratorLoopGuardMiddleware`, which is out of scope.

Watch for an import cycle: `models/loop_guard.py` imports from `researcher_context.py`, and
`custom_middleware.py` imports from both. Neither of those two imports `custom_middleware`, so
there is no cycle — but do not let `researcher_context.py` grow an import of `models`.

### 3.3a Why the think guard is merged rather than ported as a second class

Evidence that upstream's split is incidental rather than designed:

- **The split class has no enforcement.** `ConsecutiveThinkGuardMiddleware` implements only
  `awrap_tool_call`. It has no `wrap_model_call` / `awrap_model_call` and no `_filter_tools`.
  All it does is mutate `state.think_blocked`, which `ResearcherLoopGuardMiddleware._filter_tools`
  reads to actually withdraw `think`. Two classes coupled by a shared mutable flag on a
  `ContextVar` dataclass, where one does the deciding and the other does the doing.
- **The config was already unified.** `max_consecutive_thinks` lives on
  `ResearcherLoopGuardConfig`. Upstream's factory therefore has to reach into one component's
  config to construct a different component:
  `ConsecutiveThinkGuardMiddleware(max_consecutive_thinks=researcher_loop_guard.max_consecutive_thinks)`.
- **The state was already unified.** `consecutive_think_count` and `think_blocked` are fields
  on `ResearcherRunGuardState`, alongside the source-budget fields.
- **`enabled: false` is not honored today.** `ResearcherLoopGuardConfig.enabled=False` makes
  `_filter_tools` return early, but the separately-constructed think guard keeps counting and
  keeps rewriting think results with the `WARNING:` text — prompt-visible enforcement from a
  guard the operator switched off. Merging makes one flag disable the whole circuit breaker.
- **Reuse for planner / writer / orchestrator is not available, and is broken upstream.**
  `CURRENT_RESEARCHER_GUARD_STATE.set` appears in exactly one place on the source branch —
  `tools/research.py:94`, inside `_run_research_query`. Those three agents never run in that
  scope, so `.get()` returns `None` for them by construction; that is the sole reason the
  `self._consecutive_think_count` fallback exists. And that fallback is unsafe in both repos:
  `AdaptiveResearcherAgent.__init__` builds `middleware_set` once (`adaptive/agent.py:208`),
  exactly as `DeepResearcherAgent.__init__` does (`agent.py:142`), so one instance — and one
  counter — serves every run. In this repo it is worse than stale state: `register.py` reuses
  the module-level agent built at line 238 whenever no sandbox is configured and `data_sources`
  do not filter the tool list (the CLI and eval paths), so that counter interleaves across
  **concurrent** requests.
- **Its dedicated tests cover dead code.** All ten tests in upstream's
  `TestConsecutiveThinkGuardMiddleware` (test file lines 749–841) assert on
  `mw._consecutive_think_count` with no `ContextVar` set — i.e. they exercise only the fallback
  path this port does not use. Meanwhile the one test covering the real interaction
  (`test_alternating_think_and_same_search_is_bounded`) has to hand-compose both middlewares
  through a nested handler closure; that scaffolding disappears on merge.

**Cost accepted.** If an orchestrator/planner/writer think guard is wanted later, it is a new
class, not a reuse of this one. That is the right shape anyway: those agents need a *differently
shaped* bound, not just a differently scoped one. `ResearcherRunGuardState` carries `depth`,
`source_call_count`, and `source_signature_counts`, all meaningless for an orchestrator that
holds no source tools and delegates through `run_research_batch`; and an orchestrator wants a
per-**request** bound rather than a per-**invocation** one — the job upstream gave to
`OrchestratorLoopGuardMiddleware` (§8), which is safe there only because that instance is built
per-request in `build_adaptive_research_graph`.

**Structure.** One dispatch, two private handlers:

```python
async def awrap_tool_call(self, request, handler):
    state = CURRENT_RESEARCHER_GUARD_STATE.get()
    tool_call = getattr(request, "tool_call", None)
    if not self._config.enabled or state is None or not isinstance(tool_call, dict):
        return await handler(request)
    name = tool_call.get("name")
    if name == _THINK_TOOL:
        return await self._guard_think(request, handler, state, tool_call)
    # Any non-think call breaks a think streak. Resetting here — before the source budget
    # checks rather than after a successful dispatch — is safe: every block path calls
    # _mark_exhausted, and _filter_tools withdraws `think` whenever state.exhausted, so the
    # counter can never be consulted again once a call has been blocked.
    state.consecutive_think_count = 0
    if name not in self._source_tool_names:
        return await handler(request)
    return await self._guard_source_call(request, handler, state, tool_call, name)
```

`_filter_tools`, `wrap_model_call`, and `awrap_model_call` are unchanged from upstream — they
already handle both `state.exhausted` (withdraw source tools **and** `think`) and
`state.think_blocked` (withdraw `think` only).

Keep the two result-rewrite behaviours distinct: the budget nudge **appends**
(`_append_nudge`, so the final search's evidence survives) while the think warning
**overwrites**. Do not unify them.

**Behaviour changes to document**, both intended:

1. `enabled: false` now also disables think-loop protection (note this in the config field
   description in §3.8).
2. The planner, writer, and orchestrator get no think guard at all — see §3.5b.

### 3.4 Modify: `src/aiq_agent/agents/deep_researcher/tools/research.py`

Wrap the body of `_run_research_query` (lines ~83–107) so the guard state is set for the
duration of the researcher invocation and reset in a `finally`. The source-branch diff applies
cleanly to this file — `HEAD` and `develop` are byte-identical here:

```python
async with semaphore:
    guard_state = ResearcherRunGuardState(invocation_id=uuid4().hex)
    guard_token = CURRENT_RESEARCHER_GUARD_STATE.set(guard_state)
    try:
        ... existing body unchanged ...
    finally:
        CURRENT_RESEARCHER_GUARD_STATE.reset(guard_token)
```

Simpler than upstream, which also computes
`depth=normalize_research_depth(getattr(query, "depth", None))` here. Iteration 1 has no depth,
so the state needs nothing from `query` — note the guard state no longer depends on the
`ResearchQuery` at all, which is why `ResearchQuery` is untouched (§2).

**Second change in this file: stop inheriting the orchestrator's `recursion_limit`.** Add one
line to `researcher_invoke_config` (line 67-75), beside the existing pops:

```python
    config.pop("run_id", None)
    config.pop("configurable", None)
    # The orchestrator graph runs at recursion_limit 2000 (factory.py:669) and that value
    # propagates into every child config. Drop it so the researcher subgraph uses the limit
    # set on the runnable itself (factory.py, §3.5c) rather than a bound meant for the parent.
    config.pop("recursion_limit", None)
```

**Why pop here rather than rely on `.with_config` winning.** `recursion_limit` is a first-class
propagating config key (`langchain_core/runnables/config.py:129`), and `researcher_invoke_config`
passes an explicit `config=` to `ainvoke`, so whether the runnable-bound value or the invoke-time
value wins depends on `merge_configs` precedence — which is exactly the kind of thing that
silently regresses on a dependency bump. Popping the inherited value removes the contest: the
runnable's own limit becomes the only source. Assert the *effective* limit in a test (§5), not
where it was set.

The `set` must be **inside** `async with semaphore` (as upstream does) so a queued worker does
not hold a guard state while blocked, and so each admitted worker gets a fresh one.

`getattr(query, "depth", None)` is deliberate: it works with and without the Phase-2 `depth`
field, so §3.4 never needs revisiting.

### 3.5 Modify: `src/aiq_agent/agents/deep_researcher/factory.py`

**(a) `build_deep_research_middleware_set`** — accept
`researcher_loop_guard: ResearcherLoopGuardConfig | None = None` and insert the single guard
into the **researcher** stack only, immediately *before* `ToolRetryMiddleware`.

> **Make the parameter optional with a default, not required.** `researcher_loop_guard or
> ResearcherLoopGuardConfig()` matches how `resource_limits` is already handled
> (`agent.py:118`, `research.py:205`) and avoids breaking every existing call site. There are
> more than you would expect: `tests/aiq_agent/agents/deep_researcher/test_factory.py` calls
> `build_deep_research_middleware_set` at lines 98, 195, 200, 224, `build_deep_research_graph` at
> 355, 390, 451, and `build_researcher_runnable` at 511; `test_agent.py` constructs
> `DeepResearcherAgent(...)` at roughly ten places. A required kwarg turns a focused change into
> a wide test diff for no benefit — and a default of "guard enabled with sane limits" is the
> behaviour you want anyway when a caller says nothing.

```python
researcher_middleware = common()
tool_retry_index = next(
    i for i, mw in enumerate(researcher_middleware) if isinstance(mw, ToolRetryMiddleware)
)
researcher_middleware.insert(
    tool_retry_index,
    ResearcherLoopGuardMiddleware(
        source_tool_names=tool_set.source_tool_names,
        config=researcher_loop_guard,
    ),
)
```

Ordering rationale (worth a comment in the code): the guard must sit **outside**
`ToolRetryMiddleware` so a single logical source request that `ToolRetryMiddleware` retries 3×
is counted **once**, not three times — retries of a transient failure must not consume the
research budget. It must sit **after** `ToolNameSanitizationMiddleware` so the name it matches
against `source_tool_names` is the sanitized one.

Note `tool_set.source_tool_names` is the correct set even though the researcher is bound to the
*adapted* tools from `adapt_source_tools_for_research` — those wrappers preserve the original
tool name (`source_tool_batching.py:300,336`).

**(b) Leave `planner` / `writer` / `orchestrator` stacks alone.** The source branch attaches a
bare `ConsecutiveThinkGuardMiddleware()` to those three (`adaptive/factory.py:134,173,174`).
That usage is unsafe in both repos and must not be carried over — see §3.3a for the evidence.
Because the think handling is merged into `ResearcherLoopGuardMiddleware`, and that class
short-circuits when `CURRENT_RESEARCHER_GUARD_STATE.get()` is `None`, attaching it to those
stacks would also be a silent no-op. So: researcher only. A think guard for the other three
agents is genuinely useful — `think` is bound to every stack via `helper_tools`, and `develop`
has no think protection anywhere today — but it is a different bound over a different lifetime
and belongs in its own change (§8).

**(c) `build_researcher_runnable`** — bound the researcher subgraph explicitly. Paired with the
pop in §3.4; together they guarantee the subgraph no longer runs at the orchestrator's 2000.

```python
# Non-search turns the worker legitimately spends: plan read, skill read, think, synthesis,
# and the structured-response retry. Added to the search budget to size the turn allowance.
_RESEARCHER_NON_SEARCH_TURN_SLACK = 10

# Two graph steps per turn (model + tool node), plus headroom for the structured-output
# correction pass. Mirrors shallow_researcher/agent.py:358.
recursion_limit = (config.max_source_calls_per_query + _RESEARCHER_NON_SEARCH_TURN_SLACK) * 2 + 10
return create_agent(...).with_config({"recursion_limit": recursion_limit})
```

Deriving from `max_source_calls_per_query` rather than hard-coding means raising the search
budget to 30 auto-raises the ceiling to 90; a fixed constant would start crashing legitimate runs
the first time someone tunes the budget up (§6 says that number *will* be tuned). At the default
of 10 this yields **50 graph steps ≈ 25 model turns**.

> **Applies even when `enabled: false` — deliberate exception to the all-or-nothing rule
> (§3.2).** The inherited 2000 is a pre-existing defect on `develop`, not guard enforcement, and
> `enabled: false` should not restore it. Nothing is withdrawn, counted, or blocked when the
> guard is off; the subgraph merely gets a sane graph bound instead of a bound meant for its
> parent. Say so in the config field description so the exception is discoverable.

**(d) `build_deep_research_graph`** — accept `researcher_loop_guard` and pass the three prompt
values into the researcher prompt render:

```python
system_prompt=context.render_prompt(
    "researcher",
    tools=context.tool_set.tools_info,
    execution_enabled=context.runtime.execution_enabled,
    researcher_max_source_calls=researcher_loop_guard.max_source_calls_per_query,
    researcher_max_identical_source_calls=researcher_loop_guard.max_identical_source_calls,
    researcher_loop_guard_enabled=researcher_loop_guard.enabled,
),
```

### 3.6 Modify: `src/aiq_agent/agents/deep_researcher/prompts/researcher.j2`

Port only the loop-guard-relevant hunks of the source branch's researcher prompt (the diff also
carries adaptive-only `depth: low` skip-the-plan-read and skip-the-skill-check language — **do
not port those** in iteration 1, they are meaningless without a `depth` field and they weaken the
skill contract).

**Do NOT replace the existing budget lines (57–58).** They are the researcher's *behavioural*
instruction and must survive whether or not the guard is enabled:

```
- Default source budget per ResearchQuery: one primary source-tool call, plus at most one fallback or corroboration call.
- For broad survey queries, make at most one extra targeted follow-up after the first results, only when needed for target_components.
```

Leave them **unconditional**, and *append* a hard-ceiling sentence that renders only when the
guard is on:

```jinja
{% if researcher_loop_guard_enabled | default(false) %}
- Hard limit (runtime-enforced): at most {{ researcher_max_source_calls }} source-tool calls for this ResearchQuery, and at most {{ researcher_max_identical_source_calls }} call(s) with identical tool arguments. These are a backstop, not a target — follow the default budget above. Once either limit is reached the source tools and `think` are withdrawn; return `ResearchNotes` immediately from the evidence already gathered and record every unsupported component as an explicit `ResearchGap`. Do not guess.
{% endif %}
```

Three things this gets right that the earlier draft did not:

1. **`| default(false)` is mandatory, not defensive.** `render_prompt_template` builds the
   template with `jinja2.StrictUndefined` (`src/aiq_agent/common/prompt_utils.py:80`), so a bare
   `{% if researcher_loop_guard_enabled %}` **raises** `UndefinedError` (wrapped as
   `PromptError`) when the variable is absent — the `{% if %}` does not silently treat undefined
   as falsey. Any renderer that does not pass the new kwargs would break the researcher prompt
   outright.
2. **The soft guidance is not conditional.** If it moved inside the `{% if %}`, disabling the
   guard would leave the researcher with *no* budget guidance at all — a behaviour regression
   caused by turning a safety feature off. Keeping it outside means `enabled: false` restores
   exactly today's prompt, byte for byte.
3. **"identical tool arguments", not "normalized request."** `_canonical_source_signature`
   canonicalizes JSON key *order* only (`json.dumps(..., sort_keys=True)`); it does **not**
   casefold or collapse whitespace. `"AI research"` and `"ai research"` are therefore different
   signatures and both may run. The `_normalize_text` helper that does casefold/NFKC belongs to
   `OrchestratorLoopGuardMiddleware`, which is **not ported** (§8). Upstream's prompt wording
   ("the same normalized source request") overstates the guarantee — do not copy it, and use the
   same accurate phrasing in the middleware docstrings and blocked-call messages.

(In iteration 2 the ceiling line becomes the three-way `low`/`medium`/`high` table from the
source branch; in iteration 1 it renders one number, identically for every query.)

### Second prompt change: graceful termination in `## Handling Failures`

The `## Tool Use` edit above states the *ceiling*. This one states what to **do** when it is
reached — the instruction that makes termination graceful instead of a loop. Append to
`## Handling Failures` (line 65-66), which already carries a one-line version of this idea
(*"Do NOT get stuck retrying - proceed with available information and return the structured
ResearchNotes"*) — extend that section rather than adding a new one:

```jinja
{% if researcher_loop_guard_enabled | default(false) %}
- **When source tools are withdrawn, research is over — write the notes.** The runtime withdraws the source tools and `think` once this query reaches its hard limit. You will see one of: a tool result ending in a `[SYSTEM — ...]` notice, a result beginning `Source tool not executed: researcher loop guard reached ...`, or the source tools simply no longer being callable. All three mean the same thing: stop researching and return your `ResearchNotes` immediately.
- Do not retry the withdrawn tool, do not look for another route to it, and do not substitute `ls`, `read_file`, `glob`, `grep`{% if execution_enabled %}, or `execute`{% endif %} for further research — those remain available for reading context, not for continuing to gather evidence.
- The list under "Tool Availability and Prioritization" is the set you started with, not the set still callable. After a withdrawal it is out of date; trust what you can actually call.
- Report honestly rather than guessing: put every unresolved item in `gaps` as a `ResearchGap` with a concrete `description`, its `impact`, and `suggested_follow_up_queries`. Keep the findings you did ground in sources. Never fill a missing fact from memory.
- Set `evidence_judgment` to reflect the truncation — lower `relevance_score`, `confidence: "low"` where warranted, and a `rationale` stating the research was cut short by the source-call limit. The orchestrator and writer rely on this to tell a truncated note from a complete one.
{% endif %}
```

Why each bullet earns its place — all five come from reading the current prompt and runtime, not
from generic prompt-writing instinct:

1. **Names all three observable signals.** The researcher sees the withdrawal three different
   ways (appended nudge, blocked-call `ToolMessage`, tools silently gone). Only the first two
   carry text; the third is invisible unless the prompt says to expect it.
2. **Blocks the substitute loop — the most important line, and it has no equivalent today.** The
   guard withdraws source tools and `think`, but `get_verified_sources` and every tool in
   `FILESYSTEM_TOOL_NAMES` stay bound. Everything else in this prompt tells the researcher to
   keep working, so "keep working with the tools I still have" is the *compliant* reading
   without this line. This is the concrete loop §9 exists to catch.
3. **Corrects a real prompt/runtime mismatch.** `## Tool Availability and Prioritization`
   (line 98-101) renders the **static** build-time tool list and asserts *"You can ONLY use the
   tools provided below."* That list never updates when the middleware withdraws tools, so after
   a withdrawal the system prompt actively advertises tools that are no longer bound.
4. **Names real schema fields.** `ResearchGap` is exactly `description` / `impact` /
   `suggested_follow_up_queries` (`subagent_contracts.py:181-186`).
5. **Reuses an existing signalling channel.** `ResearchNotes.evidence_judgment`
   (`subagent_contracts.py:189-198`) already carries `relevance_score` / `confidence` /
   `rationale`, and nothing currently tells the researcher to use it to flag a truncated note —
   which is how the orchestrator and writer could distinguish one.

**Model-agnostic by design.** `response_format=ResearchNotes` resolves per request to either
`ProviderStrategy` (native structured output, no tool) or `ToolStrategy` (a bound schema tool),
auto-detected from model capability (`langchain/agents/factory.py:1219-1231`). So the wording
must stay "return your `ResearchNotes`" — line 70's existing phrasing — and must **not** say
"call the ResearchNotes tool," which is wrong on provider-strategy models.

**Placement.** Above the `{#- === KV CACHE BOUNDARY === -#}` marker at line 89. The rendered
values are config-derived and stable per process, so this text is cacheable.

**Align the middleware strings to this vocabulary** (§3.3) so the in-context trigger and the
standing instruction reinforce each other: keep `_RESEARCHER_BUDGET_NUDGE` and the blocked-call
message using "stop searching / return `ResearchNotes` / record `ResearchGap` entries", match
`orchestrator.j2:118`'s "budget is exhausted" phrasing, and add a pointer to setting
`evidence_judgment` honestly.

**Heading collision — do not name the §3.6 budget section "Research Depth."** `researcher.j2` already has
a `## Depth Requirements` section at line 28, and it is about *output richness*, not search
effort. "Source-call budget (runtime-enforced)" is deliberately distinct; keep it that way.

**Good news: the orchestrator prompt already handles budget exhaustion.** Current
`orchestrator.j2:118` reads: *"If the error says the source-tool circuit is open or a
source/query budget is exhausted, do not retry research. Proceed immediately to writer-agent
with the successful notes and verified sources already available, and require the writer to
state material evidence gaps..."* So the guard's blocked-call `ToolMessage` lands in a prompt
that already knows what to do with it. **No orchestrator prompt change is needed in iteration 1** —
but word the guard's blocked-call text to match this existing vocabulary ("budget is exhausted")
so the model's pattern-match fires.

### 3.7 Modify: `src/aiq_agent/agents/deep_researcher/agent.py`

- `DeepResearcherAgent.__init__`: add `researcher_loop_guard: ResearcherLoopGuardConfig | None =
  None`, store `self.researcher_loop_guard = researcher_loop_guard or ResearcherLoopGuardConfig()`
  (same `... or Default()` idiom already used for `resource_limits` at line 118).
- Pass it into `build_deep_research_middleware_set(...)` at line 142.
- Pass it into `build_deep_research_graph(...)` in `_build_orchestrator_agent` (line ~195).

### 3.8 Modify: `src/aiq_agent/agents/deep_researcher/register.py`

- `DeepResearchAgentConfig`: add

  ```python
  researcher_loop_guard: ResearcherLoopGuardConfig = Field(
      default_factory=ResearcherLoopGuardConfig,
      description="Per-researcher circuit breaker: source-call budget, repeated-request and "
                  "consecutive-think limits for one run_research_batch worker. Setting "
                  "enabled=false disables all three.",
  )
  ```

  The config class is `extra="forbid"` (line 65) — a YAML typo fails fast at startup, which is
  the desired behaviour.
- **Two** `DeepResearcherAgent(...)` construction sites need the new kwarg: line 238 and the
  per-request rebuild inside `_run` at line 275. Missing the second one is the easy bug here —
  it is the path actually taken whenever `data_sources` are set or a sandbox is configured,
  i.e. **the normal UI path**.

### 3.9 Chat researcher — **no code change required**

`chat_researcher/register.py:335` resolves the deep researcher by function name
(`builder.get_function("deep_research_agent")`) and line 339 reads
`builder.get_function_config("deep_research_agent")`. It never builds its own graph, tool set,
or middleware set — there are no other callers of `build_deep_research_middleware_set` /
`build_deep_research_graph` / `build_researcher_runnable` anywhere in `src/`.

So configuring `researcher_loop_guard` under the `deep_research_agent` block automatically
applies to chat researcher's deep path. Add one integration assertion (§5) to lock this in.

`shallow_research_agent` is a separate agent with its own graph and is **not** affected.

---

## 4. Config files: where to add `researcher_loop_guard`

The block goes under the `deep_research_agent` function definition, as a sibling of
`enable_source_router` / `max_research_concurrency`. It is **optional everywhere** — omitting it
yields the Pydantic defaults — so nothing breaks in files you skip.

> **Recommendation for iteration 1: add the block to no config file at all.** Ship on the code
> defaults and let the first eval run calibrate `max_source_calls_per_query` (§6). Since that
> number is expected to change, writing `10` into eleven YAML files converts a one-line retune in
> `models/loop_guard.py` into an eleven-file edit — and any file that gets missed silently keeps
> the stale value. There is precedent: **no config in `configs/` currently sets
> `resource_limits`** either; they all ride the defaults.
>
> The table below therefore answers *"where would this go, if we wanted it?"* — it is a map for
> after the number settles, not an iteration-1 task list. Revisit it once the eval has confirmed
> a value, and even then add the block only to profiles that genuinely want something different
> from the default.

### `configs/` — all 11 files declare `_type: deep_research_agent`

| Config file | Line of `_type: deep_research_agent` | Recommendation |
| :-- | --: | :-- |
| Config file | Line of `_type: deep_research_agent` | If/when the block is added |
| :-- | --: | :-- |
| `config_cli_default.yml` | 142 | First candidate — the reference profile; `./scripts/start_cli.sh` uses it, so it is the documenting example |
| `config_web_frag.yml` | 184 | First candidate — the primary web/FRAG profile |
| `config_web_default_guardrails.yml` | 247 | First candidate — default web deployment |
| `config_web_frag_mcp_auth.yml` | 233 | Authed MCP sources loop most easily on inaccessible data |
| `config_web_opensearch.yml` | 191 | Only if it wants non-default limits |
| `config_web_azure_ai_search.yml` | 217 | Only if it wants non-default limits |
| `config_web_default_llamaindex.yml` | 216 | Only if it wants non-default limits |
| `config_mcp.yml` | 117 | Only if it wants non-default limits |
| `config_domain_routing_and_skills.yml` | 213 | Only if it wants non-default limits |
| `config_frontier_models.yml` | 167 | Plausible exception — a larger ceiling is defensible here |
| `config_openshell.yml` | 186 | Only if it wants non-default limits |

If, after the eval settles the number, you do want the knob visible in the shipped profiles, the
three "first candidate" rows are the tightest useful set: one CLI reference, one FRAG reference,
one default web deployment. The other eight inherit the code default.

### Also carrying a `deep_research_agent` block (outside `configs/`)

These are eval harnesses, not shipped profiles — leave them on defaults unless you are
specifically A/B-ing the guard:

- `frontends/benchmarks/freshqa/configs/config_full_workflow.yml`
- `frontends/benchmarks/deepsearch_qa/configs/config_deepsearch_qa.yml`
- `frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml`
- `frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench_profiling.yml`

Reminder from prior work: Harbor eval runs use a **pinned Docker image** for code and read
config from the host. A code change like this one needs an image rebuild + pin bump before the
eval picks it up; a YAML-only change does not.

### The YAML block

```yaml
    # Per-researcher circuit breaker. Bounds ONE run_research_batch worker: caps its
    # source-tool calls, blocks a repeated identical source request, and withdraws the
    # source tools and `think` once a limit is hit so the worker returns ResearchNotes with
    # explicit ResearchGaps instead of looping. Complements resource_limits, which bounds
    # the whole job rather than a single researcher.
    researcher_loop_guard:
      enabled: true
      # One flat limit, applied identically to every ResearchQuery.
      max_source_calls_per_query: 10
      max_identical_source_calls: 2
      max_consecutive_thinks: 3
```

Every value here equals the code default from §3.2, so this block is **documentation rather than
necessity** — shown so the shape is unambiguous when someone does need it. Not part of
iteration 1 (see the note above §4's table). Note that the adaptive branch's `1/3/6` budgets are
calibrated for a shallow single-shot lookup and would truncate real deep research; do not copy
them.

---

## 5. Tests

Add to `tests/aiq_agent/agents/deep_researcher/test_custom_middleware.py` — port
`TestResearcherLoopGuardMiddleware` from
`tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py` (lines 553–744 on the
source branch), adjusting imports. **Drop the `[("low", 1), ("medium", 3), ("high", 6)]`
parametrize** on `test_each_depth_enforces_its_configured_budget` — there is no depth in
iteration 1. Replace it with a single non-parametrized test that the one configured
`max_source_calls_per_query` is enforced, plus a second asserting that two `ResearchQuery`
objects with different content receive the *same* limit (the property the user asked for: one
limit, applied identically to every query).

**Do not port `TestConsecutiveThinkGuardMiddleware` (lines 749–841).** All ten of its tests
assert on `mw._consecutive_think_count` with no `ContextVar` set — the fallback path that no
longer exists after the merge. Replace it with think cases inside the loop-guard test class,
where a `ResearcherRunGuardState` is active (`setup_method` already installs one):

- below threshold, the think result is returned unmodified;
- at and beyond `max_consecutive_thinks`, the content is **overwritten** with the `WARNING:`
  text and `state.think_blocked` becomes `True`;
- `think` is then withdrawn by `_filter_tools` while the source tools stay visible — this is
  the assertion that used to require composing two middlewares by hand;
- the handler is always awaited for `think` (the guard warns, it never blocks thinking);
- an immutable / non-Pydantic result does not raise (the warning is best-effort);
- any non-think tool resets `state.consecutive_think_count` to 0;
- with `enabled=False`, a think streak past the threshold produces **no** warning and leaves
  `think_blocked` unset — the regression test for the bug the merge fixes.

**Termination tests** (§3.4, §3.5c, §3.6):

- `build_researcher_runnable` produces a runnable whose **effective** `recursion_limit` is the
  derived value and is **not** `2000`. Assert on the effective config after
  `researcher_invoke_config` has been applied — that is the composition the pop in §3.4 and the
  `with_config` in §3.5c exist to make deterministic, and asserting on either half alone would
  pass while the other regressed.
- `researcher_invoke_config` strips an inherited `recursion_limit` from a parent config carrying
  `2000`, alongside the existing `run_id` / `configurable` pops.
- The derived limit **scales with the budget**: raising `max_source_calls_per_query` raises the
  recursion limit.
- The recursion limit is applied **even when `enabled=False`** — the deliberate exception in
  §3.5c. Without a test this quietly reverts to inheriting 2000 the first time someone
  "consistently" gates it behind `enabled`.
- Prompt rendering with `researcher_loop_guard_enabled=True` contains the withdrawal-termination
  bullets; with `False` it contains none of them **and** still contains today's
  `## Handling Failures` line.

One characterization test, still worth keeping now that §9's turn budget remains deferred — it
documents current behaviour rather than asserting desired behaviour, and becomes the regression
anchor if §9 is taken up:

- a researcher whose source budget is exhausted but which keeps calling filesystem tools
  (`ls` / `read_file`) is **not** stopped by the guard. Assert that `_filter_tools` still returns
  those tools. This is the loop the prompt discourages and the `recursion_limit` bounds, but
  which nothing currently stops *gracefully*.

Cases to keep from the ported class, each pinning a distinct guarantee:

- budget exhaustion appends the nudge **and** withdraws `{source tools, think}` while leaving
  `get_verified_sources` visible;
- the configured `max_source_calls_per_query` is enforced, and the call past it is **not
  executed** (`handler.await_count` unchanged);
- an immutable/non-Pydantic tool result still exhausts the budget (the nudge is best-effort,
  the tool withdrawal is the hard guarantee);
- the third identical request is blocked with `exhaustion_reason == "repeated source-call
  signature"`;
- `think → same_search → think → same_search` terminates via the signature rule — now a single
  middleware, so the nested-handler scaffolding upstream needed is gone;
- signature is stable across mapping key order; distinct args do not collide;
- non-source tools are never counted;
- `asyncio.gather` of two source calls against a budget of 1 executes exactly one;
- two concurrent `ContextVar` invocations do not see each other's counts;
- `enabled=False` is a pure pass-through.

New tests for this branch specifically:

- `tests/.../test_factory.py`: `build_deep_research_middleware_set` places
  `ResearcherLoopGuardMiddleware` **before** `ToolRetryMiddleware` in `middleware_set.researcher`,
  and places it in **no** other stack (`planner` / `writer` / `orchestrator`).
- `tests/.../tools/` (or `test_agent.py`): `_run_research_query` sets a fresh
  `ResearcherRunGuardState` per invocation and resets the `ContextVar` on both the success and
  the exception path. Also assert the guard state is constructed **without reading any field off
  `ResearchQuery`** — the structural guarantee that iteration 1 leaves that schema alone.
- `tests/aiq_agent/agents/deep_researcher/test_agent.py` or a chat-researcher test: a
  `researcher_loop_guard` set on `DeepResearchAgentConfig` reaches the middleware built by the
  **per-request** `DeepResearcherAgent` rebuilt inside `register.py::_run` (§3.8) — this is the
  path chat researcher uses.
- Round-trip: `DeepResearchAgentConfig.model_validate({... "researcher_loop_guard": {...}})`
  accepts the YAML block above and rejects an unknown key (`extra="forbid"`).

**Disabled-path and prompt-rendering tests** — these cover the two failure modes that would
otherwise only show up at runtime (both raised in review; see §3.6):

- `enabled: false` disables **all three** enforcement paths in one go — source budget, repeated
  request, and consecutive think. Assert no counting, no blocking, no warning injection, and
  `_filter_tools` returning the tool list untouched. One test per path, so a partial regression
  is legible.
- **The researcher prompt renders when the new kwargs are absent entirely.** This is the
  `StrictUndefined` regression test: call `render_prompt_template` on `researcher.j2` passing
  none of `researcher_loop_guard_enabled` / `researcher_max_source_calls` /
  `researcher_max_identical_source_calls`, and assert it does not raise `PromptError`. Without
  `| default(false)` this fails.
- **Disabled mode retains today's guidance.** Render with `researcher_loop_guard_enabled=False`
  and assert the "Default source budget per ResearchQuery: one primary source-tool call..."
  line is still present and the "Hard limit (runtime-enforced)" line is not.
- **Enabled mode adds the ceiling without removing the guidance.** Both lines present.
- **Both agent construction paths receive the config** — the module-level `DeepResearcherAgent`
  built at `register.py:238` *and* the per-request rebuild inside `_run` at `register.py:275`.
  The second is the path the UI and chat researcher actually take (§3.8), so a test that only
  covers the first would pass while the feature is inert in production.

Validation commands (per `CLAUDE.md`):

```bash
uv run pytest tests/aiq_agent/agents/deep_researcher -q
uv run ruff check . && uv run ruff format --check .
uv run pytest                      # before opening the PR
```

Local caveat: this workstation has no compiler, so `uv run` / pytest fail while building
`annoy`. Verify structure with `python -m py_compile` + AST extraction locally and run the real
suite in CI or the container.

---

## 6. Recommended default changes

Upstream's budgets were calibrated for adaptive shallow tiers, where a `single_shot` lookup
should stop after one or three searches. The deep researcher is the opposite workload. Proposed
defaults in `models/loop_guard.py`:

| Field | Adaptive equivalent | Proposed deep default | Why |
| :-- | --: | --: | :-- |
| `max_source_calls_per_query` | `medium: 3` | **10** | a deep-research worker legitimately runs several corroborating and follow-up searches; 3 truncates it |
| `max_identical_source_calls` | 2 | 2 | unchanged — two attempts at the identical request is already generous |
| `max_consecutive_thinks` | 3 | 3 | unchanged |

### Resolving "3 or 10?" — the question dissolves once the prompt is decoupled

Raised in review: `3` preserves today's prompt behaviour, `10` allows more research but shifts
cost. To be precise about the current state of this document — it says `10` throughout; the `3`
above is upstream's value in a comparison column, not a competing recommendation. But the
underlying concern is real and worth answering properly.

The trap in the earlier draft was rendering the ceiling *as* the budget: a prompt saying *"you
may make at most 10 source-tool calls"* tells the researcher it may do roughly **five times**
what today's prompt allows ("one primary call, plus at most one fallback"). That is a behaviour
change even on runs where the guard never fires — and it would have shipped silently.

§3.6 now separates the two:

- **Behavioural budget — unchanged from today.** "One primary call, plus at most one fallback or
  corroboration call," rendered unconditionally. This is what the model actually steers by.
- **Hard ceiling — the circuit breaker.** `max_source_calls_per_query`, rendered only when the
  guard is enabled, and explicitly labelled *"a backstop, not a target."*

With that split the ceiling should sit **well above** normal behaviour, because a circuit breaker
that fires on healthy runs is miscalibrated. `10` is right for that role and `3` is not — at `3`
the breaker would trip on legitimate multi-source research, converting normal runs into
gap-filled partial notes. Conversely, `10` no longer carries the cost risk that motivated
preferring `3`, because it never appears as an instruction to search ten times.

**Decided: ship `10`, tune from the first eval run.** It is reasoned rather than measured — the
value the adaptive branch shipped for its `medium` tier, the closest analogue to a deep-research
worker — and it has not been validated against this workload. Treat the first eval as the
calibration:

- **Breaker never fires** → it is doing its job as a backstop. Leave it, or lower it only if
  traces show researchers routinely running far below the ceiling and you want a tighter net.
- **Breaker fires on healthy runs** → too low. Raise it; a circuit breaker that trips during
  normal research is miscalibrated by definition, and the symptom will be `ResearchNotes` with
  spurious `ResearchGap` entries rather than an obvious failure.
- **Breaker fires only on the pathological runs it was built for** → correct, keep it.

Log lines to read for this: the guard emits `Researcher source-call budget reached | invocation=
… calls=%d/%d` at INFO on exhaustion and a WARNING on each blocked call, so the fire rate is
greppable from an eval run without extra instrumentation.

Because `10` is expected to change, do **not** hard-code it into the §4 config files — leave
those profiles on the code default so a retune is a one-line change in `models/loop_guard.py`
rather than an edit across eleven YAML files.

**Second-order effect to remember when retuning:** `max_source_calls_per_query` also derives the
researcher's `recursion_limit` (§3.5c). Raising the budget raises the turn ceiling with it, which
is intended — but it means a retune changes two things, so re-read both the breaker fire rate and
any `GraphRecursionError` count after adjusting.

See "Source-call accounting semantics" in §1 for what this number actually bounds — model search
*iterations*, not retrieval volume. That distinction belongs in the config field description.

---

## 7. Suggested commit sequence

1. `feat(deep-research): add per-researcher loop guard state and config models`
   — `researcher_context.py`, `models/loop_guard.py`, `models/__init__.py` exports, unit tests
   for the config field constraints (`ge=1`, `extra="forbid"`).
2. `feat(deep-research): add the researcher loop guard middleware`
   — `custom_middleware.py` (one class covering the source budget, repeated requests, and
   think loops) + ported middleware tests. No wiring yet, so this commit is inert.
3. `feat(deep-research): scope loop guard state to one researcher invocation`
   — `tools/research.py` `ContextVar` set/reset + tests.
3b. `fix(deep-research): bound the researcher subgraph instead of inheriting recursion_limit 2000`
   — the `recursion_limit` pop in `researcher_invoke_config` (§3.4) plus the derived limit on
   `build_researcher_runnable` (§3.5c), with the effective-limit tests. **Its own commit, and
   reviewable on its own:** the inherited 2000 is a pre-existing defect on `develop` independent
   of the loop guard, and this is the change most likely to need eval evidence, since it can cut
   off long-running researchers that complete today. If an eval regresses, this is the commit to
   revert without losing the guard.
4. `feat(deep-research): wire the researcher loop guard into the graph and config`
   — `factory.py`, `agent.py`, `register.py`, `prompts/researcher.j2`, factory + config tests.
   This is the commit that turns the feature on.
5. `docs: document the deep-researcher loop guard`
   — `docs/source/` page covering the knobs, the accounting semantics from §1 (logical
   invocations vs concrete calls — operators will otherwise misread the limit), and the
   chat-researcher inheritance (required by `CLAUDE.md`: docs stay canonical when configuration
   changes).

No config commit in iteration 1 — the shipped profiles stay on the code default until the eval
settles `max_source_calls_per_query` (§4, §6). The follow-up, once tuned, is a one-line change
in `models/loop_guard.py`, optionally plus the §4 YAML blocks.

Every commit needs `git commit -s` (DCO). Branch off `develop`; PR into `develop`.

---

## 8. Explicitly out of scope

- `OrchestratorLoopGuardMiddleware` — bounds the *whole request* (batch count, total delegated
  queries, duplicate query signatures, orchestrator turns). It is the real answer to "the
  orchestrator keeps spawning fresh researchers, each of which passes its own guard," and it is
  a genuine gap that this port does **not** close. It depends on `AdaptiveRequestTerminationConfig`
  and on `declare_effort_tier` for its per-tier budgets, so porting it needs a tier-free redesign.
  Worth a follow-up plan; the source material is
  `misc/adaptive-researcher-request-termination-plan.md`.
- `ComplexityRouterMiddleware`, `SingleShotShallowDelegationMiddleware`, `declare_effort_tier`,
  `tiers.py`, the shallow sub-agent, and the whole `adaptive_researcher` package.
- **Per-query `depth` in every form** — the field on `ResearchQuery`, the `low`/`medium`/`high`
  budget model, `normalize_research_depth`, the depth sections in `orchestrator.j2` /
  `planner.j2` / `researcher.j2`, and the `depth: low` plan-read and skill-check skips.
  Iteration 1 applies **one flat limit to every research query**. Depth is implemented only
  after `ResearcherLoopGuardMiddleware` has landed on this branch — see §2 for the full
  not-ported table and the iteration-2 approach.
- A think guard for the orchestrator / planner / writer. Real and currently unaddressed —
  `think` is bound to every stack via `helper_tools` and `develop` has no think protection
  anywhere — but it needs request-scoped state that `ResearcherRunGuardState` does not provide,
  and a per-request rather than per-invocation bound. Do **not** implement it by copying
  upstream's standalone `ConsecutiveThinkGuardMiddleware`: its instance counter is shared
  across runs in both repos, and across *concurrent* requests on this repo's CLI/eval path
  (§3.3a). It naturally belongs with the `OrchestratorLoopGuardMiddleware` work above, which
  already carries a correct per-request lifetime.

---

## 9. The exceeded case — two fixes promoted, one still deferred

**Status: mostly addressed in iteration 1.**

| Fix | Status |
| :-- | :-- |
| Prompt: terminate gracefully when tools are withdrawn | **In iteration 1** — §3.6 |
| A. Explicit researcher `recursion_limit` | **In iteration 1** — §3.4 + §3.5c, commit 3b |
| B. Turn budget with a graceful exit | **Still deferred** — below |

Rationale for the split: the prompt handles the cooperative case, `recursion_limit` bounds the
non-cooperative case cheaply and with no design surface, and B is the only one left that needs a
new config field, a new state field, a second exhaustion path, and a number nobody has evidence
for. With the other two in, B should be unnecessary — take it only if evals show researchers
actually grinding past exhaustion.

### The caveat, restated (what B would still fix)

The guard bounds **tool calls, not turns**, and it withdraws only the source tools and `think`.
A researcher that has exhausted its budget still holds `get_verified_sources`, every tool in
`FILESYSTEM_TOOL_NAMES`, and any skills tools, and can loop on them indefinitely. Because each
such call resets `consecutive_think_count`, the think branch never fires either.

The eventual stop is `GraphRecursionError` — but at **~1000 model turns**, because the researcher
subgraph inherits the orchestrator's `recursion_limit: 2000` rather than LangGraph's default of
25 (full derivation in §1, "Termination semantics"). The error then destroys that worker's
gathered evidence entirely: the exception path produces no `ResearchNotes` at all, so a run that
should have degraded to "partial notes plus explicit gaps" degrades to "failed query" instead.

**Net effect on the feature as advertised:** withdrawing tools is a *nudge that usually works*,
not a guarantee. The guarantee holds only for a cooperative model.

### Fix A — explicit researcher recursion limit ✅ PROMOTED to iteration 1

Specified in §3.4 (pop the inherited value in `researcher_invoke_config`) and §3.5c (derive and
bind the limit in `build_researcher_runnable`), shipped as commit 3b.

The open question recorded here earlier — whether the invoke-time `config=` beats
`.with_config(...)` — was **sidestepped rather than answered**: popping `recursion_limit` from
the inherited parent config removes the contest entirely, so the outcome no longer depends on
`merge_configs` precedence that could shift on a dependency bump.

Residual risk, unchanged: this is a **behaviour change independent of the loop guard**. The
inherited 2000 is a pre-existing defect on `develop`, and any explicit limit can truncate a
long-running researcher that completes today. Hence its own commit and its own eval read.

### Fix B — turn budget with a graceful exit (still deferred)

Add `max_researcher_turns` to `ResearcherLoopGuardConfig` and `turn_count` to
`ResearcherRunGuardState`; count in `wrap_model_call` / `awrap_model_call`; at the limit, mark
exhausted and have `_filter_tools` withdraw **every** tool. With no tools bound the model must
emit content, and `StructuredResponseTextFallbackMiddleware._promote` converts schema-valid JSON
into `ResearchNotes` — yielding partial notes with gaps instead of a crash.

Supporting evidence that a tools-empty call is a proven shape here:
`StructuredResponseTextFallbackMiddleware._correction_request` already invokes the model with
`tools=[]`, `tool_choice=None`, `response_format=None` (`custom_middleware.py:137–143`).

**Open question now RESOLVED — B is safer than when it was deferred.** The concern was that
`create_agent(response_format=ResearchNotes)` might bind a structured-output *tool* that
"withdraw everything" would strip. Verified against the installed `langchain` 1.3.4: the
structured-output tools are appended to `final_tools` **after** every `wrap_model_call`
middleware has run (`langchain/agents/factory.py:1237-1242` — `final_tools = list(request.tools)`
then `.extend(structured_tools)`). They are **never present in `request.tools`**, so middleware
cannot see or remove them. Consequences:

- `_filter_tools` returning `[]` still leaves the schema tool bound — the model can always
  produce its `ResearchNotes`. B's core mechanism is sound as written.
- The same fact guarantees iteration 1's narrower `_filter_tools` can never break structured
  output either.
- Strategy is auto-detected per model — `ProviderStrategy` when the model supports native
  structured output, else `ToolStrategy` (`factory.py:1219-1231`) — and B works under both.

What remains open for B is only sizing: the value of `max_researcher_turns`, for which there is
no evidence yet. That is the reason it stays deferred, not any technical doubt.

### Related, and probably the better frame

Fix B is a per-*invocation* turn bound. The orchestrator-level version of the same problem —
an orchestrator that keeps authoring fresh `run_research_batch` calls, each spawning researchers
that individually behave — is `OrchestratorLoopGuardMiddleware` (§8), which already tracks
`max_orchestrator_turns` upstream. If that work is taken up, revisit A and B together with it
rather than in isolation; the three share one question (what bounds a request in turns, not just
in tool calls) and answering it once is likely cleaner than three separate budgets.

### Also worth noting while here

`run_research_batch` raises when *any* worker fails (`tools/research.py:293`), even though
successful workers' notes were already persisted and registered. That error is retryable by
`ToolRetryMiddleware(max_retries=3)` in the orchestrator stack (`factory.py:274`), so one stuck
researcher can trigger up to three full re-runs of its entire batch, successful siblings
included. The retries are ultimately bounded by the `consumed_queries` ledger
(`research.py:236`) hitting the 20-query per-job cap. Independent of the loop guard; noted so it
is not rediscovered as a guard regression.
