# Adaptive Researcher — Research Sub-Agent Loop Guard Issue and Implementation Plan

## Status

Implemented on 2026-07-23. Phases 1 and 2 are complete: adaptive researcher
workers now have context-isolated source-call and repeated-signature limits,
and the consecutive-think guard withdraws `think` after its threshold. Phase 3
remains an optional evaluation item and is not part of this change.

## Problem statement

An `adaptive_researcher` job selected the `standard` effort tier for:

> How has Apple's total net sales changed over time?

The orchestrator delegated a `depth: high` query through `run_research_batch`. The
researcher sub-agent was asked to find annual 10-K net-sales history, but its preferred
tool set contained only `knowledge_search`. The configured collection returned quarterly
10-Q documents rather than the requested annual 10-K data.

The researcher then entered this loop:

```text
think
knowledge_search("Apple 10-K annual report net sales fiscal year 2024 ...")
think
knowledge_search("Apple 10-K annual report net sales fiscal year 2024 ...")
...
```

Each retrieval returned substantially the same documents. The researcher recognized the
coverage gap but issued the same query again instead of returning `ResearchNotes` with a
`ResearchGap`. The orchestrator remained blocked awaiting the researcher result, making
the whole job appear stuck even though FastAPI, Dask, PostgreSQL, and the container were
healthy.

## Investigation result

### Is `ConsecutiveThinkGuardMiddleware` attached to this researcher?

Yes.

`build_adaptive_research_middleware_set()` appends a
`ConsecutiveThinkGuardMiddleware()` instance to the researcher middleware list:

```python
researcher=[*common(), ConsecutiveThinkGuardMiddleware()]
```

`build_adaptive_research_graph()` passes that list to
`build_researcher_runnable()`, which passes it to LangChain's `create_agent()`.
The `standard` path calls that runnable through `run_research_batch`.

The tier does not conditionally enable the guard. It is attached to adaptive researcher,
planner, writer, and orchestrator loops regardless of whether the selected tier is
`standard` or `deep`.

Relevant code:

- `src/aiq_agent/agents/adaptive_researcher/factory.py`
  - `build_adaptive_research_middleware_set`
  - `build_adaptive_research_graph`
- `src/aiq_agent/agents/deep_researcher/factory.py`
  - `build_researcher_runnable`
- `src/aiq_agent/agents/adaptive_researcher/tools/research.py`
  - `build_adaptive_research_batch_tool`

### Why did the guard not stop this loop?

The guard detects only uninterrupted consecutive calls to the `think` tool.

Its state transition is:

```python
if name == "think":
    count += 1
else:
    count = 0
```

Therefore the observed sequence behaves as follows:

| Tool call | Guard count afterward |
| :-- | --: |
| `think` | 1 |
| `knowledge_search` | 0 |
| `think` | 1 |
| `knowledge_search` | 0 |
| `think` | 1 |

The default threshold of three is never reached. This is expected from the current
implementation and is explicitly codified by
`test_counter_resets_on_non_think`.

`ConsecutiveThinkGuardMiddleware` is consequently not a general ReAct-loop guard. It
addresses only:

```text
think → think → think
```

It does not address:

```text
think → search → think → search
search → search → search
same search signature repeated after equivalent results
too many distinct source-tool calls
```

### Additional limitations found

1. **The guard is advisory, not a hard restriction.**

   At the threshold it overwrites the `think` result with a warning. It does not withdraw
   the `think` tool, force structured output, or terminate the worker. A model can ignore
   the warning and continue calling `think`.

2. **Research-depth budgets are prompt-only.**

   `researcher.j2` currently describes:

   - `low`: one source-tool call;
   - `medium`: up to roughly three calls;
   - `high`: roughly four to six calls.

   Nothing deterministically enforces those limits for researcher sub-agents. The
   `single_shot_search_budget` hard cap belongs to the orchestrator's optional direct-tool
   `single_shot` path and explicitly does not apply to `standard` or `deep`
   `run_research_batch` workers.

3. **The graph recursion limit is not an appropriate research budget.**

   The adaptive graph uses `recursion_limit=2000`. That protects against an actually
   unbounded graph only at a very high ceiling; it does not provide useful source-call or
   cost control for one researcher.

4. **The guard instance is scoped to the parent agent run, not necessarily one researcher
   invocation.**

   One reusable researcher runnable is invoked concurrently for every query in a batch.
   Its middleware instances are reused by those invocations. Mutable counters stored only
   on the middleware instance can therefore mix activity from concurrent researcher
   workers. Even if individual counter mutations are safe on the event loop, the state is
   not semantically isolated per query.

5. **Existing tests cover only the narrow consecutive-think contract.**

   There is no regression test for alternating `think` and source-tool calls, repeated
   identical retrievals, depth-budget enforcement, or isolation between concurrent
   researcher workers.

## Root cause

The immediate root cause is a mismatch between the guard's narrow trigger and the actual
loop shape:

```text
ConsecutiveThinkGuardMiddleware watches consecutive `think` calls.
The incident alternated `think` with a successful non-`think` source call.
Every source call reset the guard.
```

The enabling design gap is that researcher search budgets are expressed only as prompt
guidance. There is no deterministic per-researcher source-call ceiling or repeated-call
detector.

The source-routing decision amplified the problem: the query needed annual external data,
but the researcher was constrained to an internal knowledge source containing only
quarterly reports. A robust researcher must terminate with an explicit evidence gap when
its allowed sources cannot satisfy the query.

## Recommended solution

Implement a hard, per-researcher `ResearcherLoopGuardMiddleware`. Keep
`ConsecutiveThinkGuardMiddleware` focused on pure think loops, but do not rely on it for
source-tool loop control.

The new guard should combine two deterministic protections:

1. A total source-tool-call budget derived from the query's `depth`.
2. A repeated source-call signature limit that detects no-progress retries before the
   total budget is exhausted.

### Default budgets

Match the existing researcher prompt so enforcement does not silently redefine expected
quality:

| Query depth | Maximum source-tool calls |
| :-- | --: |
| `low` | 1 |
| `medium` | 3 |
| `high` | 6 |

Use a default maximum of two executions for an identical normalized source-tool
signature. The third identical attempt should not execute. This allows one deliberate
retry while bounding the incident pattern early.

These values should be configurable and validated, but the initial defaults should remain
aligned with `researcher.j2`.

### What counts as a source-tool call?

Use the source-tool names already identified by the built `DeepResearchToolSet`, rather
than hard-coding names such as `knowledge_search` or `web_search_tool`.

Do not charge helper, filesystem, skill, structured-output, or finalization tools against
the source budget.

For batch-capable source tools, count one tool invocation as one call. The existing
`max_source_tool_batch_size` remains responsible for bounding the number of concrete
queries inside one batch call.

### Normalize repeated-call signatures

Build a stable signature from:

```text
tool name + canonical JSON tool arguments
```

Canonicalization should sort mapping keys and preserve ordered list inputs. Do not include
the returned content in the primary signature; the objective is to detect the model
issuing the same request repeatedly.

Framework-level retries inside `ToolRetryMiddleware` should remain one logical agent tool
call and should not be mistaken for a new model-directed search attempt.

### Behavior when a limit is reached

Do not raise immediately from the allowed tool call that reaches the budget. Preserve its
evidence and append a system-authored instruction similar to:

```text
[SYSTEM — researcher source budget exhausted]
Stop searching. Return ResearchNotes now using the evidence already gathered.
Represent unsupported target components as ResearchGap entries; do not guess.
```

On subsequent model calls:

- withdraw all source tools;
- withdraw `think`, because more private reasoning cannot acquire new evidence and can
  itself become another loop;
- preserve the structured `ResearchNotes` response mechanism and any tools required to
  read already-created context;
- never hide unknown framework-generated structured-output tools.

If the model somehow attempts an already-disallowed source call, return a deterministic
tool result instructing it to produce `ResearchNotes`; do not execute the source tool.

This should be a graceful completion path, not an exception. Raising would cause
`_run_research_queries()` to mark the worker failed and can fail the entire
`run_research_batch`, discarding useful partial evidence.

### Per-researcher state isolation

Do not store the new counters as unscoped mutable fields on the reusable middleware
instance.

Recommended design:

1. Add a small `ResearcherRunGuardState` structure containing:

   - invocation identifier;
   - query depth;
   - total source-call count;
   - normalized-signature counts;
   - exhausted flag and exhaustion reason.

2. Hold the current state in a `ContextVar`.

3. In `_run_research_query()`, set a fresh state immediately before
   `researcher_runnable.ainvoke()` and reset the context variable in `finally`.

4. Let `ResearcherLoopGuardMiddleware` read and mutate the state belonging to the current
   async researcher task.

This isolates concurrent queries while allowing parallel tool calls inside one researcher
invocation to share the same budget. The implementation must count before awaiting the
tool handler so simultaneous source calls cannot all pass the final remaining slot.

Place the context definition in a dependency-neutral shared module under
`deep_researcher` so the shared `_run_research_query()` can set it without importing the
adaptive package. Attaching the enforcing middleware only to adaptive researchers keeps
the initial behavior change scoped.

### Relationship to `ConsecutiveThinkGuardMiddleware`

Keep the existing middleware, but make its scope and behavior explicit:

- rename it only if API compatibility allows, or update its docstring to say it handles
  pure consecutive-think loops only;
- move its count into the same per-researcher context if it remains attached to the
  reusable researcher runnable;
- after its threshold, hide `think` on the next model call rather than relying solely on a
  warning;
- retain its existing warning so the model understands why the tool disappeared.

This strengthening is useful defense in depth, but it is not the primary fix for the
observed incident.

## Configuration plan

Add a validated nested config for the adaptive researcher, for example:

```yaml
functions:
  adaptive_research_agent:
    _type: adaptive_research_agent
    researcher_loop_guard:
      enabled: true
      source_call_budgets:
        low: 1
        medium: 3
        high: 6
      max_identical_source_calls: 2
      max_consecutive_thinks: 3
```

Suggested schema properties:

- all limits must be positive integers;
- `low <= medium <= high`;
- defaults preserve the current prompt's intended budgets;
- disabling the guard should be explicit and intended only for controlled evaluation;
- do not key the sub-agent budget directly from the parent effort tier. The parent tier
  controls orchestration width, while each `ResearchQuery.depth` controls one worker's
  sequential depth.

Update `researcher.j2` to render the configured values rather than retaining duplicated
literal numbers that can drift from enforcement.

## Implementation steps

### Phase 1 — Deterministic researcher budget

1. Add the per-invocation guard-state/context module.
2. Set and reset the context in `_run_research_query()`.
3. Implement `ResearcherLoopGuardMiddleware` with:

   - source-name-based counting;
   - per-depth total budgets;
   - canonical signature tracking;
   - last-result nudge;
   - post-budget source/`think` tool withdrawal.

4. Add the middleware to the adaptive researcher stack after tool sanitization/retry
   handling and before observability middleware.
5. Thread validated configuration from `register.py` through `agent.py` and `factory.py`.
6. Render the effective limits into `researcher.j2`.
7. Add structured logs for:

   - invocation ID;
   - depth;
   - source call count and maximum;
   - normalized tool name;
   - exhaustion reason (`total_budget` or `repeated_signature`).

   Do not log tool arguments or retrieved content at warning level.

### Phase 2 — Harden pure-think handling

1. Make `ConsecutiveThinkGuardMiddleware` state per researcher invocation.
2. Add model-call tool filtering that removes `think` after the threshold.
3. Preserve the current corrective warning.
4. Verify the behavior separately on orchestrator/planner/writer instances, where
   per-agent-run middleware instances may already be sufficiently isolated.

### Phase 3 — Optional whole-run guard

Evaluate whether the `standard` orchestrator also needs a hard maximum number of
`run_research_batch` calls. This is separate from the current incident:

- the observed unbounded loop was inside one researcher;
- a researcher budget fixes that loop;
- the orchestrator can still launch another batch after receiving valid notes.

Add a standard-tier batch-call ceiling only if evaluation shows repeated batch delegation.
Do not conflate orchestration width with per-query research depth in Phase 1.

## Likely files to change

| File | Planned change |
| :-- | :-- |
| `src/aiq_agent/agents/deep_researcher/tools/research.py` | Establish/reset per-researcher invocation guard context |
| `src/aiq_agent/agents/deep_researcher/researcher_context.py` | New dependency-neutral context and state model |
| `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py` | Add hard researcher loop guard; strengthen/document think guard |
| `src/aiq_agent/agents/adaptive_researcher/factory.py` | Wire guard and source-tool names into researcher middleware |
| `src/aiq_agent/agents/adaptive_researcher/register.py` | Add validated loop-guard configuration |
| `src/aiq_agent/agents/adaptive_researcher/agent.py` | Thread configuration into middleware construction |
| `src/aiq_agent/agents/adaptive_researcher/prompts/researcher.j2` | Render configured budgets and finalization behavior |
| `configs/config_web_default_llamaindex.yml` | Document or explicitly set production defaults |
| `tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py` | Unit and concurrency tests |
| `tests/aiq_agent/agents/adaptive_researcher/test_agent.py` | Researcher/batch integration regression coverage |
| `docs/source/architecture/agents/deep-researcher.md` | Document deterministic worker budgets if shared architecture behavior changes |

The exact test file for graph integration should follow the current adaptive-researcher
test layout; do not create a new file if the scenario fits an existing focused module.

## Test plan

### Existing guard characterization

- Confirm three uninterrupted `think` calls trigger the existing warning.
- Confirm `think → search → think → search` never reaches the consecutive-think threshold.
- Confirm the test documents this as the old guard's scope, not as acceptable total-loop
  behavior.

### Source-budget unit tests

- `low` permits exactly one source call.
- `medium` permits exactly three source calls.
- `high` permits exactly six source calls.
- Helper and filesystem calls do not consume source budget.
- The final allowed result preserves original evidence and appends the stop instruction.
- Source and `think` tools are absent from the next model request after exhaustion.
- Structured-output tools remain available.
- An immutable tool result degrades safely while hard tool withdrawal still applies.
- Parallel tool calls cannot exceed the configured total.

### Repeated-signature tests

- Equivalent dict arguments with different key order normalize to the same signature.
- Different tool names or materially different arguments do not collide.
- The third identical call is blocked with the default repeat limit of two.
- A framework retry inside one logical tool call does not consume another signature slot.

### Concurrency tests

- Two researcher queries invoked concurrently receive independent counts.
- One `low` worker exhausting its budget does not hide tools from a concurrent `high`
  worker.
- Parallel calls within one worker share and respect one budget.
- Context state is reset after success, exception, and cancellation.

### Integration regression

Use a deterministic fake model that emits:

```text
think → knowledge_search(same args) → think → knowledge_search(same args) → ...
```

Assert that:

- source executions are bounded;
- the researcher is prompted to return structured `ResearchNotes`;
- unsupported evidence becomes a `ResearchGap`;
- `run_research_batch` returns rather than hanging;
- the parent `standard` orchestrator can continue to finalization.

Also cover a legitimate `depth: high` chain with distinct searches to ensure the guard
allows useful multi-hop work up to the configured maximum.

### Validation commands

Run the narrow checks first:

```bash
uv run pytest tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py
uv run pytest tests/aiq_agent/agents/adaptive_researcher/test_agent.py
uv run ruff check \
  src/aiq_agent/agents/adaptive_researcher \
  src/aiq_agent/agents/deep_researcher/tools/research.py \
  tests/aiq_agent/agents/adaptive_researcher
uv run ruff format --check \
  src/aiq_agent/agents/adaptive_researcher \
  src/aiq_agent/agents/deep_researcher/tools/research.py \
  tests/aiq_agent/agents/adaptive_researcher
```

Then run the relevant adaptive/deep researcher suite and an end-to-end standard-tier job
against both:

- a source-complete query;
- a source-incomplete query matching the Apple annual-report incident.

Record source-call counts, completion status, latency, token usage, and citation quality
before and after the change.

## Acceptance criteria

- The observed alternating `think/search` pattern cannot execute indefinitely.
- Every adaptive researcher invocation has a deterministic source-call ceiling.
- Repeated identical source requests terminate earlier than the total depth budget.
- Budget exhaustion returns partial structured notes with explicit gaps instead of failing
  the entire batch.
- Concurrent researcher workers have isolated guard state.
- Legitimate high-depth, multi-hop research remains possible up to its configured budget.
- Standard- and deep-tier orchestration behavior is unchanged except that their adaptive
  researcher workers are bounded.
- Logs make budget exhaustion diagnosable without exposing query arguments, retrieved
  content, credentials, or other sensitive data.

## Non-goals

- Changing effort-tier selection for the Apple query.
- Removing researcher sub-agents from the `standard` tier.
- Solving incorrect source routing solely through the loop guard.
- Treating the graph's `recursion_limit` as the normal source-call budget.
- Restarting or scaling Docker/Dask as a fix for an agent-level loop.
