# Adaptive Researcher — Request-Wide Termination and Orchestrator Loop Guard Plan

## Status

Proposed.

This plan addresses a failure mode discovered after the per-researcher loop guard
was implemented in commit `6d2c09e`. That commit is present in the tested
container and is working as designed. The remaining problem is at the parent
orchestrator level: a request can repeatedly create new, individually bounded
researcher invocations and therefore run for hours.

This document is an implementation plan, not a record of completed changes.

## Executive summary

The adaptive researcher currently has several local safety limits:

- `single_shot` direct searches have a hard search-call budget;
- each `standard` or `deep` researcher invocation has a depth-based source-call
  budget;
- identical source calls are bounded within one researcher invocation;
- uninterrupted `think` loops are bounded;
- each `run_research_batch` call accepts only a configured number of queries.

These controls limit one component at a time, but they do not limit the complete
top-level request.

The observed request selected the `standard` tier and repeatedly searched for
Apple FY2024/FY2025 filings that were not present in the configured knowledge
base. Individual researchers correctly stopped at their `3/3` or `6/6` limits.
The parent orchestrator then started another `run_research_batch`, which created
new researcher invocation IDs with fresh budgets. The request continued for
hours even though every individual guard fired correctly.

The required fix is a request-wide termination envelope:

1. Add an orchestrator guard that limits research batches, total delegated
   queries, repeated delegated queries, and orchestrator turns for the entire
   request.
2. Add a hard wall-clock deadline around the complete workflow.
3. Give the model one bounded opportunity to finalize from evidence already
   collected.
4. If model-driven finalization does not complete, return a deterministic,
   citation-safe partial result instead of leaving the job running.
5. Reduce the graph recursion limit to a realistic last-resort ceiling.
6. Add per-researcher deadlines so one model or source call cannot hold the
   parent request indefinitely.

The core invariant should be:

> Every adaptive research request reaches a terminal state within a configured
> maximum duration. It returns either a complete report, a clearly marked
> partial report, or a bounded failure response. It never remains active
> indefinitely.

## Incident summary

### User query

The affected top-level query was:

> How have Apple's inventory levels changed across these quarters and what
> might this indicate about their supply chain management?

The adaptive agent selected:

```text
standard
```

The relevant tier selection was logged at `2026-07-23 10:11:49 UTC`.

### What the request did

The orchestrator delegated queries for Apple inventory data across fiscal 2024
and 2025. The available `knowledge_search` collection primarily contained Apple
quarterly documents from 2022 and 2023. Some searches also returned irrelevant
NVIDIA filings.

Researchers repeatedly recognized that the requested periods were absent, but
continued trying query variants such as:

```text
Apple Q3 FY2024 10-Q inventory June 2024
Apple FY2024 10-K inventory September 2024
Apple Q1 FY2025 10-Q inventory December 2024
Apple Q2 FY2025 10-Q inventory March 2025
```

### What the existing guard did

The per-researcher guard worked:

```text
Researcher source-call budget reached |
invocation=<id> depth=high tool=knowledge_search calls=6/6
```

It also blocked repeated source calls:

```text
Researcher loop guard blocked repeated source call |
invocation=<id> depth=high tool=knowledge_search
repeats=2/2 reason=repeated_signature
```

The researcher was instructed to stop searching and return `ResearchNotes`
containing explicit gaps.

### Why the request still continued

The guard state is deliberately scoped to one call to
`_run_research_query()`. When that invocation finishes, its context is reset.
The next `run_research_batch` creates a new invocation:

```text
parent orchestrator
    └── run_research_batch
          └── researcher invocation A: reaches 6/6 and stops

parent orchestrator
    └── run_research_batch again
          └── researcher invocation B: fresh 0/6 budget

parent orchestrator
    └── run_research_batch again
          └── researcher invocation C: fresh 0/6 budget
```

The local guard therefore prevented an infinite loop inside invocation A, but
did not prevent an unbounded sequence of A, B, C, and later invocations.

### Container verification

This was not caused by an old Docker image:

- the image was built shortly before the fix was committed;
- it was built from the working tree that already contained the fix;
- the new `loop_guard.py` was present in `/app/src`;
- all inspected runtime files had byte-identical SHA-256 hashes between the
  running container and commit `6d2c09e`;
- guard activation messages were present throughout the logs.

The container was running the fix. The fix bounded the scope it was designed to
bound.

### Separate log duplication issue

Many callback records appeared repeatedly with the same invocation ID and
timestamp, sometimes roughly eleven times. This is likely duplicate callback or
logging registration.

That duplication makes the incident appear even larger and may add overhead,
but it is not the primary termination failure. The presence of many distinct
researcher invocation IDs shows that the parent workflow was also creating new
research work.

Callback deduplication should be fixed, but it must not be treated as a
substitute for request-wide termination.

## Current protection boundaries

| Existing control | Scope | What it prevents | What it does not prevent |
| :-- | :-- | :-- | :-- |
| `single_shot_search_budget` | One `single_shot` request | Too many direct source calls | `standard`/`deep` batch loops |
| `ResearcherLoopGuardMiddleware` | One researcher invocation | Too many source calls or identical searches within that invocation | New researcher invocations |
| `ConsecutiveThinkGuardMiddleware` | One active agent loop | Uninterrupted `think` calls | Alternating tools or new sub-agents |
| `max_research_concurrency` | One `run_research_batch` call | Too many queries in one batch | Repeated batches |
| Source-tool timeout | One retrieval operation | One source call waiting forever | Whole workflow duration |
| `recursion_limit=2000` | One LangGraph execution | Eventually stops an extreme graph loop | Timely, useful termination |

The missing boundary is the complete top-level request.

## Root cause

### Immediate root cause

`standard` and `deep` preserve `run_research_batch` after tier selection, and
there is no runtime counter for how often the orchestrator may call it.

The prompt tells `standard` to author a small number of queries and use the
fewest batches, but that is advisory. The runtime currently enforces only:

```python
len(queries) <= max_research_concurrency
```

for each individual call.

It does not enforce:

```text
number of run_research_batch calls in this request
total ResearchQuery items submitted in this request
whether a query was already researched in an earlier batch
total orchestrator turns
total request duration
```

### Enabling conditions

Several conditions amplified the failure:

1. The requested FY2024/FY2025 evidence was not available from the enabled
   source.
2. The model treated missing evidence as a reason to try more query variants
   instead of accepting a `ResearchGap`.
3. Every new researcher invocation received a fresh source-call budget.
4. `recursion_limit=2000` allowed a very large number of graph transitions.
5. The `timeout: 300` value in `config_adaptive_frag.yml` belongs to one
   knowledge retrieval call. It is not a five-minute workflow deadline.
6. There is no deterministic fallback report when the orchestrator does not
   finalize voluntarily.

## Goals

### Primary goals

1. Guarantee bounded termination for every adaptive research request.
2. Preserve useful evidence already collected when a limit is reached.
3. Return explicit evidence gaps instead of retrying unavailable information.
4. Keep request state isolated under concurrency.
5. Preserve the intended quality difference between `single_shot`, `standard`,
   and `deep`.
6. Make the termination reason visible in logs and job metadata.
7. Allow operators to tune limits without editing prompt text.

### Non-goals

This work should not:

- redesign effort-tier classification;
- solve source-routing quality in the same change;
- guarantee that every report is complete;
- silently fabricate missing evidence;
- remove the existing per-researcher guard;
- collapse `standard` or `deep` into the `single_shot` execution model;
- rely only on prompts for termination;
- treat the graph recursion limit as the main budgeting mechanism.

## Required termination semantics

Every request must finish through one of these paths:

| Terminal path | Meaning |
| :-- | :-- |
| Complete | Normal `submit_final_report` or writer output completed within budget. |
| Partial | The research or time budget was exhausted, but collected evidence was synthesized safely. |
| Bounded failure | No safe report could be synthesized; return a clear error/gap response within the deadline. |
| Cancelled | The caller cancelled the job; outstanding work and sandbox resources are cleaned up. |

“Still running” after the configured request deadline is not a valid state.

## Proposed solution

Use defense in depth. No single model-facing nudge can guarantee termination.

### Layer 1: request-wide orchestrator loop guard

Add an `OrchestratorLoopGuardMiddleware` to the adaptive orchestrator.

The middleware instance is created per top-level adaptive request, so its state
can safely live on the middleware instance. If graph reuse later changes that
assumption, move the state to a request-scoped `ContextVar` as was done for
researcher workers.

The guard should track:

```text
declared tier
orchestrator model-turn count
run_research_batch call count
total ResearchQuery count across all batches
normalized ResearchQuery signature counts across all batches
research phase: active | finalizing | terminal
exhaustion reason
request start time / deadline
```

#### Suggested default budgets

These are initial values for evaluation, not immutable product policy:

| Limit | `standard` | `deep` / delta |
| :-- | --: | --: |
| Maximum `run_research_batch` calls | 2 | 4 |
| Maximum total delegated `ResearchQuery` items | 6 | 16 |
| Maximum executions of one normalized query | 1 | 1 |
| Maximum orchestrator model turns | 18 | 60 |
| Finalization turns after exhaustion | 1 | 1 |

`single_shot` keeps its existing direct-search budget. `direct` and meta paths
perform no delegated research.

The final values should be calibrated with evals, but all must be finite by
default.

#### Why allow more than one batch?

`standard` usually needs one batch. A second permits a small targeted follow-up
or resubmission of a failed worker without opening an unlimited retry path.

`deep` may legitimately split a plan into multiple batches when the planned
query count exceeds concurrency or when ordered research is required.

The total-query limit remains authoritative even when more batch calls are
allowed.

#### Count before execution

Count batches and queries before awaiting `run_research_batch`. This prevents
parallel calls in one orchestrator turn from racing past the limit.

If a proposed batch would exceed the remaining total-query budget:

- do not partially execute an ambiguous model-authored batch;
- return a deterministic tool result explaining the remaining budget is
  insufficient;
- mark the request as finalizing.

This is easier to reason about than silently dropping selected queries.

### Layer 2: cross-batch duplicate detection

The existing repeated-signature guard is local to one researcher. Add a
cross-batch signature at the orchestrator boundary.

A practical normalized signature should include:

```text
normalized query text
normalized subqueries
target components
preferred tools
depth
```

Normalization should:

- Unicode-normalize text;
- trim and collapse whitespace;
- case-fold text;
- sort sets such as preferred tool names;
- preserve ordered subqueries when order is meaningful;
- omit free-form rationale text so changing an explanation does not bypass the
  guard;
- hash the canonical representation before logging.

An exact normalized query should execute once per top-level request by default.
If the orchestrator needs a follow-up, it must express a materially different
question or target component. The finite total-query budget protects against
semantic paraphrases that evade exact normalization.

Do not attempt embedding-based semantic deduplication in the first version. It
adds latency, nondeterminism, and threshold tuning. Exact canonical
deduplication plus a hard aggregate budget is sufficient for guaranteed
termination.

### Layer 3: controlled transition to finalization

When any orchestrator research limit is reached:

1. Set the request phase to `finalizing`.
2. Do not execute the disallowed batch.
3. Return a deterministic `ToolMessage` that states:
   - which budget was reached;
   - that existing notes and registered sources must be used;
   - that missing components must be represented as gaps;
   - that no further research may be requested.
4. Remove `run_research_batch` from subsequent model calls.
5. Remove direct source tools, if any are present.
6. Remove `think` so the orchestrator cannot enter a reasoning-only loop.
7. Permit one finalization turn.

The allowed finalization tools depend on the active path:

- `standard` inline: `get_verified_sources` and `submit_final_report`;
- planned `standard`, `deep`, and delta: allow the existing writer completion
  path and read-only access to already persisted plan/notes;
- no path may regain source or research-batch tools after exhaustion.

The guard must not rely indefinitely on the model following this instruction.
After the configured finalization turn, control moves to the forced fallback.

### Layer 4: hard workflow deadline

Wrap the complete `agent.ainvoke(...)` call in
`AdaptiveResearcherAgent.run()` with `asyncio.timeout`.

Suggested initial deadlines:

| Tier | Deadline |
| :-- | --: |
| Meta / direct | 60 seconds |
| `single_shot` | 180 seconds |
| `standard` | 300 seconds |
| `deep` / delta | 1,200 seconds |

There is a routing challenge because the tier is chosen inside the graph. Two
reasonable implementations are:

#### Option A: one conservative outer deadline

Use a configurable request deadline, such as 1,200 seconds, around the entire
graph. Tier-specific batch and turn limits still constrain cheaper paths.

This is simpler and is the recommended first implementation.

#### Option B: dynamically tighten the deadline after tier declaration

Start with the deep ceiling, then publish the declared tier into request context
and enforce its earlier deadline.

This is more precise but adds coordination between middleware and
`AdaptiveResearcherAgent.run()`. It can follow after the core termination path
is proven.

The deadline must cover:

- orchestrator LLM calls;
- planner/researcher/writer calls;
- source tools;
- sandbox operations;
- final synthesis.

A source-specific timeout is not a workflow timeout.

### Layer 5: bounded forced finalization

When an orchestrator budget, graph-turn limit, or workflow timeout is reached,
the system should attempt to return useful partial work.

Use this ordered fallback:

#### Step 1: reuse an already completed report

If `output.md` or the final-report metadata already exists, process and return
it through the normal citation verification and sanitization path.

#### Step 2: bounded tool-free synthesis

If research notes or registered sources exist but no report exists:

- invoke a dedicated finalizer once;
- provide only persisted `ResearchNotes`, compact verified-source metadata,
  the original question, and the exhaustion reason;
- expose no tools;
- use a short independent timeout, for example 60 seconds;
- require explicit “Limitations” or “Evidence gaps” text;
- prohibit unsupported claims.

This call must not re-enter the adaptive graph.

#### Step 3: deterministic partial response

If the finalizer fails or times out, build a deterministic Markdown response
without another model call:

```text
# Partial research result

Research stopped after reaching the configured safety limit.

## What was found
<safe summaries/claims already present in validated ResearchNotes>

## Evidence gaps
<ResearchGap entries and missing periods/components>

## Sources
<verified compact source entries>
```

If there are no validated notes or sources, return a bounded failure explaining
that the configured sources did not provide sufficient evidence.

The deterministic fallback is what makes termination independent of model
compliance.

### Layer 6: per-researcher deadline

The per-researcher source-call budget cannot stop a single LLM or source call
that never returns.

Wrap each `researcher_runnable.ainvoke(...)` in a timeout. Suggested defaults:

| Research depth | Worker deadline |
| :-- | --: |
| `low` | 60 seconds |
| `medium` | 180 seconds |
| `high` | 300 seconds |

On worker timeout:

- cancel the worker;
- return a structured per-query failure to `_run_research_queries`;
- retain successful sibling notes;
- allow the parent to finalize from partial evidence;
- do not automatically resubmit the same normalized query if its request-wide
  execution count is already exhausted.

Worker timeouts must remain below the overall request deadline.

### Layer 7: realistic graph recursion limits

Replace `recursion_limit=2000` with a lower last-resort ceiling.

Suggested initial values:

| Tier/path | Recursion limit |
| :-- | --: |
| `single_shot` | 25 |
| `standard` | 80 |
| `deep` / delta | 250 |

If tier-aware graph configuration is awkward initially, use a single
conservative value such as `250`, then tighten by tier later.

The recursion limit should raise a recognizable exception that enters the same
partial-finalization path. It should not discard gathered evidence.

Batch, query, turn, and time budgets remain the primary controls because they
map directly to product behavior. Recursion count is an implementation-level
safety net.

### Layer 8: callback registration deduplication

Investigate why identical callback log records are emitted repeatedly.

Likely checks:

- callbacks appended more than once while building nested agents;
- the same callback list passed through both graph config and agent
  construction;
- researcher callbacks inherited repeatedly for every batch;
- logging handlers installed more than once per process or request.

Add an identity-based deduplication boundary when composing callbacks, and test
that one logical tool event produces one callback event.

Do not make request termination dependent on this cleanup.

## Configuration proposal

Add validated configuration under `AdaptiveResearchAgentConfig`.

One possible shape:

```yaml
request_termination:
  enabled: true

  max_batch_calls:
    standard: 2
    deep: 4

  max_total_research_queries:
    standard: 6
    deep: 16

  max_identical_research_queries: 1

  max_orchestrator_turns:
    standard: 18
    deep: 60

  finalization_turns: 1
  workflow_timeout_seconds: 1200
  fallback_finalizer_timeout_seconds: 60

  researcher_timeout_seconds:
    low: 60
    medium: 180
    high: 300

  recursion_limit: 250
```

Use immutable Pydantic models with:

- `extra="forbid"`;
- positive integer validation;
- ordering checks where applicable;
- validation that worker and fallback timeouts are less than the workflow
  timeout;
- validation that `standard` budgets do not exceed `deep` budgets;
- defaults enabled so production is safe without requiring operators to know
  about the new feature.

Consider naming the model `AdaptiveRequestTerminationConfig` rather than
expanding `ResearcherLoopGuardConfig`. The two configurations protect different
lifetimes:

- researcher guard: one delegated researcher;
- request termination: the top-level adaptive request.

Keeping them separate makes the safety boundary clear.

## Runtime state proposal

A request-scoped structure could contain:

```python
@dataclass
class OrchestratorRunGuardState:
    request_id: str
    declared_tier: str | None
    phase: Literal["active", "finalizing", "terminal"]
    model_turn_count: int
    batch_call_count: int
    total_query_count: int
    query_signature_counts: dict[str, int]
    exhaustion_reason: str | None
    started_at_monotonic: float
```

Use monotonic time for deadline calculations.

Do not log raw query arguments in guard messages. Log:

```text
request ID
tier
phase
batch count / limit
query count / limit
hashed signature
turn count / limit
elapsed time
termination reason
```

## Detailed request lifecycle

### Normal `standard` request

```text
route to standard
  → batch 1 with 2–3 bounded queries
  → researchers return notes
  → get verified sources
  → submit final report
  → terminal: complete
```

### Missing-evidence request

```text
route to standard
  → batch 1 returns notes with gaps
  → optional targeted batch 2
  → aggregate batch budget reached
  → run_research_batch withdrawn
  → one finalization turn
  → terminal: complete or partial
```

### Non-compliant orchestrator

```text
aggregate budget reached
  → research tools withdrawn
  → model does not finalize
  → finalization-turn limit reached
  → bounded tool-free finalizer
  → deterministic fallback if needed
  → terminal: partial or bounded failure
```

### Hung dependency

```text
LLM/source/sandbox call does not return
  → worker or workflow deadline fires
  → outstanding tasks cancelled
  → completed notes retained
  → bounded fallback
  → terminal: partial or bounded failure
```

## Cancellation and cleanup

Timeout and cancellation paths must clean up resources:

- cancel outstanding researcher tasks;
- await task cancellation with bounded cleanup;
- reset all request and researcher `ContextVar` tokens in `finally`;
- invoke the existing sandbox interruption cleanup;
- avoid persisting a “running” checkpoint after terminal timeout;
- mark the job with a terminal reason;
- do not leave background LLM or retrieval tasks detached from the request.

Cancellation cleanup itself must have a timeout. Cleanup failure should be
logged but must not keep the user request open indefinitely.

## Files likely to change

| File | Planned change |
| :-- | :-- |
| `src/aiq_agent/agents/adaptive_researcher/models/request_termination.py` | Add immutable validated request-wide budget and timeout configuration. |
| `src/aiq_agent/agents/adaptive_researcher/models/__init__.py` | Export the new configuration models. |
| `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py` | Add request-wide batch/query/turn tracking, duplicate detection, phase transition, and tool withdrawal. |
| `src/aiq_agent/agents/adaptive_researcher/factory.py` | Wire the request guard and replace the fixed recursion limit with the configured value. |
| `src/aiq_agent/agents/adaptive_researcher/agent.py` | Apply the workflow deadline, catch budget/timeout/recursion exits, and run bounded fallback finalization. |
| `src/aiq_agent/agents/adaptive_researcher/tools/research.py` | Add worker deadlines and structured timeout handling; expose query metadata needed for aggregate accounting only if middleware interception is insufficient. |
| `src/aiq_agent/agents/adaptive_researcher/register.py` | Add and forward `request_termination` configuration. |
| `src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2` | Explain finite request-wide budgets and mandatory gap-aware finalization. |
| `configs/config_adaptive_frag.yml` | Add explicit evaluated defaults. |
| `tests/aiq_agent/agents/adaptive_researcher/` | Add middleware, lifecycle, timeout, fallback, concurrency, and regression coverage. |
| `docs/source/architecture/agents/deep-researcher.md` | Document termination guarantees and partial-result semantics. |

Avoid changing the general `deep_researcher` unless a shared primitive is
clearly beneficial. The observed gap belongs to the adaptive orchestrator and
its effort tiers.

## Implementation phases

### Phase 1: configuration and pure budget state

1. Add validated request-termination models.
2. Add a small request-state object.
3. Implement canonical `ResearchQuery` signatures.
4. Unit-test validation, normalization, and concurrent accounting.

No behavior should change until the middleware is wired.

### Phase 2: orchestrator batch/query enforcement

1. Add `OrchestratorLoopGuardMiddleware`.
2. Intercept `declare_effort_tier` and `run_research_batch`.
3. Count batches and queries before execution.
4. Block repeated or over-budget batches deterministically.
5. Withdraw research tools after exhaustion.
6. Permit one bounded finalization turn.
7. Add metadata-only logging.

This phase directly fixes the observed repeated-new-invocation loop.

### Phase 3: forced finalization and hard workflow timeout

1. Add the outer `asyncio.timeout`.
2. Add a recognizable internal termination exception/result.
3. Reuse completed output when available.
4. Add one tool-free partial-report finalizer with its own timeout.
5. Add deterministic fallback Markdown.
6. Ensure citation verification and sanitization still run on partial reports
   where applicable.
7. Add cleanup for cancellation and timeout.

This phase provides the actual end-to-end termination guarantee.

### Phase 4: worker deadlines and recursion limit

1. Add depth-specific researcher deadlines.
2. Preserve successful sibling notes when one worker times out.
3. Prevent automatic duplicate resubmission.
4. Lower and configure the graph recursion limit.
5. Route recursion-limit exits through the same fallback.

### Phase 5: callback deduplication and observability

1. Identify duplicate callback registration.
2. Deduplicate callbacks by identity at the correct composition boundary.
3. Add termination counters and latency metrics.
4. Confirm one logical event creates one log/callback record.

### Phase 6: evaluation and rollout

1. Run focused unit and integration tests.
2. Reproduce the Apple missing-evidence case.
3. Run representative `single_shot`, `standard`, `deep`, and delta queries.
4. Tune defaults from measured latency and quality.
5. Enable in the dev adaptive config.
6. Observe partial-result and timeout rates before widening rollout.

## Test plan

### Configuration tests

- Defaults are finite and enabled.
- Unknown fields are rejected.
- Zero and negative budgets are rejected.
- `standard` values cannot exceed corresponding `deep` values.
- Worker/finalizer deadlines cannot exceed the workflow deadline.
- Config values are forwarded through both reusable and per-request agent
  construction paths.

### Orchestrator guard unit tests

- First allowed batch executes.
- Batch count is scoped to one request.
- A batch over the per-call concurrency limit keeps existing behavior.
- A second/third batch is blocked at the configured tier limit.
- Total-query count is enforced across differently sized batches.
- An identical normalized query is blocked across batches.
- Rationale-only changes do not bypass duplicate detection.
- A materially different target component receives a different signature.
- Parallel batch calls cannot overshoot the limit.
- `single_shot` behavior remains governed by its existing direct-search guard.
- Independent concurrent requests do not share counts.

### Finalization transition tests

- Exhaustion removes `run_research_batch`.
- Exhaustion removes source tools and `think`.
- `standard` inline retains verified-source and submit tools.
- Planned writer paths retain only the tools necessary to finish from existing
  notes.
- The model receives exactly one finalization turn.
- Further non-terminal behavior triggers forced fallback.

### Timeout tests

- A hung researcher is cancelled at its worker deadline.
- Successful sibling notes survive one worker timeout.
- A hung orchestrator is cancelled at the workflow deadline.
- A hung fallback finalizer is cancelled at its shorter deadline.
- Cancellation resets context and invokes resource cleanup.
- Cleanup failure does not prevent terminal response.

### Fallback tests

- Existing `output.md` is preferred.
- Existing inline final-report metadata is preferred.
- Validated notes produce a partial report.
- `ResearchGap` entries appear in the limitations section.
- Only verified sources appear as citations.
- Empty evidence produces a clear bounded failure.
- Sanitization still removes unsafe URLs.

### Recursion tests

- A synthetic infinite model loop reaches the configured recursion limit.
- The recursion exception enters fallback rather than escaping as an
  unstructured server error.
- Gathered evidence remains available after the recursion exit.

### Incident regression test

Simulate a `standard` Apple inventory query where every source call returns only
2022/2023 documents and every researcher records FY2024/FY2025 gaps.

Assert:

- each researcher remains within its depth budget;
- the orchestrator does not exceed its batch or total-query budget;
- no normalized query executes twice;
- the request returns a partial report;
- the report explicitly states the unavailable periods;
- the workflow terminates within the test deadline;
- the final state is not `running`.

### Callback tests

- One logical LLM/tool event reaches each callback once.
- Nested researcher callbacks are not appended repeatedly.
- Callback composition remains isolated across concurrent jobs.

## Observability

Add structured events for:

```text
adaptive_request_started
adaptive_tier_declared
adaptive_batch_started
adaptive_batch_blocked
adaptive_query_duplicate_blocked
adaptive_request_budget_exhausted
adaptive_request_timeout
adaptive_worker_timeout
adaptive_forced_finalization_started
adaptive_partial_report_returned
adaptive_request_completed
```

Recommended metrics:

| Metric | Purpose |
| :-- | :-- |
| Request duration by tier and terminal path | Detect slow or timeout-prone tiers. |
| Batch calls per request | Detect orchestrator retry behavior. |
| Delegated queries per request | Measure fan-out and enforce tuning assumptions. |
| Researcher source calls by depth | Verify local guard behavior. |
| Duplicate query blocks | Detect unavailable-source retry patterns. |
| Partial-report rate | Identify budget or source-coverage problems. |
| Workflow timeout rate | Detect dependency or model regressions. |
| Worker timeout rate by source/model | Localize slow components. |
| Callback events per logical event | Detect duplicate registration. |

Do not log raw source arguments, retrieved content, credentials, or complete user
queries in guard-specific logs.

## Rollout strategy

1. Land request-wide batch/query enforcement with conservative limits.
2. Keep a temporary metric-only comparison for one development run if needed,
   but do not ship an unbounded “disabled” production default.
3. Enable the hard workflow deadline and fallback in development.
4. Reproduce the original incident and confirm bounded partial completion.
5. Run evals for answer quality and citation correctness.
6. Tune `standard` and `deep` separately.
7. Deploy to a limited environment and monitor:
   - completion latency;
   - partial-result rate;
   - timeout rate;
   - average batch/query counts;
   - user-visible quality.
8. Expand rollout after the partial-result rate is understood.

If limits are too strict, adjust finite values. Do not restore unbounded
behavior.

## Risks and mitigations

### Risk: valid deep research is cut off too early

Mitigation:

- use separate `standard` and `deep` budgets;
- evaluate real multi-hop workloads;
- return partial evidence rather than discarding work;
- tune finite limits based on data.

### Risk: the model paraphrases queries to bypass exact deduplication

Mitigation:

- enforce the aggregate query and batch budgets regardless of signatures;
- use canonical exact matching as an early stop, not the only stop.

### Risk: forced fallback produces low-quality prose

Mitigation:

- prefer an already completed report;
- use one short tool-free synthesis call;
- fall back to deterministic validated note rendering;
- label partial results clearly.

### Risk: cancellation leaves background tasks running

Mitigation:

- use structured task ownership;
- cancel and await child tasks;
- run bounded sandbox cleanup;
- test cancellation under concurrency.

### Risk: tool withdrawal breaks the writer branch

Mitigation:

- use path-aware finalization allow-lists;
- test `standard` inline, planned `standard`, `deep`, and delta separately;
- never withdraw access to already persisted notes needed for synthesis.

### Risk: recursion-limit behavior differs by LangGraph version

Mitigation:

- treat recursion as a last resort;
- add a version-specific regression test;
- rely primarily on explicit batch, query, turn, and time budgets.

## Acceptance criteria

The work is complete when all of the following are true:

1. A top-level adaptive request has finite batch, query, turn, and time limits.
2. New researcher invocations cannot reset the top-level request budget.
3. Identical delegated queries cannot execute repeatedly across batches.
4. Reaching a research budget transitions the request to finalization.
5. The model receives at most one configured finalization opportunity after
   exhaustion.
6. A hard outer deadline cancels any remaining work.
7. A partial or bounded failure response is returned when normal finalization
   does not complete.
8. Successful evidence and verified citations are preserved in partial results.
9. Worker and request cancellation clean up owned resources.
10. `single_shot`, `standard`, `deep`, and delta regression tests pass.
11. The Apple missing-evidence incident completes within the configured
    `standard` deadline.
12. No job remains in a running state after its deadline.
13. One logical callback event is logged once.
14. Canonical documentation describes the termination and partial-result
    semantics.

## Recommended first implementation slice

The smallest change that directly fixes the observed incident while preserving
room for follow-up work is:

1. Add request-scoped counts for `run_research_batch` calls and total delegated
   queries.
2. Add cross-batch exact query-signature detection.
3. Withdraw `run_research_batch` after exhaustion.
4. Allow one finalization turn.
5. Wrap the full workflow in a hard timeout.
6. Return a deterministic partial response if normal finalization does not
   complete.
7. Lower the recursion limit from `2000` to a finite evaluated ceiling.

Per-tier dynamic deadlines, semantic query deduplication, and richer partial
report synthesis can follow without weakening the core guarantee.
