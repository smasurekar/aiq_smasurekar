# Adaptive Researcher Loop Guard — Implementation Report

**Date:** 2026-07-23
**Scope:** Adaptive researcher sub-agent loop protection
**Source plan:** `misc/adaptive-researcher-subagent-loop-guard-plan.md`

## Executive Summary

The adaptive researcher now has deterministic, per-invocation protection against
unbounded source-tool loops. The implementation addresses the observed pattern of
alternating `think` and repeated `knowledge_search` calls, which the existing
consecutive-think middleware could not detect because every search reset its counter.

The solution adds hard depth-based source budgets, repeated-call detection,
context-isolated state for concurrent workers, graceful structured completion, validated
configuration, prompt synchronization, metadata-only logging, and regression tests.

## Incident and Root Cause

The affected standard-tier job delegated a high-depth query to a reusable researcher
worker. Its available internal source returned quarterly documents instead of the annual
data requested, and the worker repeatedly issued the same search:

```text
think → knowledge_search(same arguments) → think → knowledge_search(same arguments) → ...
```

`ConsecutiveThinkGuardMiddleware` was present, but it only detects uninterrupted
`think → think → think` sequences. A non-`think` tool call resets its counter, so the
observed alternating sequence never reached the threshold. The depth budgets in the
researcher prompt were also advisory and had no runtime enforcement.

The reusable researcher middleware instance can serve multiple concurrent batch items,
so storing new counters directly on the middleware object would also have mixed state
between independent queries. The implemented design therefore stores worker state in a
`ContextVar` created and reset around each researcher invocation.

## Runtime Behavior Added

### Source-call budgets

Each researcher invocation receives a hard ceiling based on `ResearchQuery.depth`:

| Depth | Maximum source-tool calls |
| :-- | --: |
| `low` | 1 |
| `medium` or missing | 3 |
| `high` | 6 |

Only source tools identified by `DeepResearchToolSet.source_tool_names` are charged.
Helper, filesystem, skill, and structured-output tools remain available and free.

### Repeated-call protection

A source request is identified by a SHA-256 hash of its tool name and canonical JSON
arguments. Mapping keys are sorted while list order is preserved. Two identical calls
are allowed by default; the third is blocked without executing the source tool. Raw
arguments and retrieved content are not written to guard logs.

### Graceful exhaustion

The last allowed source result is preserved and receives a system nudge instructing the
worker to return `ResearchNotes` using gathered evidence and `ResearchGap` entries for
unsupported components. Later model calls no longer expose source tools or `think`.
Unexpected attempts to call an exhausted source return a deterministic `ToolMessage`
instead of raising and failing the entire batch.

### Concurrency and retry semantics

- `_run_research_query()` creates a unique guard state for every worker invocation.
- `ContextVar` isolation prevents counts from leaking across concurrent queries.
- Calls are counted before awaiting the handler, so parallel calls share one hard limit.
- The guards wrap `ToolRetryMiddleware`; internal framework retries therefore remain one
  logical source call and do not consume extra budget or repeated-signature slots.
- The context token is reset in `finally`, including error and cancellation paths.

### Pure-think hardening

`ConsecutiveThinkGuardMiddleware` remains responsible for uninterrupted think loops.
Its researcher counter now uses the same per-invocation context. At the configured
threshold it retains the corrective warning and marks `think` for withdrawal from the
next researcher model request.

## Configuration

The adaptive agent accepts this validated nested configuration:

```yaml
researcher_loop_guard:
  enabled: true
  source_call_budgets:
    low: 1
    medium: 3
    high: 6
  max_identical_source_calls: 2
  max_consecutive_thinks: 3
```

All numeric limits must be positive, budgets must satisfy
`low <= medium <= high`, and unknown fields are rejected. The effective values are
rendered into `adaptive_researcher/prompts/researcher.j2` so prompt guidance cannot
silently drift from runtime enforcement.

## Files Changed

| File | Change |
| :-- | :-- |
| `src/aiq_agent/agents/deep_researcher/researcher_context.py` | Added dependency-neutral per-invocation state and depth normalization. |
| `src/aiq_agent/agents/deep_researcher/tools/research.py` | Establishes and resets guard context around every reusable researcher invocation. |
| `src/aiq_agent/agents/adaptive_researcher/models/loop_guard.py` | Added validated immutable budget and guard configuration models. |
| `src/aiq_agent/agents/adaptive_researcher/models/__init__.py` | Exports the new configuration models. |
| `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py` | Added source-budget/repeat enforcement and strengthened pure-think handling. |
| `src/aiq_agent/agents/adaptive_researcher/factory.py` | Wires source names, configuration, retry-safe ordering, and prompt values. |
| `src/aiq_agent/agents/adaptive_researcher/agent.py` | Stores and threads effective guard configuration. |
| `src/aiq_agent/agents/adaptive_researcher/register.py` | Adds the nested NAT function configuration and forwards it for reusable/per-request agents. |
| `src/aiq_agent/agents/adaptive_researcher/prompts/researcher.j2` | Replaces duplicated literal budgets with rendered effective limits and completion guidance. |
| `configs/config_adaptive_frag.yml` | Enables explicit defaults for the active adaptive deployment. |
| `tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py` | Covers all depth limits, repeated calls, the incident sequence, parallel calls, isolation, disabled mode, immutable results, and tool withdrawal. |
| `tests/aiq_agent/agents/adaptive_researcher/test_factory.py` | Verifies guards wrap framework retry middleware. |
| `tests/aiq_agent/agents/adaptive_researcher/test_register.py` | Covers defaults, overrides, and rejected invalid configuration. |
| `tests/aiq_agent/agents/deep_researcher/test_agent.py` | Verifies invocation context creation, default depth, and cleanup. |
| `docs/source/architecture/agents/deep-researcher.md` | Documents adaptive worker limits and concurrency semantics. |
| `docs/source/customization/prompts.md` | Adds the adaptive researcher prompt to the canonical inventory. |
| `misc/adaptive-researcher-subagent-loop-guard-plan.md` | Updates plan status to implemented for Phases 1 and 2. |

## Validation Results

- 78 focused middleware, configuration, retry-order, and context lifecycle tests passed.
- 91 broader adaptive/deep researcher tests passed with one unrelated stale assertion
  deselected.
- Ruff 0.15.1 lint passed for all changed Python files.
- Ruff formatting checks passed for all changed Python files.
- Python compilation and `git diff --check` passed.
- The regular `uv run` environment could not build `annoy` because the host lacks
  `x86_64-linux-gnu-g++`; tests were therefore run in the existing AI-Q runtime image
  with the repository mounted read-only and pinned pytest packages mounted separately.

The stale factory assertion expects the orchestrator tool list to omit
`declare_effort_tier`, although that tool already exists in the base factory code. It is
not caused by this implementation and was not changed as part of this scope.

## Deployment Issue Found After Implementation

The first rebuilt/running container rejected the adaptive config with the misleading
message `Invalid configuration: functions: Extra inputs are not permitted`. Backend logs
showed the actual cause: importing `researcher_context.py` raised `PermissionError`, so
NAT could not register `adaptive_research_agent` or `adaptive_research_workflow`.

The patch fallback had created or replaced several files with mode `0600`, while the
container runs AI-Q as a non-root user. All files changed by this implementation have now
been restored to mode `0644`. The backend image must be rebuilt/recreated after that
permission correction before inference is retried.

## Deferred and Non-Goals

- A whole-run ceiling on repeated `run_research_batch` orchestration remains an optional
  evaluation item; this change bounds each researcher worker.
- Effort-tier selection, source-routing quality, Docker worker scaling, and the graph
  recursion limit are unchanged.
