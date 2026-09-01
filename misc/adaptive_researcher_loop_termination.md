# Adaptive researcher: graceful termination after budget exhaustion

Investigation notes and design proposal.

- **Date:** 2026-07-30
- **Trigger:** eval run in tmux session `eval_auto_run`, container `aiq-agent`
  (`nvcr.io/nvidia/blueprint/aiq-agent:2.0.0`), config
  `configs/config_adaptive_frag.yml`, dataset `deepsearchqa`, `--limit 1`.
- **Status:** diagnosis confirmed from container logs and source; solution proposed, not yet implemented.

---

## 1. Incident

A single-sample eval appeared hung. The server was healthy
(`localhost:8100/health` → 200, `8000/tcp -> 0.0.0.0:8100`); the eval harness was
blocked at `Running inference: 0%|` waiting on one request.

The sample started at 10:47. From 10:52 onward the agent repeated the identical
turn every ~11–12 seconds, indefinitely:

```
[Reasoning] "...Let me search for the specific country profiles on ocindex.net."
[Tool Calls] 2 tool(s) requested
  → read_file  {'file_path': '/shared/plan.json'}
  → ls         {'path': '/shared'}
[Tool Result] "['/shared/effort_tier.json', '/shared/plan.json']"
```

Measured: 31 `read_file` + 31 `ls` in a 5-minute window, ~45 iterations by 11:01.
The reasoning text was **byte-identical every turn**. Prompt tokens grew ~744 per
turn (49,651 → 51,883 over the observed slice), so context was accumulating while
no progress was made.

The loop began immediately after the researcher source-call budgets were spent:

```
10:52:17 Researcher source-call budget reached | depth=low    tool=advanced_web_search_tool calls=1/1
10:52:37 Researcher source-call budget reached | depth=medium tool=advanced_web_search_tool calls=3/3
10:52:56 Researcher source-call budget reached | depth=medium tool=advanced_web_search_tool calls=3/3
```

No orchestrator loop-guard lines appeared, confirming the loop consumed no
`run_research_batch` calls — it was pure idle motion.

---

## 2. Relevant configuration

From `configs/config_adaptive_frag.yml`:

```yaml
researcher_loop_guard:          # per-researcher-invocation
  enabled: true
  source_call_budgets:
    low: 1
    medium: 3
    high: 6
  max_identical_source_calls: 2
  max_consecutive_thinks: 3

request_termination:            # per-request
  enabled: true
  standard: { max_batch_calls: 3, max_total_research_queries: 9,  max_orchestrator_turns: 24 }
  deep:     { max_batch_calls: 6, max_total_research_queries: 24, max_orchestrator_turns: 100 }
  max_identical_research_queries: 1
  workflow_timeout_seconds: 3600
  fallback_finalizer_timeout_seconds: 60
  recursion_limit: 250
```

Both blocks match the schema defaults exactly
(`models/loop_guard.py:16-44`, `models/request_termination.py:35-70`).

Also in play: `tavily_web_search.max_results: 2`. A `low`-depth researcher
therefore sees **at most 2 search results total** before its source tools are
withdrawn.

---

## 3. Root cause

The system *has* a graceful-exit design. It fails for three independent reasons,
which compound.

### Defect A — the exhaustion instruction is destroyed by context pruning

`ResearcherLoopGuardMiddleware._append_nudge`
(`agents/adaptive_researcher/custom_middleware.py:326`) appends
`_RESEARCHER_BUDGET_NUDGE` to the **end** of the last successful search result:

> `[SYSTEM — researcher source budget exhausted. Stop searching and return ResearchNotes now ...]`

`ToolResultPruningMiddleware(keep_last_n=10, max_chars=2000)`
(`agents/deep_researcher/custom_middleware.py:788`, wired at
`adaptive_researcher/factory.py:120`) truncates every ToolMessage older than the
last 10 to `content[:2000]`.

A web-search result is far larger than 2,000 characters, and the nudge sits at
the very end of it. Each loop iteration adds 2 tool messages, so **after 5
iterations the only instruction telling the model what to do is silently cut
off.** The model is left with its source tools missing and no explanation.

The `_blocked_result` message (`custom_middleware.py:337`) does not compensate:
it only fires when the model *attempts* a source call, and after exhaustion the
source tools are hidden, so the model never attempts one and never sees it.

### Defect B — tool starvation with somewhere to go

`_filter_tools` (`custom_middleware.py:348-360`) hides the source tools and
`think` once exhausted. It leaves the seven `FILESYSTEM_TOOL_NAMES`
(`deep_researcher/factory.py:68`): `edit_file`, `execute`, `grep`, `glob`, `ls`,
`read_file`, `write_file`.

A model that still wants to search, cannot search, and has no directive in
context will reach for the nearest available tool. `ls` and `read_file` are
always safe, always succeed, and always return the same thing — a perfect
attractor.

### Defect C — no guard covers the tools it actually calls

- `ResearcherLoopGuardMiddleware.awrap_tool_call` (`custom_middleware.py:379`)
  early-returns unless the tool is in `_source_tool_names`. So
  `max_identical_source_calls: 2` **never applies to `ls`/`read_file`**.
- `ConsecutiveThinkGuardMiddleware` counts only `think`, and any non-think call
  *resets* the counter — alternating `ls`/`read_file` defeats it structurally.
- A researcher invocation has **no turn budget and no wall-clock budget at all**.
  `max_orchestrator_turns` is orchestrator-only.
- `FilesystemToolCallGuardMiddleware` (`deep_researcher/custom_middleware.py:167`)
  only normalizes arg names and rejects placeholder paths — no repeat detection.

### Consequence

Nothing bounds the loop except the two request-level ceilings in
`agent.py:570-585`. At ~12 s/turn, `recursion_limit: 250` (~50 min) trips before
`workflow_timeout_seconds: 3600`, and `_build_partial_result` (`agent.py:498`)
returns a **deterministic stub** — literally *"No synthesized findings were
available when research stopped"* plus gaps and a source list. No LLM is
involved on that path.

A judge scores that stub zero. This is exactly the pattern recorded in
`TIER_ACCURACY_ANALYSIS.md`: *"62.3% of deep records exhaust the workflow time
limit and return incomplete answers."* Those samples are not producing bad
answers — they are producing the fallback template.

### Also noted

`fallback_finalizer_timeout_seconds: 60` is **dead config**. Its own field
description (`models/request_termination.py:100-107`) says *"Reserved for a
bounded tool-free finalizer. Unused by the deterministic first-slice fallback."*
The validator enforces it stays below `workflow_timeout_seconds`, but nothing
reads it.

Separately, `budgets_for_tier` (`models/request_termination.py:131-143`) returns
`None` for `single_shot` / `direct` / `meta` / pre-declaration, which disables
`_maybe_force_finalize_on_turns` entirely — those paths have **no turn cap at
all**.

---

## 4. Design principles

1. **Never rely on the model alone.** A directive is a hint; a withdrawn tool is
   a constraint; a hard ceiling is a guarantee. Ship all three.
2. **An exhausted agent must have exactly one exit.** If the only remaining
   action is "emit the structured response," it emits the structured response.
3. **Instructions must be durable.** Anything that can be pruned, truncated, or
   summarized out of context is not a mechanism.
4. **Degrade to partial evidence, never to a stub.** Notes with explicit gaps are
   a valid, scoreable outcome. The current fallback template is not.
5. **Bound per invocation, not per process.** Researcher runnables are built once
   and reused across invocations — guard counters must live in the
   `CURRENT_RESEARCHER_GUARD_STATE` contextvar, not on middleware instances.

---

## 5. Proposed solution

Four layers. Layers 1–2 are the minimal fix; 3–4 make it a guarantee.

### Layer 1 — durable exhaustion directive (fixes Defect A)

Stop appending the directive to a prunable ToolMessage. Inject it as a fresh
trailing `SystemMessage` on **every** model call while exhausted. Per-call
`request.override(messages=...)` is not persisted to graph state, so there is no
duplication, and `ToolResultPruningMiddleware` only touches `ToolMessage` — the
directive can never be truncated away.

This mirrors the established pattern in
`ComplexityRouterMiddleware._model_overrides` (`custom_middleware.py:269`), which
already swaps `system_message` per call.

```python
_RESEARCHER_EXHAUSTED_DIRECTIVE = (
    "SOURCE RESEARCH IS CLOSED for this researcher invocation ({reason}). "
    "The source tools have been withdrawn and will not return. Do NOT call ls, read_file, "
    "grep, glob, or any other tool looking for more evidence — no new evidence will appear. "
    "Return your structured ResearchNotes NOW from the evidence already in this conversation. "
    "For every target component you could not support, emit a ResearchGap naming the component "
    "and what was missing. Partial notes with explicit gaps are the expected, correct outcome; "
    "another tool call is a failure."
)


def _model_overrides(self, request) -> dict[str, object]:
    overrides: dict[str, object] = {"tools": self._filter_tools(request.tools)}
    state = CURRENT_RESEARCHER_GUARD_STATE.get()
    if state is not None and state.exhausted:
        directive = _RESEARCHER_EXHAUSTED_DIRECTIVE.format(
            reason=state.exhaustion_reason or "source budget exhausted"
        )
        overrides["messages"] = [*request.messages, SystemMessage(content=directive)]
    return overrides


def wrap_model_call(self, request, handler):
    return handler(request.override(**self._model_overrides(request)))


async def awrap_model_call(self, request, handler):
    return await handler(request.override(**self._model_overrides(request)))
```

`_append_nudge` can stay as a first-touch signal, but is no longer load-bearing.

### Layer 2 — starve the loop to its exit (fixes Defect B)

Once exhausted, withdraw the filesystem tools too. The researcher is built with
`response_format=ResearchNotes` (`deep_researcher/factory.py:377`), so under
LangChain's tool strategy the structured-response tool remains in the list and
becomes the only callable action.

```python
def _filter_tools(self, tools: list[object]) -> list[object]:
    state = CURRENT_RESEARCHER_GUARD_STATE.get()
    if not self._config.enabled or state is None:
        return tools
    hidden = set()
    if state.exhausted:
        hidden.update(self._source_tool_names)
        hidden.add(_THINK_TOOL)
        # Evidence gathering is over. Leaving the filesystem tools visible gives a model that
        # still "wants to search" somewhere to go, which is how an exhausted researcher
        # degenerates into an ls/read_file loop instead of returning ResearchNotes.
        hidden.update(FILESYSTEM_TOOL_NAMES)
    elif state.think_blocked:
        hidden.add(_THINK_TOOL)
    if not hidden:
        return tools
    remaining = [t for t in tools if _request_tool_name(t) not in hidden]
    if not remaining:
        # Provider-native structured output keeps no tool in the list; never hand the provider
        # an empty tool array. Fall back to withdrawing only the source tools.
        logger.warning("Researcher tool filter emptied the tool list; falling back to source-only hiding.")
        return [t for t in tools if _request_tool_name(t) not in self._source_tool_names]
    return remaining
```

**Verify before landing:** which structured-output strategy the installed
LangChain resolves for `ResearchNotes`. If it is `ProviderStrategy` rather than
`ToolStrategy`, the fallback branch fires every time and Layer 2 degrades to a
no-op — in that case keep `read_file` visible and lean on Layer 3.

### Layer 3 — universal repeated-tool-call guard (fixes Defect C)

A general net, independent of exhaustion. Hash `(tool_name, canonical_args)` for
**every** tool call and refuse a call that repeats beyond a threshold. Reuses the
existing `_canonical_source_signature` helper (content-free — only the hash is
retained).

Counters live in the contextvar state so they reset per researcher invocation,
with an instance-level fallback for orchestrator/planner/writer — the same
pattern `ConsecutiveThinkGuardMiddleware` already uses.

Requires one new field on `ResearcherRunGuardState`
(`deep_researcher/researcher_context.py:17`):

```python
tool_signature_counts: dict[str, int] = field(default_factory=dict)
```

```python
class RepeatedToolCallGuardMiddleware(AgentMiddleware):
    """Block a tool call that repeats an identical (name, args) signature.

    Scope is deliberately *every* tool, not just source tools. ResearcherLoopGuardMiddleware
    bounds evidence acquisition; this bounds pointless motion. A model that has lost its search
    tools will otherwise re-issue the same ls/read_file pair indefinitely — each call succeeds,
    so no existing guard fires, and only the graph recursion ceiling stops it.
    """

    def __init__(self, *, max_identical_tool_calls: int = 2, exempt_tool_names=frozenset()) -> None:
        self._max = max_identical_tool_calls
        self._exempt = frozenset(exempt_tool_names)
        self._fallback_counts: dict[str, int] = {}

    async def awrap_tool_call(self, request, handler):
        tool_call = getattr(request, "tool_call", None)
        if not isinstance(tool_call, dict):
            return await handler(request)
        name = tool_call.get("name")
        if name in self._exempt:
            return await handler(request)

        state = CURRENT_RESEARCHER_GUARD_STATE.get()
        counts = state.tool_signature_counts if state is not None else self._fallback_counts
        signature = _canonical_source_signature(name, tool_call.get("args", {}))
        count = counts.get(signature, 0)

        if count >= self._max:
            logger.warning(
                "Repeated tool-call guard blocked | tool=%s repeats=%d/%d signature=%s",
                name, count, self._max, signature[:12],
            )
            if state is not None:
                state.exhausted = True          # hands off to Layers 1 and 2
                state.exhaustion_reason = "repeated identical tool calls"
            return ToolMessage(
                content=(
                    f"'{name}' was not executed: this exact call has already run {count} time(s) "
                    "in this invocation and returned the same result. Repeating it cannot produce "
                    "new information. Return your structured response now from what you already "
                    "have, recording anything unresolved as an explicit gap."
                ),
                tool_call_id=tool_call.get("id", "repeated-tool-call-guard"),
                name=name or "tool",
                status="error",
            )

        counts[signature] = count + 1
        return await handler(request)
```

Wire it into `build_common_middleware` (`adaptive_researcher/factory.py:116`) so
researcher, planner, writer **and orchestrator** all inherit it — the same hole
exists at every level. Exempt `write_file` and `edit_file` (legitimately
idempotent-looking but state-changing).

### Layer 4 — hard per-researcher ceilings with deterministic notes

The guarantee. Two new bounds, both per researcher invocation:

- `max_researcher_turns` — per depth, e.g. `low: 8`, `medium: 16`, `high: 28`.
  Counted in `awrap_model_call`.
- `researcher_timeout_seconds` — wall clock, e.g. `300`.

On overflow, **do not ask the model.** Raise a dedicated exception from the
middleware and catch it where each researcher is invoked, in
`build_adaptive_research_batch_tool`
(`adaptive_researcher/factory.py:262`), converting it into a valid
`ResearchNotes` assembled from what `SourceRegistryMiddleware` already captured:

```python
class ResearcherBudgetExceeded(Exception):
    """A researcher invocation exceeded its hard turn or wall-clock ceiling."""


async def _run_one(query):
    try:
        async with asyncio.timeout(cfg.researcher_timeout_seconds):
            return await researcher_runnable.ainvoke(...)
    except (TimeoutError, ResearcherBudgetExceeded, GraphRecursionError) as ex:
        logger.warning("Researcher forcibly finalized (%s); returning deterministic notes.", type(ex).__name__)
        return _deterministic_research_notes(query, source_registry_middleware, reason=str(ex))
```

`ResearchNotes` requires (`deep_researcher/models/subagent_contracts.py:181-195`):
`query_topic`, `target_components`, `summary`, `findings`, `gaps`, `sources`,
`narrative_notes`, `language`, optional `evidence_judgment`. All of these can be
filled deterministically: components and topic from the `ResearchQuery`, sources
from the registry, one `ResearchGap` per unsupported component, and a summary
stating the researcher was cut short.

This mirrors `_build_partial_result` (`agent.py:498`) one level down. Crucially,
a stalled researcher then costs one query instead of the whole request: the
orchestrator receives usable notes-with-gaps and can still synthesize a real
answer.

### Layer 5 — orchestrator parity (smaller, same shape)

1. `OrchestratorLoopGuardMiddleware._filter_tools` (`custom_middleware.py:618`)
   currently hides only `run_research_batch` and `think` when finalizing. Also
   hide the filesystem tools, keeping `get_verified_sources` and
   `submit_final_report`.
2. Add an absolute turn ceiling that applies when `budgets_for_tier` returns
   `None`, closing the "loops before declaring a tier → no cap at all" gap.
3. Implement `fallback_finalizer_timeout_seconds` — on the workflow deadline,
   give the model that reserved budget with the gathered evidence and **no
   tools** to write a real answer, before falling back to the deterministic stub.
   This is the single highest-leverage change for the deep-tier eval numbers,
   since it converts zero-scoring templates into genuine attempts.

---

## 6. Configuration surface

Additions to `ResearcherLoopGuardConfig` (`models/loop_guard.py:36`):

```yaml
researcher_loop_guard:
  enabled: true
  source_call_budgets: { low: 1, medium: 3, high: 6 }
  max_identical_source_calls: 2
  max_consecutive_thinks: 3
  # new
  max_identical_tool_calls: 2          # applies to every tool, not just source tools
  max_researcher_turns: { low: 8, medium: 16, high: 28 }
  researcher_timeout_seconds: 300
```

Defaults must be finite and enabled, matching the stated convention in
`models/request_termination.py:15-16` ("production is bounded without operators
having to opt in"). `max_researcher_turns` needs the same `low <= medium <= high`
validator as `ResearcherSourceCallBudgets`.

Worth revisiting independently: `source_call_budgets.low: 1` combined with
`tavily_web_search.max_results: 2` gives a low-depth researcher two search
results total. That is a thin evidence base and makes exhaustion the common case
rather than the exceptional one. Raising `low` to 2 and `medium` to 4 will not
fix the loop, but will reduce how often the termination path is exercised.

---

## 7. Tests

Existing files to extend:

- `tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py`
- `tests/aiq_agent/agents/adaptive_researcher/test_orchestrator_loop_guard.py`
- `tests/aiq_agent/agents/adaptive_researcher/test_request_termination.py`

Cases:

1. **Regression for this incident** — exhaust the source budget, then drive 20
   alternating `ls`/`read_file` calls; assert the guard blocks by call 3 and the
   tool list no longer contains any filesystem tool.
2. **Directive survives pruning** — build a message list with >10 oversized
   ToolMessages after exhaustion; assert the directive is still present in the
   messages passed to the model. *This is the test that would have caught the
   original bug.*
3. **Per-invocation isolation** — two sequential invocations sharing one
   middleware instance; assert signature counts reset via the contextvar.
4. **Turn ceiling** — exceed `max_researcher_turns`; assert
   `ResearcherBudgetExceeded` and that the batch tool returns a schema-valid
   `ResearchNotes` with one gap per unsupported component.
5. **Empty tool list guard** — assert `_filter_tools` never returns `[]`.
6. **Wall clock** — a researcher that hangs is cut at
   `researcher_timeout_seconds` and yields deterministic notes.

Validation commands (`CLAUDE.md`):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/aiq_agent/agents/adaptive_researcher/
```

---

## 8. Recommended landing order

| Step | Layers | Risk | Effect |
| :-- | :-- | :-- | :-- |
| 1 | 1 + 2 | Low | Fixes the observed loop directly; small, targeted diff |
| 2 | 3 | Low | General anti-loop net at every agent level |
| 3 | 4 | Medium | Makes termination a guarantee; contains blast radius to one query |
| 4 | 5 | Medium | Orchestrator parity + activates the dead finalizer budget |

Steps 1–2 are independently shippable and should measurably reduce deep-tier
timeouts on their own.

**Suggested validation:** rerun the 38 timed-out deep samples identified in
`TIER_ACCURACY_ANALYSIS.md` and compare on that same fixed subset, so the
comparison is not confounded by a different tier/question mix.

---

## 9. Open questions

1. Which structured-output strategy does the installed LangChain pick for
   `response_format=ResearchNotes`? Determines whether Layer 2 alone is
   sufficient.
2. Should `execute` be exempt from the repeated-call guard? Re-running an
   identical command can be legitimate when sandbox state has changed.
3. Is the byte-identical reasoning across ~45 turns purely the tool-starvation
   attractor, or is decoding pinned (temperature 0 with a near-identical prompt
   prefix)? Worth confirming against the model config — if it is decoding, the
   guards still fix it, but the diagnosis is incomplete without checking.
