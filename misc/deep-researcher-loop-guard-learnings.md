# Deep researcher loop guard — learnings from the reverted implementation

Context: commits `b991418` (feat: loop guard middleware) and `f4f4d3d` (CodeRabbit
follow-up) were reverted so the PR could be re-implemented around the built-in
`ModelCallLimitMiddleware`. This file keeps what those two commits established, so the
knowledge is not lost with the code.

## Why it was reverted

Reviewer verdict: prompt- and code-heavy for the problem. Four independent limits
(`max_source_calls_per_query`, `max_identical_source_calls`, `max_consecutive_thinks`,
`max_model_turns_per_query`), a `ContextVar` + dataclass for per-invocation state,
SHA-256 source-signature hashing, consecutive-`think` detection, ~300 lines of
middleware, ~70 lines of docs, and researcher prompt changes that the *safety*
behaviour depended on.

Key points from the review:

- Prompts should improve graceful degradation, never be required for safety.
- Repeated searches and think loops are *symptoms*. A hard model-turn cap catches all
  of them, plus filesystem loops and novel looping patterns the heuristics never
  recognized.
- The existing job budgets already bound the expensive dimension:
  `resource_limits.max_source_tool_calls` (job-wide retrieval),
  `resource_limits.max_execution_seconds` (wall clock), `max_research_concurrency`,
  `max_concurrent_source_tool_calls`, `max_source_tool_batch_size`. Concurrency limits
  bound parallelism, not iterations — the only missing dimension was a **per-worker
  model-turn cap**.
- LangChain 1.3.11 (already locked) ships
  `ModelCallLimitMiddleware(run_limit=..., exit_behavior="end")`, which keeps its counts
  in invocation state. Concurrent workers are therefore already isolated: no
  `ContextVar`, no custom counter middleware.
- A code-built empty `ResearchNotes` at exhaustion is *not* acceptable on its own — it
  discards the worker's synthesis and its already-retrieved evidence. Hence the
  agreed design: research budget → reserved finalization turn (tools removed, prompt
  injected by middleware) → built-in limit as the last-resort stop.

## Technical findings worth keeping

These were verified against `langchain==1.3.11` and are what the new implementation
builds on.

### `ModelCallLimitMiddleware` (`langchain/agents/middleware/model_call_limit.py`)

- Counts in `after_model`, i.e. **per model node**, not per model invocation. A
  middleware that calls `handler(...)` twice inside one `wrap_model_call` (as
  `StructuredResponseTextFallbackMiddleware` does for its correction call) still costs
  one counted turn.
- `run_model_call_count` is `Annotated[int, UntrackedValue, PrivateStateAttr]`, so it
  resets per `ainvoke`. Every researcher worker is its own run → isolation is free.
- `before_model` checks `count >= run_limit` and, with `exit_behavior="end"`, returns
  `{"jump_to": "end", "messages": [AIMessage("Model call limits exceeded: ...")]}`.
  The run therefore ends with **no** `structured_response`, which is what the
  `_run_research_query` fallback has to handle.

### Structured output in `create_agent` (`langchain/agents/factory.py`)

`response_format=ResearchNotes` is wrapped in `AutoStrategy`, resolved per model call:

- `ProviderStrategy` when `_supports_provider_strategy(model, tools=...)` — the model is
  bound with `bind_tools(final_tools, **kwargs)` **even when `final_tools` is empty**.
  Sending `tools: []` is rejected by OpenAI-compatible providers, which is why the
  reverted code had to null out `response_format` on its final turn and lean on
  `StructuredResponseTextFallbackMiddleware` to promote raw JSON.
- `ToolStrategy` appends the structured-output tool to `final_tools` and sets
  `tool_choice="any"` whenever any structured tool exists. With `tools=[]` overridden,
  the structured-output tool is the *only* callable tool and the model is forced to
  call it → `_handle_model_output` fills `structured_response` and the run ends.
  **This is the deterministic exit** and it avoids the empty-`tools` problem entirely.
- A malformed structured tool call is not fatal: `_handle_structured_output_error`
  appends an error `ToolMessage` and loops back to the model — that is the one
  "schema-correction" turn the finalization phase has to reserve.
- `ToolStrategy` may only name tools declared upfront, so re-binding
  `ToolStrategy(schema=ResearchNotes)` mid-run is legal (same tool name as the
  `AutoStrategy` setup binding), while introducing a new schema would raise.

### Middleware ordering in `build_researcher_runnable`

Order is `[Skills, Filesystem, summarization, PatchToolCalls,
StructuredResponseTextFallback, *researcher_middleware, *visibility]`, and
`wrap_model_call` composes **first-in-list-outermost**. So anything appended through
`middleware_set.researcher` runs *inside* the text fallback and its corrective call
re-enters it. `middleware_set.researcher` is consumed only by
`build_researcher_runnable`; the planner/writer/orchestrator stacks are untouched.

For tool hooks, the reverted guard sat immediately *before* `ToolRetryMiddleware` so a
request that the retry middleware attempted three times counted once. That ordering
concern disappears with a model-turn-only budget, but it is the right anchor if any
tool-level counting is ever reintroduced. `ToolNameSanitizationMiddleware` position is
irrelevant: it rewrites names in `awrap_model_call`, so every `wrap_tool_call` wrapper
already sees sanitized names.

### Miscellaneous

- `researcher_invoke_config` must keep the inherited `recursion_limit`: the researcher
  subgraph binds none of its own, so popping it silently drops to LangGraph's default
  rather than leaving it unbounded.
- The researcher stack's tool middleware is async-only by convention
  (`ToolRetryMiddleware`, `SourceRegistryMiddleware`, …), and `_run_research_query`
  drives the worker exclusively through `ainvoke`.
- `ResearchNotes` is a strict contract: `query_topic`, `target_components`, `summary`,
  `findings`, `gaps`, `sources`, `narrative_notes`, `language` are all required;
  `evidence_judgment` is optional. Any code-built fallback note must fill all of them.

## Traps found while building the replacement

Three defects found in review of the first cut of the new design, all reproduced against
`langchain==1.3.11`. Worth keeping because each one looks correct on inspection:

- **`request.tools=[]` does not disarm the graph.** It only changes the model *binding*.
  `ToolNode` was built from the full tool list at `create_agent` time, and
  `_make_model_to_tools_edge` dispatches any tool call whose name it recognizes. A model that
  emits a tool call anyway - or hallucinates one - still executes it after exhaustion.
  Blocking has to happen in `wrap_tool_call`/`awrap_tool_call`, not by emptying the binding.
  The guard condition is `run_model_call_count > max_model_calls`, strictly greater: the
  counter is incremented after each model node, so the *last research turn* dispatches its
  tools at exactly `max_model_calls` and only the finalization turn dispatches above it.
- **A middleware nested inside `StructuredResponseTextFallbackMiddleware` can fire twice per
  counted turn.** The fallback calls its handler a second time for the corrective JSON call,
  and `run_model_call_count` is only incremented once the model *node* finishes, so both calls
  read the same count. Nested, the finalization override was re-applied and the nudge appended
  twice - two real provider calls inside one counted turn. Placing the budget pair *outside*
  the fallback makes its corrective call nested within the one reserved turn, which is what
  turns it into a schema correction rather than a second finalization chance.
- **`exit_behavior="end"` is indistinguishable from any other empty result.** It ends the run
  with a plain `AIMessage` and no `structured_response`, so a caller cannot tell budget
  exhaustion from an ordinary contract failure without string-matching LangChain's internal
  message. `exit_behavior="error"` raises `ModelCallLimitExceededError` (public API), which
  propagates unwrapped through `ainvoke`, so `_run_research_query` can catch it and attribute
  the truncated note accurately. This is the one deliberate deviation from the reviewer's
  snippet.

## What the replacement keeps

1. One YAML knob — a per-worker model-turn budget — instead of four limits and an
   `enabled` flag.
2. No diff in `prompts/researcher.j2`; the finalization instruction is injected by
   middleware only on the finalization turn.
3. Exactly one finalization turn: tools removed *and* refused at the tool node,
   `ToolStrategy(ResearchNotes)` forced, so the worker's own synthesis of the evidence it
   already gathered is what gets returned.
4. `ModelCallLimitMiddleware(run_limit=budget + 1, exit_behavior="error")` as the backstop,
   with a code-built truncated `ResearchNotes` in `_run_research_query` so the batch always
   receives a note, attributed to the cause that actually stopped the worker.
