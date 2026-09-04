# Plan: Shallow Researcher as a Compiled Sub-Agent on the Adaptive `single_shot` Path

Status: **implemented.** See §0 for where the implementation deviates from this plan.
Target: `src/aiq_agent/agents/adaptive_researcher/`
Reference library: `deepagents==0.6.8` (`.venv/.../deepagents/middleware/subagents.py`),
cross-checked against the upstream checkout at `~/Desktop/Swapnil/github_repos/deepagents`.

---

## 0. Implementation notes — where this differs from the plan

Four deviations, each forced by something the plan did not anticipate.

1. **`run_once` keys on `status`, not on whether the task raised.** The plan cleared the
   in-flight slot when the task "raised or was cancelled". That contradicts the never-raise
   contract from §5.1: a failed attempt *completes normally* with a failure payload, so the slot
   was retained and every later delegation replayed the cached failure instead of making a real
   attempt. The condition is now `task.done() and self.status != "completed"`, which covers
   raised, cancelled, and returned-a-failure. Caught by
   `test_a_failed_attempt_leaves_the_slot_retryable`.

2. **The prompt variable is `single_shot_shallow_subagent | default(false)`.** `render_prompt_template`
   uses `jinja2.StrictUndefined`, so an unguarded new variable broke 18 existing prompt tests
   that render `orchestrator.j2` with their own fixed kwargs. The `default(false)` filter matches
   how `sections | default({})` is already handled in this template, keeps every other renderer
   working, and leaves the flag-off render byte-identical (verified by diffing the rendered
   output against `git show HEAD:` for both the two-loop and single-loop cases).

3. **`_declared_tiers_in_current_tool_batch` is used only by the new middleware.** §6.3 proposed
   adopting it in `ComplexityRouterMiddleware` and `OrchestratorLoopGuardMiddleware` too. Both
   already resolve their tier through their own `awrap_tool_call`, and their tier-dependent
   effects land on the *next* model call — after the batch has finished — so the ordering hazard
   the helper solves does not apply to them. Changing `OrchestratorLoopGuardMiddleware` would
   also tighten enforcement on the `standard` / `deep` paths (a `run_research_batch` issued in
   the same turn as `declare_effort_tier` currently escapes the budget), which is a real but
   separate fix and out of scope here.

4. **`last_human_text` is public.** The factory needs the same helper to capture the original
   query at build time; importing an underscore-prefixed name across modules is worse than
   exporting one.

**Validation.** `1550 passed, 8 failed` across `tests/aiq_agent`; the 8 failures are pre-existing
and byte-identical to the pre-change baseline (7 `test_orchestrator_loop_guard` config-validation
tests plus 1 `test_factory` tool-list assertion, none of them touching this feature — verified by
`comm`-diffing the failure sets with the change stashed). 62 new tests. `ruff check` and
`ruff format --check` clean on every file touched (`tools/research.py` carries one pre-existing
E501 and was not modified). Both `configs/config_adaptive_frag.yml` and the new
`configs/config_adaptive_shallow_subagent.yml` validate against `AdaptiveResearchAgentConfig`.

Not yet done: the docs update in §12, and the live smoke / eval steps in §13 (which need the
Harbor image rebuild described in §14).

---

## 1. Goal

When the adaptive orchestrator declares the `single_shot` effort tier, hand the request to the
**existing** `ShallowResearcherAgent` (`src/aiq_agent/agents/shallow_researcher/agent.py`) —
reused as-is, no fork, no re-prompting — wired into the adaptive DeepAgents graph as a
**`CompiledSubAgent`**. The shallow agent researches the *same original user query*, and its
post-processed report becomes the authoritative run answer: the orchestrator cannot rewrite it,
although the adaptive agent's existing citation verification, sanitization, whitespace
normalization, and artifact post-processing may still make deterministic final-output changes.

Explicit non-goals:

- Tier-selection criteria do **not** change. `declare_effort_tier` and the orchestrator's
  selection prompt stay exactly as they are today (`prompts/orchestrator.j2` §Effort Selection,
  `tiers.py::TIER_PROFILES`).
- `direct` / `standard` / `deep` execution paths are unchanged. Enabling this feature does
  necessarily add `shallow-researcher` to the shared `task` description; middleware rejects that
  subtype outside `single_shot`, and regression tests/evals cover any model-behavior spillover.
- The shallow researcher's own prompt, graph, loop bounds, and citation logic are not modified
  (one small, optional, backwards-compatible constructor addition is called out in §6.5).

---

## 2. What `single_shot` does today

Two existing execution shapes, both ending at the same finalize seam:

| Mode | Config | Path |
| :-- | :-- | :-- |
| Two-loop (default) | `single_loop_single_shot: false` | orchestrator → `run_research_batch` → researcher subagent → orchestrator writes the answer inline → `submit_final_report` |
| Single-loop | `single_loop_single_shot: true` | orchestrator calls source tools **directly** (`factory.py:274-283`), capped by `single_shot_search_budget` (`custom_middleware.py:247-267`) → `submit_final_report` |

Finalize seam (unchanged by this plan, and the thing we plug into):

- `tools/finalize.py::submit_final_report` writes `/shared/final_report.md` +
  `/shared/final_report_meta.json` through `backend.upload_files` and is `return_direct=True`,
  so the graph ends the moment it executes (`tools/finalize.py:104-145`).
- `agent.py::_resolve_output_file_markdown` (`agent.py:316-340`) reads
  `/shared/output.md` → `/output.md` → `/shared/final_report.md` from the run's `files` mapping.
- `agent.py::run` then verifies citations against `SourceRegistryMiddleware.active_registry()`,
  sanitizes, re-emits to the frontend, and overwrites the last message
  (`agent.py:592-701`).

This plan adds a **third** `single_shot` mode and reuses that seam untouched.

---

## 3. Design overview

```
AdaptiveResearcherAgent.run(state)
  └─ build_adaptive_research_graph(...)                       [factory.py]
       ├─ subagents = [source-router?, planner-agent, writer-agent,
       │               shallow-researcher  ← NEW CompiledSubAgent]
       └─ orchestrator middleware
            ├─ ComplexityRouterMiddleware        (existing; gains a shallow-subagent mode)
            └─ SingleShotShallowDelegationMiddleware   ← NEW (tier-safe routing + override)

Runtime, single_shot:
  turn 1  declare_effort_tier("single_shot")
            → ComplexityRouter: expose {task, submit_final_report, get_verified_sources};
              hide run_research_batch + direct source tools
  turn 2  task(subagent_type=<whatever model supplied>, description=<whatever model supplied>)
            → delegation middleware requires a declared tier, then for single_shot
              forces subagent_type="shallow-researcher" and description=<original user question>
            → adapter runnable
                 · coalesces parallel/retried calls onto one in-flight shallow run
                   (run-scoped task, owned and cancelled by AdaptiveResearcherAgent.run)
                 · rebuilds ShallowResearchAgentState from the ORIGINAL user query
                 · shares the citation registry with the parent
                 · await ShallowResearcherAgent.run(...)   ← reused as-is
                 · records normalized, post-processed markdown into a per-request capture
                 · on failure: returns a failure notice, never raises — see §5.1
                 · returns {"messages": [...], "files": {final_report.md, meta.json}}  (safety net)
            → ComplexityRouter: `task` now hidden (one delegation per run)
  turn 3  submit_final_report(markdown=<whatever the model typed>)
            → SingleShotShallowDelegationMiddleware rewrites args to the captured
              markdown, researched=True, tier="single_shot"
            → return_direct=True ends the graph
  post    AdaptiveResearcherAgent.run reads /shared/final_report.md → verify → sanitize →
          emit_final_report → returned state
```

The design has **three independent guarantees** that the shallow report is authoritative:

1. **Tier-aware task override** forces the `single_shot` task call to the shallow researcher
   with the original query; prompt compliance is not part of the correctness boundary.
2. **Argument override** on `submit_final_report` prevents orchestrator paraphrasing.
3. **Run-scoped capture recovery** exposes a completed capture to `AdaptiveResearcherAgent.run`.
   A normal graph result also receives the sub-agent's `files` update. If `ainvoke` instead
   raises `TimeoutError` or `GraphRecursionError`, `run()` reconstructs the final-report files
   from the capture before entering the existing partial-result path. A returned `files` delta
   alone is not sufficient on exceptions because the graph result is unavailable.

And one guarantee that the *failure* path stays cheap and terminal:

4. **Bounded, non-retrying failure.** The adapter never raises, so the orchestrator's
   `ToolRetryMiddleware` cannot multiply a failed shallow run; a per-request attempt cap stops
   re-delegation; and once the cap is spent the finalize guard opens so the run ends through the
   normal seam instead of grinding to the turn budget or the workflow deadline. See §5.1 and §8.

---

## 4. Key decisions (and what was rejected)

| Decision | Rationale | Rejected alternative |
| :-- | :-- | :-- |
| Register via `deepagents.CompiledSubAgent` (`{"name", "description", "runnable"}`) | Native deepagents seam; `_get_subagents` uses the runnable as provided (`subagents.py:690-697`), parent callbacks/config propagate automatically, and the parent's `task` tool description advertises it | A bespoke `run_shallow_research` LangChain tool. Works, but bypasses the subagent registry, tracing tags (`ls_agent_type=subagent`), and the streamed `lc_agent_name` |
| Wrap `ShallowResearcherAgent` in a thin **adapter** `RunnableLambda` rather than handing over `agent.graph` | Lets us (a) force the original query, (b) map parent state → `ShallowResearchAgentState`, (c) bridge the citation registry, and (d) capture the post-processed output | Passing `shallow_agent.graph` directly: skips `run()`'s verification/sanitization/emit and gives us no capture seam |
| Treat middleware as the authoritative tier/subagent router | The shared DeepAgents `task` tool advertises every registered subagent. On `single_shot`, middleware rewrites both `subagent_type` and `description`; before tier declaration it rejects delegation, and on other tiers it rejects attempts to invoke `shallow-researcher`. This preserves the other tier contracts even when the model ignores the prompt | Prompt-only routing or log-only handling of a wrong subagent. Neither guarantees the goal |
| Coalesce adapter invocations onto one in-flight task | LangGraph may execute multiple tool calls from one assistant turn concurrently. Hiding `task` on the next turn is too late to prevent duplicate shallow runs | Setting `capture.invoked=True` only after the subagent returns |
| The run owns and cancels the in-flight task | `asyncio.create_task` detaches: cancelling the awaiting coroutine does **not** cancel the task. Without explicit ownership, the workflow deadline in `run()` (`agent.py:572`) returns a partial result while an orphaned shallow run keeps issuing LLM and source calls with no owner | Relying on parent cancellation to propagate; or running the first caller inline (then a cancelled first caller poisons every sibling waiter) |
| The adapter **never raises** — failures become an ordinary return value | The orchestrator stack runs `ToolRetryMiddleware(max_retries=3)` with the default `retry_on=(Exception,)` (`tool_retry.py:133`), and `ToolNode` converts surviving exceptions into error `ToolMessage`s (`langgraph/prebuilt/tool_node.py:753`). An exception is therefore *not* a propagation path here — it is a 4× multiplier on a full shallow research run, for a failure that is usually systematic (dead source, missing key) rather than transient | Raising `EmptySourceRegistryError` out of the adapter and expecting it to surface; or narrowing `retry_on` on the shared orchestrator middleware, which changes retry behaviour for every other tool |
| Failure semantics come from the existing empty-registry gate, not a new policy | After the attempt cap is spent the finalize guard lets `submit_final_report` through with `researched` forced to `True`. With no sources captured, `agent.py:642-654` then raises `EmptySourceRegistryError` — exactly what a failed `single_shot` run does today. Nothing new is invented, and a failed research attempt cannot be relabelled as a deliberate no-research answer | Synthesizing a "no evidence" answer in the adapter (mislabels a failure as an answer); or blocking finalize forever (livelock) |
| Middleware **overrides `submit_final_report`'s `markdown` arg** | The only way to guarantee "final response is the shallow response" without trusting the orchestrator not to paraphrase. `ToolCallRequest.override(tool_call=...)` is a supported, immutable API (`langgraph/prebuilt/tool_node.py:170`) | Trusting a prompt instruction ("echo it verbatim"); or blocking `submit_final_report` outright (leaves the loop with no terminal action) |
| Keep `submit_final_report` as the normal terminal action | It is already `return_direct=True`, writes both files, and logs tier/researched. Normal completion reuses the existing extraction/verification/emit path; `agent.py` changes only to expose and recover a completed capture after top-level timeout/recursion | Terminating from the subagent (no supported seam); a duplicate finalize tool |
| Sub-agent `files` return for normal completion; capture for exception recovery | The `files` channel accepts `{path: FileData}` and `files` is not excluded from CompiledSubAgent state updates. That works when the graph returns normally. Timeout/recursion raises without a result, so the run-scoped capture must be available to `run()` as the separate recovery seam | Assuming a graph-state delta is available after `ainvoke` raises |
| New opt-in config flag, default **off** | Matches how `single_loop_single_shot` and `dynamic_orchestrator_sections` were rolled out | Changing default `single_shot` behaviour |

---

## 5. Component 1 — the compiled sub-agent

**New file:** `src/aiq_agent/agents/adaptive_researcher/subagents/__init__.py`
**New file:** `src/aiq_agent/agents/adaptive_researcher/subagents/shallow.py`

Two things live here: a per-request **capture** object and the **adapter builder**.

```python
"""Shallow-researcher CompiledSubAgent for the adaptive orchestrator's single_shot tier.

The shallow researcher is reused exactly as it ships (`ShallowResearcherAgent`): same graph,
same prompt, same bounded tool loop, same citation post-processing. This module only adapts
its I/O to the DeepAgents subagent contract and captures its normalized, post-processed output
so the adaptive finalize seam can make it authoritative without orchestrator rewriting.
"""

SHALLOW_RESEARCHER_SUBAGENT = "shallow-researcher"

# Hard cap on shallow research attempts per request. Every attempt is a full shallow run, and
# the failures this guards against (an unusable source, a missing API key, a zero-result
# retrieval config) are systematic rather than transient — retrying them buys nothing and costs
# the workflow budget. Once the cap is spent, the delegation guard stops accepting `task` and
# opens the finalize escape hatch (§8), so the run ends through the normal seam.
MAX_SHALLOW_ATTEMPTS = 2


@dataclass
class ShallowSubagentCapture:
    """Per-request result, attempt budget, and at-most-once coordination for the shallow run.

    Created once per adaptive run; separate requests never share one. Calls *within* one request
    may still execute concurrently — ToolNode dispatches a turn's tool calls together — so
    `run_once` coalesces duplicates onto a single shallow execution rather than relying on
    next-turn tool hiding, which lands one turn too late.

    Coordination state is reached only through this class's own methods; callers never touch the
    lock or the task handle directly.
    """

    markdown: str | None = None
    researched: bool = True
    declared_tier: str | None = None  # kept current by delegation middleware, including escalation
    status: Literal["not_started", "running", "completed", "failed"] = "not_started"
    error_type: str | None = None      # metadata only; never retain/log exception text
    attempts: int = 0                  # finished attempts, successful or failed
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _task: asyncio.Task[dict[str, Any]] | None = field(default=None, repr=False)

    @property
    def invoked(self) -> bool:
        """Whether a shallow invocation is running or has succeeded (drives tool hiding)."""
        return self.status in ("running", "completed")

    @property
    def exhausted(self) -> bool:
        """Whether the attempt budget is spent with no usable report (opens the §8 escape hatch)."""
        return self.status != "completed" and self.attempts >= MAX_SHALLOW_ATTEMPTS

    async def run_once(self, factory: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """Execute `factory()` once, sharing its result with concurrently dispatched callers.

        The task is created before the first await so siblings from the same turn find it. It is
        stored on the capture rather than in a closure so `AdaptiveResearcherAgent.run` can cancel
        it: `asyncio.create_task` detaches the coroutine, and cancelling the awaiter does not stop
        the work. A finished-but-failed task is cleared so a later, budget-permitting delegation
        is a genuine new attempt instead of a replay of the stored outcome.
        """
        async with self._lock:
            if self._task is None:
                self.status = "running"
                self._task = asyncio.create_task(factory())
            task = self._task
        try:
            return await task
        finally:
            async with self._lock:
                # `.exception()` is only legal on a finished, non-cancelled task; the awaiter can
                # reach this line while the task is still running (its own cancellation), in which
                # case the handle must stay so `cancel()` can still reach it.
                if self._task is task and task.done() and (task.cancelled() or task.exception()):
                    self._task = None

    def cancel(self) -> None:
        """Cancel an unfinished shallow run. Idempotent, and a no-op once the task has finished."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()


def build_shallow_researcher_subagent(
    *,
    llm_provider: LLMProvider,
    tools: Sequence[BaseTool],
    callbacks: list[Any],
    capture: ShallowSubagentCapture,
    source_registry_middleware: SourceRegistryMiddleware,
    original_query: str | None,
    max_llm_turns: int,
    max_tool_iterations: int,
) -> dict[str, Any]:
    """Build the `shallow-researcher` CompiledSubAgent spec for `create_deep_agent`."""

    # Built once per run, mirroring how the shallow agent is built per request in its own
    # register.py: the active tool set depends on this request's data_sources / MCP tools.
    shallow_agent = ShallowResearcherAgent(
        llm_provider=llm_provider,
        tools=list(tools),
        max_llm_turns=max_llm_turns,
        max_tool_iterations=max_tool_iterations,
        callbacks=callbacks,
    )

    async def _execute_shallow_once(state: dict[str, Any]) -> dict[str, Any]:
        # ---- 1. Input query -------------------------------------------------------------
        # DeepAgents removes the parent's `messages` key and replaces it with one HumanMessage
        # containing the task description. It does not forward parent conversation history.
        # The middleware also overwrites that description with the original query, while this
        # closure remains the authoritative source and the task description is only a fallback.
        query = original_query or _last_human_text(state) or ""

        shallow_state = ShallowResearchAgentState(
            messages=[HumanMessage(content=query)],
            data_sources=state.get("data_sources"),
            user_info=state.get("user_info"),
            available_documents=state.get("available_documents"),
            collection_name=state.get("collection_name"),
        )

        # ---- 2. Share one citation registry ---------------------------------------------
        # Both agents resolve their registry as `get_session_registry() or <own registry>`.
        # Outside the chat path the two would otherwise diverge, so temporarily bind the
        # parent's instance registry. If a chat/session registry already exists, both agents
        # already resolve to that same object and no additional binding is needed.
        token = None
        if get_session_registry() is None:
            token = set_session_registry(source_registry_middleware.registry)
        try:
            result = await shallow_agent.run(shallow_state)
            markdown = str(result.messages[-1].content).strip()
            if not markdown:
                raise ValueError("shallow researcher returned an empty final report")
        except asyncio.CancelledError:
            # Never swallow cancellation — the request is being torn down (workflow deadline or
            # client disconnect). Leave `status="running"` so no partial capture is recoverable.
            raise
        except Exception as exc:
            # Deliberately terminal, and deliberately NOT raised — see §5.1. Record the failure,
            # spend one attempt, and hand the orchestrator a notice it can act on.
            capture.status = "failed"
            capture.error_type = type(exc).__name__
            capture.attempts += 1
            logger.warning(
                "single_shot shallow researcher failed (%s); attempt %d/%d",
                capture.error_type,
                capture.attempts,
                MAX_SHALLOW_ATTEMPTS,
            )
            return {"messages": [AIMessage(content=_failure_notice(capture))]}
        finally:
            if token is not None:
                reset_session_registry(token)

        capture.markdown = markdown
        capture.researched = True
        capture.error_type = None
        capture.attempts += 1
        capture.status = "completed"

        # ---- 3. Return to the parent -----------------------------------------------------
        # `messages` is mandatory. `files` is not excluded by DeepAgents, so it merges into
        # parent state on normal graph completion. Exception recovery uses `capture` directly
        # because an `ainvoke` exception does not return that merged graph state.
        meta = json.dumps(
            {"researched": True, "tier": "single_shot", "source": SHALLOW_RESEARCHER_SUBAGENT}
        )
        return {
            "messages": [AIMessage(content=markdown)],
            "files": {
                FINAL_REPORT_PATH: create_file_data(markdown),
                FINAL_REPORT_META_PATH: create_file_data(meta),
            },
        }

    async def _run_shallow(state: dict[str, Any]) -> dict[str, Any]:
        """Entry point for the `task` tool: at-most-once execution per attempt."""
        return await capture.run_once(lambda: _execute_shallow_once(state))

    return {
        "name": SHALLOW_RESEARCHER_SUBAGENT,
        "description": (
            "Shallow researcher - answers one bounded, factual question with a short, "
            "citation-backed Markdown report. Available only for the single_shot effort tier; "
            "the routing middleware supplies the original user query."
        ),
        "runnable": RunnableLambda(_run_shallow),
    }
```

Notes / gotchas to honour when implementing:

- The module needs explicit `asyncio`, `field`, `Literal`, `Callable`, and `Awaitable` imports
  for the coordination state shown above; keep the public export surface limited to the
  constants, capture, graph-run bundle if colocated, and builder.
- **Async only.** `RunnableLambda` built from a coroutine function supports `ainvoke` but not
  `invoke`. DeepAgents' sync `task` path (`subagents.py::task`) would fail. AI-Q always drives
  the graph through `ainvoke` (`agent.py:573`), so this is fine — but add a sync shim that
  raises a clear `RuntimeError("shallow-researcher subagent requires the async path")` rather
  than letting a confusing LangChain error surface.
- **`create_file_data`** comes from `deepagents.backends.state` (already imported in
  `deep_researcher/deepagents_runtime.py:32`).
- `_last_human_text(state)` is only a fallback for the DeepAgents-created HumanMessage that
  contains `task.description`. Parent messages are **not** passed through: DeepAgents excludes
  the parent `messages` key and replaces it with that one message. The captured original query
  is therefore the authoritative input.
- The shallow agent's `run()` already calls `emit_final_report` on its callbacks; passing the
  parent callbacks means the frontend sees the shallow draft immediately and the adaptive
  layer's final re-emit (`agent.py:683-686`) overwrites it with the verified/sanitized text.

### 5.1 Failure contract — why the adapter never raises

Raising out of the adapter looks like "preserve the standalone `ShallowResearcherAgent`
semantics", but in this stack it does not propagate anywhere useful. Two verified facts:

- The orchestrator middleware stack includes `ToolRetryMiddleware(max_retries=3)`
  (`factory.py:120`) whose default is `retry_on=(Exception,)` (`tool_retry.py:133`). It retries
  on **any** exception.
- `ToolNode` converts a surviving tool exception into an error `ToolMessage` via
  `_default_handle_tool_errors` (`langgraph/prebuilt/tool_node.py:753`) — it does not reach
  `AdaptiveResearcherAgent.run`.

So a raised `EmptySourceRegistryError` costs **four full shallow research runs** and still ends
as an error message to the model. Worse, a `failed` capture leaves `invoked` False, so `task`
becomes visible again and the model can re-delegate — another four runs per attempt — while the
premature-finalize guard blocks the only exit. On the most likely cause (a misconfigured or dead
source, identical on every retry) that consumes the whole request budget before the deadline
produces an expensive partial.

The contract instead is:

1. **`asyncio.CancelledError` always re-raises.** Cancellation is teardown, not failure.
2. **Every other exception is caught**, recorded as `status="failed"` + `error_type` (type name
   only — never the message, which can carry source content or credentials), and spends one
   attempt. The adapter returns a normal result containing only a short `_failure_notice(capture)`
   message: what failed by category, how many attempts remain, and what to do next. No `files`
   are written, so nothing can be mistaken for a report.
3. **`MAX_SHALLOW_ATTEMPTS` bounds re-delegation.** The delegation guard (§8) rejects `task` once
   `capture.exhausted` is true.
4. **The existing empty-registry gate provides the failure semantics.** Once exhausted, the
   finalize guard lets `submit_final_report` through but forces `researched=True`. If the shallow
   attempts captured no sources, `agent.py:642-654` raises `EmptySourceRegistryError` — precisely
   what a failed `single_shot` run does today. If partial evidence *was* captured, the answer is
   verified against it normally. Forcing `researched=True` is what prevents a failed research
   attempt from being relabelled as a deliberate no-research answer; that distinction was the
   original reason not to degrade to `researched=False`, and it is preserved here without
   inventing a new failure policy.

---

## 6. Component 2 — plugging it into the adaptive deep agent

### 6.1 `factory.py::build_adaptive_research_graph`

New keyword arguments: `single_shot_shallow_subagent: bool = False`,
`shallow_subagent_max_llm_turns: int`, `shallow_subagent_max_tool_iterations: int`.
The function now returns an `AdaptiveResearchGraphRun` bundle containing the runnable and its
optional run-scoped capture; §6.4 uses that capture for exception recovery.

```python
@dataclass(frozen=True)
class AdaptiveResearchGraphRun:
    runnable: Any
    shallow_capture: ShallowSubagentCapture | None = None


# --- Shallow-researcher subagent (opt-in, single_shot only) ------------------------------
# Parent-report deltas are excluded here, at the canonical mode definition, so every later
# prompt/tool/middleware branch inherits the safety decision.
shallow_mode = (
    single_shot_shallow_subagent
    and "single_shot" in enabled_tiers
    and not context.parent_report_context_available
)
shallow_capture = ShallowSubagentCapture() if shallow_mode else None

subagents = build_deep_research_subagents(context)
for spec in subagents:
    if spec["name"] == PLANNER_AGENT:
        spec["response_format"] = AdaptiveResearchPlan
if shallow_mode:
    subagents.append(
        build_shallow_researcher_subagent(
            llm_provider=llm_provider,
            # The raw NAT tools (already filtered by data_sources upstream), NOT
            # tool_set.researcher_tools: the requirement is to run the shallow researcher
            # exactly as it runs standalone, where it receives the plain tool list.
            tools=tools,
            callbacks=callbacks,
            capture=shallow_capture,
            source_registry_middleware=source_registry_middleware,
            original_query=_last_human_text_from_state(state),
            max_llm_turns=shallow_subagent_max_llm_turns,
            max_tool_iterations=shallow_subagent_max_tool_iterations,
        )
    )
```

`create_deep_agent(..., subagents=subagents, ...)` needs no `subagents` signature change —
`CompiledSubAgent` and `SubAgent` specs coexist in the same list. Wrap its configured runnable
in `AdaptiveResearchGraphRun(runnable=agent.with_config(...), shallow_capture=shallow_capture)`
instead of returning the bare runnable.

### 6.2 Precedence against `single_loop_single_shot`

Both flags target the same tier. Rule: **shallow subagent wins.**

```python
# Direct source tools are only wired for the legacy inline single_shot loop. With the shallow
# subagent active the orchestrator must not hold source tools at all — the shallow researcher
# owns retrieval on this path.
direct_source_tools = (
    list(context.tool_set.research_source_tools)
    if (single_loop_single_shot and not shallow_mode)
    else []
)
```

Add a `model_validator` on the config (§8) that raises when both flags are `true`, so the
precedence is a config error rather than a silent surprise.

### 6.3 Middleware wiring

`ComplexityRouterMiddleware` gains `shallow_subagent_capture: ShallowSubagentCapture | None`
and is attached whenever `shallow_mode` is on (add `or shallow_mode` to the condition at
`factory.py:384`). Introduce one small `_declared_tiers_in_current_tool_batch(state)` helper
and use it consistently in `ComplexityRouterMiddleware`, `OrchestratorLoopGuardMiddleware`, and
the new delegation middleware whenever a tier affects a sibling tool call. This prevents those
per-request caches from diverging when tool wrappers execute concurrently and ensures none of
them cache conflicting same-turn declarations. The new delegation middleware is appended after
the existing guards:

```python
if shallow_mode:
    orchestrator_middleware = [
        *orchestrator_middleware,
        SingleShotShallowDelegationMiddleware(
            capture=shallow_capture,
            original_query=_last_human_text_from_state(state) or "",
        ),
    ]
```

### 6.4 Exception recovery and cancellation ownership

`AdaptiveResearcherAgent._build_orchestrator_agent` returns the `AdaptiveResearchGraphRun`
bundle. `run()` invokes `built.runnable`.

**Cancellation first.** `asyncio.create_task` detaches the shallow run from its awaiter, so when
the workflow deadline fires (`agent.py:570-579`) the graph invocation is cancelled but the
shallow task is not — it keeps issuing LLM and source-tool calls after the request has already
returned a partial result, owned by nobody and counted by nothing. The run must therefore own it.
Wrap the invocation so cancellation happens on **every** exit, not only the two handled
exceptions (a client disconnect surfaces as `CancelledError` in `register.py:362`, which `run()`
does not catch):

```python
try:
    async with asyncio.timeout(timeout_seconds):
        result = await built.runnable.ainvoke(state, config=...)
    ...
finally:
    # No-op after a completed shallow run; cancels an in-flight one on timeout, recursion
    # abort, client disconnect, or any other error path.
    if built.shallow_capture is not None:
        built.shallow_capture.cancel()
```

**Then recovery.** In both the `TimeoutError` and `GraphRecursionError` handlers, check for a
completed capture whose current `declared_tier` is still `single_shot`. If present, merge
`FINAL_REPORT_PATH` and `FINAL_REPORT_META_PATH` into a copied state and pass that state to the
existing `_build_partial_result` path:

```python
def _state_with_completed_shallow_capture(
    state: AdaptiveResearchAgentState,
    capture: ShallowSubagentCapture | None,
) -> AdaptiveResearchAgentState:
    if (
        capture is None
        or capture.status != "completed"
        or capture.declared_tier != "single_shot"
        or not capture.markdown
    ):
        return state
    files = {
        **state.files,
        FINAL_REPORT_PATH: create_file_data(capture.markdown),
        FINAL_REPORT_META_PATH: create_file_data(
            json.dumps({"researched": capture.researched, "tier": "single_shot"})
        ),
    }
    return state.model_copy(update={"files": files})

# In both exception handlers:
recovery_state = _state_with_completed_shallow_capture(state, built.shallow_capture)
return self._build_partial_result(recovery_state, reason=...)
```

This deliberately does not recover a `running`/`failed` capture, and it does not reuse a
completed shallow report after the orchestrator has escalated to another tier. Add a helper
rather than duplicating the merge in both exception handlers.

### 6.5 Optional shallow-agent change

None is strictly required. One *optional*, backwards-compatible addition that would remove the
contextvar juggling in §5 step 2: add `source_registry: SourceRegistry | None = None` to
`ShallowResearcherAgent.__init__`, and skip the `self.source_registry.clear()` at
`shallow_researcher/agent.py:349` when the registry was injected. Prefer the contextvar bridge
for the first iteration — it keeps `sources/`-adjacent code untouched.

---

## 7. Component 3 — invoking it when `single_shot` is selected

### 7.1 Tool gating (`ComplexityRouterMiddleware._filter_tools`)

Insert a shallow-mode branch **before** the existing `single_loop_single_shot` branch
(`custom_middleware.py:247-267`):

```python
if self._shallow_capture is not None:
    if self._declared_tier == "single_shot":
        # single_shot is delegated wholesale to the shallow researcher:
        #   * hide run_research_batch and every direct source tool (retrieval belongs to
        #     the subagent), and
        #   * hide `task` on later model turns once execution has started/completed.
        # The adapter's in-flight coalescing—not this next-turn filter—is the at-most-once guard
        # for multiple task calls emitted together in one assistant turn.
        hidden = {_RUN_RESEARCH_BATCH_TOOL} | self._direct_source_tool_names
        if self._shallow_capture.invoked:
            hidden.add(_TASK_TOOL)
        return [t for t in tools if _request_tool_name(t) not in hidden]
    # Tier not declared yet, or a non-single_shot tier: fall through to the normal rules.
```

Also fix the shallow-only-preset interaction in `hidden_tools_for_ceiling`
(`custom_middleware.py:95-113`): with `enabled_tiers: [single_shot]` the ceiling is
`single_shot`, which today hides `task` and `write_todos`. Add a
`allow_shallow_subagent: bool = False` parameter that keeps `task` visible (still hiding
`write_todos`), and pass `allow_shallow_subagent=shallow_mode` from the factory. Without this,
the shallow-only fast-lane config — the most likely place to enable this feature — would hide
the very tool that invokes the sub-agent.

This filter controls which tool names the model sees, but cannot select among the subagent
names advertised inside the shared `task` tool. `SingleShotShallowDelegationMiddleware` (§8)
is therefore the correctness boundary for tier-aware `subagent_type` enforcement. Sibling tool
wrappers may run in either order, so the middleware derives an **effective tier** from the
current AIMessage's tool-call batch before consulting its historical cache. A same-turn
`declare_effort_tier` + `task` is therefore routed consistently in either scheduling order;
a task with no declaration in the current or an earlier turn is rejected.

### 7.2 Prompt (`prompts/orchestrator.j2`)

Add a third branch to the `single_shot` workflow block (currently lines 169-183), rendered when
a new `single_shot_shallow_subagent` template variable is true:

```jinja
{% if single_shot_shallow_subagent -%}
**Delegated path** (the `shallow-researcher` subagent performs the research):
1. Do not search yourself and do not call `run_research_batch` — neither is available here.
2. Call `task` exactly once with `subagent_type="shallow-researcher"` and `description` set to
   the user's question **verbatim** (do not rewrite, summarise, or add instructions). The runtime
   enforces both arguments, so these instructions optimize model behavior rather than provide
   the correctness guarantee.
3. The subagent returns a complete, cited Markdown answer. Pass it through **unchanged**:
   `submit_final_report(markdown=<the subagent's answer>, researched=true, tier="single_shot")`.
   Do not edit, re-order, shorten, or re-cite it. The runtime replaces these finalizer arguments
   with the captured values even if you do.
{%- elif single_loop_single_shot -%}
... existing direct-tool path ...
```

Render it from `_orchestrator_render_kwargs` (`factory.py:301-328`) alongside the existing
`single_loop_single_shot` key.

Watch the Jinja whitespace control when inserting this branch. The surrounding block already
mixes `{% ... -%}` and `{%- ... %}`, and a misplaced hyphen changes the **flag-off** render by a
newline. That is not cosmetic here: the dynamic-sections work depends on each mode rendering a
byte-stable prefix for prompt KV-cache reuse, and the existing prompt tests assert flag-off
output. Diff the flag-off render before and after (§11 test 20).

**Dynamic-sections interaction:** `SECTION_PRESETS["single_shot"]` (`tiers.py:191-205`) does
*not* include the `subagents` section, so a `dynamic_orchestrator_sections` run would trim away
the block that enumerates subagents. That is fine **because the instructions above live in the
`workflow` section**, which is on for `single_shot`, and they name the subagent explicitly.
Keep them there; do not move them into `## Subagents`. Add a test asserting the rendered
`single_shot` prompt contains `shallow-researcher`.

### 7.3 Why the orchestrator still gets a turn at all

An alternative would be short-circuiting the graph the moment `declare_effort_tier("single_shot")`
fires. There is no supported middleware seam to terminate a DeepAgents run from a tool call, and
`submit_final_report`'s `return_direct=True` already collapses the tail to a single cheap turn
whose content we overwrite anyway. Cost is one extra orchestrator turn; robustness is worth it.

---

## 8. Component 4 — response propagation (the core requirement)

**New middleware:** `SingleShotShallowDelegationMiddleware` in `custom_middleware.py`.
It receives both the capture and the original query. Returning an error `ToolMessage` follows
the existing `OrchestratorLoopGuardMiddleware` pattern and keeps rejected calls inside the
agent loop without executing the wrong subagent or finalizer.

```python
class SingleShotShallowDelegationMiddleware(AgentMiddleware):
    """Make shallow delegation and finalization deterministic for single_shot.

    Responsibilities:
    1. Resolve the effective tier from a declaration in the current tool-call batch, falling
       back to the cached prior-turn tier; reject conflicting same-turn declarations.
    2. Cache every accepted declared tier in both middleware and capture (including escalation).
    3. Reject `task` when no tier is known. For `single_shot`, force the task subtype and
       description to `shallow-researcher` and the original query, and reject further delegation
       once the attempt budget is spent. On every other tier, reject an attempted shallow subtype
       while leaving existing planner/writer/router calls alone.
    4. Reject premature `single_shot` finalization while a shallow attempt is still viable.
    5. Replace the accepted finalizer's markdown/researched/tier arguments with captured values —
       or, when the attempt budget is spent, let the finalizer through with `researched=True` so
       the existing empty-registry gate decides the outcome (§5.1). Rejecting forever would
       livelock the run: with `task` exhausted and finalize blocked there is no terminal action
       left, and only the turn budget or the workflow deadline would end it.
    """

    def _effective_tier(self, request) -> tuple[str | None, bool]:
        """Return (tier, conflict) using only the current AI tool-call batch plus cache.

        Historical message scanning remains unnecessary/unreliable. The current ToolNode state
        does contain the AIMessage whose sibling tool calls are being executed, so this makes
        declare+task and escalation+finalize independent of coroutine scheduling order.
        """
        declared_here = _declared_tiers_in_current_tool_batch(request.state)
        if len(declared_here) > 1:
            return None, True
        if declared_here:
            return next(iter(declared_here)), False
        return self._declared_tier, False

    def _blocked(self, request, message: str) -> ToolMessage:
        tool_call = request.tool_call
        return ToolMessage(
            content=message,
            tool_call_id=tool_call.get("id", "shallow-delegation-guard"),
            name=tool_call.get("name", "task"),
            status="error",
        )

    async def awrap_tool_call(self, request, handler):
        name = _request_tool_call_name(request)
        args = dict(request.tool_call.get("args") or {})
        effective_tier, conflicting_declarations = self._effective_tier(request)
        if conflicting_declarations:
            return self._blocked(
                request,
                "Conflicting effort tiers were declared in one turn. Declare exactly one tier "
                "before continuing.",
            )

        if name == _DECLARE_EFFORT_TIER_TOOL:
            tier = args.get("tier")
            if isinstance(tier, str) and tier:
                self._declared_tier = tier
                self._capture.declared_tier = tier
            return await handler(request)

        if name == _TASK_TOOL:
            if effective_tier is None:
                return self._blocked(request, "Declare the effort tier before delegating.")
            requested_type = args.get("subagent_type")
            if effective_tier == "single_shot":
                if self._capture.exhausted:
                    # Attempt budget spent. Stop re-delegating (each attempt is a full shallow
                    # run) and point the model at the now-open finalize path.
                    return self._blocked(
                        request,
                        "The shallow researcher could not complete this request after "
                        f"{MAX_SHALLOW_ATTEMPTS} attempts. Do not delegate again; call "
                        "submit_final_report with whatever the gathered evidence supports.",
                    )
                args["subagent_type"] = SHALLOW_RESEARCHER_SUBAGENT
                args["description"] = self._original_query
                request = request.override(tool_call={**request.tool_call, "args": args})
                return await handler(request)
            if requested_type == SHALLOW_RESEARCHER_SUBAGENT:
                return self._blocked(
                    request,
                    "shallow-researcher is available only on the single_shot tier; follow the "
                    "declared tier's existing workflow.",
                )
            return await handler(request)

        if name == _FINALIZE_TOOL and effective_tier == "single_shot":
            if self._capture.status != "completed" or not self._capture.markdown:
                if not self._capture.exhausted:
                    # A shallow attempt is still viable: make the model delegate first.
                    return self._blocked(
                        request,
                        "The single_shot shallow researcher has not completed. Call task with the "
                        "shallow-researcher before finalizing.",
                    )
                # Escape hatch (§5.1). The budget is spent and there is no report to enforce, so
                # let the finalizer run — but force researched=True so a failed research attempt
                # cannot be recorded as a deliberate no-research answer. With no sources captured
                # this reaches the existing EmptySourceRegistryError path, exactly as a failed
                # single_shot run does today.
                args["researched"] = True
                args["tier"] = "single_shot"
                logger.warning(
                    "single_shot: shallow researcher exhausted after %d attempt(s) (last error: "
                    "%s); allowing orchestrator-authored finalize",
                    self._capture.attempts,
                    self._capture.error_type or "unknown",
                )
                return await handler(request.override(tool_call={**request.tool_call, "args": args}))
            if args.get("markdown", "").strip() != self._capture.markdown:
                logger.info(
                    "single_shot: replacing orchestrator-authored final report (%d chars) with "
                    "the shallow-researcher report (%d chars)",
                    len(args.get("markdown", "")),
                    len(self._capture.markdown),
                )
            args["markdown"] = self._capture.markdown
            args["researched"] = self._capture.researched
            args["tier"] = "single_shot"
            request = request.override(tool_call={**request.tool_call, "args": args})

        return await handler(request)
```

`_declared_tiers_in_current_tool_batch` reads only the last AIMessage in the current
ToolNode state, collects non-empty `tier` strings from `declare_effort_tier` calls, and never
scans historical messages. It must handle dict and Pydantic state shapes. The same helper is
used by every tier-aware middleware noted in §6.3.

`ToolCallRequest.override(...)` is the supported immutable mutation API
(`langgraph/prebuilt/tool_node.py:170-178`); direct attribute assignment is deprecated.

### 8.1 The full propagation chain

| Step | Where | What carries the answer |
| :-- | :-- | :-- |
| 1 | delegation middleware | Forces `task.subagent_type` and `description`; rejects premature finalization |
| 2 | shallow `run()` | Returns citation-verified and sanitized `messages[-1].content` |
| 3 | adapter | Stores normalized Markdown in `capture`; returns the same text in `messages` plus final-report `files` |
| 4a | normal graph path | DeepAgents merges `files`; finalizer args are overwritten from `capture`; `backend.upload_files` writes the authoritative report |
| 4b | timeout/recursion path | `run()` reads the completed run-scoped capture and reconstructs the two final-report files in a copied state |
| 5 | adaptive post-processing | Existing citation verification, sanitization, callback emission, message replacement, and optional artifact processing run normally |
| 6 | logging | `_read_tier` returns `single_shot` from the effort-tier file; escalation prevents capture recovery/override |
| — | failure path | No shallow report exists to carry. After `MAX_SHALLOW_ATTEMPTS` the finalize guard opens with `researched=True` forced, and the existing empty-registry gate decides the outcome (§5.1) |

### 8.2 Output identity contract

"Authoritative" means the orchestrator cannot alter the shallow report. The capture stores
`ShallowResearcherAgent.run()`'s already verified/sanitized content after leading/trailing
whitespace normalization, and both the finalizer and exception-recovery path consume that
captured value. It does **not** mean byte identity with the shallow message at every later
boundary: `submit_final_report` normalizes outer whitespace, and the adaptive layer deliberately
re-verifies citations, sanitizes, and may resolve or append sandbox artifacts. Tests should
assert equality at the capture/finalizer boundary and the expected normalized/post-processed
answer at the public return boundary.

### 8.3 Citation registry bridge

Without the registry bridge in §5 step 2, retrieval happens entirely inside the shallow
sub-graph, whose `tool_node_with_source_capture` writes to *its* registry
(`shallow_researcher/agent.py:292-318`), while `AdaptiveResearcherAgent.run` verifies against
`SourceRegistryMiddleware.active_registry()`. The adaptive side would see zero sources and
`agent.py:642-654` would raise `EmptySourceRegistryError` on a perfectly good answer.

The bridge (binding the session registry to `source_registry_middleware.registry` for the
duration of the sub-run) makes both sides use one registry, so:

- the shallow agent's own `verify_citations` works as it does standalone;
- the adaptive re-verification (`agent.py:617-641`) runs against a registry that already
  contains every source cited and is expected to be citation-idempotent; the public-output
  contract still permits its deterministic normalization;
- `get_source_entries(mode="compact")` falls back to all sources when
  `_compact_source_keys` is empty (`custom_middleware.py:663-669`), so the reference list is
  still correct without calling `register_research_note_sources`.

Belt-and-suspenders alternative if the contextvar bridge proves awkward under Dask/threads:
after the sub-run, call
`source_registry_middleware.register_compact_sources(shallow_registry.all_sources())`
(`custom_middleware.py:566-577`) to copy entries across. Implement the bridge; keep this
documented as the fallback.

---

## 9. Config surface

`register.py` — `AdaptiveResearchAgentConfig`:

```python
single_shot_shallow_subagent: bool = Field(
    default=False,
    description=(
        "Route the single_shot tier to the shallow_research_agent, wired into the adaptive "
        "graph as a DeepAgents CompiledSubAgent. Tier selection is unchanged; when single_shot "
        "is declared the orchestrator delegates the original user query to the shallow "
        "researcher via `task` and prevents orchestrator rewriting of its report. "
        "Mutually exclusive with "
        "single_loop_single_shot. Off by default for rollout safety."
    ),
)
shallow_subagent_max_llm_turns: int = Field(default=10, ge=1, description="...")
shallow_subagent_max_tool_iterations: int = Field(
    default=5, ge=1,
    description="Bounded tool-call budget inside the shallow subagent (its own loop guard).",
)

@model_validator(mode="after")
def _validate_single_shot_mode(self):
    if self.single_shot_shallow_subagent and self.single_loop_single_shot:
        raise ValueError(
            "single_shot_shallow_subagent and single_loop_single_shot both control the "
            "single_shot execution path; enable at most one."
        )
    return self
```

Thread through: `register.py` (both `AdaptiveResearcherAgent(...)` construction sites —
build-time at `register.py:275` and the per-request rebuild at `register.py:315`) →
`agent.py::__init__` (store on `self`) → `agent.py::_build_orchestrator_agent`
(`agent.py:237-259`) → `factory.py::build_adaptive_research_graph`.

Keep the already-declared-but-unused `single_shot_researcher_llm` field out of this change.
A follow-up may use it for the compiled shallow agent, but must pass a dedicated shallow-only
provider/model override. Reconfiguring `LLMRole.RESEARCHER` on the shared provider would also
change the `standard` and `deep` researcher model and violate this plan's non-goals.

Example config (`configs/config_adaptive_shallow_subagent.yml`):

```yaml
workflow:
  _type: adaptive_research_agent
  enabled_tiers: [direct, single_shot, standard, deep]
  single_shot_shallow_subagent: true
  single_loop_single_shot: false
  shallow_subagent_max_tool_iterations: 5
```

---

## 10. Edge cases and failure modes

| Case | Handling |
| :-- | :-- |
| Orchestrator supplies a paraphrased final report | Finalizer args are overwritten with the capture; only length metadata is logged |
| Orchestrator finalizes before shallow research completes | Delegation middleware returns an error `ToolMessage`; the finalizer is not executed and `task` remains available. The rejection is conditional on the attempt budget still allowing a shallow run — see the exhausted row below |
| Model chooses the wrong subagent on `single_shot` | Middleware overwrites `subagent_type` and `description` with `shallow-researcher` and the original query before execution |
| Model chooses `shallow-researcher` on another tier | Middleware rejects only that subtype; existing planner/writer/source-router delegation remains unchanged |
| Model emits `task` concurrently with one tier declaration | Effective tier is read from the current tool-call batch, so routing is identical regardless of wrapper scheduling order |
| Model emits `task` with no current or cached declaration | Task is rejected without invoking any subagent |
| Model emits conflicting tier declarations in one turn | Delegation/finalization is rejected with a corrective error; no ambiguous tier-dependent action executes |
| Model emits two or more `task` calls in one turn | `capture.run_once` shares a single task; all callers receive the same result and only one shallow run executes. Next-turn filtering then hides `task` |
| Normal graph completion without `submit_final_report` | CompiledSubAgent `files` merge into the returned graph state and `_resolve_output_file_markdown` finds the report |
| Timeout/recursion after shallow completion | The returned-files delta is unavailable, so `run()` reconstructs the files from the completed capture and uses `_build_partial_result` |
| Timeout/cancellation while shallow is still running | Do not use an incomplete capture; preserve the existing deterministic partial-result behavior. `run()`'s `finally` calls `capture.cancel()` so the detached shallow task stops instead of running on unowned after the request returns (§6.4) |
| Shallow run finds no sources (`EmptySourceRegistryError`) | Caught, **not raised** (§5.1): raising would be retried 3× by `ToolRetryMiddleware` — three more full shallow runs — and would still end as an error `ToolMessage`. The capture records `status="failed"` + the error type and spends an attempt; `researched=False` is still never used, so the deliberate-no-research meaning is preserved |
| Other shallow exception | Same path: exception type only (never the message), one attempt spent, failed task cleared so a budget-permitting re-delegation is a genuine new attempt |
| Shallow attempt budget exhausted | `task` is rejected with a corrective message and the finalize guard opens with `researched=True` forced, so the run ends through the normal seam. With no sources captured this lands on the existing `EmptySourceRegistryError` path — same outcome a failed `single_shot` run has today — rather than livelocking until the turn budget or deadline |
| `asyncio.CancelledError` inside the adapter | Always re-raised; never converted into a failure notice or an attempt. Status stays `running`, so nothing partial is recoverable |
| Orchestrator escalates to `standard`/`deep` after shallow output | Tier recapture updates `capture.declared_tier`; finalizer override and exception recovery require it still equal `single_shot`, so the shallow result is unused |
| Parent-report delta request | The canonical `shallow_mode` expression excludes delta before subagent/prompt/middleware construction |
| `request_termination` budgets | Existing single-shot request budgets do not count nested subagent work. The shallow loop guard and workflow deadline still bound it; adding a dedicated task-invocation budget is a follow-up |
| Sandbox/artifact post-processing | Remains enabled after extraction and may deterministically change the public report; this is explicitly allowed by the output identity contract |

---

## 11. Tests

Add to `tests/aiq_agent/agents/adaptive_researcher/`:

`test_shallow_subagent.py` (new)

1. Builder returns a valid CompiledSubAgent spec with `name`, `description`, and `runnable`.
2. Adapter uses the captured original query rather than an untrusted task description.
3. Successful output contains `messages`, `FINAL_REPORT_PATH`, and the meta sidecar.
4. Sources captured by the shallow agent are visible through the adaptive registry bridge.
5. Two concurrent adapter invocations execute `ShallowResearcherAgent.run()` exactly once and
   receive the same captured result/files.
6. `EmptySourceRegistryError` does **not** escape the adapter: the call returns a failure notice
   with no `files`, sets `status="failed"` and `error_type`, spends exactly one attempt, and
   leaves the slot retryable. Assert with a `ShallowResearcherAgent.run` stub that counts calls
   that a raising shallow agent produces exactly one run per delegation — not four — proving
   `ToolRetryMiddleware` is never triggered.
7. Empty final content is rejected rather than producing an empty capture.
8. After `MAX_SHALLOW_ATTEMPTS` failures, `capture.exhausted` is true and no further shallow run
   executes.
9. `asyncio.CancelledError` from the shallow agent propagates unchanged, spends no attempt, and
   leaves `status="running"`.
10. `capture.cancel()` cancels an in-flight task and is a no-op on a completed one; a cancelled
    awaiter does not leave a detached task running (assert `task.cancelled()` after teardown).

`test_custom_middleware.py` (extend)

11. The shared current-batch tier helper handles dict/Pydantic state; all tier-aware middleware
    agree under reversed sibling-wrapper ordering and ignore conflicting declarations.
12. An accepted `declare_effort_tier` updates middleware and capture, including escalation.
13. A task with no current/cached declaration is rejected; declare+task produces identical
    routing under both simulated sibling-wrapper orders; conflicting declarations are rejected.
14. On `single_shot`, wrong/missing task arguments are overwritten with the shallow subtype and
    exact original query.
15. On other tiers, shallow delegation is rejected while planner/writer delegation is untouched.
16. Premature `single_shot` finalization is rejected while attempts remain and the capture is
    absent/running/failed.
17. **Escape hatch:** with `capture.exhausted`, `task` is rejected *and* `submit_final_report` is
    allowed through with `researched=True` forced and `markdown` left as the model wrote it.
    Assert the pair together — a test that only checks the rejection would pass on a livelocking
    implementation.
18. Completed `single_shot` finalizer args are overwritten from the capture; other tiers are
    untouched, and same-turn or prior-turn escalation disables the override regardless of
    sibling-wrapper order.
19. `_filter_tools` hides batch/source tools on shallow mode and hides `task` on later turns.
20. `hidden_tools_for_ceiling(..., allow_shallow_subagent=True)` preserves only `task`, not
    `write_todos`, for a shallow-only ceiling.

`test_factory.py` (extend)

21. `shallow_mode` requires the flag, an enabled `single_shot` tier, and no parent-report delta.
22. The returned `AdaptiveResearchGraphRun` exposes the same capture used by middleware/adapter.
23. Direct source tools are absent in shallow mode.
24. Full and dynamic single-shot prompt renders name `shallow-researcher`; the flag-off render is
    compared **byte-for-byte** against the pre-change output (§7.2 whitespace-control risk).
25. Standard/deep tool routing still permits planner/writer and rejects shallow only through the
    new middleware.

`test_register.py` (extend)

26. Config validation rejects both single-shot modes enabled together and permits both disabled.
27. Both `AdaptiveResearcherAgent` construction sites forward all new fields.

`test_agent.py` (extend)

28. Normal graph completion without `submit_final_report` returns the report from merged files.
29. Timeout and recursion exceptions after a completed, still-single-shot capture reconstruct the
    report and enter the existing verified/sanitized partial-result path.
30. Running, failed, empty, or escalated captures are not recovered after an exception.
31. `capture.cancel()` is invoked on **every** exit path — normal completion, `TimeoutError`,
    `GraphRecursionError`, and an arbitrary exception — so the `finally` placement in §6.4 cannot
    regress to per-handler cleanup.
32. Capture/finalizer equality and public normalized/post-processed output match the §8.2 contract.

Add one integration-style test using the real DeepAgents CompiledSubAgent task contract (with
stubbed LLM/tool behavior) so mocks cannot conceal changes to `_EXCLUDED_STATE_KEYS`, `Command`
state merging, or async `ainvoke` selection.

Local environment limitations are preflight constraints, not acceptance criteria. If `annoy`
prevents local pytest collection, record that limitation and run `python -m py_compile` plus
focused static checks for quick feedback; the PR must still run and pass the targeted pytest
suite in CI or another build-capable environment before merge.

---

## 12. File-by-file change list

| File | Change |
| :-- | :-- |
| `adaptive_researcher/subagents/__init__.py` | **new** — export `SHALLOW_RESEARCHER_SUBAGENT`, `ShallowSubagentCapture`, `build_shallow_researcher_subagent` |
| `adaptive_researcher/subagents/shallow.py` | **new** — capture (coalescing, attempt budget, cancellation) + adapter builder with the never-raise failure contract (§5, §5.1) |
| `adaptive_researcher/custom_middleware.py` | authoritative tier/subagent routing, attempt-capped delegation, premature-finalize guard **plus its exhausted-budget escape hatch**, finalizer override, shallow branch in `_filter_tools`, and ceiling exception for `task` |
| `adaptive_researcher/factory.py` | canonical delta-safe `shallow_mode`; append CompiledSubAgent; return `AdaptiveResearchGraphRun`; attach middleware; new render kwarg |
| `adaptive_researcher/prompts/orchestrator.j2` | third `single_shot` branch (delegated path; runtime enforcement remains authoritative) |
| `adaptive_researcher/agent.py` | store/forward config; invoke the graph bundle; cancel the in-flight shallow task in a `finally` on every exit path; recover completed single-shot captures on timeout/recursion |
| `adaptive_researcher/register.py` | three config fields, the mutual-exclusion validator, and forwarding at **both** construction sites |
| `configs/config_adaptive_shallow_subagent.yml` | **new** example config |
| `tests/aiq_agent/agents/adaptive_researcher/*` | §11 |
| `docs/source/...` | document the new flag alongside `single_loop_single_shot` |

Deliberately **unchanged**: `shallow_researcher/` (unless the optional §6.5 constructor arg is
taken), `tiers.py`, `tools/finalize.py`, `models/state.py`, and the whole `deep_researcher/`
package.

---

## 13. Rollout and validation

1. Land behind `single_shot_shallow_subagent: false`; default-path behavior remains unchanged.
2. Run the targeted unit and integration tests in §11, then `uv run ruff check .` and
   `uv run ruff format --check .`. Local static checks never substitute for the pytest gate.
3. Smoke with the explicit example config. Assert one underlying shallow execution, captured
   tier `single_shot`, authoritative finalizer replacement, valid citations, and final output.
4. Fault-injection smoke: after shallow completion, force both timeout and recursion exits and
   confirm capture recovery. Cancel during shallow execution and confirm incomplete output is not
   returned **and that the detached shallow task actually stops** (no source-tool calls in the
   logs after the request returns). Then break retrieval deliberately (revoke the search key) and
   confirm the run makes exactly `MAX_SHALLOW_ATTEMPTS` shallow attempts, ends through the
   finalize escape hatch, and surfaces the existing empty-registry failure — rather than looping
   to the turn budget or the workflow deadline. Compare the wall-clock and token cost of that
   broken-source run against a flag-off single_shot failure; it should not be materially worse.
5. Regression-smoke `direct`, `standard`, `deep`, and parent-report delta requests with the flag
   enabled. Confirm shallow delegation is rejected outside `single_shot`, while existing
   planner/writer routing and delta safety remain intact.
6. A/B FreshQA flag-off vs. flag-on. Compare accuracy, token usage, and latency against both the
   existing `single_loop_single_shot` path and standalone `shallow_research_workflow`; standalone
   shallow quality is the target.
7. Flip the default only if quality holds, cost/latency improve, and the failure-path tests pass.

---

## 14. Resolved review decisions and follow-ups

1. **Query scope for iteration one:** send only the current original user turn. DeepAgents does
   not forward parent messages, and this adapter intentionally does not add conversation history.
   Include ambiguous follow-up queries in evaluation; adding explicit history is a separate,
   evidence-driven change because it changes standalone-shallow parity and token cost.
2. **Double verification:** retain it. It preserves one adaptive-agent output contract and is
   part of the explicitly allowed post-processing boundary. Consider a skip marker only if
   profiling demonstrates material cost and equivalence tests prove it safe.
3. **Request budgets:** the workflow deadline plus the shallow agent's own loop bounds are the
   initial hard limits. A dedicated `task`-invocation/subagent budget in
   `request_termination` is a follow-up, informed by traces. For now three narrower mechanisms
   bound the shallow path: at-most-once coalescing (no parallel multiplication),
   `MAX_SHALLOW_ATTEMPTS` (no serial re-delegation), and the never-raise contract (no
   `ToolRetryMiddleware` multiplication). If traces show the cap being hit routinely, that is a
   signal about source configuration, not a reason to raise it.
4. **`single_shot_researcher_llm`:** wire it only in a follow-up through a shallow-only provider
   or direct model override. Never mutate the shared `LLMRole.RESEARCHER` binding, which would
   change `standard` and `deep` behavior.
