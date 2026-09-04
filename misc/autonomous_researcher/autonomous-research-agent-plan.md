# Autonomous Research DeepAgent — Additive Design & Implementation Plan

Adding an autonomous, description-driven LangChain DeepAgent alongside the existing
`deep_research_agent` and tier-based `adaptive_research_agent` architectures.

---

## 0. Baseline — post-rebase state

This plan is written against `dev/smasurekar/aiq-autonomous-agent` **rebased onto
`upstream/develop` (AI-Q 2.2.0)**. All 17 branch commits replayed; three conflicts were resolved
(`common/callbacks.py`, `deploy/Dockerfile` + `pyproject.toml`, `deep_researcher/tools/research.py`
+ its test). The rebase changed several facts this plan depends on — they are folded in below, but
the headline items are:

| Change | Effect on this plan |
| :-- | :-- |
| `sources/gsf`, `sources/you_com`, `sources/nimble_web_search` now carry real source | The "broken packages" caveat is **retired**. Nine source packages are live; the tool inventory in §5.4 grows accordingly. |
| `DeepResearchGraphContext` gained 3 required fields — `resource_limits`, `final_report_tracker`, `state_budget` | The new `autonomous_researcher/factory.py` **must** supply these. See §8. |
| New `deep_researcher/resource_limits.py` | A tier-independent hard-limits layer already exists upstream. The "flattened `request_termination`" work should **reuse** it, not reinvent it. See §3.3. |
| New `common/logging_utils.log_content_metadata` | Prompts/responses/tool payloads must never be logged raw. See §3.3. |
| New `src/aiq_agent/guardrails/` with `deep_agent` / `shallow_agent` / `workflow` adapters | The autonomous researcher needs a guardrails path; there is no `autonomous` adapter yet. See §3.3. |
| New `runtime-tools` dependency group in `pyproject.toml` | Any new source package the autonomous agent ships with must be added there to reach the container image. |

**Prerequisites — RESOLVED.** The baseline had 38 failures; all are fixed and
`uv run pytest tests/aiq_agent/agents/adaptive_researcher/ tests/aiq_agent/agents/deep_researcher/`
now reports **854 passed, 6 skipped**.

1. **31 failures** — `adaptive_researcher/factory.py` called `DeepResearchGraphContext(...)` without
   the three new upstream fields (`TypeError: ... missing 3 required positional arguments`). Rebase
   fallout: git merged the file textually but the upstream API moved underneath it. **Fixed** by
   mirroring `deep_researcher/factory.py:578-600` — `build_adaptive_research_graph` gained optional
   `resource_limits` / `final_report_tracker` / `state_budget` parameters that default to one
   `DeepResearchResourceLimits()`, one `FinalReportCommitTracker()`, and a
   `StateBudgetLedger(limits=limits, files=state.files, sandbox_enabled=runtime.execution_enabled)`
   per graph build (i.e. per request).
2. **7 failures** — `tests/.../test_orchestrator_loop_guard.py` built configs whose `standard`
   budgets exceeded the fixture's `deep` budgets, violating
   `AdaptiveRequestTerminationConfig._validate_relationships` ("escalating effort must not reduce
   any limit"). Pre-existing bug in branch commit `69e5b47`. **Fixed** in the `_cfg` helper, which
   now widens `deep` element-wise to stay ≥ `standard` unless a test sets `deep` explicitly —
   preserving the intent of all 7 tests without touching their call sites.
3. **1 further failure, unmasked by fix 1** — `test_factory.py` asserted a 4-tool orchestrator list
   that predated `declare_effort_tier`. It had been hidden behind the `TypeError`. **Fixed** by
   adding `declare_effort_tier` to the expected list.

> **Noted, not blocking — this is a POC branch and no PR is planned, so CI is not a gate.**
> Recorded only so the failures are not mistaken for regressions introduced by this work.
> The full suite has 4 remaining failures in
> `tests/aiq_agent/test_default_model_profiles.py`. Two come from untracked local scratch configs
> and vanish when those are removed. The other two are real: upstream 2.2.0 added
> `test_deprecated_model_and_endpoint_references_are_absent`, which flags
> `base_url: https://inference-api.nvidia.com/v1` — used by every branch config
> (`config_adaptive_frag*.yml`, `config_adaptive_shallow_subagent.yml`, `config_shallow_frag.yml`,
> and the FreshQA variants). Upstream now treats that endpoint as deprecated. Resolving it means
> either repointing the configs (which changes the endpoint evals actually hit) or scoping them out
> of the test — a call for the branch owner. **`configs/config_autonomous_researcher.yml` (§8) must
> not reintroduce the deprecated endpoint.**
>
> Separately, `ruff check` reports one E501 in `adaptive_researcher/tools/research.py:61`
> (121 > 120) — pre-existing branch code, would fail the CI gate.

---

## 1. Problem

`adaptive_research_agent` encodes adaptivity as an **enumerated tier system**. Each request is
classified into one of `direct | single_shot | standard | deep`, and that single label then drives
four separate machines:

| Machine | Where | What the tier controls |
| :-- | :-- | :-- |
| Prompt text | `prompts/orchestrator.j2` + `tiers.py:TIER_PROFILES` | Which `### <tier>` procedure blocks render, plus a per-tier "when / planner / writer / width / tools / finalize" table |
| Prompt assembly | `tiers.py:SECTION_PRESETS`, `sections_for_tier`, `sections_for_catalog` | Which of 16 `SECTION_FLAGS` render, swapped mid-run by `ComplexityRouterMiddleware` |
| Tool exposure | `custom_middleware.py:hidden_tools_for_ceiling`, `ComplexityRouterMiddleware._filter_tools` | Hides `advanced_web_search_tool`, `task`, `write_todos`, `run_research_batch`, or the source tools by tier |
| Runtime budgets | `models/request_termination.py:budgets_for_tier` | Per-tier `max_batch_calls` / `max_total_research_queries` / `max_orchestrator_turns` |

This is predictable, and that predictability is why it exists. But it is **not generic**:

1. **Adding a capability means editing the tier tables.** A new source tool must be reasoned about
   against `TierProfile.tools` (`"basic web / knowledge retrieval"`, `"full set incl. advanced web
   search"`) *and* against `hidden_tools_for_ceiling`, which hides `advanced_web_search_tool` by
   literal name. There is no path where a tool is described richly and simply picked up. This is
   most acute for **per-user MCP tools**, which arrive at request time carrying only a description
   and no tier table to be added to.
2. **The tier vocabulary redundantly re-describes the tools.** The prompt explains what `standard`
   means instead of letting tool and subagent descriptions say what they are for.
3. **Combinatorial branching.** `single_shot` alone has three mutually-exclusive execution paths
   (`run_research_batch` / `single_loop_single_shot` / `single_shot_shallow_subagent`), each forking
   the prompt, the tool filter, the `TierResolver` compatibility matrix, and the finalize override.
4. **Three runtime sources of truth for the tier** — `TierResolver._tier`, each middleware's own
   `_declared_tier` cache, and `/shared/effort_tier.json` — plus an inference fallback that logs at
   WARNING whenever the model fails to declare.
5. **Known dead weight**: `single_shot_researcher_llm` is declared but unused
   (`register.py:193-201`); `prompts/source_registry.j2` is never loaded (`agent.py:250`).

Scale of the machinery: `tiers.py` 325 lines · `custom_middleware.py` 1,565 lines ·
`orchestrator.j2` 360 lines across 16 gated sections · `factory.py` 544 lines.

## 2. Goal

Add a description-driven `autonomous_research_agent` alongside the existing deep and adaptive agents.
Do not change default routing until evaluation establishes a reason to do so.

Within the new sibling agent, use *emergent* rather than *declared* adaptivity, in the shape the
`deepagents` library itself uses: **one research system prompt + richly-described tools +
richly-described subagents**, with the orchestrator choosing depth as an ordinary reasoning step
rather than as a classification step.

"Simple question → one search → answer" and "complex question → plan, fan out, write" become
behaviors of one undifferentiated loop rather than named modes.

**Explicit non-goal:** beating the tier agent on the first eval run. The goal is an architecture
where capability is added by **writing a description**, not by editing a table.

## 3. Decisions

| | Decision |
| :-- | :-- |
| **Delivery** | **New sibling package** `src/aiq_agent/agents/autonomous_researcher/`, `_type: autonomous_research_agent`. `adaptive_researcher/` untouched, so both run side-by-side on FreshQA / DeepSearchQA. |
| **Retrieval** | **Full menu, always visible.** The orchestrator holds *all* source tools directly, *and* `run_research_batch`, *and* `task()` into every subagent. Nothing is hidden by anything. It decides how to research from the descriptions alone. |
| **Subsystems** | Keep `request_termination` (flattened), sandbox + skills, and parent-report delta mode. Citation verification kept throughout. |
| **Exit contract** | **API- and report-compatible with the existing agents** — see §3.1. |
| **`general-purpose` subagent** | **Suppressed — no usable general-purpose delegation path.** The deepagents default is unsafe here (inherits `submit_final_report` and `run_research_batch`, bypasses `SourceRegistryMiddleware`, and advertises itself for research). Only `researcher`, `planner`, and `writer` can retrieve, research, or finalize. See §4.2.1 / §4.2.2. |

### 3.1 The API and report contract is invariant

The new agent preserves the integration contract: same response shape, same report extraction order
in `agent.py` (`/shared/output.md` → `/shared/final_report.md` →
`/shared/final_report_meta.json`), same both-exits finalize contract (writer → marker, inline →
`submit_final_report`), same citation-verification behavior, and compatible workflow wrapper
semantics.

This is not a behavioral-equivalence claim. Tool and subagent choices, answer depth, output quality,
token cost, latency, and trace shape are expected to differ because those are the dimensions the
new architecture is intended to explore. The existing agents and their default routing remain
unchanged while those differences are evaluated.

The eval harnesses (`frontends/benchmarks/freshqa`, `deepsearch_qa`, `deepresearch_bench`) and the
UI must run against the new agent with **no harness changes**. For the end user and for eval, the
response and report artifacts remain compatible; the observed research behavior and performance
may differ.

Practical consequence: `submit_final_report` keeps its signature (only the `tier=` argument — pure
tier observability — is dropped), and `agent.py`'s extraction path is ported verbatim rather than
rewritten.

### 3.2 The three carried-over subsystems, in plain terms

These are bolted onto the tier agent but are **not part of the tier machinery**. Each is
independently either carried over or left out; all three are carried.

| Subsystem | What it actually is | What changes |
| :-- | :-- | :-- |
| **`request_termination`** | The safety net that stops a run researching forever — hard caps on research calls, total queries, orchestrator turns, wall-clock seconds, and graph recursion. Without it, a query whose evidence does not exist can run for hours. | Today the caps are looked up **per tier** (`budgets_for_tier()` returns `standard` budgets, `deep` budgets, or `None`). Becomes **one flat set** that always applies. This is a strict improvement: `single_shot` and `direct` currently resolve to `None`, i.e. no request-wide guard at all. **See §3.3, which revises this to reuse upstream's `DeepResearchResourceLimits` as the base layer rather than porting a flattened config.** |
| **Sandbox + skills** | Modal code execution (`execute`) for charts, tables and calculations; artifact harvesting so figures embed via `artifact://`; plus the 6 `deep_research_skills` markdown files teaching chart generation, data-table analysis, forecast analysis, and long-form report writing. | Carried unchanged — purely additive, no tier coupling. Cost: Modal becomes a dependency of any smoke test exercising it. |
| **Parent-report delta mode** | The "follow up on an existing report" path: a previous report is mounted at `/shared/original_report.md` with its source allowlist, and the agent returns a complete revised report rather than a patch. Used by the UI follow-up flow and `report_rewriter`. | Today it is a hard-coded **override that bypasses tier selection entirely** (`is_delta` short-circuits the router, forces the full prompt, forces the writer pipeline). In this design it becomes a conditional context block plus a prompt rule — much simpler. |

### 3.3 Upstream 2.2.0 subsystems the autonomous agent must integrate with

The rebase introduced machinery that did not exist when this design was first sketched. None of it
is tier-coupled, so all of it is inherited rather than redesigned — but the new factory has to wire
it explicitly.

**`deep_researcher/resource_limits.py` — a tier-independent hard-limits layer.** Upstream now ships
`DeepResearchResourceLimits`, `StateBudgetLedger`, and `DeepResearchExecutionTimeout` with 16
tier-independent `DEFAULT_MAX_*` constants: `MAX_RESEARCH_QUERIES=20`,
`MAX_SOURCE_TOOL_CALLS=100`, `MAX_RESEARCH_EXECUTION_SECONDS=3600`, `MAX_TOTAL_STATE_BYTES=24MB`,
`MAX_TODO_ITEMS=20`, and so on.

This substantially overlaps the "flattened `request_termination`" plan in §3.2. **Revised
recommendation:** do not port a flattened `AdaptiveRequestTerminationConfig` wholesale. Use
`DeepResearchResourceLimits` as the base layer — it is already exactly the "one flat set of caps"
the plan called for — and keep only the genuinely additive adaptive budgets on top
(`max_batch_calls`, `max_orchestrator_turns`, `max_identical_research_queries`, the workflow
deadline), which `resource_limits.py` does not cover. That removes the per-tier lookup *and* a
duplicated limits layer in one step.

**Finalize integrity needs a autonomous dual-exit tracker.** The upstream
`RequiredOutputFileMiddleware(tracker=...)` remains correct inside the writer, but
`RequiredWriterDelegationMiddleware(tracker=...)` is not correct for the autonomous orchestrator: it
would force every run through the writer and eliminate the valid inline `submit_final_report`
exit. The autonomous factory instead constructs one `AutonomousFinalReportCommitTracker` that accepts
either a writer-owned `/shared/output.md` commit or an orchestrator-owned
`/shared/final_report.md` commit. A autonomous finalization middleware gives one bounded corrective
turn only when neither exit has committed. It never forces writer delegation merely because the
inline exit was chosen.

**`common/logging_utils.log_content_metadata` — redaction.** Prompts, model responses, tool
payloads, and exception details must never reach logs raw; they are replaced by
`chars=<n> ref=sha256:<12>`. The autonomous agent's own logging must follow this. Concretely: no
`logger.info(prompt)`, no logging of `ResearchNotes` bodies or tool results.

**`src/aiq_agent/guardrails/`** ships `deep_agent`, `shallow_agent`, and `workflow` adapters. There
is **no `autonomous` adapter**, so one is needed for parity — otherwise a guardrails-enabled
deployment silently runs the autonomous agent unguarded. This is a genuine scope addition the original
plan missed; size it before committing to it.

**`deep_researcher/researcher_context.py`** (`CURRENT_RESEARCHER_GUARD_STATE`,
`ResearcherRunGuardState`, `normalize_research_depth`) now lives in `deep_researcher/` and is
shared. The autonomous researcher reuses it as-is — it is depth-keyed, not tier-keyed.

---

## 4. Reference architecture and library facts

The `deepagents` library ships the target pattern in `examples/deep_research/agent.py`:

```python
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)
```

Properties we adopt: no effort classification anywhere; depth guidance as soft heuristics in one
workflow prompt; budgets stated per-agent rather than per-tier.

### 4.1 Build against 0.6.8, not the reference checkout

AI-Q pins `deepagents>=0.6.5` and resolves to **0.6.8**; the reference checkout is **0.6.12**.
The rebase onto 2.2.0 did **not** change this — the pin and the locked version are both unchanged,
re-confirmed against `pyproject.toml` and the regenerated `uv.lock`. The two versions differ in
ways that matter, so the following is verified against the *installed* 0.6.8:

- `create_deep_agent(model, tools, *, system_prompt, middleware, subagents, skills, memory,
  permissions, backend, interrupt_on, response_format, state_schema, context_schema, checkpointer,
  store, debug, name, cache)` — `graph.py:235`.
- `system_prompt` is `str | SystemMessage | None`. The structured `SystemPromptConfig`
  (`prefix`/`base`/`suffix`) **does not exist in 0.6.8** — do not plan around it.
- Built-ins injected automatically: `write_todos`, `ls`, `read_file`, `write_file`, `edit_file`,
  `glob`, `grep`, `execute`, `task` (`graph.py:260-265`).
- `SubAgent`: required `name`, `description`, `system_prompt`; optional `tools` (**`NotRequired` —
  omitting it inherits the parent's tools**), `model`, `middleware`, `interrupt_on`, `skills`,
  `permissions`, `response_format` (`middleware/subagents.py:27,83,115`).
- `CompiledSubAgent`: `name`, `description`, `runnable` (`middleware/subagents.py:155`).

### 4.2 Library behaviors that shape the work

1. **Our prompt is a *prefix*, not the whole prompt.** `graph.py:832` computes
   `base_prompt = _apply_profile_prompt(_profile, BASE_AGENT_PROMPT)` and places the caller's
   `system_prompt` *before* it. `BASE_AGENT_PROMPT` (`graph.py:69`) already supplies "Core
   Behavior", "Professional Objectivity", "Doing Tasks" (understand → act → verify) and "Clarifying
   Requests". We are writing the research-specific layer on top of a autonomous agent prompt that
   already exists — not authoring one from scratch.
2. **Delegation guidance ships with the `task` tool.** `SubAgentMiddleware` appends
   `TASK_SYSTEM_PROMPT` (when/when-not to delegate, the Spawn → Run → Return → Reconcile lifecycle)
   and `TASK_TOOL_DESCRIPTION` (parallelization guidance and worked examples), then auto-appends
   `"Available subagent types:"` followed by each subagent's `description`. **This is the mechanism
   that replaces the tier table**, and it is free.
3. **A `general-purpose` subagent is auto-injected by default** (`graph.py:687-693`), and in this
   design it is not merely redundant — it is actively unsafe. See §4.2.1.

#### 4.2.1 Why the default `general-purpose` subagent must go

`graph.py:724-728` builds it as:

```python
general_purpose_spec: SubAgent = {
    **GENERAL_PURPOSE_SUBAGENT,
    "model": model,
    "tools": _tools or [],        # the parent's ENTIRE tool list
    "middleware": gp_middleware,  # a fresh default stack, NOT AI-Q's
}
```

Three verified consequences, all specific to this architecture:

- **Its description competes head-on with `researcher`.** Verbatim
  (`middleware/subagents.py:423`): *"General-purpose agent for **researching complex questions**,
  searching for files and content, and executing multi-step tasks... This agent has access to all
  tools as the main agent."* In a design where descriptions *are* the routing logic, that is the
  worst possible collision — and the wording is not ours to control.
- **It inherits `submit_final_report`** (`@tool(return_direct=True)`), so a subagent could commit
  the run's final report. It also inherits `run_research_batch`, giving nested fan-out — N sub-runs
  inside a sub-run, invisible to the orchestrator's request-wide loop guard.
- **It bypasses AI-Q's middleware entirely.** `gp_middleware` is rebuilt from `TodoListMiddleware`,
  `FilesystemMiddleware`, summarization, and `PatchToolCallsMiddleware`; the stack passed to
  `create_deep_agent(middleware=...)` reaches only the main agent. So a general-purpose sub-run has
  **no `SourceRegistryMiddleware`** — its retrieved sources never enter the citation registry, so
  any citation it produces fails verification — plus no loop guards and no tool-name sanitization.

This does not affect `researcher` / `planner` / `writer`: those specs carry AI-Q middleware
explicitly, as `adaptive_researcher` already does.

#### 4.2.2 Mechanism — two options, and the trade-off

`graph.py:693` suppresses auto-injection when **either** condition holds:

```python
if gp_profile.enabled is not False and not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
```

| | **A — explicit spec** (recommended) | **B — harness profile** |
| :-- | :-- | :-- |
| How | Supply `{"name": "general-purpose", "tools": [], "description": "Not used by this agent; use `researcher` instead.", "system_prompt": …}` in `subagents` | `register_harness_profile(key, HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))` |
| Scope | Per-agent. No global state | **Process-global.** `_HARNESS_PROFILES` is a module-level dict (`harness_profiles.py:935`) and registrations merge additively |
| Removes the 3 hazards above | Yes — no tools, no inherited finalizer, our description | Yes |
| Residue | One advertised name in the `task` tool description | None |
| Risk | Cosmetic only | All three arms use `nemotron_super_llm` and resolve to the same key, so this **also changes the deep and adaptive control arms**, violating §3.1 |

`create_deep_agent` has no profile parameter (`graph.py:235`), and for pre-built model instances
`_model_spec` is `None` (`graph.py:547`) so the key is derived from `get_model_provider` /
`get_model_identifier` — meaning the key is not knowable in advance and must be discovered
empirically.

**Recommendation: option A.** It eliminates every real hazard; the only thing it does not remove is
one line of prompt text. Option B removes that line too, but pays for it with process-global state
that silently mutates the control arms — and the contamination is real in-process (the
`chat_deepresearcher_agent` path routes to shallow/deep in one process), even though isolated
`nat eval` runs would be unaffected. If option B is chosen anyway, the isolation seam must exist
*before* implementation; silently allowing the default subagent is not an acceptable fallback.

Tool exclusion uses `HarnessProfile(excluded_tools=…)`, while disabling the auto-added subagent uses
`HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))`;
`FilesystemMiddleware` and `SubAgentMiddleware` are protected scaffolding. Custom middleware merges
into the default stack **by `.name`** — a same-named middleware replaces in place rather than
stacking.

### 4.3 Follow-up recorded, out of scope

deepagents **0.6.12** ships `profiles/harness/_nvidia_nemotron_3_ultra.py` — 1,826 lines of
Nemotron-specific middleware (`NemotronToolCallShim`, `NemotronTextToolCallParser`,
`NemotronReasoningTagCleanupMiddleware`, `NemotronProgressBudgetMiddleware`,
`NemotronPolicyNudgeMiddleware`, `FinalAnswerGuardMiddleware`) that overlaps substantially with
AI-Q's hand-rolled `EmptyContentFixMiddleware`, `ToolNameSanitizationMiddleware`, and
`ConsecutiveThinkGuardMiddleware`. **It is absent from 0.6.8.**

Caveat before anyone counts on it: it registers under keys like
`nvidia:nvidia/nemotron-3-ultra-550b-a55b`, while AI-Q configures `_type: nim` with
`model_name: nvidia/nvidia/nemotron-3-ultra`. The profile would very likely **not** match and so
would not auto-apply. Worth investigating as a separate deepagents-upgrade task.

---

## 5. Tools available to the main DeepAgent — by category

Names below are the exact strings the model calls.

### 5.1 Planning & reflection
| Tool | Origin |
| :-- | :-- |
| `write_todos` | deepagents `TodoListMiddleware`. Always available — in the tier design it was *hidden* under a `single_shot` ceiling. |
| `think` | AI-Q `helper_tools` (`deep_researcher/factory.py:185`). `ConsecutiveThinkGuardMiddleware` caps consecutive calls. |

### 5.2 Research fan-out
| Tool | Origin |
| :-- | :-- |
| `run_research_batch(queries: list[ResearchQuery])` | `adaptive_researcher/tools/research.py:76`. Runs ≤ `max_research_concurrency` queries in parallel against the `researcher` runnable; returns `ResearchNotes` JSON and persists each note under `/shared/`. Each `ResearchQuery` carries `query`, `preferred_tools`, `target_components`, `rationale`, `depth`. |

### 5.3 Delegation
| Tool | Origin |
| :-- | :-- |
| `task(subagent_type, description)` | deepagents `SubAgentMiddleware`. Single entry point to every subagent in §6. |

### 5.4 Retrieval sources — held **directly** by the orchestrator

Resolved per request from `data_source_registry` (`src/aiq_agent/common/data_source_registry.py`),
filtered by the request's `data_sources` and the config's `exclude_tools`.

Unlike the tier design — where source tools were exposed only on the `single_loop_single_shot`
`single_shot` path and hidden everywhere else by `ComplexityRouterMiddleware` — these are **always
in the orchestrator's tool list**, alongside `run_research_batch` and `task`. Their names go into
the `ToolNameSanitizationMiddleware` allowlist unconditionally.

All nine `sources/` packages are live post-rebase (upstream 2.2.0 restored `gsf`, `you_com`, and
`nimble_web_search`). Registered `_type` names verified against the current tree:

**Web search**
| `_type` | Package | Config-wired today? |
| :-- | :-- | :-- |
| `tavily_web_search` | `sources/tavily_web_search` | Yes — instantiated twice as `web_search_tool` (max_results 5) and `advanced_web_search_tool` (advanced, max_results 2) |
| `exa_web_search` | `sources/exa_web_search` | Source present, never config-wired |
| `nimble_web_search` | `sources/nimble_web_search` | **New post-rebase.** Depth (`lite`/`fast`/`deep`) and focus modes (`general`/`news`/`location`/`shopping`/`geo`/`social`). In `runtime-tools`, so it ships in the container image |
| `you_web_search` | `sources/you_com` | **New post-rebase.** Livecrawl + freshness controls |

**News**
| `duckduckgo_news_search` | `sources/duckduckgo_news_search` | Wired only in `config_domain_routing_and_skills.yml` |

**Academic / scholarly**
| `paper_search` | `sources/google_scholar_paper_search` | Serper / SerpAPI / SearchAPI backends |

**Internal knowledge / RAG**
| `knowledge_retrieval` | `sources/knowledge_layer` | Instantiated as `knowledge_search`. Backends: `llamaindex`, `foundational_rag`, `opensearch`, `azure_ai_search`. Note: every current adaptive config sets `exclude_tools: [knowledge_search]` |

**Prediction markets**
| `polymarket_search` | `sources/polymarket_prediction_market` | Wired in `config_domain_routing_and_skills.yml` |

**Finance**
| `you_finance_research` | `sources/you_com` | **New post-rebase.** Finance-optimized index (SEC filings, earnings, equity prices, macro); returns cited markdown |

**Agentic / synthesizing research APIs**
| `you_research` | `sources/you_com` | **New post-rebase.** Returns a *synthesized cited answer*, not raw results. See the caveat below |

**URL / content extraction**
| `you_contents` | `sources/you_com` | **New post-rebase.** Up to 10 URLs → clean markdown/HTML/metadata. The only tool that fetches a *specific* URL rather than searching |

**Enterprise structured data / text-to-SQL / forecasting** — a NAT **function group**, so it exposes
prefixed children
| `gsf__catalog_search`, `gsf__text_to_sql`, `gsf__text_to_pql` | `sources/gsf` | **New post-rebase.** Semantic catalog lookup, validated SQL over authorized enterprise data, and Kumo/PQL predictions |

**Per-user MCP sources** — resolved at request time via `per_user_auth` on a registry entry, merged
by `register_tool_sources()`. Names are unknown until the request lands. This is the strongest
single argument for the whole change.

> ⚠️ **Caveat — `you_research` overlaps the agent itself.** It runs its own multi-source research
> and returns a finished cited answer. Handing that to a research orchestrator invites the model to
> delegate the whole task to it and pass the result through — bypassing `run_research_batch`, the
> loop guards, and AI-Q's citation registry (its citations would never appear in
> `get_verified_sources`, so the finalizer would reject them or strip them). Recommend
> **excluding it** from the autonomous agent, or describing it explicitly as a last-resort
> corroboration tool. The same reasoning applies more weakly to `you_finance_research`.
>
> This is a concrete instance of the §6.1 overlap risk: the autonomous design has no middleware left
> to catch a bad delegation choice, so the exclusion must be a deliberate config decision.
>
> Packaging note: `gsf__*` and `you_*` are **dev-only** — they are not in the `runtime-tools`
> dependency group, so enabling them for a deployed image also requires adding them there and to
> the Dockerfile install chain.

### 5.5 Citation integrity
| Tool | Origin |
| :-- | :-- |
| `get_verified_sources(mode="compact"\|"full")` | `deep_researcher/tools/source_registry.py:30`. The whitelist of URLs actually captured by `SourceRegistryMiddleware`. For the autonomous agent, compact mode is the union promoted or persisted by all three research paths in §6.1. Must be called before writing any cited answer. |

### 5.6 Filesystem / scratchpad
`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` — deepagents `FilesystemMiddleware`
over `DeepAgentsRuntime`'s `CompositeBackend` (`StateBackend` for `/shared/`, sandbox FS elsewhere,
`FilesystemBackend(virtual_mode=True)` for skills). Gated per agent by `context.permissions(...)`.
`write_file` cannot overwrite an existing file; use `edit_file`.

### 5.7 Code execution & artifacts
| Tool | Origin |
| :-- | :-- |
| `execute` | deepagents built-in; real only when the backend implements `SandboxBackendProtocol` (Modal). `ExecuteTimeoutClampMiddleware` clamps model-supplied timeouts. Sandbox-written files are harvested by `ArtifactHarvestMiddleware` and embeddable via `artifact://`. |

### 5.8 Finalize
| Tool | Origin |
| :-- | :-- |
| `submit_final_report(markdown, researched)` | New `autonomous_researcher/tools/finalize.py`, `@tool(return_direct=True)`. It commits the inline exit to the autonomous dual-exit tracker. `researched=true` means the answer is backed by any research path: direct source calls, `task(researcher)`, or `run_research_batch`. It is not defined only in terms of batch research. |

The autonomous agent owns this tool rather than changing
`adaptive_researcher/tools/finalize.py`; the adaptive control arm keeps its tier-specific
documentation and metadata semantics. The `tier=` argument and `declare_effort_tier`
(`adaptive_researcher/tools/finalize.py:97`) have no counterpart in the autonomous design.

---

## 6. Subagents available to the main DeepAgent

All reached through the one `task(subagent_type=…, description=…)` tool. Each **description string
is the routing mechanism** — it replaces the tier table, and must answer "when should the
orchestrator pick me?" without reference to any effort level.

| `subagent_type` | Tools | Purpose |
| :-- | :-- | :-- |
| `researcher` | all source tools + `think` | Investigates **one** topic in an isolated context and returns structured cited findings. Reachable directly through `task` and in parallel behind `run_research_batch`; both use the same runnable and loop guards, while the task path adds the persistence side effects described in §6.1. |
| `planner` | all source tools + `think` | Turns a complex or multi-part request into an answer strategy plus a set of `ResearchQuery` objects. `response_format=AdaptiveResearchPlan`; `PlanPersistenceMiddleware` writes `/shared/plan.json`. Optional for inline answers, but mandatory before any writer delegation. |
| `writer` | `read_file`, `get_verified_sources`, `execute` (when sandboxed), writer skills | Synthesizes a long-form cited report from `/shared/plan.json` + `ResearchNotes`, writing `/shared/output.md`. Invoked only after `/shared/plan.json` exists. Parent-report deltas therefore run planner before writer. |

There is deliberately **no usable `general-purpose` subagent** — the deepagents default is
suppressed (§4.2.1 for why, §4.2.2 for how). Under the recommended mechanism the `task` tool
advertises `researcher`, `planner`, `writer`, and a zero-tool `general-purpose` stub whose
description points back at `researcher`; under the alternative it advertises only the first three.
Either way no delegation path exists that can retrieve, research, or finalize outside the three
defined subagents.

**Planner/writer invariant:** any path delegating to `writer` must first produce
`/shared/plan.json`, including parent-report deltas. `writer.j2:6` requires
`answer_strategy.answer_type`, `.title`, `.required_components`, and `constraints` from that
file. Adaptivity remains emergent because the orchestrator decides whether the answer needs the
writer; once it chooses writer publication, planner-first sequencing is deterministic and enforced
before the `task(writer)` call is accepted.

**New:** `researcher` is exposed as a `task`-reachable `SubAgent` *in addition to* remaining the
runnable behind `run_research_batch`. In the tier design it was only the latter. Both use
`build_researcher_runnable` and the same loop guards, but equivalent evidence state requires an
explicit persistence wrapper on the direct task path.

### 6.1 Three research paths: routing and evidence-state contract

Giving the orchestrator the full menu is the decision, and it is the right one for a
description-driven architecture — but it means three tools can answer "go find this out":

| Path | Cost | Where results land |
| :-- | :-- | :-- |
| A source tool called directly | 1 tool call | Raw results in orchestrator context; newly captured sources are promoted into the compact verified-source set |
| `task(subagent_type="researcher")` | 1 sub-run | One structured `ResearchNotes` file under `/shared/`; its source locators are registered as compact sources |
| `run_research_batch([...])` | N parallel sub-runs | N `ResearchNotes` files under `/shared/`; their source locators are registered as compact sources |

The evidence contract is path-independent:

- A autonomous direct-source middleware snapshots the source registry around each orchestrator source
  call and promotes newly captured entries into the compact set. This prevents direct evidence from
  disappearing from compact `get_verified_sources` after a batch has established a compact subset.
- The `task(researcher)` runnable is wrapped so a successful structured `ResearchNotes` result is
  persisted under a collision-safe `/shared/research_note_*.json` path and passed to
  `SourceRegistryMiddleware.register_research_note_sources(...)` on the run's registry instance
  (`deep_researcher/custom_middleware.py:1027` — a method, not a module-level function) before
  returning its digest to the orchestrator.
- `run_research_batch` keeps its existing note persistence and source-registration behavior via the
  same method.
- The writer and inline finalizer consume the union through compact `get_verified_sources`; neither
  needs to know which research path produced a source.

If the descriptions do not differentiate these sharply, the model will pick inconsistently — the
failure mode this architecture is most exposed to, even though evidence persistence is now
consistent across the three paths.

**Mitigation:** differentiate on *context isolation and scale*, never on effort level, and say so in
both the tool descriptions and the prompt's workflow section:

- **direct source tool** — one quick lookup where you want the raw result in front of you;
- **`task(researcher)`** — one topic needing iterative, multi-hop investigation, digested back;
- **`run_research_batch`** — several *independent* questions to investigate at once.

This is the first thing to read the smoke matrix (§9.3) for, and the most likely reason to need a
second prompt iteration.

### 6.2 Dropped subagents

| | Why |
| :-- | :-- |
| `source-router-agent` | Advisory domain routing; already `enable_source_router: false` in every adaptive config. Rich tool descriptions are precisely the mechanism it substitutes for. |
| `shallow-researcher` | A `CompiledSubAgent` reachable only on the `single_shot` tier and force-routed by `SingleShotShallowDelegationMiddleware`. `task(subagent_type="researcher")` is the same capability without the tier gate or the forcing middleware. Removes `subagents/shallow.py` (337 lines), `ShallowSubagentCapture`, the cancellation/recovery path in `agent.py:393-412`, and 3 config knobs. |
| `general-purpose` | Not merely a redundant fourth delegation route: the deepagents default inherits the parent's **entire** tool list — including `submit_final_report` (`return_direct=True`) and `run_research_batch` — and runs on a fresh default middleware stack with **no `SourceRegistryMiddleware`**, so its citations can never verify. Its shipped description also advertises it for "researching complex questions", competing directly with `researcher`. Full evidence in §4.2.1; suppression mechanism in §4.2.2. |

---

## 7. What the autonomous system prompt carries

Because `BASE_AGENT_PROMPT` and `TASK_SYSTEM_PROMPT` already cover baseline agent behavior and
delegation mechanics (§4.2), our prompt carries only what is research-specific plus what the tiers
were smuggling in. Everything the library supplies is deliberately *not* restated.

1. **Role** — a research orchestrator; match effort to the question. 2–3 lines.
2. **Workflow** — assess → (optionally `write_todos`) → research by whichever path fits →
   `get_verified_sources` → either answer inline, or ensure `task(planner)` has produced
   `/shared/plan.json` before `task(writer)` → finalize. Planning remains optional only for
   inline publication.
3. **How to choose a research path** — the §6.1 differentiation, stated as capability not effort.
   **This is the most important paragraph in the prompt**; it is what stops the three overlapping
   paths being chosen at random.
4. **Calibration heuristics, as guidance not modes** — a bounded factual question needs one lookup;
   a genuinely multi-part request warrants `planner` and a wider batch; a `writer` delegation only
   when the answer is report-shaped; no research at all only for chit-chat. **A few lines replacing
   a 4×6 tier table plus four `### <tier>` procedure blocks.**
5. **Depth guidance** — port `## Research Depth (per query)` (`orchestrator.j2:78-91`) largely
   as-is. Already tier-independent: `depth` is a per-`ResearchQuery` knob orthogonal to width, and
   it is what `ResearcherLoopGuardMiddleware` budgets against (low=1 / medium=3 / high=6).
6. **Anti-memory rule** — from `orchestrator.j2:63`: when a fact could be time-sensitive, or you are
   not fully certain, research it; never answer from memory. **This is the highest-risk deletion in
   the whole change.** The tier design enforced it structurally (a research-capable tier had to be
   selected); in this design it is prompt-only. State it emphatically; it is the first thing the smoke
   tests check.
7. **Citation contract** — port `orchestrator.j2:286-292` and state that compact verified
   sources are the path-independent union described in §6.1.
8. **Stopping & evidence failure** — port `orchestrator.j2:264-267` as-is; already tier-independent.
   Includes the rule that an honest "could not verify" finalizes with `researched=true` when any
   direct, delegated, or batched research path was attempted, rather than downgrading to an
   unresearched answer.
9. **Delta / parent-report rule** — port `orchestrator.j2:96-97`, but as a plain
   `{% if parent_report_context_available %}` block rather than a tier override. The block
   requires planner completion and `/shared/plan.json` before writer delegation.
10. **Finalize protocol** — two valid exits tracked by one autonomous commit tracker: delegated to
    `writer` only after a plan exists → writer commits `/shared/output.md` and the orchestrator
    returns its marker; wrote inline → the autonomous `submit_final_report` commits
    `/shared/final_report.md` exactly once. Do not attach
    `RequiredWriterDelegationMiddleware` to the autonomous orchestrator.
11. **Context block** — datetime, user info, uploaded documents, clarifier result, retrieval tool
    names + descriptions. Port `orchestrator.j2:310-360` unchanged, keeping the `KV CACHE BOUNDARY`
    comment so static text stays cacheable.

**Target: ~150 lines with no `{% if S %}` section gating**, versus 360 lines across 16 gated
sections today. Jinja conditionals remain only for optional *capabilities* (sandbox present,
documents uploaded, parent report present) — never for effort.

**The descriptions do the routing.** `SubAgentMiddleware` renders each subagent's `description`
into the `task` tool, and retrieval tool descriptions render into the context block. Describing a
capability well *is* the routing logic — so the strings in §5–§6 are load-bearing and should be
reviewed as carefully as code.

---

## 8. Files

### New — `src/aiq_agent/agents/autonomous_researcher/`

| File | Content |
| :-- | :-- |
| `register.py` | `AutonomousResearchAgentConfig(FunctionBaseConfig, name="autonomous_research_agent")` + `autonomous_research_workflow`. Reuses the `get_all_tool_refs()` / `exclude_tools` pattern from `adaptive_researcher/register.py:279-290`. **Drops** `enabled_tiers`, `enforce_tier_tools`, `single_loop_single_shot`, `dynamic_orchestrator_sections`, `single_shot_search_budget`, `single_shot_shallow_subagent`, `shallow_subagent_max_*`, `single_shot_researcher_llm` (already dead), `source_router_llm`, `enable_source_router`, `domain_catalog_path`. **Keeps** the LLM refs, `tools` / `exclude_tools`, `enable_citation_verification`, `researcher_loop_guard`, `request_termination` (flattened), `skills`, `sandbox`, `max_research_concurrency`, `max_concurrent_source_tool_calls`, `max_source_tool_batch_size`. |
| `agent.py` | `AutonomousResearcherAgent`, adapted from `adaptive_researcher/agent.py`. **Extraction path unchanged.** Removes `_read_tier` and the shallow-capture recovery path. Owns the autonomous dual-exit tracker for each run. |
| `factory.py` | One `create_deep_agent` call, one prompt render. No `_render_orchestrator`, no `TierResolver`, no `ComplexityRouterMiddleware`, no `SingleShotShallowDelegationMiddleware`, no `hidden_tools_for_ceiling`. `orchestrator_tools = [*helper_tools, run_research_batch, submit_final_report, *research_source_tools]` — source tools wired in **unconditionally**, with their names added to the `ToolNameSanitizationMiddleware` allowlist. Explicit subagent specs: `researcher`, `planner`, `writer`, plus the zero-tool `general-purpose` stub that suppresses deepagents' auto-injection per §4.2.2 option A (no global harness-profile registration). The factory wraps `task(researcher)` persistence, enforces plan-before-writer delegation, and attaches autonomous dual-exit finalization middleware rather than `RequiredWriterDelegationMiddleware`. **Must construct `DeepResearchGraphContext` with the three fields upstream added — `resource_limits`, `final_report_tracker`, `state_budget`.** Copy the now-working form from `adaptive_researcher/factory.py` (§0 fix 1): `limits = resource_limits or DeepResearchResourceLimits()`, `final_report_tracker or FinalReportCommitTracker()`, and `state_budget or StateBudgetLedger(limits=limits, files=state.files, sandbox_enabled=runtime.execution_enabled)`. Note the ledger takes the run's real `state.files` and the runtime's actual sandbox flag — not empty/`True` placeholders. |
| `custom_middleware.py` | Autonomous-agent-only evidence and integrity seams: promote sources from direct orchestrator retrieval into the compact registry; persist successful `task(researcher)` results as `ResearchNotes` and register their compact sources; reject `task(writer)` until `/shared/plan.json` exists; and accept either tracked final-report exit without forcing writer publication. |
| `tools/finalize.py` | Autonomous `build_submit_final_report_tool` with neutral `researched` semantics covering direct, delegated, and batched research. Writes the inline report/meta files and commits the inline side of the dual-exit tracker; contains no tier argument or tier metadata. |
| `models/request_termination.py` | `AutonomousRequestTerminationConfig` — one flat budget set replacing `AdaptiveTierBudgets` + `budgets_for_tier()`. Per §3.3, this should hold **only** what `DeepResearchResourceLimits` does not already cover (`max_batch_calls`, `max_orchestrator_turns`, `max_identical_research_queries`, workflow deadline) rather than duplicating the upstream limits layer. |
| `prompts/orchestrator.j2` | New, per §7. |
| `prompts/planner.j2`, `researcher.j2`, `writer.j2` | Ported from `adaptive_researcher/prompts/`, tier vocabulary stripped. `researcher.j2` keeps the `depth`→budget interpolation (the loop guard still uses it). |

### Reused unchanged (imported, not copied)

- `deep_researcher/deepagents_runtime.py` — `DeepAgentsRuntime`, backends, sandbox, skills
- `deep_researcher/custom_middleware.py` — `SourceRegistryMiddleware`,
  `ToolNameSanitizationMiddleware`, `ToolResultPruningMiddleware`, `EmptyContentFixMiddleware`,
  `FilesystemToolCallGuardMiddleware`, `ExecuteTimeoutClampMiddleware`, `ArtifactHarvestMiddleware`,
  `PlanPersistenceMiddleware`, `RequiredOutputFileMiddleware`, `TodoSuppressionMiddleware`
- `deep_researcher/tools/source_registry.py` — `get_verified_sources`
- `common/citation_verification.py` · `common/data_source_registry.py`
- `adaptive_researcher/tools/research.py` — `build_adaptive_research_batch_tool` (reused for the
  batch path only; direct task persistence is autonomous-agent-specific)
- `adaptive_researcher/models/{state,subagent_contracts,loop_guard}.py`
- `adaptive_researcher/custom_middleware.py` — `ConsecutiveThinkGuardMiddleware`,
  `ResearcherLoopGuardMiddleware`, and `OrchestratorLoopGuardMiddleware` (the last needs a small
  change to take a flat budget instead of a tier lookup)

### New configs

`configs/config_autonomous_researcher.yml` — mirrors `config_adaptive_frag_sandbox.yml` (same
`nemotron_super_llm`, same Tavily + knowledge tools, same `deep_research_skills` +
`deep_research_sandbox`) with the tier block deleted. Plus a FreshQA variant under
`frontends/benchmarks/freshqa/configs/` for the A/B.

**Decision, not a recommendation:** the config sets
`exclude_tools: [you_research, you_finance_research]` whenever those functions are registered. Per
§5.4 they return their own synthesized cited answers, which would bypass `run_research_batch`, the
loop guards, and the citation registry. Leaving this to prompt guidance is not sufficient — there
is no middleware in the autonomous design to catch the delegation. (The A/B baseline config wires only
the Tavily + knowledge tools, so this is a guard for whoever widens the source set later.)

### Registration

Add the package to `src/aiq_agent/agents/__init__.py` and the `nat.plugins` entry point, following
`adaptive_researcher`'s pattern. Optionally register `autonomous_researcher` in
`frontends/aiq_api/src/aiq_api/registry.py` (~lines 105-133) alongside the other async-job agent
types.

### Untouched

`src/aiq_agent/agents/adaptive_researcher/**` — the whole tier implementation remains as the
control arm, along with its 11 test files, **except** the two §0 baseline fixes:
`factory.py:268` (the `DeepResearchGraphContext` construction) and the
`test_orchestrator_loop_guard.py` fixture budgets. Both are prerequisites, not part of the autonomous
design; nothing else in this tree changes.

---

## 9. Sequencing

0. **Prerequisite** — fix the two known-failing test groups from §0. Item 1 (the
   `DeepResearchGraphContext` signature) is a hard blocker: the autonomous factory copies that
   construction path, so it must be correct in `adaptive_researcher/factory.py` first.
1. Scaffold the package: `register.py` config schema + `models/request_termination.py` flat budget.
   Suppress the default `general-purpose` subagent per §4.2.2 option A — a zero-tool spec of that
   name in the `subagents` list, requiring no global registration. If option B (harness profile) is
   chosen instead, the isolation seam must exist first; fail closed rather than falling back to an
   auto-injected subagent.
2. `custom_middleware.py` + `tools/finalize.py` — implement direct-source compact promotion,
   persisted `task(researcher)` notes, plan-before-writer enforcement, and the dual-exit commit
   tracker/finalization guard.
3. `factory.py` — the single `create_deep_agent` call, researcher runnable,
   `run_research_batch`, exactly three subagent specs (`researcher`, `planner`, `writer`),
   the autonomous middleware stack, and no `RequiredWriterDelegationMiddleware`.
4. `agent.py` — lifecycle, prompt loading, citation verification, dual-exit tracker ownership,
   extraction path.
5. Prompts — `orchestrator.j2` first (the substance of the change), then the three ported ones.
6. `OrchestratorLoopGuardMiddleware` — flat-budget variant.
7. Configs + registration.
8. Tests, then the smoke matrix, then the eval A/B.

---

## 10. Verification

1. **Unit** — new `tests/aiq_agent/agents/autonomous_researcher/`:
   - The rendered prompt contains no tier labels; assert structural tier artifacts such as
     `declare_effort_tier`, tier procedure headings, tier metadata, and tier-based tool filtering
     are absent.
   - The orchestrator tool list matches §5 and `task` advertises exactly `researcher`, `planner`,
     and `writer` as the only delegable agents. Assert deepagents' **default** `general-purpose`
     is never built: whatever `general-purpose` entry exists must have an empty tool list and must
     not hold `submit_final_report` or `run_research_batch`. Assert the autonomous agent's
     construction leaves deep/adaptive subagent construction byte-identical (guards against the
     §4.2.2 option-B contamination if anyone switches mechanism later).
   - A successful `task(researcher)` call writes one collision-safe `ResearchNotes` file and
     registers its locators in compact sources. A mixed direct-source + batch run returns the union
     from compact `get_verified_sources`, including the direct-only source.
   - `task(writer)` is rejected before `/shared/plan.json` exists and accepted after planner
     persistence; parent-report delta tests assert planner precedes writer.
   - Inline `submit_final_report` satisfies the dual-exit guard without writer delegation; a writer
     commit also satisfies it; a run committing neither exit receives one bounded corrective turn.
     The autonomous `researched` flag is true for direct, delegated, and batched evidence.
   - Both finalization paths reach the same extraction order and citation-verification boundary.
   Confirm the existing `tests/aiq_agent/agents/adaptive_researcher/` files pass, after the §0
   fixes (the factory signature fix, and the loop-guard fixture budgets — the only adaptive test
   file this change may edit) —
   which first requires clearing the 38 failures recorded in §0. Baseline for comparison:
   `uv run pytest tests/aiq_agent/agents/adaptive_researcher/ tests/aiq_agent/agents/deep_researcher/`
   currently reports **38 failed, 816 passed, 6 skipped**; all 38 are in `adaptive_researcher` and
   none are in `deep_researcher`.
2. **Lint** — `uv run ruff check .` and `uv run ruff format --check .`. Both pass at this baseline.
3. **Smoke, by shape** — against `configs/config_autonomous_researcher.yml`, reading the trace for the
   behavior the tier used to force:
   - `"Hi"` → no research, one `submit_final_report`.
   - `"What is the capital of France?"` → at most one small lookup.
   - `"What's the latest NVIDIA GPU announcement?"` → **researches rather than answering from
     memory**. This is the anti-memory rule and the single highest-risk regression.
   - `"Compare A and B on X, Y, Z"` → one batch with several queries, answered inline.
   - `"Comprehensive report on …"` → todos → `planner` → batch → `writer` → `/shared/output.md`.
   - A follow-up on an existing report → delta path with planner before writer, parent citations preserved.
   No trace may **invoke** `general-purpose` (under option A the name is advertised but has no
   tools; under option B it is absent). Also read every trace for §6.1: is the
   research-path choice *consistent* across similar queries, and does compact verified-source state
   contain evidence from every path that ran?
4. **Eval A/B** — FreshQA against the adaptive arm, then DeepSearchQA. Compare accuracy **and**
   tokens/latency. The autonomous agent is expected to cost more on trivial queries; measuring that
   trade-off is the point of the exercise.
5. **Manual** — `./scripts/start_cli.sh` with the new config.
