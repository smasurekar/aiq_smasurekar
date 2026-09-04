# `lc_deep_research` — LangChain DeepAgents Deep-Research Example inside AI-Q

Porting the upstream **LangChain DeepAgents `deep_research` example** into AI-Q as a third research
agent alongside `adaptive_research_agent` and `autonomous_research_agent`, so all three can be A/B'd
on the same Harbor eval harness with the same NAT-provided Nemotron Ultra model.

- **Ground truth**: `/home/smasurekar/Desktop/Swapnil/github_repos/deepagents/examples/deep_research`
- **DeepAgents library**: `/home/smasurekar/Desktop/Swapnil/github_repos/deepagents/libs/deepagents` (v0.7.6 at HEAD)
- **Harbor evals**: `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals`
- **Target branch**: `dev/smasurekar/aiq-autonomous-agent`

---

## 0. Design contract (what "no accuracy impact" means here)

The whole point of this example is to be a **faithful reference implementation** of the upstream
LangChain deep-research agent, running under AI-Q's plumbing. So this plan draws a hard line:

| Layer | Rule |
| :-- | :-- |
| **Prompts** (`prompts.py`) | **Byte-for-byte verbatim** from ground truth. No AI-Q prose, no `.j2` conversion, no citation-registry instructions. |
| **Tools** (`tools.py`) | **Verbatim**, except one mandated change: lazy `TavilyClient` construction (AI-Q forbids crashing at import when a secret is absent — `CLAUDE.md` "Security and auth rules"). |
| **Agent topology** (`agent.py`) | Verbatim: one `create_deep_agent`, one `research-agent` subagent, same instruction concatenation, same `max_concurrent_research_units=3` / `max_researcher_iterations=3`. Only the `model=` line is replaced with the NAT-resolved LLM. |
| **AI-Q post-processing** | **Not applied.** No `verify_citations`, no `sanitize_report`, no `SourceRegistryMiddleware`, no loop guards, no `submit_final_report`. Every one of those rewrites the report text and would make this arm no longer a measurement of the upstream design. |
| **AI-Q I/O contract** | **Fully adopted.** NAT `register_function`, `FunctionBaseConfig`, state in / state out for the UI, and a `ChatResponse`-returning workflow wrapper for evals — identical in shape to `autonomous_research_workflow`. |

Everything AI-Q adds sits **outside** the agent graph (registration, LLM wiring, config, output
extraction). Nothing AI-Q adds sits **inside** it.

---

## 1. Baseline facts established during research

### 1.1 Ground-truth example

`examples/deep_research/` is 5 files, ~460 lines total:

| File | Lines | Role |
| :-- | --: | :-- |
| `agent.py` | 59 | Builds one `create_deep_agent` with `[tavily_search, think_tool]`, one `research-agent` subagent, and the concatenated orchestrator prompt. |
| `research_agent/prompts.py` | 172 | `RESEARCH_WORKFLOW_INSTRUCTIONS`, `RESEARCHER_INSTRUCTIONS`, `SUBAGENT_DELEGATION_INSTRUCTIONS`, `TASK_DESCRIPTION_PREFIX` (the last is defined but unused by `agent.py`). |
| `research_agent/tools.py` | 116 | `tavily_search` (Tavily URL discovery → `httpx` fetch → `markdownify`) and `think_tool` (reflection no-op). |
| `research_agent/__init__.py` | 20 | Re-exports. |
| `utils.py` | 94 | `rich` notebook display helpers — **not needed**, do not port. |

Behaviorally load-bearing details worth stating explicitly, because they must survive the port:

- The final report is written by the orchestrator to the **virtual filesystem path `/final_report.md`**
  (`RESEARCH_WORKFLOW_INSTRUCTIONS` step 5). It is *not* the last chat message. The user request is
  mirrored to `/research_request.md` (step 2).
- The workflow depends on `write_todos` (step 1) — supplied by DeepAgents' default
  `TodoListMiddleware`. See §6, this is the single biggest version hazard.
- `tavily_search(max_results=1)`: **one URL per query**, but the *full page* is fetched and converted
  to markdown. This is deliberate (depth over breadth) and is a real accuracy knob. Keep the default.
- Citations are model-authored `[1]`, `[2]` with a trailing `### Sources` section. There is no
  registry and no verification — the model is trusted.
- `RESEARCHER_INSTRUCTIONS` is `.format(date=...)`-templated and applies **only to the subagent**.
  The orchestrator never sees it.

### 1.2 AI-Q integration surface (from `autonomous_researcher`)

`src/aiq_agent/agents/autonomous_researcher/register.py` is the template to copy structurally:

- `AutonomousResearchAgentConfig(FunctionBaseConfig, name="autonomous_research_agent")` →
  `@register_function(..., framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])` →
  `yield FunctionInfo.from_fn(_run, description=...)` where `_run(state) -> state`.
- A second, thin `..._workflow` registration (`register.py:296-318`) that takes a **plain string**,
  wraps it in `HumanMessage`, invokes the agent function, and returns
  `_create_chat_response(content, response_id="research_response", model=workflow_id)`.
- Plugin discovery is via `pyproject.toml` `[project.entry-points."nat.plugins"]`
  (`aiq_autonomous_researcher = "aiq_agent.agents.autonomous_researcher.register"`).
- `builder.get_llm(config.x_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)` returns a LangChain
  `BaseChatModel` that can be handed straight to `create_deep_agent(model=...)`.
- State model: `DeepResearchAgentState` (`deep_researcher/models/state.py:39`) carries
  `messages`, `data_sources`, `files`, `todos`, `tools_info`, … — the shape the UI and the
  `chat_deepresearcher_agent` path already understand.

### 1.3 Harbor eval contract (what actually has to hold)

From `ai-q-harbor-evals/src/aiq_harbor_evals/agents/aiq_runner.py`:

1. `load_workflow(config_file)` → `session.run(instruction)` → `runner.result(to_type=str)`.
2. `_extract_text()` (line 482) accepts **only** `str` or `nat.data_models.api_server.ChatResponse`,
   takes `choices[0].message.content`, strips it, and **raises on empty**.
3. The result is written verbatim to `/workspace/answer.txt`.
4. `_validate_required_components()` (line 535) hard-fails if any name in `required_functions` is
   missing from `config.functions ∪ config.function_groups`, or any `required_data_sources` id is
   missing from the `data_source_registry` block. With `strict_tools: true` every listed data-source
   tool must also resolve.

**Consequence**: returning a `ChatResponse` from a `*_workflow` function — exactly what
`autonomous_research_workflow` does — is the entire contract. Nothing in the eval repo's *code*
needs to change. A **new eval YAML** is still needed (§5) because the config file path and
`required_functions` are per-arm config values.

> ⚠️ **Image pin.** `deepsearchqa_*_frag.yaml` pins `AIQ_RUNTIME_IMAGE: aiq-harbor:c075362751ce` /
> `AIQ_REVISION: c075362751ce…`. AI-Q *code* lives inside that image; only the AI-Q *config* is read
> live from the host. So a new agent package under `src/aiq_agent/` **requires a Harbor runtime image
> rebuild and a pin bump** before it can be evaluated. Budget for this — it is not optional.

---

## 2. Package layout

```
src/aiq_agent/agents/lc_deep_research/
├── __init__.py                 # docstring + re-exports; no side effects
├── agent.py                    # graph construction (ported from examples/deep_research/agent.py)
├── register.py                 # NAT config schema + two register_function bodies
└── research_agent/             # kept as a subpackage so `diff` against upstream stays trivial
    ├── __init__.py             # verbatim
    ├── prompts.py              # verbatim (byte-for-byte)
    └── tools.py                # verbatim + lazy Tavily client
```

**Why the nested `research_agent/` package**: the upstream files use absolute imports
(`from research_agent.prompts import ...`). Keeping the directory name means the only edit is the
import prefix, and a future upstream refresh is a copy-paste. AI-Q's `.j2`-prompt convention is
deliberately *not* followed here — see §0. Note this deviation in the module docstring so reviewers
don't "fix" it.

---

## 3. File-by-file implementation

### 3.1 `research_agent/prompts.py` — verbatim

Copy unchanged. All four constants, including the unused `TASK_DESCRIPTION_PREFIX` (keeping it makes
the file a literal copy). Add **no** license header edits beyond the AI-Q SPDX block if pre-commit
requires one — if the hook does require it, prepend the header *above* the existing docstring and
change nothing else.

### 3.2 `research_agent/tools.py` — verbatim + one mandated change

Only change, at module scope:

```python
# Ground truth constructs TavilyClient() at import time, which raises when TAVILY_API_KEY is
# absent. AI-Q loads every registered plugin at startup regardless of which agent the config
# selects, so an import-time raise would break unrelated configs. Defer construction to first
# use; behavior after the first call is identical.
_tavily_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client
```

and `tavily_client.search(...)` → `_get_tavily_client().search(...)` inside `tavily_search`. The
tool docstrings — which are the LLM-facing tool descriptions — stay untouched.

`fetch_webpage_content`, the `User-Agent` header, the `10.0`s timeout, the `## {title}` result
formatting, the `🔍 Found N result(s)` envelope, and `think_tool` in full: **unchanged**.

### 3.3 `agent.py` — graph construction

Ported from ground truth with three edits:

1. **Model comes from NAT.** Delete the `init_chat_model(...)` / `ChatGoogleGenerativeAI` lines and
   the `langchain_google_genai` import. Take a `BaseChatModel` parameter instead.
2. **Build per request, not at import.** Wrap the module body in
   `build_lc_deep_research_graph(model, *, max_concurrent_research_units=3, max_researcher_iterations=3)`.
   `current_date` is then computed per build rather than per process — strictly more correct for a
   long-running server, and behaviorally identical for the ground truth's intended usage.
3. **Recursion limit is passed at invoke time** (§3.4), not baked into the graph.

Everything else is literal: the `INSTRUCTIONS` concatenation
(`RESEARCH_WORKFLOW_INSTRUCTIONS + "\n\n" + "="*80 + "\n\n" + SUBAGENT_DELEGATION_INSTRUCTIONS.format(...)`),
the `research_sub_agent` dict with `name="research-agent"` and its exact `description` string, and
`create_deep_agent(model=..., tools=[tavily_search, think_tool], system_prompt=INSTRUCTIONS, subagents=[research_sub_agent])`.

Same file also owns the **output extraction**, which is where AI-Q's contract meets the ground
truth's virtual filesystem:

```python
FINAL_REPORT_PATH = "/final_report.md"

def extract_final_report(result) -> str:
    """Resolve the answer: /final_report.md first, last AI message as fallback.

    Ground truth writes the report to the virtual filesystem (RESEARCH_WORKFLOW_INSTRUCTIONS
    step 5); the terminal chat message is usually just an acknowledgement. Mirrors the resolution
    order in autonomous_researcher/agent.py::_resolve_output_file_markdown, minus the AI-Q-specific
    /shared/output.md writer path which this agent has no writer for.
    """
```

Handle all three shapes DeepAgents' `StateBackend` can put in `files[path]` — `str`, `bytes`, and
`{"content": ...}` — exactly as `autonomous_researcher/agent.py:258-282` does. Fall back to the last
message's content when the file is absent or blank, and raise a clear `ValueError` when both are
empty (an empty answer fails the Harbor runner with a much worse message at line 501).

**No `sanitize_report`. No `verify_citations`.** Both are explicitly out of scope per §0.

### 3.4 `register.py` — NAT wiring

Deliberately small config surface. Every field either selects the model or bounds runaway execution;
none of them change how the agent reasons.

```python
class LcDeepResearchAgentConfig(FunctionBaseConfig, name="lc_deep_research_agent"):
    model_config = ConfigDict(extra="forbid")

    llm: LLMRef                      # single model — ground truth uses one model everywhere
    verbose: bool = True
    max_concurrent_research_units: int = 3   # ground-truth value
    max_researcher_iterations: int = 3       # ground-truth value
    recursion_limit: int = 100               # see note below
    workflow_timeout_seconds: int | None = None   # None = faithful to ground truth (no timeout)
```

- **`llm` is a single ref**, not four. Ground truth passes one `model` to `create_deep_agent`, and
  the subagent inherits it. Adding per-role refs would be a divergence with no upstream analogue.
- **`recursion_limit`**: LangGraph's default is **25**, which a 3-iteration multi-subagent research
  run will blow through — it would surface as `GraphRecursionError`, i.e. a *crashed* eval trial, not
  a degraded one. Raising it does not change any reasoning step; it only permits the graph to finish.
  Passed as `config={"recursion_limit": ...}` on `ainvoke`. **Decision needed** (§8, D2) on the
  default value.
- **`verbose`** attaches `VerboseTraceCallback` (`aiq_agent.common`) so the UI streams intermediate
  steps and Harbor's ATIF trajectory capture sees the same event shape as the other two arms.

Two registrations, mirroring `autonomous_researcher/register.py`:

```python
@register_function(config_type=LcDeepResearchAgentConfig,
                   framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def lc_deep_research_agent(config, builder):
    model = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    graph = build_lc_deep_research_graph(model, ...)

    async def _run(state: DeepResearchAgentState) -> DeepResearchAgentState:
        # invoke graph with {"messages": state.messages}, extract /final_report.md,
        # append AIMessage(report), emit_final_report on callbacks, return state
        ...
    yield FunctionInfo.from_fn(_run, description="LangChain DeepAgents deep-research reference agent.")


class LcDeepResearchWorkflowConfig(FunctionBaseConfig, name="lc_deep_research_workflow"):
    """String-in / ChatResponse-out wrapper. Response shape matches the adaptive and autonomous
    workflows, so the Harbor eval harness runs against it unchanged."""


@register_function(config_type=LcDeepResearchWorkflowConfig,
                   framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def lc_deep_research_workflow(config, builder):
    agent_fn = await builder.get_function("lc_deep_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        state = DeepResearchAgentState(messages=[HumanMessage(content=query)])
        result = await agent_fn.ainvoke(state)
        return _create_chat_response(result.messages[-1].content,
                                     response_id="research_response", model=workflow_id)
    yield FunctionInfo.from_fn(_run, description="LC deep research workflow (accepts string query).")
```

`state.data_sources` is accepted and **ignored** (the agent's search tool is hard-wired to Tavily by
the ground truth). Log a one-line `INFO` when a non-empty `data_sources` arrives so a UI user
toggling sources is not silently misled.

### 3.5 `pyproject.toml`

```toml
[project.entry-points."nat.plugins"]
aiq_lc_deep_research = "aiq_agent.agents.lc_deep_research.register"
```

Dependencies — add to `[project] dependencies` (**not** a dev group; these must reach the runtime
container image):

```toml
"tavily-python>=0.7.26",   # ground truth's TavilyClient; langchain-tavily is a different API
"markdownify>=1.2.2",      # ground truth's HTML→markdown conversion
```

`httpx` is already present transitively (0.28.1 installed) but should be declared explicitly since
`tools.py` imports it directly.

`deepagents` pin: see §6.

---

## 4. `configs/config_lc_deep_research_frag.yml`

A near-copy of `configs/config_autonomous_frag.yml`, differing only in the `functions` and `workflow`
blocks. Keeping `general`, `llms`, and the `data_sources` registry identical is what makes the A/B
honest and lets the Harbor eval YAML be a three-line diff.

```yaml
general:            # identical to config_autonomous_frag.yml (uvloop, console logging, aiq_api front end)
llms:
  nemotron_super_llm:
    _type: nim
    model_name: nvidia/nvidia/nemotron-3-ultra
    base_url: "https://inference-api.nvidia.com/v1"
    temperature: 0.7
    top_p: 0.7
    max_tokens: 65536
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: true

functions:
  data_sources:                      # declared for UI + Harbor preflight parity ONLY
    _type: data_source_registry
    sources:
      - id: web_search
        name: "Web Search"
        description: "Search the web for real-time information."
        tools: [web_search_tool, advanced_web_search_tool]
      - id: knowledge_layer
        ...
  web_search_tool: { _type: tavily_web_search, max_results: 5, max_content_length: 1000 }
  advanced_web_search_tool: { _type: tavily_web_search, max_results: 2, advanced_search: true }
  knowledge_search: { _type: knowledge_retrieval, ... }

  lc_deep_research_agent:
    _type: lc_deep_research_agent
    llm: nemotron_super_llm
    verbose: true
    max_concurrent_research_units: 3
    max_researcher_iterations: 3
    recursion_limit: 100

workflow:
  _type: lc_deep_research_workflow
```

⚠️ **The `data_sources` / `web_search_tool` / `knowledge_search` block is inert for this agent.** The
LC agent's only search path is the ground truth's own `tavily_search`, which talks to Tavily directly
via `TavilyClient` and never touches AI-Q's `tavily_web_search` NAT function. The block is retained
so that (a) the Harbor preflight's `required_data_sources: [web_search]` + `strict_tools: true` check
passes with the same eval YAML shape as the other arms, and (b) the AI-Q UI's source toggles still
render. A prominent comment must say this in the config itself, or the next reader will assume
`max_results: 5` applies to this agent. It does not — the effective value is `max_results=1` inside
`research_agent/tools.py`.

**Alternative** if the inert block is judged too confusing: drop it and set
`required_data_sources: []` in the eval YAML. This is cleaner but makes the eval configs diverge.
See §8, D3.

`temperature: 0.7` is inherited from the adaptive/autonomous arms for A/B comparability. Ground truth
uses `temperature=0.0`. **Decision needed** — §8, D4.

---

## 5. Harbor eval wiring (`ai-q-harbor-evals`, config only)

New file `configs/deepsearchqa_lc_deep_research_frag.yaml`, a verbatim copy of
`deepsearchqa_autonomous_frag.yaml` with three edits:

```yaml
      config_file: /home/.../aiq_smasurekar/configs/config_lc_deep_research_frag.yml
      required_functions:
        - lc_deep_research_agent
        - web_search_tool
        - advanced_web_search_tool
    env:
      ...
      AIQ_RUNTIME_IMAGE: aiq-harbor:<new-tag>      # rebuilt image containing lc_deep_research
      AIQ_REVISION: <new-sha>
```

`required_env` (`NVIDIA_API_KEY`, `TAVILY_API_KEY`), `required_data_sources: [web_search]`,
`strict_tools: true`, `datasets`, and `artifacts` are unchanged. **No Python change in the eval
repo.**

Steps, in order:
1. Land the AI-Q changes on the branch.
2. Rebuild the Harbor AI-Q runtime image from that commit; note the new tag + SHA.
3. Add the eval YAML with the new pin.
4. Smoke-run 2–3 `deepsearchqa` trials before a full sweep.

---

## 6. DeepAgents version — the one real hazard

**Current state**: AI-Q pins `deepagents>=0.6.5`; **0.6.8** is what's installed. The ground-truth
example pins `deepagents>=0.6.12`. The local `deepagents` checkout is at **0.7.6**.

Verified: every API the example touches — `create_deep_agent(model, tools, system_prompt=,
subagents=[{name, description, system_prompt, tools}])` — exists **unchanged** in the installed
0.6.8. The example runs today with no library change at all.

**Upgrading to 0.7.x is not a version bump; it is a migration.** From
`libs/deepagents/CHANGELOG.md` §0.7.0 "⚠ BREAKING CHANGES":

| Break | Impact on `lc_deep_research` | Impact on the three existing AI-Q agents |
| :-- | :-- | :-- |
| `create_deep_agent` no longer includes `TodoListMiddleware`; `write_todos` and the `todos` channel are gone | **Direct accuracy hit.** `RESEARCH_WORKFLOW_INSTRUCTIONS` step 1 is literally "Create a todo list with `write_todos`". Must pass `middleware=[TodoListMiddleware()]` on the main agent *and* on the subagent to restore. | `deep_researcher/custom_middleware.py:821-889` builds `TodoQuotaMiddleware` / `TodoSuppressionMiddleware` **on the assumption that DeepAgents attaches `TodoListMiddleware` by default**. All three agents silently lose `write_todos`. |
| Base agent prompt is now empty; `BASE_AGENT_PROMPT`, `TASK_SYSTEM_PROMPT`, `FILESYSTEM_SYSTEM_PROMPT`, `SUMMARIZATION_SYSTEM_PROMPT`, `EXECUTION_SYSTEM_PROMPT` removed/deprecated | The harness prose the ground truth's prompts were tuned against disappears. | Same, across all agents. |
| `write_file` now overwrites instead of erroring on an existing file | Low — the report is written once. | Guardrails that relied on the file-exists error need review. |
| `FilesystemBackend` / `LocalShellBackend` default to `virtual_mode=True`; backend **factories** no longer accepted (concrete `BackendProtocol` only); `files_update` removed from `WriteResult`/`EditResult` | None (default `StateBackend`). | `deep_researcher/deepagents_runtime.py` composes `CompositeBackend` / `FilesystemBackend` / `StateBackend` directly — needs a full audit. |
| `read_file` gutter format changed; `ls`/`glob` render `No files found` | None. | Any AI-Q code parsing raw tool output needs review. |

**Recommendation for iteration 1**: pin **`deepagents>=0.6.12,<0.7.0`**. That satisfies the ground
truth's own floor, is the newest release with the behavior the example was authored and tuned
against, and touches nothing in the existing three agents. Ship the accuracy measurement first.

The user's stated requirement is "latest version". Concretely, "latest" (0.7.6) means: restore
`TodoListMiddleware` on both the orchestrator and the subagent, re-tune against lean base prompts,
and separately migrate `deep_researcher`'s backend and todo-middleware stack — a change that touches
all four agents and invalidates the existing adaptive/autonomous eval baselines. That should be its
own tracked piece of work, not a side effect of adding an example. **Decision needed — §8, D1.**

---

## 7. Tests and validation

### 7.1 New tests — `tests/aiq_agent/agents/lc_deep_research/`

Mirroring the shape of `tests/aiq_agent/agents/autonomous_researcher/`:

| File | Covers |
| :-- | :-- |
| `test_prompts_fidelity.py` | **The most valuable test.** Asserts each prompt constant is byte-identical to a checked-in fixture copied from ground truth, so an accidental "improvement" fails CI. Same for the `research-agent` subagent `description`. |
| `test_tools.py` | `tavily_search` result formatting against a stubbed client; `fetch_webpage_content` error path returns the `Error fetching content from {url}: …` string rather than raising; lazy client is **not** constructed at import with `TAVILY_API_KEY` unset. |
| `test_agent.py` | `extract_final_report` across all `files[path]` shapes (`str`/`bytes`/`{"content":…}`), the last-message fallback, and the raise-on-empty path. |
| `test_register.py` | Config schema defaults match ground-truth values (3/3); `extra="forbid"`; the workflow wrapper returns a `ChatResponse` whose `choices[0].message.content` is the report. |

### 7.2 Commands

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/aiq_agent/agents/lc_deep_research/
uv run pytest tests/aiq_agent/agents/deep_researcher/ \
              tests/aiq_agent/agents/adaptive_researcher/ \
              tests/aiq_agent/agents/autonomous_researcher/   # regression, only if deepagents moves
```

Known-unrelated: `tests/aiq_agent/test_default_model_profiles.py` failures come from scratch configs
plus the 2.2.0 endpoint deprecation and are pre-existing on this branch.

### 7.3 Live smoke

```bash
dotenv -f deploy/.env run nat serve --config_file configs/config_lc_deep_research_frag.yml --port 8000
```

Then a short factual query and a comparison query ("Compare X vs Y") — the second should visibly
fan out to multiple parallel `task(research-agent)` calls per
`SUBAGENT_DELEGATION_INSTRUCTIONS`. Confirm in the response: `## `-level headings, inline `[1]`
markers, and a trailing `### Sources` list.

---

## 8. Decisions needed before implementation

| # | Decision | Options | Recommendation |
| :-- | :-- | :-- | :-- |
| **D1** | `deepagents` version | (a) `>=0.6.12,<0.7.0` — zero blast radius, matches ground truth's own floor. (b) `>=0.7.6` — "latest", but requires restoring `TodoListMiddleware` here *and* migrating the three existing agents; invalidates existing eval baselines. | **(a) for iteration 1**, with (b) tracked as a separate cross-agent migration. Flagging because the stated requirement was "latest". |
| **D2** | `recursion_limit` default | LangGraph's 25 (faithful, will likely crash multi-subagent runs) vs 100 vs the autonomous arm's 250. | **100.** Enough headroom for 3 iterations × 3 units; still a real backstop. Not a reasoning change. |
| **D3** | Inert `data_sources` block in the AI-Q config | Keep (eval-YAML parity + UI toggles render) vs drop (honest, but eval configs diverge). | **Keep, heavily commented.** Parity is worth more than tidiness for an A/B arm. |
| **D4** | `temperature` | `0.7` (matches adaptive/autonomous arms — comparable A/B) vs `0.0` (matches ground truth). | **0.7** if the goal is "which architecture wins on Nemotron Ultra"; **0.0** if the goal is "reproduce the LangChain example". Needs the user's call — it materially affects results. |
| **D5** | Prompts as `.py` vs `.j2` | Verbatim `prompts.py` (upstream-diffable) vs AI-Q's `.j2` convention. | **`.py`, verbatim**, per §0. Document the deviation in the module docstring. |

---

## 9. Implementation order

1. **D1–D5 resolved.**
2. Create the package; copy `prompts.py`, `tools.py`, `research_agent/__init__.py` verbatim; fix
   import prefixes; apply the lazy-Tavily change. *(No AI-Q coupling yet.)*
3. Write `agent.py`: `build_lc_deep_research_graph` + `extract_final_report`.
4. Write `register.py`: config schema + agent registration + workflow wrapper.
5. `pyproject.toml`: entry point, `tavily-python`, `markdownify`, `httpx`, deepagents pin. `uv sync`.
6. `configs/config_lc_deep_research_frag.yml`.
7. Tests (§7.1) + lint + targeted pytest.
8. Live smoke against `nat serve` (§7.3).
9. Rebuild the Harbor AI-Q runtime image; note tag + SHA.
10. Add `configs/deepsearchqa_lc_deep_research_frag.yaml` in the eval repo with the new pin; smoke a
    few trials, then run the full sweep alongside the adaptive and autonomous arms.

---

## 10. Explicit non-goals

- No AI-Q citation registry, citation verification, or report sanitization on this arm.
- No loop guards, request-termination envelope, tier logic, or source router.
- No sandbox / skills / artifact support (`create_deep_agent` default `StateBackend` only).
- No `clarifier` integration and no `chat_deepresearcher_agent` wiring — this is a standalone
  workflow arm, as `config_autonomous_frag.yml` is.
- No changes under `sources/`; the LC agent does not register a NAT tool.
- No changes to Python **code** in the eval repo.

---

## 11. Implementation status — 2026-08-14

**Built and verified.** Decisions taken: **D1 = (a)** `deepagents>=0.6.12,<0.7.0`; **D2 = 100**;
**D3 = keep, commented**; **D4 = 0.4** (middle ground between upstream's 0.0 and the sibling arms'
0.7, to be tuned); **D5 = verbatim `.py`**.

### Files

| Path | Note |
| :-- | :-- |
| `src/aiq_agent/agents/lc_deep_research/research_agent/prompts.py` | Byte-identical to upstream (only an EOF newline added). |
| `src/aiq_agent/agents/lc_deep_research/research_agent/tools.py` | Verbatim except the lazy `TavilyClient`. |
| `src/aiq_agent/agents/lc_deep_research/research_agent/__init__.py` | Import prefixes only. |
| `src/aiq_agent/agents/lc_deep_research/agent.py` | `build_lc_deep_research_graph` + `extract_final_report`. |
| `src/aiq_agent/agents/lc_deep_research/register.py` | `lc_deep_research_agent` + `lc_deep_research_workflow`. |
| `configs/config_lc_deep_research_frag.yml` | Nemotron Ultra @ temperature 0.4. |
| `tests/aiq_agent/agents/lc_deep_research/` | 42 tests, incl. prompt-digest fidelity tripwires. |
| `pyproject.toml` | Entry point, `tavily-python`/`markdownify`/`httpx`, deepagents pin, ruff per-file ignores. |
| `.pre-commit-config.yaml` | `trailing-whitespace` excludes `prompts.py` — upstream has a trailing space *inside* `RESEARCHER_INSTRUCTIONS`. |

### Validation performed

- `uv run ruff check` / `ruff format --check`: clean.
- New tests: **42 passed**.
- Regression after the deepagents 0.6.8 → 0.6.12 bump:
  `deep_researcher` + `adaptive_researcher` + `autonomous_researcher` = **945 passed, 6 skipped**.
- Full root suite: **2707 passed, 13 skipped, 4 failed** — all four are the pre-existing
  `test_default_model_profiles.py` failures (scratch configs + the `inference-api.nvidia.com`
  deprecation, which already lists `config_adaptive_frag.yml` and `config_autonomous_frag.yml`).
  Failure count is unchanged by this work.
- `nat validate --config_file configs/config_lc_deep_research_frag.yml`: valid.
- **Live end-to-end run** against Nemotron Ultra ("What is the current population of Tokyo?"):
  completed in ~5 min, delegated once to `research-agent`, produced inline `[n]` citations and a
  `### Sources` section.

### Live-run observations (behaviour, not defects — do not "fix" by editing prompts)

1. **The sub-agent made 43 `tavily_search` calls** against `RESEARCHER_INSTRUCTIONS`' stated hard
   limit of 2–3 (simple) / 5 (complex). Upstream enforces that budget *by prompt compliance alone* —
   no middleware — and Nemotron Ultra does not comply the way Claude Sonnet 4.5 does. This is
   precisely the kind of finding the A/B exists to surface. It also means cost and latency on this
   arm will be materially higher than the sibling arms, which enforce budgets in middleware.
2. **`recursion_limit` is load-bearing.** At LangGraph's default of 25 this run would have died with
   a `GraphRecursionError` well before finishing. 100 was enough.
3. **No `/final_report.md` was written** on this run; the orchestrator answered inline despite step 5
   of the workflow prompt. `extract_final_report`'s fallback returned the inline answer, which is why
   that fallback exists. Worth watching whether the file is written for longer, report-shaped
   queries — if it is skipped consistently, that is a model-compliance datapoint, not a port bug.
4. One sub-agent LLM call carried **~147k prompt tokens**, a consequence of upstream fetching whole
   pages (`max_results=1` but full-page markdown) and never pruning tool results.

### Remaining work (not done here)

1. **Rebuild the Harbor AI-Q runtime image** from this commit and note the new tag + revision SHA.
   `lc_deep_research` lives in AI-Q code, which is baked into the image; only the config is read live
   from the host.
2. **Add the eval config** in `ai-q-harbor-evals` — config only, no Python change. Copy
   `configs/deepsearchqa_autonomous_frag.yaml` to `configs/deepsearchqa_lc_deep_research_frag.yaml`
   and change exactly these lines:

   ```yaml
         config_file: /home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/config_lc_deep_research_frag.yml
         required_functions:
           - lc_deep_research_agent
           - web_search_tool
           - advanced_web_search_tool
       env:
         AIQ_RUNTIME_IMAGE: aiq-harbor:<new-tag>
         AIQ_REVISION: <new-sha>
   ```

   `required_env`, `required_data_sources: [web_search]`, `strict_tools: true`, `datasets`, and
   `artifacts` stay as-is. (Not written from this repo: `CLAUDE.md` scopes AI-Q changes to this
   checkout.)
3. **Smoke 2–3 trials before a full sweep**, given observation 1 — a 43-search sub-agent run at
   10 concurrent trials is a lot of Tavily volume.
4. Docs under `docs/source/` were **not** updated: this is a POC arm on a branch with no PR planned.
