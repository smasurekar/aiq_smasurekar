# Adaptive Researcher — `single_loop_single_shot` Feature + Root-Cause Fix

This document records every change made to implement the `single_shot` loop-collapse
token-reduction feature for the `adaptive_researcher` agent, **and** the root-cause fix
for the bug where the feature had no effect in the deployed (Dockerized) API path.

---

## 1. What the feature does

For `single_shot`-tier queries, collapse the two-loop architecture into one loop:

- **Before:** orchestrator loop → `run_research_batch` → researcher subagent loop. The same
  evidence is re-transmitted across two LLM contexts (~4× token cost, ~114k input tokens).
- **After (`single_loop_single_shot=True`):** for `single_shot` the orchestrator holds the
  source tools directly and searches inline — no researcher subagent, one loop
  (target ~30k input tokens). `standard` / `deep` keep the two-loop subagent architecture
  unchanged.

The switch is a rollout flag, `single_loop_single_shot` (default `False`), so existing
deployments are unaffected until it is explicitly enabled in config.

---

## 2. The bug (why the feature appeared to do nothing)

After the feature was implemented and `single_loop_single_shot: true` was set in config,
`single_shot` queries **still called `run_research_batch`**. Every rebuild reproduced it.

### Root cause

API jobs do **not** construct the agent through the NAT `@register_function`
(`register.py`). The deployed request path is:

```
API request
  → Dask worker (async job)
    → frontends/aiq_api/src/aiq_api/jobs/runner.py :: run_agent_job()
      → _create_agent_instance()          ← the agent is actually built HERE
        → AdaptiveResearcherAgent(...)
```

`_create_agent_instance()` has its own hardcoded constructor kwarg list
(`_ADAPTIVE_RESEARCH_AGENT_KWARGS`) and construction branch that **predated the
`single_loop_single_shot` feature**. It never passed `single_loop_single_shot` to the
agent, so the constructor defaulted it to `False` → `ComplexityRouterMiddleware` was never
attached → `run_research_batch` stayed visible → `single_shot` never collapsed.

The config, `register.py`, `agent.py`, `factory.py`, and the middleware were **all correct**
end-to-end — they simply were not on the API job path.

### How it was confirmed

A probe injected into the live Dask worker (`_create_agent_instance`) showed:

```
isinstance Adaptive = True
fn_config.single_loop_single_shot = True   ← config was always correct
ctor accepts adaptive kwargs = False        ← gate failed → minimal fallback → all config dropped
```

The gate `_constructor_accepts_explicit_kwargs(agent_cls, _ADAPTIVE_RESEARCH_AGENT_KWARGS)`
failed (compounded by a stale in-memory worker), causing the code to fall through to the
minimal fallback constructor (`llm_provider`, `tools`, `verbose`, `callbacks` only), which
drops **all** config — `enabled_tiers`, `enforce_tier_tools`, and `single_loop_single_shot`.

### The fix

`frontends/aiq_api/src/aiq_api/jobs/runner.py`:

1. Added `"single_loop_single_shot"` to `_ADAPTIVE_RESEARCH_AGENT_KWARGS`.
2. Added `single_loop_single_shot=fn_config.single_loop_single_shot` to the adaptive-branch
   `AdaptiveResearcherAgent(...)` construction.

`single_shot_researcher_llm` is intentionally **not** wired — the agent constructor does not
accept it yet (reserved config field for future dynamic model switching).

### Deployment note (important)

A plain `docker compose restart aiq-agent` did **not** refresh the long-lived Dask worker —
it kept running stale in-memory code. Use a full recreate so the worker re-imports the
fixed code:

```bash
cd deploy/compose
docker compose up -d --build --force-recreate aiq-agent
```

---

## 3. Complete list of changes

### The fix (the change that actually made the feature take effect)

| File | Change |
| :-- | :-- |
| `frontends/aiq_api/src/aiq_api/jobs/runner.py` | Added `single_loop_single_shot` to `_ADAPTIVE_RESEARCH_AGENT_KWARGS` and passed `single_loop_single_shot=fn_config.single_loop_single_shot` in `_create_agent_instance`'s adaptive branch. **This is the root-cause fix.** |

### Feature implementation

| File | Change |
| :-- | :-- |
| `src/aiq_agent/agents/adaptive_researcher/register.py` | Added config fields `single_loop_single_shot: bool = False` and `single_shot_researcher_llm: LLMRef \| None = None`; threaded `single_loop_single_shot` into both `AdaptiveResearcherAgent(...)` construction sites (shared + per-request). |
| `src/aiq_agent/agents/adaptive_researcher/agent.py` | Added `single_loop_single_shot` constructor param, stored on `self`, computed `direct_source_tool_names`, and passed it through the middleware set and `build_adaptive_research_graph`. |
| `src/aiq_agent/agents/adaptive_researcher/factory.py` | `build_adaptive_research_graph` accepts `single_loop_single_shot`; wires source tools into the orchestrator ToolNode and excludes them from the prompt's "Your Tools" list; attaches `ComplexityRouterMiddleware` when `enforce_tier_tools or single_loop_single_shot`; threaded `direct_source_tool_names` through the orchestrator middleware / sanitizer allowlist. |
| `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py` | Extended `ComplexityRouterMiddleware` with the dynamic tool swap: caches the declared tier via `awrap_tool_call` (intercepting `declare_effort_tier`), then in `_filter_tools` removes `run_research_batch` and exposes source tools for `single_shot`, or hides source tools and keeps `run_research_batch` for other tiers. |
| `src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2` | Made the "Available Subagents", `single_shot` workflow, "Retrieval Tools", and "User Uploaded Documents" sections conditional on `single_loop_single_shot` (direct-tool path vs. `run_research_batch` path); tightened the `standard` tier to 2–3 queries defaulting to `depth: low`. |
| `src/aiq_agent/agents/adaptive_researcher/prompts/researcher.j2` | Depth-wise hint for the research sub-agent (from a prior commit; carried in this branch). |
| `configs/config_adaptive_frag.yml` | Set `single_loop_single_shot: true` to enable the feature in the dev config. |

### Tests

| File | Change |
| :-- | :-- |
| `tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py` | **New file.** Tests for `awrap_tool_call` tier caching and `_filter_tools` static-ceiling + single-shot swap behavior. |
| `tests/aiq_agent/agents/adaptive_researcher/test_register.py` | Added tests: `single_loop_single_shot` defaults `False`, `single_shot_researcher_llm` defaults `None`, and the flag can be enabled. |

---

## 4. Key architectural lessons (for future feature work)

- **Two construction paths exist.** NAT direct invocation builds the agent via
  `register.py`; the deployed async API path builds it via
  `aiq_api/jobs/runner.py::_create_agent_instance`. **Any new agent constructor param must
  be wired in BOTH places**, and its name must be added to the relevant
  `_*_AGENT_KWARGS` gate set in `runner.py` or the tailored branch is silently skipped in
  favor of a minimal fallback that drops all config.
- **The Dask worker is long-lived.** It imports the agent modules at container start and
  does not hot-reload. A plain `restart` may keep stale code — a full
  `up --build --force-recreate` is needed to guarantee the worker runs fresh code.
- **`configs/` is bind-mounted (live); `src/` is baked into the image.** Config edits apply
  on restart; code edits require a rebuild.
