# Plan: Enable orchestrator prompt sections dynamically by tier

**Goal:** send the orchestrator only the prompt sections a tier actually needs, so
cheap tiers (`direct`, `single_shot`) pay far fewer prompt tokens than `deep`/delta.

**Status:** design/ideation. Template scaffolding already exists in
`orchestrator.restructured.j2` (every section is wrapped in `{% if S.get('<flag>', True) %}`).
This document is the wiring plan, not the implementation.

---

## 1. The core problem (read this first)

The prompt is rendered **once, at agent-build time** (`factory.py` → `render_prompt("orchestrator", ...)`).
But the tier is chosen **later, at request time**, by the model itself when it calls
`declare_effort_tier(tier=...)` as its first tool call.

So there is a chicken-and-egg gap:

> To strip sections *by tier*, we must know the tier.
> But today the tier isn't known until the model has already read the full prompt.

Everything below is about closing that gap. There are two honest ways to do it.

---

## 2. Two strategies

### Strategy A — Re-inject a trimmed prompt *after* the tier is declared  ✅ recommended

Keep model-driven tier selection. Swap the system prompt the moment the tier is known.

```
Turn 1  ── model sees a small ROUTER prompt (just: how to pick a tier)
        └─ model calls declare_effort_tier(tier="single_shot")
Turn 2+ ── middleware replaces the system prompt with the SINGLE_SHOT preset
           (only the sections single_shot needs) for every following model call
```

- **Pros:** no extra classifier call; model still selects; escalation stays possible;
  reuses hooks that already exist in `ComplexityRouterMiddleware`.
- **Cons:** turn 1 pays the (small) router prompt; savings start from turn 2.
  For short `single_shot` runs (~3–4 model calls) turns 2–N are still the majority,
  so the win is real.
- **Why it's safe here:** `ComplexityRouterMiddleware` **already hides the source
  tools before the tier is declared**, so the model literally cannot start
  researching in turn 1 — its only useful move is to declare. That guarantees a
  clean seam to swap the prompt on.

### Strategy B — Pre-classify the tier *before* rendering (triage middleware)

A cheap classifier decides the tier up front; the prompt is assembled trimmed from turn 1.

- **Pros:** maximum savings (turn 1 trimmed too); one stable prompt per tier.
- **Cons:** adds a classification step (latency + cost + a new failure mode); removes
  model-driven selection; escalation forces a re-render. This is a bigger change to
  the run's control flow.

**Recommendation:** ship **Strategy A** first (small, reversible, reuses existing
middleware). Treat **Strategy B** as a later optimization if turn-1 tokens still matter.
The rest of this plan details Strategy A.

---

## 3. Section → tier map (the presets)

These are the `sections` dicts middleware will pass to `render_prompt`. A key set to
`False` is stripped; any key omitted defaults to **on** (`S.get(key, True)`), so only
list what you turn **off** in real code — the full form is shown here for clarity.

| Section flag           | router | direct | single_shot | standard-inline | standard-writer | deep | delta |
| :--------------------- | :----: | :----: | :---------: | :-------------: | :-------------: | :--: | :---: |
| `intro`                |   ✅   |   ✅   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `effort_catalog`       |   ✅   |   ❌   |     ❌      |       ❌        |       ❌        |  ❌  |  ❌   |
| `effort_selection`     |   ✅   |   ❌   |     ❌      |       ❌        |       ❌        |  ❌  |  ❌   |
| `research_depth`       |   ❌   |   ❌   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `delta_rule`           |   ✅   |   ❌   |     ❌      |       ❌        |       ❌        |  ❌  |  ✅   |
| `subagents`            |   ❌   |   ❌   |     ❌      |       ❌        |       ✅        |  ✅  |  ✅   |
| `research_routing`     |   ❌   |   ❌   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `filesystem`           |   ❌   |   ❌   |     ❌      |       ❌        |       ✅        |  ✅  |  ✅   |
| `sequential_handoffs`  |   ❌   |   ❌   |     ❌      |       ❌        |       ✅        |  ✅  |  ✅   |
| `workflow`             |   ❌   |   ✅   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `research_loop`        |   ❌   |   ❌   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `escalation`           |   ❌   |   ❌   |     ✅*     |       ✅*       |       ✅*       |  ❌  |  ❌   |
| `stopping`             |   ❌   |   ❌   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `finalize`             |   ✅   |   ✅   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |
| `citation_contract`    |   ❌   |   ❌   |     ✅      |       ✅        |       ❌        |  ❌  |  ❌   |
| `important`            |   ✅   |   ✅   |     ✅      |       ✅        |       ✅        |  ✅  |  ✅   |

`✅*` = include `escalation` only when a **higher tier is enabled** (nothing to step up to otherwise).

**Two extra levers beyond the `sections` flags:**

1. **`enabled_tiers`** — the `## Workflow` block renders one `### <tier>` sub-block per
   enabled tier. Pass `enabled_tiers=[chosen_tier]` in the trimmed render so only the
   chosen tier's procedure appears (drops the other 3 sub-blocks for free).
2. **`workflow` off for direct** is wrong — `direct` still needs its tiny `### direct`
   sub-block. Keep `workflow=True` for direct but pass `enabled_tiers=["direct"]`.

> Keep these presets in **one place** (e.g. `tiers.py`, next to `enabled_tier_profiles`)
> as a `SECTION_PRESETS: dict[str, dict[str, bool]]`, so the map above is code, not prose.

---

## 4. Where the code changes go

Three touch-points. Nothing else needs to move.

### 4.1 `tiers.py` — declare the presets and a helper

```python
# one row per tier key from the table in §3
SECTION_PRESETS: dict[str, dict[str, bool]] = {
    "router":      {"effort_catalog": True, "effort_selection": True, "delta_rule": True,
                    "workflow": False, "research_loop": False, "stopping": False, ...},
    "direct":      {"effort_catalog": False, "effort_selection": False, "research_depth": False, ...},
    "single_shot": {...},
    "standard":    {...},   # inline vs writer handled by enabled_tiers, not a separate preset
    "deep":        {...},
    "delta":       {...},
}

def sections_for_tier(tier: str, *, higher_tier_enabled: bool) -> dict[str, bool]:
    preset = dict(SECTION_PRESETS[tier])
    preset["escalation"] = higher_tier_enabled and preset.get("escalation", False)
    return preset
```

### 4.2 `factory.py` — render the router prompt at build time

Change the single build-time render so the **initial** prompt is the *router* preset
(small; only teaches tier selection), and pass the sections dict through:

```python
system_prompt=context.render_prompt(
    "orchestrator",
    ...,                                   # unchanged args
    sections=sections_for_tier("router", higher_tier_enabled=...),
),
```

Point `render_prompt` at `orchestrator.restructured.j2` (rename to `orchestrator.j2`
once validated).

### 4.3 `custom_middleware.py` — swap the prompt after declaration

`ComplexityRouterMiddleware` already caches `self._declared_tier` in `awrap_tool_call`.
Add prompt substitution in its existing `awrap_model_call`:

```python
async def awrap_model_call(self, request, handler):
    if self._declared_tier is not None:
        request = self._with_tier_prompt(request, self._declared_tier)
    return await handler(request)

def _with_tier_prompt(self, request, tier):
    # render the trimmed prompt for `tier` and replace the system message
    sections = sections_for_tier(tier, higher_tier_enabled=self._higher_tier_enabled)
    new_system = render_prompt_template(
        self._orchestrator_template,
        sections=sections,
        enabled_tiers=[tier],           # collapse ## Workflow to this tier
        **self._static_render_context,  # same vars factory passed at build time
    )
    return replace_system_message(request, new_system)
```

The middleware needs, at construction time, the template string + the static render
context (datetime, tools, retrieval_tools, flags). Pass those in from `factory.py`
when the middleware is built (it is already built there around `factory.py:258`).

---

## 5. Escalation and delta — don't let them break

- **Escalation** (`single_shot → standard`): when the model steps up, it must call
  `declare_effort_tier` again with the higher tier. The `awrap_tool_call` handler
  updates `_declared_tier`, so the **next** `awrap_model_call` re-renders with the
  higher tier's (larger) preset automatically. Requirement: escalation must go through
  a re-declaration, not a silent switch. The `escalation` section text must say so.
- **Delta** is state-driven, not tier-driven. When `parent_report_context_available`
  is true, force the **`delta`** preset regardless of what the model would pick — do
  this in `factory.py` (render the delta preset directly and skip the router phase),
  because delta must never be handled inline.

---

## 6. KV-cache impact (important)

Swapping the system prompt mid-run changes the cached prefix. Two rules keep the hit
rate high:

1. **One stable prompt per (tier × config), byte-for-byte.** `sections_for_tier` must
   be deterministic and section order in the template is fixed, so
   `single_shot` always produces the identical prefix. Do not build the dict with
   set iteration or anything order-unstable.
2. **Swap at most once per run.** Turn 1 = router prefix; turns 2+ = one tier prefix.
   Re-declaration on escalation is the only allowed second swap. Never re-render every
   turn "just in case."

Net effect: instead of 1 shared prefix you get N stable prefixes (router + one per
tier). Each is still cache-friendly *within its own turns*.

---

## 7. Implementation checklist

- [ ] Add `SECTION_PRESETS` + `sections_for_tier()` to `tiers.py` (encode §3 table).
- [ ] Validate `orchestrator.restructured.j2` renders for every preset (see §8) and
      promote it to `orchestrator.j2`.
- [ ] `factory.py`: render the **router** preset at build time; force **delta** preset
      when `parent_report_context_available`.
- [ ] `factory.py`: pass the template string + static render context into
      `ComplexityRouterMiddleware`.
- [ ] `custom_middleware.py`: implement `_with_tier_prompt` and call it from
      `awrap_model_call` when `_declared_tier` is set.
- [ ] Confirm pre-declaration tool-hiding still forces `declare_effort_tier` to be the
      only useful first move (so the router→tier seam stays clean).
- [ ] Update `escalation` section text to require re-declaration on step-up.

---

## 8. How to validate (no full build needed)

Section rendering is pure Jinja2 — test it directly, the way the drafts were checked:

```python
import jinja2
from aiq_agent.agents.adaptive_researcher.tiers import SECTION_PRESETS, sections_for_tier

tpl = open(".../prompts/orchestrator.restructured.j2").read()
for tier in SECTION_PRESETS:
    out = jinja2.Template(tpl, undefined=jinja2.StrictUndefined).render(
        sections=sections_for_tier(tier, higher_tier_enabled=True),
        enabled_tiers=[tier] if tier != "router" else ["direct","single_shot","standard","deep"],
        **mock_static_context,   # datetime, tools, retrieval_tools, flags
    )
    # assert the tier's required headings are present and the stripped ones are absent
```

Then measure the token delta per preset (e.g. count characters / a tokenizer) to
confirm `direct`/`single_shot` really are much smaller than `deep`.

Behavioral checks worth adding to `tests/`:
- router preset contains `## Choosing Effort`, omits `## Research Loop`.
- `single_shot` preset omits `## Available Subagents`, keeps `## Inline Citation Contract`.
- `deep` preset keeps everything except `## Inline Citation Contract` and `## Choosing Effort`.
- delta path forces the delta preset even for a trivially-worded request.

---

## 9. Open decisions (need a call before coding)

1. **Router prompt scope** — minimal (only tier-selection) vs. "selection + direct/meta
   inline" (so trivial `direct`/meta answers finish in turn 1 without a second render).
   Leaning minimal; revisit if a re-render for one-shot `direct` feels wasteful.
2. **`effort_catalog` split (B1/B2)** — currently one flag. Do we also want to inject
   the *chosen* tier's one-line profile into the trimmed prompt (execution guidance),
   or is the `### <tier>` workflow block enough? If yes, split the flag in the template.
3. **Standard inline vs writer** — handled by `enabled_tiers`, or a distinct preset?
   Recommend `enabled_tiers`, keep one `standard` preset.
