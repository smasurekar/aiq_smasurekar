<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Autonomous Researcher — Analysis of Three Review Comments

**Status:** §1 and §2 **implemented 2026-08-18** — see §1.9 / §1.10 and §2.8 for what landed.
§3 remains analysis only.
**Author date:** 2026-08-18
**Subject:** feasibility and caveats of three review comments raised against the autonomous
research agent as of commit `b87022a` ("Added section wise well structured prompt for Autonomous
Agent Orchestrator").

## Scope and absolute paths

Source under review:

- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/factory.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/custom_middleware.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/prompts/orchestrator.j2`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/tools/research.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/models/request_termination.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/config_autonomous_frag.yml`

Upstream behavior was read from the installed `deepagents` in the repo virtualenv:

- `.venv/lib/python3.13/site-packages/deepagents/middleware/subagents.py`
- `.venv/lib/python3.13/site-packages/deepagents/middleware/filesystem.py`
- `.venv/lib/python3.13/site-packages/deepagents/graph.py`
- `.venv/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py`

Empirical claims are quoted from
`misc/autonomous_researcher/autonomous-researcher-token-and-f1-analysis.md`, which analyzed Harbor
job `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13`
(90 DeepSearchQA tasks).

## Important dating qualification

The eval job is dated **2026-08-13** and therefore measured commit `f0a484e`. The two prompt
revisions (`531a966`, `b87022a`) landed **2026-08-17** and are the *untested response* to that
eval. Every behavioral number below describes the pre-revision prompt. No claim in this document
should be read as measuring the prompt currently in the tree.

## The three comments

1. For subagent delegation, can't we just have it in the subagent description? The orchestrator
   should be able to read those.
2. Can we use middleware for budgets rather than defining them in the prompt?
3. We provide the researcher subagent as a tool through `run_research_batch` *and* as a subagent.
   Remove the tool and let deepagents delegate to the subagent.

## Summary of verdicts

| # | Comment | Verdict | Recommended action |
| :-- | :-- | :-- | :-- |
| 1 | Routing belongs in subagent descriptions | Partly already implemented; the prompt was a third copy | **Done (§1.9):** both prompt sections folded into the descriptions as a full union |
| 2 | Budgets belong in middleware | **Adopted.** Strongest of the three, and it restates this repo's own eval recommendation #3 | **Done (§2.8):** request guard extended with four budgets; the prompt's `# Budgets` section deleted outright |
| 3 | Remove `run_research_batch`, delegate via `task` | Adopt the intent, **invert the fix** | Keep the typed, budgeted path; either unify budgets across both routes or drop `researcher-agent` from the `task` list |

---

## 1. Subagent delegation guidance in descriptions

### 1.1 The mechanism works, and renders twice

deepagents renders each subagent `description` in two places:

| Location | Source |
| :-- | :-- |
| Into the `task` tool description | `subagents.py:484-488` (`_build_task_tool`) |
| Appended to the orchestrator system prompt under `"Available subagent types:"` | `subagents.py:684-685` (`SubAgentMiddleware.__init__`) |

So the premise of the comment is correct: the orchestrator does read them, twice.

### 1.2 This is already the design

`factory.py:128-155` defines `RESEARCHER_SUBAGENT_DESCRIPTION`, `PLANNER_SUBAGENT_DESCRIPTION`,
and `WRITER_SUBAGENT_DESCRIPTION` as request-property triggers, and the module docstring states
that descriptions *are* the routing logic. The comment is therefore not proposing a new mechanism —
it is implicitly observing that the prompt duplicates it.

### 1.3 The finding is duplication, not absence

`prompts/orchestrator.j2:63-133` — the `# Subagent Delegation Instructions` section, roughly 70 of
266 lines — restates the same triggers a third time:

| Source | Text |
| :-- | :-- |
| `PLANNER_SUBAGENT_DESCRIPTION` (`factory.py:137`) | "three or more distinct deliverables … the answer's structure must be fixed before research … a parent report is mounted" |
| `orchestrator.j2:69` | "Delegate when the request carries three or more distinct deliverables, when its structure must be fixed before research, or when a parent report is mounted" |

The prompt's own opening comment instructs the author to "never restate middleware-supplied text."
This section violates that rule.

### 1.4 What cannot move into descriptions

1. **Cross-agent ordering.** "Planner first, or not at all," "writer runs last," and "one dependent
   subagent per assistant turn" are properties of the *set* of subagents, not of any one member. A
   description is rendered inside a flat bulleted list of peers, where sequencing constraints read
   as noise.
2. **The `task()` brief templates.** These are Jinja-conditional on
   `parent_report_context_available`, `execution_enabled`, and skills availability. They *could* be
   built per-context — `build_autonomous_subagents(context)` has the context — but see §1.5.

### 1.5 Caveat: moving text into a description costs roughly double

Because deepagents renders descriptions into both the tool schema and the system prompt
(§1.1), any text moved from prompt to description is paid for twice per turn. The eval's
recommendation #8 is already "reduce the stable autonomous prompt/tool-schema footprint," and
§1.3 of that report measured a mean peak orchestrator prompt of 34,433 tokens against adaptive's
20,812. Migrating the ~20-line brief templates into descriptions moves in the wrong direction.

### 1.6 Caveat: this hypothesis has already failed once

Description-only routing was the explicit bet of the original design. Observed routing across the
90-task job (`autonomous-researcher-token-and-f1-analysis.md` §2.2):

| Route | Observed use |
| :-- | --: |
| `run_research_batch` | 120 calls, 286 worker contexts |
| `task(... researcher-agent ...)` | 3 calls, all in one trial |
| `task(... planner-agent ...)` | 0 |
| `task(... writer-agent ...)` | 0 |

Removing prompt reinforcement before re-measuring the 08-17 revisions would double down on the
side of the experiment that lost.

### 1.7 Related trap: the upstream `task` tool description works against this agent

`TASK_TOOL_DESCRIPTION` (`subagents.py:280-300`) is prepended to the rendered subagent list and
contains usage note 7 — *"When only the general-purpose agent is provided, you should use it for
all tasks"* — plus a worked `general-purpose` example. This competes directly with the inert
`general-purpose` stub at `factory.py:157-175`.

It is overridable, but only through `_profile.tool_description_overrides.get("task")`
(`graph.py:772`). `HarnessProfile` is process-global — the same objection already documented at
`factory.py:157-166` for `HarnessProfile(...enabled=False)` — so overriding it would silently
affect the deep and adaptive control arms sharing the same model key. Not a blocker, but not a
one-line change either.

### 1.8 Recommendation (superseded by §1.9)

The original recommendation was to cut only the duplicated trigger prose from
`orchestrator.j2:63-133` and leave the ordering rules and `task()` brief templates in the prompt.
The reviewer asked instead for the **full union** in the descriptions, which is what shipped. §1.9
records the result.

### 1.9 What was implemented (2026-08-18)

Both prompt sections — `# Subagents` and `# Subagent Delegation Instructions`, 94 lines — were
removed and folded into the subagent descriptions. Each description now answers four questions in
a fixed order: **WHEN TO CHOOSE IT** (the original routing triggers), **SEQUENCING** (the prompt's
set-level "their order is fixed" list, distributed so each agent states its own constraint),
**WHAT IT PRODUCES**, and **DELEGATION BRIEF** (the verbatim `task()` template).

Because the planner and writer briefs are request-conditional, those two descriptions became
builder functions rather than module constants:

- `build_planner_subagent_description(*, parent_report_context_available)`
- `build_writer_subagent_description(*, parent_report_context_available, execution_enabled)`

`RESEARCHER_SUBAGENT_DESCRIPTION` has no conditionals and stayed a constant.

Content that did **not** move into a description, and why. The final layout separates three
concerns, each with one owner:

| Concern | Owner | Rule |
| :-- | :-- | :-- |
| **Routing + delegation briefs** — when to pick an agent, where it sits in the order, the verbatim `task()` template | Subagent descriptions (`factory.py`) | The reviewer's item 1 |
| **Delegation contract** — what one `ResearchQuery` must contain (`query`, `preferred_tools`, `target_components`, `rationale`, `depth`) and what the call returns | `_RESEARCH_BATCH_DESCRIPTION` (`tools/research.py`) | Properties of the call, rendered next to the schema |
| **Loop control** — running the research cycle across turns | Prompt, `# The Research Loop` | Orchestrator behavior, not delegation mechanics |

The loop section was rebuilt rather than restored verbatim. It absorbed three previously scattered
blocks — the old `The research loop:` list, the `When a lookup keeps failing:` ladder that had been
sitting under `# Tool Instructions`, and the `Stopping, and when the evidence falls short:` rules —
and reorganizes them as three terse stages: **Each pass** (ledger the query text, read every result
including its `evidence_judgment`, name what the answer still lacks), **When a pass comes back thin
or fails** (change the target not the phrasing, keywords not URLs, revise only failed queries,
escalate once, then leave), and **Leaving the loop** (sufficient/partial evidence, the
same-year/same-definition/same-source-family check for superlatives, the empty-registry
inability-to-verify rule, and `get_verified_sources` before writing).

It is deliberately compressed to **4 content lines / 1,404 characters**, against 997 characters for
the `The research loop:` list it replaced — and it also absorbs two blocks that used to live
elsewhere in the prompt, so the net prompt effect is a reduction. Prose is plain and declarative
("Write down every query you send", "Search tools need keywords, not URLs") rather than the
em-dash-heavy register used elsewhere in the file: this is the orchestrator's hot path, re-read on
every turn of every run, so it is optimized for fast parsing rather than for nuance.

`test_research_loop_stays_concise` fails the build above 6 content lines or 1,600 characters, and
`test_prompt_owns_the_research_loop` asserts all seventeen rules by content rather than by phrasing,
so a future reword that quietly drops one still fails. The scope boundary for maintainers is a Jinja
`{# ... #}` comment, verified unrendered by `test_research_loop_maintainer_note_is_not_sent_to_the_model`,
so it costs zero tokens per turn.

Four rules were dropped from the section outright, each verified to be single-homed elsewhere rather
than lost: routing independent unknowns to a batch versus a chain to `researcher-agent`
(`## Deciding what to do` plus the tool and subagent descriptions), "never fan out two queries at the
same unresolved fact" (the researcher description's SEQUENCING block), task-delegated notes being
persisted like batch notes (the same description's WHAT IT PRODUCES block), and budget exhaustion
ending research (`# Budgets`).

A token-coverage audit over the 59 excised content lines found exactly one line below 75% coverage
in its new home: the bare section header `The research loop:`, which carries no content.

That audit was too coarse at the sub-clause level. A later line-by-line trace found one clause with
no home: *"query IDs are meaningless to them"*. The token audit had passed the line because its
other clause, "full standalone context", matched elsewhere. The guard is still live —
`ResearchQuery.target_components` holds `ComponentId` values like `latest_price_anchor`, so an
orchestrator transcribing planner queries can name a component instead of spelling out the topic —
and it was restored to `_RESEARCH_BATCH_DESCRIPTION`'s `query` field guidance. Lesson for the
remaining items: audit moved prose by clause, not by line.

A cross-surface check confirms each concern now resolves to exactly one owner. The single string
appearing on all three surfaces is `evidence_judgment`, which is not duplicated guidance: the tool
description says the batch call returns it, the researcher description says that agent produces it,
and the prompt says to read it. Three different relationships to one field.

The prompt went from 266 to 180 lines.

#### The measured token result contradicts §1.5's hope

§1.5 warned that description text is paid for twice. That was confirmed empirically — a unique
marker placed in a subagent `description` appears in **both** the `task` tool description and
`SubAgentMiddleware.system_prompt`, because `create_deep_agent` does not pass a `system_prompt` to
the middleware, so it falls back to `TASK_SYSTEM_PROMPT` and the `"Available subagent types:"`
appendix is always built (`graph.py:764-774`, `subagents.py:684-685`).

Measured, in characters of stable per-turn footprint (prompt + descriptions counted twice + the
`run_research_batch` description), using the real bound tool descriptions:

| | prompt | descs x2 | tool | total |
| :-- | --: | --: | --: | --: |
| Baseline (HEAD) | 21,527 | 3,874 | 1,444 | 26,845 |
| After the consolidation alone | 14,921 | 12,268 | 1,585 | 28,774 (+7.2%) |
| After also cutting the duplicated tool lists (§1.10) | 10,977 | 12,268 | 1,585 | **24,830** (−7.5%) |

The consolidation on its own costs 7.2%, because the brief templates moved from a surface billed
once to a surface billed twice. §1.10 more than pays it back, so the net effect of item 1 is a
**7.5% reduction** against baseline.

That is an accepted trade, not an oversight: the reviewer's goal was a single authoritative home
for delegation guidance, and the previous arrangement had the briefs sitting one section away from
triggers that contradicted them in delta mode (see the table above). But it means item 1 should
**not** be credited with a token saving, and if the footprint becomes the binding constraint the
cheapest reversal is to move the three `DELEGATION BRIEF` blocks — roughly 3,400 characters,
doubled to 6,800 — back into the prompt while leaving triggers and sequencing in the descriptions.

Validation: `uv run pytest tests/aiq_agent/agents/` → 1344 passed, 54 skipped;
`uv run ruff check src/ tests/` and `ruff format --check` clean. Fourteen new tests in
`TestDelegationGuidanceLivesInDescriptions` pin the union, the prompt's ownership of the loop,
the absence of loop control from the tool description, the conditional briefs, the delta-mode
override, and the absence of the removed prompt sections.

---

## 2. Budgets in middleware rather than in the prompt

This is the strongest of the three comments. Enforcement is already a hybrid, and the split falls
in the wrong place.

### 2.1 What middleware already enforces

`AutonomousOrchestratorLoopGuardMiddleware` (`custom_middleware.py:567-766`) plus
`DeepResearchResourceLimits` and `ResearcherLoopGuardMiddleware` cover: batch-call count, total
delegated queries, duplicate `ResearchQuery` signatures, orchestrator model turns, research-tool
withdrawal on finalization, per-worker source-call budget by `depth`, and per-job byte and count
ledgers.

### 2.2 What the prompt asserts with nothing behind it

`orchestrator.j2:14-25`:

| Prompt claim | Actual enforcement | Observed behavior |
| :-- | :-- | :-- |
| "Batches: **one** per request" | `max_batch_calls: 6` in `config_autonomous_frag.yml:130` | 1.33 batches per trial |
| "`depth: high`: at most **one** per request" | Not enforced anywhere | 120 of 286 delegated queries (**42.0%**) declared `high`, with no aggregate F1 return |
| "Your own direct source-tool calls: at most **2** per request" | Not counted; only withdrawn once finalizing | **480** direct orchestrator source calls across 90 trials; outliers at 61 and 62 |
| "Never issue the same query twice, on any path — **the runtime blocks the duplicate**" | `awrap_tool_call` returns early for every tool except `run_research_batch` (`custom_middleware.py:699`) | — |

### 2.3 The false-promise row is the priority

The duplicate-blocking claim is factually wrong for the direct-source path and the `task` path. A
false promise is worse than silence: it tells the model not to track duplicates itself because the
runtime will. The cost is visible in the two worst trials:

| Trial | Input tokens | Direct source calls | Orchestrator calls | Result |
| :-- | --: | --: | --: | :-- |
| `deepsearchqa-0545` | 4,612,249 | 61 | 63 | F1 = 0, partial-result fallback |
| `deepsearchqa-0060` | 3,862,976 | 62 | 63 | F1 = 0, partial-result fallback |

In `deepsearchqa-0545` the parent prompt grew from ~11.5k to ~97k tokens while repeatedly
alternating Statistics Canada keyword searches and raw CSV URLs.

Note that both runaway trials terminated at 63 orchestrator calls — *inside* the
`max_orchestrator_turns: 100` envelope. The turn cap did not save them.

### 2.4 This restates the repo's own recommendation

`autonomous-researcher-token-and-f1-analysis.md` §3, priority 3: *"Add a hard orchestrator
direct-search budget and normalized duplicate guard, separate from `run_research_batch` budgets. A
small default such as 5–10 calls is consistent with the intended quick-lookup path."*

Priority 6 covers the `depth` clamp: *"Bias default query depth to low/medium and reserve high for
demonstrably chained retrieval."*

### 2.5 Achievability

High. Every prerequisite already exists in `AutonomousOrchestratorLoopGuardMiddleware`:
request-scoped counters on `self`, a documented count-before-await discipline that makes concurrent
tool calls in one turn share a single ceiling, `_blocked_result`, `_mark_finalizing`, and
`_filter_tools`. The work is to extend `awrap_tool_call` to also branch on
`name in self._source_tool_names` and on `TASK_TOOL`, and to reuse
`_canonical_research_query_signature` against the source tool's query argument.

Four additions, in decreasing order of expected value:

1. Direct source-tool call budget (request-scoped counter, small default).
2. Normalized duplicate guard covering direct source calls. **Narrowed during implementation
   (§2.8): direct-vs-direct only.** A genuinely cross-path guard — blocking a direct call that
   repeats an earlier `run_research_batch` query — would block the documented purpose of a direct
   call, which the prompt defines as verifying or disambiguating something a researcher returned.
   The runaway trials repeated direct against direct, which is the shape that is now caught.
3. `depth: "high"` clamp beyond the first occurrence per request.
4. Count `task(researcher-agent)` invocations against the same request budget as batches.

### 2.6 Caveats

1. ~~**Prompt text is still required — different text.**~~ **Overstated; corrected in review.**
   The claim that the prompt must keep restating intent alongside the middleware was right about
   *which* prose matters and wrong about *where it already lived*. A clause-by-clause audit (§2.8)
   found that all but two clauses of `# Budgets` were already single-homed in the research-loop
   section, the `run_research_batch` description, or the `# Tool Instructions` bullets. The section
   was duplication, not intent. The residual concern is real but smaller than stated: a first-time
   block costs one turn, which is why the direct-search path warns in-context via the nudge on the
   last allowed call rather than relying on the block alone.
2. ~~**If numbers stay in the prompt, render them from the config object.**~~ **Superseded:**
   no number stays in the prompt, so nothing needs rendering. This was the first implementation
   attempt and the reviewer rejected it — rendering `{{ request_budgets.max_batch_calls }}` fixes
   drift but still spends the tokens on every turn and still states a ceiling the model does not
   need until it hits one. The pattern is worth recording anyway, because it is the right answer
   whenever a prompt genuinely must quote a configured value: `factory.py` passes
   `researcher_source_call_budgets=researcher_loop_guard.source_call_budgets.model_dump()` into
   `researcher.j2`. The orchestrator prompt is the one place that hard-codes them instead. One
   source of truth permanently removes the "1 versus 6" contradiction.
3. **Withdrawal versus rejection.** Withdrawing a tool via `_filter_tools` costs no turn — the model
   never sees it. Rejecting costs one turn. But withdrawal mutates the tool list mid-run, which
   invalidates prompt caching for every subsequent turn. This is free today (both saved jobs report
   `n_cache_tokens = 0`) and becomes a liability the moment recommendation #8 is acted on.
4. **Clamp `depth`, do not reject the batch.** Silently downgrading the second and later `high`
   queries to `medium` avoids a wasted turn and loses no work. It must be logged — `_log_research_depths`
   in `tools/research.py` is the natural place — or the trajectory becomes misleading.
5. **Decide which number is true.** If the intent really is one batch, set `max_batch_calls: 2`
   (one batch plus one prerequisite-consumption batch) and let the middleware state it. The prompt
   currently errs conservative, which is the less harmful direction but is still drift.

### 2.7 Recommendation

Adopt. Land this change alone, without items 1 or 3, so the next eval attributes cleanly.

### 2.8 What was implemented (2026-08-18)

All four additions from §2.5 landed, and the prompt's `# Budgets` section was deleted. The guard
now owns every ceiling; the prompt states none.

**New config fields** on `AutonomousRequestTerminationConfig`, with `max_batch_calls` retuned:

| Field | Default | Replaces |
| :-- | --: | :-- |
| `max_batch_calls` | 6 (unchanged) | the prompt's "one per request" prose |
| `max_direct_source_calls` | **2** | nothing — this budget did not exist |
| `max_identical_direct_source_calls` | **1** | nothing |
| `max_high_depth_queries` | **1** | the prompt's unenforced "at most one per request" |

`max_batch_calls` keeps its existing value of 6. It was tried at 2 and then 3 before the per-trial
data showed a tightened ceiling was not supportable — see §2.10. What changed is its *scope*:
`task(researcher-agent)` now spends it too, closing the escape hatch. The prompt's old "one batch
per request" prose is gone either way, so the 1-versus-6 contradiction is resolved by deleting the
claim rather than by moving the number.

**`awrap_tool_call` became a dispatcher.** It previously early-returned for every tool except
`run_research_batch`, so direct source calls and `task` were invisible to it. It now routes to one
of three guards (`_guard_direct_source_call`, `_guard_delegation`, `_guard_research_batch`), with
everything else — `think`, `get_verified_sources`, the finalizer, the filesystem tools — still
passing through untouched.

Three design points in the implementation are load-bearing and easy to "fix" into a regression;
each is pinned by a test.

1. **Exhausting the direct-search budget does not finalize the request.** `_mark_finalizing`
   withdraws `run_research_batch` and `think` as well, which would push the model to *answer* when
   the intended response is to *delegate*. `_filter_tools` therefore has two independent
   withdrawal branches: finalizing hides everything, a spent direct budget hides only the source
   tools. This is eval recommendation #4 expressed as a mechanism, and it is where the design
   deliberately diverges from adaptive's single-shot budget, where exhaustion *does* mean finalize.
2. **`task` is never withdrawn.** `task(writer-agent)` is one of the two ways a run legitimately
   ends, so hiding the tool would strand a finalizing run with no writer exit. The researcher
   delegation is gated by `subagent_type` inside `awrap_tool_call` instead — the same reason
   `PlanBeforeWriterMiddleware` gates there rather than in a tool filter.
3. **Extra `high` queries are clamped, not rejected.** Rejecting a five-query batch to correct one
   `depth` field discards four good queries and spends a model turn saying so. The allowance is
   committed only once the batch is admitted, so a batch rejected by a later check does not
   silently consume it. Clamping is invisible to duplicate detection because the request-wide
   signature excludes `depth` — an earlier version did not, which made the clamp a duplicate
   bypass (§2.9, defect 1).

**Everything was reused rather than invented.** `_canonical_source_signature` came from
`adaptive_researcher/custom_middleware.py:1099` (this module already imported
`_canonical_research_query_signature` from the same place), the reserve-before-await discipline and
the nudge-plus-withdraw pattern from `ComplexityRouterMiddleware`, and the
`request.override(tool_call=...)` arg rewrite from `SingleShotShallowDelegationMiddleware`.
`AutonomousResearchQuery` was deliberately left alone: it is an *alias* for `AdaptiveResearchQuery`
(`models/__init__.py`), so a model-level clamp would have silently moved the control arm.

**Prompt: `# Budgets` was deleted outright.**

The first attempt kept the section and rendered every number from
`request_budgets=request_termination.model_dump()`. That fixes drift but misses the point of the
review comment — the budgets were still *defined in the prompt*, still re-sent on every turn, and
still stating ceilings the model does not need until it reaches one. The reviewer corrected it and
the section was removed.

Before deleting, each clause was audited for an existing home — the lesson recorded in §1.9 was to
audit moved prose by clause, not by line. All but two clauses were already single-homed:

| Clause | Existing owner |
| :-- | :-- |
| one batch; a second only for a resolved prerequisite | `_RESEARCH_BATCH_DESCRIPTION` |
| never re-ask in different words; a thin pass needs a different target | `# The Research Loop` |
| a batch of one is legitimate | `_RESEARCH_BATCH_DESCRIPTION` |
| `low` / `medium` / `high` selection | `_RESEARCH_BATCH_DESCRIPTION` |
| direct calls are for verification; their results persist in context | `# Tool Instructions` |
| never repeat a query; a repeat is blocked and costs a turn | `# The Research Loop` |
| what to do once the direct budget is spent | `_DIRECT_SOURCE_BUDGET_NUDGE`, in-context when it fires |
| what to do once research is over | the blocked `ToolMessage` bodies |

So `# Budgets` was overwhelmingly duplication, the same finding as §1.10's tool lists. Two clauses
were genuinely single-homed and were rehomed rather than dropped:

- **The per-batch query ceiling** moved into the `run_research_batch` description, rendered from the
  same `max_research_concurrency` the tool validates against. This is the one ceiling worth telling
  the model in advance, because exceeding it is rejected outright — a guaranteed wasted turn.
- **"Re-running a worker's question yourself is not a repeat — that is verification"** moved into
  the `# Tool Instructions` direct-source bullet, which already owned the surrounding judgment.
  Without it the truthful duplicate rule reads as forbidding the one thing direct calls are for.

Net: the section removed 21 lines / 2,384 characters, and the orchestrator prompt is now 115 lines
/ ~10.2k characters. `render_prompt("orchestrator", ...)` no longer passes `request_budgets` or
`max_research_concurrency` at all.

**Validation.** 1377 passed, 54 skipped across `tests/aiq_agent/agents/`; ruff clean. Thirty new
tests, six of them the §2.9 regressions. Two mutations were used to confirm the tests are not vacuous: widening the direct-budget
withdrawal to include `run_research_batch` fails
`test_spent_direct_budget_withdraws_sources_but_keeps_the_delegation_paths`, and moving the direct
counter after the `await` fails `test_parallel_direct_calls_share_one_hard_ceiling` plus two others.
(Four pre-existing `test_default_model_profiles` failures are unrelated — they reproduce on a
stashed tree and come from scratch configs plus the 2.2.0 endpoint deprecation.)

### 2.9 Four defects found in review, and fixed

A second review pass found four real defects in the §2.8 implementation. All four were reproduced
before fixing and are now pinned by regression tests.

**1. The clamp was a duplicate bypass.** `depth` was part of the request-wide query signature while
the clamp rewrote `depth`, so the same `high` question re-sent in a later batch was clamped to
`medium`, hashed differently from its own earlier run, and executed again. The §2.8 note claiming
the clamp-then-sign order *produced* collisions had it exactly backwards.

Fixed by giving the request-wide ledger its own signature, `_canonical_request_query_signature`,
which excludes `depth`. That is the right key on its own merits: asking the same question harder is
still asking the same question, and "never re-ask for more depth" is already the stated rule. It
also closes a case the original never covered — the same query sent at `low` and then at `high` by
the model's own choice, with no clamp involved. The shared
`_canonical_research_query_signature` is left alone, because adaptive is the control arm.

**2. Duplicates *inside* one batch were not blocked.** Every signature was checked against the
committed ledger before any was recorded, so two copies of a query in one batch each saw a count of
zero and both ran, recording a count of 2 against `max_identical_research_queries: 1`. Fanning two
workers at one question is precisely the waste the guard exists to stop, and the eval's
`deepsearchqa-0543` failure did exactly this. Fixed with a `seen_in_batch` tally consulted
alongside the ledger. The existing test only covered duplicates across two calls.

**3. The direct-call duplicate guard was exact-JSON, not normalized.** `_canonical_source_signature`
sorts keys but leaves string *values* untouched, so `"same"` and `"  SAME  "` hashed differently
and both executed — contradicting the config field's own "normalized signature" wording. Fixed with
`_canonical_direct_source_signature`, which applies the same `_normalize_text` (NFKC,
whitespace-collapsed, casefolded) the query signature uses. Left the adaptive helper alone: inside
one researcher invocation the budget is small and short-lived, whereas this ledger spans the whole
request.

**4. Budget counts survived in the bound tool description.** `# Budgets` was deleted from the system
prompt, but `_RESEARCH_BATCH_DESCRIPTION` still said "Issue ONE batch per request" and "`high`: at
most one per request". A tool description is model input exactly like the prompt, so this did not
remove the drift — it moved it somewhere less visible, and configuring `max_batch_calls` or
`max_high_depth_queries` differently recreates the original mismatch. The counts are gone; the
routing intent ("prefer a single well-formed batch", "reserve `high` for a genuine chain") stays.
The one surviving number is the per-call query cap, interpolated from the same
`max_research_concurrency` the tool validates against, so it cannot drift.
`test_bound_tool_descriptions_state_no_budget_counts` now scans every bound description, not just
the system prompt.

The through-line: three of the four are the same mistake in different places — trusting a
normalization or an ordering without testing the adversarial input. The fourth is the same
*category* error the whole item was meant to fix, committed one file over.

### 2.10 Where the ceilings were placed, and how

The three new ceilings were checked against the per-trial data in
`jobs/2026-08-13__19-24-13/per_trial_behaviour.csv` rather than set by intuition. Two held; one did
not.

**`max_direct_source_calls: 2` — holds.** The distribution is bimodal, and the mean of 5.33 is
misleading: the *median* is 0.

| direct calls | 0 | 1 | 2 | 3–9 | 10–62 |
| :-- | --: | --: | --: | --: | --: |
| trials | 47 | 15 | 3 | 13 | 12 |

65 of 90 trials (72%) already sit within the cap. And F1 falls monotonically with direct-search
volume — trials exceeding 2 average F1 0.3626 against 0.5665 for the rest; trials exceeding 10
average 0.1414. The cap bites the failing tail, not the working median.

**`max_high_depth_queries: 1` — holds by construction.** It clamps rather than rejects, so no work
is lost and no turn is spent. The eval found trials using `high` scored about the same as those
that did not, so there is little upside to protect.

**`max_batch_calls` — tightening did not hold; left at 6.**

| ceiling | binds on | mean F1 of bound trials | mean F1 of the rest |
| --: | --: | --: | --: |
| 2 | 11/90 (12%) | **0.5379** | 0.5059 |
| 3 | 4/90 (4%) | 0.3750 | 0.5161 |
| 6 | 0/90 | — | — |

At 2 the trials being cut scored *above* average. At 3 they score below it, but only 4 trials are
involved. At 6 nothing binds at all: the most batches any trial issued was 5.

The original rationale — "one-batch runs scored best" — was true but incomplete: the three-plus
bucket (0.5379) also beats the two-batch bucket (0.3744), so forcing those trials into a two-batch
shape is not obviously an improvement. The decision was to leave the ceiling at 6 as a **runaway
backstop rather than a shape control**, for three reasons:

1. It is the harshest ceiling to hit — exceeding it calls `_mark_finalizing`, withdrawing
   `run_research_batch`, `think`, *and* every source tool. It does not refuse one call, it ends
   research. A ceiling with that failure mode should only fire on unambiguous runaway.
2. The runaway trials were **direct-search** loops (61 and 62 direct calls), not batch loops. The
   budget that addresses them is `max_direct_source_calls`, which does bind.
3. With duplicate detection now correct (§2.9, defects 1 and 2), a third or fourth batch can only
   carry *genuinely new* queries. Re-asking is blocked on its own merits, which is the behavior the
   batch cap was standing in for.

What did change is the ceiling's scope: `task(researcher-agent)` now spends it alongside
`run_research_batch`, so the delegation door is no longer a free path around it.

All of these buckets are observational and task difficulty is confounded — harder questions
plausibly cause both more searching and lower F1. They bound where a ceiling is *defensible*, not
where it is optimal.

### 2.11 Caveats carried forward

- **Attribution is now confounded.** §4 recommended landing item 2 alone. Item 1 is already in the
  working tree uncommitted, so the next eval carries consolidated subagent descriptions, ~3.9k
  fewer duplicated characters per turn (§1.10), *and* these budgets. The writeup must say so rather
  than attributing any delta to the guards alone.
- **`max_batch_calls: 2` cuts the three-plus-batch bucket** (11 trials / 12%). Favorable on the
  measured data, but worth watching for tasks that genuinely need staged research.
- **Prompt caching.** Withdrawing tools mid-run mutates the tool list and invalidates the cache for
  every later turn. Free today (`n_cache_tokens = 0` in both saved jobs) and a liability the moment
  eval recommendation #8 is acted on. The direct-budget withdrawal adds a second trigger for this.
- **Out of scope, but noticed.** `config_autonomous_frag.yml` sets the *researcher* `source_call_budgets`
  to `low: 5 / medium: 10 / high: 20`, against model defaults of `1 / 3 / 6` — a roughly 3×
  amplification of worker source calls per query, entirely separate from anything changed here, and
  a plausible contributor to the remaining token gap.

---

## 3. Removing `run_research_batch` in favor of `task` delegation

The underlying observation is legitimate: one capability behind two doors, with observed usage
collapsing almost entirely onto one of them (120 batch calls versus 3 `task(researcher-agent)`
calls). The proposed remedy removes the door that is actually used.

### 3.1 Three objections that do not apply

| Concern | Finding |
| :-- | :-- |
| Loss of parallelism | Does not apply. `langgraph/prebuilt/tool_node.py:858` executes tool calls under `asyncio.gather`, and `TASK_TOOL_DESCRIPTION` usage note 1 explicitly instructs the model to fan out in a single message. |
| Loss of structured output | Does not apply. `subagents.py:506-513` serializes `structured_response` into the returned `ToolMessage`, so `ResearchNotes` survives the `task` boundary. |
| Loss of note persistence and source registration | Does not apply. `ResearcherTaskPersistenceMiddleware` (`custom_middleware.py:276-390`) already gives the `task` path both side effects on the run's own registry instance. |

`files` state also merges rather than clobbers across concurrent `task` returns: the key is
annotated with `DeltaChannel(_file_data_delta_reducer)` at `filesystem.py:310`.

### 3.2 What actually breaks

1. **`depth` disappears, and with it the per-worker loop guard.** `TaskToolSchema`
   (`subagents.py:265-278`) carries exactly `description` and `subagent_type`. With no
   `ResearchQuery` there is no `depth` to seed `CURRENT_RESEARCHER_GUARD_STATE`, so
   `ResearcherLoopGuardMiddleware` passes through. `factory.py:_researcher_subagent_spec` already
   documents this for the one-off case and argues it is an acceptable trade *there*. Making `task`
   the only path makes **every worker unbudgeted**. This is the primary blocker.
2. ~~**The request-wide guard becomes a no-op.**~~ **Substantially weakened by §2.8.** This
   objection rested on `custom_middleware.py:699` returning early for every tool except
   `run_research_batch`. That early return is gone: `task(researcher-agent)` now spends
   `max_batch_calls` and `max_total_research_queries` and is refused once the request is
   finalizing. What still degrades is *per-query* accounting rather than request-wide accounting —
   a free-text delegation is budgeted as one query whatever it actually does, and
   `max_identical_research_queries` cannot fire against it because there is no `ResearchQuery` to
   normalize (see point 4). So this is now a real but second-order cost, not the blocker it was
   written as. Point 1 remains the primary blocker.
3. **`preferred_tools` is lost.** That field is the mechanism the prompt's entire
   `## Retrieval Tools` block exists to feed. Free-text delegation cannot steer source selection.
   `target_components`, `rationale`, and `subqueries` are lost with it.
4. **Duplicate detection degrades.** `_canonical_research_query_signature` normalizes over query
   text, sorted target components, sorted preferred tools, and depth. Against free-text task
   descriptions the same question phrased two ways will not collide.
5. **The per-job ledger needs reimplementing.** `build_autonomous_research_batch_tool` tracks
   `consumed_queries`, `consumed_query_chars`, `persisted_note_count`, and `persisted_note_bytes`,
   with rollback on persistence failure. All of it is keyed to queries.
6. **No concurrency cap.** The batch tool hard-fails past `max_research_concurrency`. Nothing
   prevents the model from issuing twelve `task` calls into one `asyncio.gather`. Enforcing a
   per-turn cap across separate tool calls requires new machinery (grouping by parent AI message
   id), because `awrap_tool_call` fires once per call.
7. **State inheritance changes.** `_validate_and_prepare_state` (`subagents.py:534-540`) hands the
   subagent the parent's entire state minus `_EXCLUDED_STATE_KEYS`, including `files`. Batch
   workers receive a purpose-built state instead. This may well be an improvement — the researcher
   could read `/shared/plan.json` — but it is a behavior change to measure, not to assume.

### 3.3 The inversion

A one-query batch is already documented as legitimate (`tools/research.py`,
`_RESEARCH_BATCH_DESCRIPTION`). A one-query batch at `depth: "high"` delivers exactly the
open-ended multi-hop chain that `task(researcher-agent)` exists to provide — with a budget, with
tool steering, and with duplicate detection. `run_research_batch` therefore strictly subsumes the
`task` route.

If the goal is a single door, the correct move is to **keep `run_research_batch` and drop
`researcher-agent` from the subagent list** — the inverse of the comment. That also removes most of
the reason `ResearcherTaskPersistenceMiddleware` and the `general-purpose` stub need to exist.

### 3.4 The counter-argument, stated fairly

The comment implicitly bets that the model ignores `researcher-agent` *because* the batch tool
exists, and would use it if it were the only option. That is plausible and not refuted by the data.
But it is the same "descriptions will route correctly" bet that produced 0 planner and 0 writer
calls in §1.6, and this time it would be riding on the one path the agent actually uses.

### 3.5 Recommendation

Do not remove `run_research_batch`. Address the duplication either by unifying the request budget
across both routes (which is item 2, addition 4) or by removing `researcher-agent` from the `task`
list. Whichever is chosen, do not land it in the same eval cycle as item 1 — both change routing
behavior and would confound attribution.

---

## 4. Sequencing

1. **Item 2 first, alone.** It has hard evidence behind it, a ready-made home in existing
   middleware, and no design argument against it.
2. **Re-run the eval** to separate the 08-17 prompt revisions from the new guards.
3. **Items 1 and 3 afterwards, separately from each other.** Both alter routing behavior; landing
   them together makes the next eval uninterpretable.

## 5. Open questions for the reviewer

1. ~~Is `max_batch_calls: 6` or the prompt's "one batch per request" the intended contract?~~
   **Answered:** neither exactly — `max_batch_calls: 2`, i.e. one batch plus one
   prerequisite-consumption batch, which is what the prompt's prose already described. Both
   numbers are now read from the same config object, so the question cannot recur.
2. ~~Should the direct source-tool budget block (costs a turn, explicit message) or withdraw the
   tools (costs no turn, breaks prompt caching)?~~ **Answered by existing precedent:** do both, as
   `ComplexityRouterMiddleware` already does. Withdraw on the next model call, append a nudge to
   the result of the call that spends the last slot so the withdrawal is explained in-context, and
   keep the block purely as the backstop for a call already in flight.
3. If item 3's duplication is to be resolved by deletion rather than by budget unification, is
   removing `researcher-agent` from the `task` list acceptable, given that it also removes the
   documented no-depth-cap escape hatch for open-ended chains?


### 1.10 Removing the prompt's duplicated tool lists (2026-08-18)

Raised in review: if deepagents injects the subagent list, does it not also inject the tools — and
was `## Your Tools (callable)` / `## Retrieval Tools` therefore a second copy?

It was. Tool `name` and `description` reach the model through `bind_tools`
(`langchain/agents/factory.py:1367`), which is what the provider renders as the tool-definitions
block. The prompt then re-rendered the same two fields from
`tools_info = [{"name": t.name, "description": t.description} for t in tools]`
(`deep_researcher/factory.py:217`) — the identical values, not a summary.

Verified before cutting, by binding a recording chat model to the real graph and dumping every
tool the orchestrator model actually receives:

- **14 tools bound**, each with its full description.
- The **6** tools the prompt listed were all bound, with **byte-identical** descriptions.
- **8** tools — `task`, `write_todos`, `read_file`, `write_file`, `edit_file`, `ls`, `glob`,
  `grep` — were *never* listed in the prompt and are used successfully from the schema alone.
  That last point is the load-bearing evidence: schema-only tool knowledge already works in this
  agent for most of its tools.

Both blocks were removed. The one clause in the `## Retrieval Tools` preamble not already stated
in `# Tool Instructions` — *why* a direct call is expensive, namely that its raw results stay in
the conversation and are re-sent every turn — was folded into the existing "Direct source tools"
bullet. Net **−3,944 characters per orchestrator turn**.

A second reason beyond token cost: the prompt copy could not be kept in step with the schema.
`AutonomousOrchestratorLoopGuardMiddleware._filter_tools` withdraws `run_research_batch`, `think`,
and every source tool once the request is finalizing. The bound schema loses them; a rendered
prompt list cannot. The model was being told it could call tools that had already been taken away.

`test_prompt_does_not_duplicate_bound_tool_schemas` now fails the build if any bound tool's
description (>40 chars) reappears in the prompt.

**Scope caveat.** This pattern is repo-wide — `deep_researcher/prompts/orchestrator.j2:177` and
`adaptive_researcher/prompts/orchestrator.j2:354,359` do the same — and was changed for the
autonomous agent only. For adaptive it is *not* duplication on the `standard` and `deep` tiers,
where the orchestrator deliberately does not hold source tools, so `## Retrieval Tools` is the only
place those names appear; a blind port would break it. Because adaptive is the control arm, the
next autonomous-vs-adaptive comparison now differs in two ways — consolidated subagent descriptions
*and* ~3.9k fewer duplicated characters per turn — and the writeup must say so rather than
attributing any delta to routing alone.
