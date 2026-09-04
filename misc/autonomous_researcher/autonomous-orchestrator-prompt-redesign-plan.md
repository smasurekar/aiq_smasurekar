<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Autonomous Researcher — Orchestrator Prompt Redesign Plan

**Status:** Appendix A (prompt) and Appendix B (descriptions) **implemented 2026-08-17**, with the
routing section revised once post-review (Revision 2). §8.1's probe set is **implemented and run**
— 22/24, see its results table. **Revision 3 (§4.6) is specified and not yet implemented.** §6
(C1–C7) remains deferred until after the Harbor eval, except that C3 is now *confirmed* rather than
speculative. See §7 for a post-implementation token correction.
**Target file:** `src/aiq_agent/agents/autonomous_researcher/prompts/orchestrator.j2`
**Author date:** 2026-08-17

---

> **Revision 2 (2026-08-17) — the opening-move ladder was rejected and removed.**
>
> The A–E ladder specified in §4.2 and drafted in Appendix A was implemented, reviewed, and then
> pulled. The objection, which is correct: an enumerated set of request kinds that the model
> selects from before acting **is** a tier system, whether or not middleware enforces the
> selection. The mapping was close to 1:1 — A↔`direct`, B↔`single_shot`, C↔`standard`-inline,
> E↔`deep` — because the ladder was derived by reading the adaptive prompt. Removing
> `declare_effort_tier`, section swapping, and tool hiding changed *who enforces* the taxonomy,
> not *whether there is one*. This agent exists to drop the taxonomy.
>
> **What shipped instead:** a section that states two *properties the model reads off the request*
> rather than categories it sorts the request into — do the unknowns depend on each other, and is
> the deliverable's structure part of the ask. Neither is an effort level; both are capability
> facts about the tools (independence is exactly what parallel workers can and cannot exploit; a
> fixed structure is what `/shared/plan.json` exists to record). They compose, so a request can
> exhibit both and get both. There is nothing to classify into and no ordering to climb.
> See §4.2R below; it supersedes §4.2 and the ladder block in Appendix A.
>
> Everything else in this plan survived the reversal unchanged. The answer-set contract (§4.4),
> the failed-lookup exit (§4.5), the stated budgets (§4.3), the prerequisite rule, and the
> numeric-comparison check are behavioral rules that apply regardless of how much work a request
> warrants — they were never tier-shaped and they are what target the measured precision and token
> defects. Only the routing section was taxonomic, and only it was replaced.
>
> §8.1's probe set needs rewording: assert **which route** a request takes, not which "shape" it
> is assigned. The four buckets and their assertions are otherwise still valid.

> **Revision 3 (2026-08-17) — pipeline ordering, agent identity, and output parity. NOT YET
> IMPLEMENTED; specified in §4.6.**
>
> Four gaps raised in review, all of which the shipped prompt leaves implicit or absent:
>
> 1. **The planner must be used first**, not part-way through a run.
> 2. **The writer must be used last**, as the terminal step.
> 3. **The identity line is too thin** — "You are a research orchestrator" says nothing about what
>    this agent is, what it produces, or what it guarantees.
> 4. **Output format must match `deep_researcher` / `adaptive_researcher`.** Investigation found
>    the real divergence is not where it looks: the writer *prompt* is already byte-identical
>    across all three arms, but autonomous does not pin the writer's **delegation message**, which
>    is also part of the writer's input. That is the actual parity gap.
>
> Rules 1 and 2 are ordering invariants, not a taxonomy — they constrain *sequence*, not *how much
> work a request deserves* — so they do not reopen Revision 2.

---

## 0. Purpose and scope

The autonomous researcher was built to replace the adaptive researcher's tier-driven
prompt/tool/budget switching with **rich tool descriptions + rich subagent descriptions + one
generic orchestrator prompt**, letting the orchestrator infer depth implicitly. Adaptivity was
meant to be *emergent* rather than *declared*.

Two independent analyses of the same 90-task DeepSearchQA job show the emergence did not happen.
The agent did not become adaptive; it became **uniformly deep**, at the worst F1 and the highest
cost of the four arms measured.

This plan diagnoses that outcome **as a prompt-design failure**, specifies a replacement
orchestrator prompt, and separates the parts a prompt can fix from the parts it cannot. It does
**not** propose reintroducing effort tiers, `declare_effort_tier`, per-tier tool hiding, or
`dynamic_orchestrator_sections`. The design stays deepagents-idiomatic: one prompt, full tool
menu, no enforcement machinery. What changes is that the prompt stops *describing* capabilities
and starts *prescribing a decision procedure with worked examples and stated budgets*.

**Deliverable of this document:** the diagnosis, the section-by-section design, a complete draft of
the new prompt (Appendix A), the companion description edits (Appendix B), and a validation plan.
Implementation is a separate step.

### Scope boundary

| In scope | Out of scope (this plan) |
| :-- | :-- |
| `prompts/orchestrator.j2` rewrite | Reintroducing tiers or a router middleware |
| Subagent description strings in `factory.py` | Researcher / planner / writer prompt rewrites |
| `run_research_batch` tool description in `tools/research.py` | New retrieval tools (URL fetch) — flagged, not designed here |
| A validation/probe suite for routing behavior | Harbor image rebuild mechanics |

---

## 1. Evidence base

Everything below is grounded in these absolute paths. No claim in §2 is inferred without one.

**Analysis reports**

- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/misc/autonomous_researcher/autonomous-researcher-token-and-f1-analysis.md`
- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/AUTONOMOUS_AGENT_ANALYSIS.md`

**Eval jobs**

- Autonomous: `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/`
- Adaptive (A/B control): `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21/` and `/2026-08-03__11-44-56/`
- Adaptive V2: `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-07__16-40-55/`
- Chat researcher: `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-07-31__18-57-45/`

**Source under change**

- `src/aiq_agent/agents/autonomous_researcher/prompts/orchestrator.j2` (current, 202 lines)
- `src/aiq_agent/agents/autonomous_researcher/factory.py` (subagent descriptions, lines 110–165; tool menu, lines 445–470)
- `src/aiq_agent/agents/autonomous_researcher/tools/research.py` (`_RESEARCH_BATCH_DESCRIPTION`, lines 59–75)
- `configs/config_autonomous_frag.yml`

**Reference prompts consulted**

- `src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2` — the tier-gated control that wins on F1 at 43 % of the tokens
- `src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2` — the workhorse of the best-scoring arm (chat researcher, F1 0.6078)
- `/home/smasurekar/Desktop/Swapnil/github_repos/deepagents/examples/deep_research/research_agent/prompts.py` — LangChain's own deep-research orchestrator prompt
- `/home/smasurekar/Desktop/Swapnil/github_repos/deepagents/libs/deepagents/deepagents/graph.py` (`_LEGACY_BASE_AGENT_PROMPT`)
- `/home/smasurekar/Desktop/Swapnil/github_repos/deepagents/libs/deepagents/deepagents/middleware/subagents.py` (`TASK_TOOL_DESCRIPTION`)
- `/home/smasurekar/Desktop/Swapnil/github_repos/deepagents/libs/deepagents/deepagents/profiles/harness/_nvidia_nemotron_3_ultra.py` (`_SYSTEM_PROMPT_SUFFIX`, lines 1730–1790)

### 1.1 The numbers the redesign must move

| Metric | Autonomous | Adaptive (control) | Chat researcher (best) |
| :-- | --: | --: | --: |
| Mean F1 | **0.5098** | 0.5311 | **0.6078** |
| Precision | **0.5191** (worst of 4) | 0.5678 | 0.6156 |
| Recall | 0.5501 | 0.5385 | 0.6232 |
| Fully correct | 30.0 % | 28.9 % | **42.2 %** |
| Correct-but-excessive | 13.3 % | 10.0 % | **6.7 %** |
| Avg input tokens | **999,938** | 431,533 | **295,080** |
| Avg output tokens | 31,787 | 8,712 | 8,394 |

Routing census, 90 trials:

| Route | Calls | Trials using it |
| :-- | --: | --: |
| `run_research_batch` | 120 calls / 286 workers | 74 / 90 |
| direct source tools (orchestrator) | **480** | — |
| `task(researcher-agent)` | **3** | 1 / 90 |
| `task(planner-agent)` | **0** | **0 / 90** |
| `task(writer-agent)` | **0** | **0 / 90** |

Run-shape outcome table (the single most actionable result in the whole analysis):

| Run shape | Trials | Mean F1 | Avg input tokens |
| :-- | --: | --: | --: |
| No delegation (inline only) | 15 | 0.5039 | 667,089 |
| **One batch** | **43** | **0.5709** | **644,062** |
| Two batches | 21 | **0.3744** | 1,463,692 |
| Three or more | 11 | 0.5379 | 1,959,632 |

---

## 2. Diagnosis — nine defects, all of them in the prompt

Each defect states the mechanism, the prompt text responsible, and the measurement that confirms it.

### D1 — The prompt has no decision surface for *how much work*, only for *which tool*

`## Choosing a research path` (lines 34–44) is the section the model reads when it decides to
research. It differentiates the three paths on **"context isolation and scale, not by how hard the
question is"** — explicitly disclaiming the axis the model actually needs to resolve first.

`## Calibrating effort` (lines 47–56) does address that axis, but it is placed *after* the routing
menu, is framed as advisory (**"These are heuristics, not modes — nothing is gated on them"**), and
gives no worked examples and no opening move.

The adaptive control does the opposite. Its prompt opens with `## Choosing Effort` /
**"Before anything else:"** and — under `catalog_mode` — carries a literal *"First action by level"*
table naming the exact tool call to emit now. That table is the mechanism the autonomous prompt
deleted along with the tiers, and nothing replaced it.

> **Confirmation:** mean *final* orchestrator turn count is **3.0** against a
> `max_orchestrator_turns` cap of 100, and 43/90 runs are the identical shape (one batch → answer).
> The model is not choosing depth; it has one default and applies it everywhere.

### D2 — The planner is absent from the routing menu that the model actually consults

`## Choosing a research path` lists **three** options: direct source call, `researcher-agent`,
`run_research_batch`. `planner-agent` appears nowhere in it. Its only mention in the entire
workflow is one bullet inside the advisory `## Calibrating effort` block, gated on the phrases
*"comprehensive report"*, *"deep dive"*, *"analyze X across Y and Z"* — stylistic triggers that
never occur in a DeepSearchQA item.

`writer-agent` is unreachable by construction: `PlanBeforeWriterMiddleware`
(`custom_middleware.py:394`) rejects a writer call before `/shared/plan.json` exists, and only the
planner writes that file. **Zero planner calls therefore forces zero writer calls.**

Worse, the prompt offers a cheaper competitor for the same job. `## Workflow` step 2 tells the
model to **"Track multi-part work with `write_todos`"**. Given a multi-part request, the model can
satisfy the stated instruction with one cheap local tool instead of a delegation.

> **Confirmation:** `write_todos` 19 calls, `planner-agent` **0** calls, `writer-agent` **0** calls,
> across 90 trials. In a design where descriptions are the routing logic, the planner description
> is not winning a single routing decision.

### D3 — Direct source calls are advertised as the cheap path, with no budget and no exit rule

Line 38 reads: *"**Call a source tool directly** — … Cheapest path: one tool call, no sub-run."*
That is the opposite of true in this architecture. Every raw search result lands in the long-lived
parent context and is re-billed on every subsequent orchestrator turn (`n_cache_tokens = 0` across
all 90 trials, so there is no caching to absorb it).

Nothing in the prompt caps direct calls, and nothing tells the model what to do when a direct
lookup fails repeatedly.

> **Confirmation:** 480 orchestrator direct source calls (5.33/trial), 393 of them the expensive
> `advanced_web_search_tool`. Orchestrator input tokens: **31.9 M autonomous vs 7.7 M adaptive** —
> a 24.2 M gap that more than explains the whole 15.7 M net job difference. Worker prompt cost per
> call is effectively identical between arms (17,843 vs 17,930), so the extra spend is *entirely*
> parent-context growth.
>
> The failure mode has a name in the data: **thrash-to-zero**. `deepsearchqa-0060` — 0 batches,
> 62 direct `advanced_web_search_tool` calls, **73.4 %** duplicate queries, 3.86 M input tokens,
> F1 = 0, never reached `submit_final_report`. `deepsearchqa-0545` — 151 queries, 41.7 % duplicates,
> a Statistics Canada table id re-issued ×17, F1 = 0. Both trials pasted a **URL into a keyword
> search tool**, which cannot fetch it, and re-issued it until the run died.

### D4 — No answer-set contract; the citation contract governs citations, not answer *shape*

`## Inline citation contract` is thorough about locators and says nothing about **which entities
are allowed to appear in the answer**. The DeepSearchQA set grader extracts candidate names from
the answer body; a name listed *in order to reject it* still counts as an excessive item.

> **Confirmation:** 153 correct items vs **153 excessive items** (adaptive: 159 / 84 — 82 % fewer
> excessive). `deepsearchqa-0314` (exhaustive *Tyrannosaurus* types) has recall **1.0** and
> precision **0.1333** because the answer also names 13 invalid species, synonyms, and nomina nuda
> — each correctly labeled invalid in the prose, each counted against it. Adaptive: precision 1.0,
> 0 excessive.
>
> This is not verbosity. Autonomous answers are *shorter* on average (2,625 chars vs 2,871). It is
> a **candidate-selection** defect: the agent refuses to commit to a set.

### D5 — No prerequisite/dependency rule, so chained questions get parallelized

The only anti-pattern the prompt states is *"Do not use `run_research_batch` for a single query"*.
There is no rule against fanning out a question whose second half depends on the first half's
answer — which is the dominant shape in DeepSearchQA (*identify entity X, then list X's Y*).

> **Confirmation:** `deepsearchqa-0543` (shuttle with the second-fewest construction milestones →
> that shuttle's EAFB missions). Autonomous submitted **two parallel, near-duplicate `depth:"high"`
> queries about the same prerequisite**, then made 21 direct parent-context calls, selected
> Atlantis instead of Endeavour, and returned 12 Atlantis missions — every one excessive.
> P = 0, R = 0, F1 = 0, 2.62 M input tokens. Adaptive used two *staged* batches, selected
> Endeavour, F1 = 0.9231 on 348 k tokens.

### D6 — A second batch has no stated purpose, and the model uses it to re-ask

The prompt says *"Repeat as needed while each new step is likely to add material evidence"* and
*"If a first attempt returns thin evidence … send a distinct, refined query, widen the fan-out, or
step up to a plan."* All three of those licence a second batch as **more of the same**, which is
precisely the compounding-bad-framing failure.

> **Confirmation:** two-batch runs are the **worst bucket in the job** — n = 21, F1 **0.3744**,
> 1.46 M tokens, 14.3 % fully correct — against the one-batch bucket's F1 0.5709 at 644 k tokens.
> A second batch as currently prompted is actively harmful.

### D7 — Depth is over-declared because `high` is described but never priced

`## Research depth (per query)` defines the three levels neutrally and names `medium` the default.
Nothing says `high` is expensive or that it should be rare.

> **Confirmation:** 120/286 queries marked `high` (42 %) vs adaptive's 88/351 (25 %); only 15 % low.
> Trials with ≥1 high-depth query score **the same** F1 as trials without (0.5116 vs 0.5076) while
> costing **1.15 M vs 812 k** tokens. The depth escalation buys nothing measurable.

### D8 — The meta / no-research path is one sentence with no exit contract

The autonomous prompt's entire treatment is line 56 (*"Only chit-chat, greetings, and questions
about your own capabilities need no research at all"*) plus one clause in `## Finishing a run`.
The adaptive prompt, by contrast, gives the meta path **its own named section with a literal
worked `submit_final_report(...)` call**.

For the user's criterion *"handle small meta/direct queries very easily without any tool call"*,
the current prompt supplies a permission but not a procedure — and it sits immediately under
`## Never answer from memory`, a section written in maximally forbidding terms. The two sections
pull against each other and the forbidding one is louder.

### D9 — Three overlapping sections make the prompt long *and* vague at the same time

`## Choosing a research path`, `## Calibrating effort`, and `## Research depth` all answer
overlapping questions with different vocabularies (isolation/scale · effort/shape · width/depth).
The base orchestrator prompt is ~3.4 k tokens and the first orchestrator turn is 11,499 tokens
(adaptive: 9,716). The prompt pays for redundancy and gets ambiguity in return.

### D-obs — Non-prompt observation worth verifying

`configs/config_autonomous_frag.yml` sets `model_name: nvidia/nvidia/nemotron-3-ultra`. The
deepagents Nemotron 3 Ultra **harness profile** registers only against the specs listed in
`_NEMOTRON_ULTRA_MODEL_SPECS` (e.g. `NVIDIA:nvidia/nemotron-3-ultra-550b-a55b`). If the resolved
key does not match, this arm is running **without** `NemotronProgressBudgetMiddleware`
(repeat-call and turn budgets), `FinalAnswerGuardMiddleware`, and the `<loop_control>` /
`<final_answer_completeness>` prompt suffix — exactly the three guards whose absence the thrash and
over-answering data describe. `factory.py:152` assumes a profile *does* resolve for all three arms.

**VERIFIED 2026-08-17 — the profile does not resolve. This arm runs unprotected.**

```python
from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
from deepagents._models import get_model_identifier, get_model_provider
from langchain_nvidia_ai_endpoints import ChatNVIDIA          # NAT's `_type: nim` resolves here
m = ChatNVIDIA(model="nvidia/nvidia/nemotron-3-ultra",
               base_url="https://inference-api.nvidia.com/v1", api_key="dummy")  # pragma: allowlist secret
get_model_identifier(m)   # 'nvidia/nvidia/nemotron-3-ultra'
get_model_provider(m)     # 'NVIDIA'
p = _harness_profile_for_model(m, None)
p.system_prompt_suffix    # None
p.extra_middleware        # ()
p.tool_description_overrides  # {}
```

The lookup key is `NVIDIA:nvidia/nvidia/nemotron-3-ultra`; the registry holds
`NVIDIA:nvidia/nemotron-3-ultra-550b-a55b`. No match, and the provider-prefix fallback finds
nothing because the Nemotron profile is registered per model spec, not provider-wide.

**What this arm is therefore missing**, all of it directly relevant to the measured failures:

| Missing | What it would have done |
| :-- | :-- |
| `NemotronProgressBudgetMiddleware` | Caps 16 model calls, 48 tool results, and **3 consecutive identical tool calls** per user turn, then short-circuits to a fallback answer built from results so far. This is precisely the guard `deepsearchqa-0060` needed — 62 direct calls, 73.4 % duplicates, 3.86 M tokens, no answer. |
| `FinalAnswerGuardMiddleware` + `<final_answer_completeness>` | "Answer each question from its matching tool output; do not substitute an entity from another subtask" — the `deepsearchqa-0543` wrong-shuttle shape. |
| `<loop_control>` suffix | "Never re-issue the same failing call unchanged." |
| `NemotronTextToolCallParser`, `ChatNVIDIAMessageCompatibilityMiddleware`, `NemotronToolCallShim` | Repair tool calls emitted as text, and normalize empty tool results. |

`factory.py:152`'s comment assumes a profile *does* resolve for all three research arms; that
assumption is wrong, though its conclusion (use the per-agent general-purpose stub rather than
`HarnessProfile(enabled=False)`) stays correct for other reasons.

C3 in §6 is therefore **confirmed, not speculative** — and it is a one-line config change. Note it
would change the deep and adaptive arms too if they share the model spec, so it must be its own
measured arm, not folded into the prompt A/B.

---

## 3. Design principles for the replacement prompt

These are the rules the rewrite is held to. They are deliberately stated so the draft in Appendix A
can be audited against them.

**P1 — Emergent, not declared, still means *prescribed*.**
Removing tiers removed the enforcement *and* the decision procedure. Only the enforcement was
supposed to go. The new prompt gives the model an explicit ladder with worked examples and stated
budgets, but adds **no** `declare_effort_tier` tool, **no** section swapping, **no** tool hiding,
and **no** middleware that blocks a route. (Revision 2: the ladder this principle originally
defended was itself the problem — see §4.2R. Removing enforcement is necessary but not sufficient;
the taxonomy had to go too.)

**P2 — Decision first, menu second.**
The opening section is the shape ladder. Tool mechanics come after. This mirrors adaptive's
`## Choosing Effort` ("Before anything else:") and LangChain's own deep-research prompt, whose
delegation section opens with `**DEFAULT: Start with 1 sub-agent** for most queries` followed by
four literal example questions.

**P3 — Examples beat adjectives.**
LangChain's deep-research prompt routes on *examples* ("Compare OpenAI vs Anthropic vs DeepMind AI
safety approaches → 3 parallel sub-agents"), not on adjectives. The current autonomous prompt is
all adjectives ("genuinely multi-part", "open-ended", "bounded factual"). Every shape in the new
ladder carries 2–3 concrete example questions.

**P4 — Every budget is a number in the text.**
`shallow_researcher/researcher.j2` — the workhorse of the *best-scoring arm* — says
**"Max 2 calls per tool"** and **"Do not keep searching when one call provides enough evidence."**
LangChain's researcher prompt has a `<Hard Limits>` block with counts. Numbers in prompt text are
load-bearing; the current autonomous prompt has exactly one (`max_research_concurrency`).

**P5 — Make the measured-best shape the stated default.**
One batch, then answer, is empirically the best *and* cheapest shape (43 trials, F1 0.5709,
644 k tokens). The prompt should name it the default and require a *reason* — not merely
"thin evidence" — to go past it.

**P6 — Demote the parent's own search from "cheap" to "exceptional".**
The parent context is the expensive place to put raw results. Direct source calls become a
capped verification affordance, never the primary research route. This also implements the user's
criterion 2 (*shallow research ⇒ one sub-agent call*).

**P7 — Constrain the answer, not just the citations.**
Add an answer-set contract as a first-class section, at the same level as the citation contract.

**P8 — Net-neutral token budget.**
The rewrite must not grow the base prompt. Sections merged in §4 free the space the new sections
consume. Target: **≤ 3.4 k tokens** for the static prefix, i.e. no worse than today (§7).

**P9 — Do not fight the harness suffix.**
If the Nemotron harness profile resolves, `<approach>`, `<loop_control>`, `<grounding>`,
`<tool_selection>`, and `<final_answer_completeness>` are appended *after* our prompt. The new
prompt must not contradict them — in particular `<loop_control>`'s "never re-issue the same failing
call unchanged" is an ally, and our direct-search rule should be phrased to reinforce it.

---

## 4. The new prompt architecture

### 4.1 Section map — before and after

| Current section | Disposition | Rationale |
| :-- | :-- | :-- |
| Role line | Keep, tighten | — |
| `## Workflow` (5 steps) | **Merge** into the new opening section | Its step 3 ("Research by whichever path fits") is a pointer with no content |
| `## Choosing a research path` | **Replace** by `## Deciding what to do` (§4.2R) | D1, D2 |
| `## Calibrating effort` | **Deleted** — effort calibration is not modelled at all | D9 |
| `## Research depth (per query)` | **Compress** to 5 lines, add the price of `high` | D7, D9 |
| `## Never answer from memory` | Keep, shorten, reconcile with the meta path | D8 |
| `## Research loop` | Keep, add the second-batch rule and the direct-search cap | D3, D6 |
| `## Filesystem` / sandbox | Keep verbatim | Unrelated to the regression |
| `## Delegation sequencing` | Keep, absorb the prerequisite rule | D5 |
| `## Parent-report delta` | Keep verbatim (conditional block) | Unrelated |
| `## Stopping and evidence failure` | Keep, add the numeric-comparison completeness check | §2 / `deepsearchqa-0309` |
| — | **NEW** `## What goes in the answer` | D4 |
| — | **NEW** `## When a lookup keeps failing` | D3 thrash-to-zero |
| `## Finishing a run` | Keep, add the meta worked example | D8 |
| `## Inline citation contract` | Keep verbatim | Working as intended |
| Context block (post-cache-boundary) | Keep, one edit to the Retrieval Tools preamble | D3 |

Net: **−3 sections, +2 sections**, with the two largest overlapping blocks collapsed into one.

### 4.2R Two properties, not a taxonomy (**supersedes §4.2 — this is what shipped**)

The section that replaced `## Choosing a research path` + `## Calibrating effort` asks the model to
work out what it needs to know that it cannot already verify, and then reads **two properties** off
that set of unknowns. If the set is empty, it answers.

| Property | Question | Route it implies | Why it is a capability fact, not an effort level |
| :-- | :-- | :-- | :-- |
| **Independence** | Can I state every unknown *now*? | yes → one `run_research_batch`; no → `task(researcher-agent)` for the chain | Parallel workers cannot see each other. This is a hard property of the fan-out mechanism, true regardless of how hard the question is. |
| **Fixed structure** | Did the user fix the sections, comparison axes, or format? | yes → `task(planner-agent)` first, then `writer-agent` | `/shared/plan.json` is the artifact that records an output contract, and `writer-agent` is gated on it. Also a mechanism fact. |

Why this is not the ladder wearing a hat:

- **Nothing to select from.** There is no enumerated list of request kinds. The model does not
  assign the request a label before acting; it answers two questions about the work.
- **No ordering.** Neither property is "higher" than the other and neither implies more effort.
  The ladder had a climb; this has none.
- **Compositional.** A request can have dependent unknowns *and* a fixed structure and gets both
  routes. Tiers and shapes both forced a single selection.
- **Grounded in the tools, not in the question.** Independence is not a judgment about difficulty —
  it is a fact about whether a worker could have written the query. That is why it survives the
  objection that killed the ladder.

The `deepsearchqa-0543` fix is preserved and is arguably sharper: "never fan out two queries aimed
at the same unresolved fact" is a direct statement of the independence property, where the ladder
expressed it indirectly as a climb test from C to D.

The planner fix is also preserved. Its trigger was the load-bearing part — stylistic ("deep dive")
loses, structural wins — and "the deliverable's structure is part of the request" is structural
without enumerating request categories. The `write_todos` competition clause carries over verbatim.

### 4.2 The opening-move ladder — **REJECTED, see Revision 2 and §4.2R**

Retained for the record; this is what the objection was raised against.


This replaces `## Choosing a research path` + `## Calibrating effort`. It is the first thing after
the role line, and it names the tool call to emit **in the first assistant turn**.

Five shapes, ordered cheapest-first, with a "climb only when the test fails" rule:

| Shape | Opening move | Test to climb past it |
| :-- | :-- | :-- |
| **A — Answer now** | `submit_final_report(..., researched=false)`, alone | The request asks for a fact about the world |
| **B — One lookup** | one `run_research_batch` with **one** `ResearchQuery`, `depth:"low"` | The answer needs more than one independent fact |
| **C — One fan-out** | one `run_research_batch` with 2–5 independent queries | The sub-questions depend on each other |
| **D — Chained** | `task(subagent_type="researcher-agent", ...)` | Two or more independent chains, or the output shape must be fixed first |
| **E — Plan first** | `task(subagent_type="planner-agent", ...)` | — (top of the ladder) |

Three design decisions inside this table deserve to be called out, because they are the ones a
reviewer is most likely to want to change:

**(a) Shape B is a one-query batch, not a direct source call.** This is the deliberate reversal of
the current prompt's "cheapest path" line. It costs one sub-run but keeps the raw search trail out
of the parent context, and it matches the user's stated criterion that shallow research should be
*"probably single sub-agent call"*. The alternative — keep the direct call as shape B — is the
status quo that produced 480 parent-context searches and a 24.2 M-token orchestrator overrun.

**(b) Shape D is the *default* for prerequisite chains**, and the ladder's climb test from C to D
is explicitly *dependency*, not difficulty. This is the `deepsearchqa-0543` fix.

**(c) Shape E's trigger is structural, not stylistic.** Three concrete triggers replace
"comprehensive report / deep dive":

1. three or more distinct deliverables in one request;
2. the answer's *structure* must be decided before research (a sectioned report, a comparison
   matrix, a briefing) — i.e. you intend to publish through `writer-agent`;
3. a parent-report delta (already mandatory).

Each shape carries worked examples. Draft set (Appendix A has them in place):

- A — "Hi", "What can you do?", "Summarize what you just told me."
- B — "Who is the current CEO of Intel?", "What was NVIDIA's Q3 FY26 data-center revenue?"
- C — "Compare the 2025 flagship GPUs from NVIDIA, AMD, and Intel on memory bandwidth."
- D — "Which shuttle had the second-fewest construction milestones, and what were its EAFB missions?"
- E — "Write me a briefing on the state of solid-state battery commercialization, covering
  chemistry, manufacturing readiness, and the top five players."

The shuttle example is lifted verbatim from the failing trial. That is intentional: it is the
cheapest possible way to teach the dependency rule.

### 4.3 Budgets stated in prompt text

| Budget | Value | Source of the number |
| :-- | --: | :-- |
| Direct source calls by the orchestrator | **≤ 2 per request**, verification only | Report recommendation #3 proposed 5–10; the observed *median useful* count is far lower and 82 % of them were the expensive advanced tool. Start at 2, relax if the probe set shows real loss. |
| Identical/near-identical direct query | **never twice** | 11 % of all queries were exact repeats; the worst trial was 73.4 % |
| Batches per request | **1 by default; a 2nd only to consume a resolved prerequisite** | One-batch bucket is best & cheapest; two-batch bucket is worst |
| Queries per batch | 1–5 | matches `max_research_concurrency` |
| `depth:"high"` per request | **at most 1**, and only for a genuine chain | 42 % high with zero measurable F1 return |
| Answer-set commitment | 1 entity for a "which X" question | 153 excessive items |

These are **stated**, not enforced. Enforcement is §6.

### 4.4 `## What goes in the answer` (new section)

The precision fix. Rules, in the order they matter:

1. Name the exact set the question asks for: one entity, an exhaustive list, a number, a date.
2. Put **only qualifying members** in the answer body. Do not list rejected candidates, close
   alternatives, historical synonyms, deprecated names, or "commonly confused with" entries —
   **not even to say they do not qualify.** A name in the answer is read as an answer.
3. If two candidates remain and the evidence favors one, **commit to it** and state the residual
   uncertainty in a single clause. Do not hedge by listing both.
4. No research-process commentary — no "I searched", no "sources disagreed", no tool names.
5. Numeric comparison / superlative questions (`most`, `fewest`, `highest`, `minimum`): before you
   select a winner, confirm every candidate has a value on the **same year, same definition, and
   same source family**. If one candidate is missing a comparable value, go get it — do not drop
   the candidate and do not compare across definitions.

Rule 5 is the `deepsearchqa-0309` fix (five countries by IDP count → agriculture-employment
comparison; autonomous accepted a secondary-source Syria value, answered Colombia, F1 = 0; adaptive
found a consistent value, answered Syria, F1 = 1).

### 4.5 `## When a lookup keeps failing` (new section)

The thrash-to-zero fix, phrased to reinforce the harness `<loop_control>` suffix:

- A search tool takes **keywords, not URLs**. Pasting a URL or a database table id into a search
  tool does not fetch it and will not start working on the third attempt.
- If the same target has failed twice, **change the target, not the wording**: search for the
  publishing organization plus the figure, for a news or summary page that quotes it, or for a
  mirror.
- If it fails again, stop searching directly and hand the whole chain to
  `task(subagent_type="researcher-agent", ...)` **once**, with everything you already tried spelled
  out so it does not repeat you.
- If that also fails, finish with an honest partial answer under `## Stopping`. **Never** end a run
  without an exit — a run that stops mid-search scores zero.

The last clause targets the two trials (`0060`, `0545`) that never reached `submit_final_report`.

### 4.6 Revision 3 — pipeline ordering, identity, and output parity (**specified, not yet built**)

#### 4.6.1 Planner first, writer last

The shipped prompt gets the *conditions* right (plan when the deliverable's structure is fixed;
the writer needs `/shared/plan.json`) but never states the *ordering* as an invariant. Both arms we
are matching do state it. `deep_researcher/prompts/orchestrator.j2` numbers the pipeline
explicitly — planning is Step 2, writer is the last delegated step, and Step 5/6 is "return only
the completion marker" — under a `## Sequential Handoffs` section that says "Start with
`planner-agent`".

Why this matters beyond tidiness. A planner invoked *after* research has already run produces a
plan that does not know what has been found, so either it re-plans work already done (wasted
batches — and the two-batch bucket is the worst-scoring in the eval at F1 0.374) or the
orchestrator ignores the parts of the plan it has already covered, which silently voids the output
contract the writer will later read. Symmetrically, anything after the writer is either dead work
or a second, conflicting answer.

Half of the writer rule is already enforced in code — `PlanBeforeWriterMiddleware`
(`custom_middleware.py:394`) rejects a writer call before `plan.json` exists — but nothing enforces
or states "planner before research" or "nothing after the writer".

Draft text, to sit in `## Delegation sequencing`:

```
**Order is fixed when you use these agents.** `planner-agent` runs *first* — before any
research — or not at all. Deciding to plan after results are already in hand is too late:
the plan cannot account for what you found, and you will either re-run work or quietly
break the output contract the writer reads. If you did not plan first, finish the run
yourself instead.

`writer-agent` runs *last*. It is the final action of the run: after it returns, report its
completion marker and stop. Never research, verify, or call another subagent after it.
```

#### 4.6.2 A concrete identity

Current opening: *"You are a research orchestrator."* That is a job title, not a definition. It
does not say what product this is, what it delivers, what it guarantees, or how it relates to its
subagents — all of which the model needs in order to resolve exactly the ambiguities the probe
exposed (§8.1: it answered "Who are you?" in prose partly because it has no crisp account of what
it is).

Draft replacement:

```
You are the AI-Q research agent — the orchestrator of an NVIDIA AI-Q Blueprint deployment.

You answer questions by commissioning research and returning a verifiable, cited answer:
a short direct answer when the question is narrow, a structured report when it is broad.
Every factual claim you deliver traces to a source captured during this run. You do not
answer factual questions from memory, and you do not present an unsourced claim as a
finding.

You coordinate; you do not personally do the searching. Specialist subagents plan, research,
and write. Your judgment is about which of them a request needs, in what order, and when the
evidence on hand is enough to answer.
```

That last paragraph also reinforces the direct-search budget (§4.3) from the identity itself rather
than only from a rule 60 lines later.

#### 4.6.3 Output-format parity — the real gap is the delegation message

The obvious check comes back clean. `writer.j2`, `planner.j2`, and `researcher.j2` are
**byte-identical** between `autonomous_researcher` and `adaptive_researcher`:

```bash
diff -q src/aiq_agent/agents/autonomous_researcher/prompts/writer.j2 \
        src/aiq_agent/agents/adaptive_researcher/prompts/writer.j2   # identical
```

So the writer path already emits the same artifact. But an identical writer prompt is **not
sufficient for identical output**, because the writer's other input is the `task(description=...)`
message the orchestrator composes. `deep_researcher` pins that with a literal
`## Writer Delegation Template`; `adaptive_researcher` spells the same instructions out in its
Planned Writer Pipeline steps. **Autonomous pins nothing** — outside the parent-report delta block
it never tells the writer what to read or what to return, so each run improvises the brief. That is
where output drift enters, and it is invisible to a prompt diff.

The template carries load-bearing content that autonomous currently drops entirely:

| Instruction in deep's template | Consequence of omitting it |
| :-- | :-- |
| "Read `/shared/plan.json`, all research note files under `/shared/`, and the verified sources" | Writer may synthesize from the delegation message alone and under-use gathered evidence |
| "For broad reports, produce a cross-synthesized narrative with developed paragraphs; do not compress into a checklist of short component summaries" | The failure mode that makes reports read like bullet dumps |
| "Return only the short completion marker `Wrote /shared/output.md`. Do not return JSON and do not echo the full Markdown" | Writer echoes the whole report into the parent context — pure token waste on the most expensive path |
| Chart/artifact embedding rules (`artifact://`, only when source-anchored) | Broken or fabricated figures under `execution_enabled` |
| Delta-mode preservation clause | Already present in autonomous, but only in the delta block |

**Action:** port a `## Writer delegation` template into the autonomous orchestrator prompt,
adapted from `deep_researcher/prompts/orchestrator.j2:122-142`, keeping the `execution_enabled` and
`parent_report_context_available` conditionals. This is the concrete deliverable of point 4.

##### One tension to resolve, not paper over

`## What goes in the answer` (§4.4) is autonomous-only — deep and adaptive have no equivalent. It
was added to fix a measured precision defect (153 excessive items; the *Tyrannosaurus* trial at
precision 0.133), and it instructs terseness: "Which X gets one X", "include only qualifying
members".

Deep's writer template instructs the opposite for broad requests: "a cross-synthesized narrative
with developed paragraphs". These do not contradict on a narrow factual question, but on a
report-shaped request they pull apart, and *report-shaped is exactly the writer path*.

Recommended resolution: **scope the answer-set contract to the inline path only.** It exists to
satisfy a set grader on short-answer benchmarks, which is the inline exit; the writer path is
governed by the writer prompt and the delegation template, which both arms already share. Concretely,
retitle it and open with a scoping line:

```
## What goes in the answer you write yourself

These rules govern the answer you compose and submit with `submit_final_report`. When
`writer-agent` composes the report instead, its own contract governs — do not restate these
rules to it.
```

Without that scoping, this section is a live source of the very output divergence point 4 asks us
to eliminate.

#### 4.6.4 Cost and verification

Estimated additions: identity ~+90 tokens, ordering rules ~+70, writer template ~+200 (mostly the
verbatim block), scoping line ~+30. Roughly **+390 tokens**, taking the static prefix to ~3.0 k
(from 2.62 k; baseline 2.24 k). That reopens P8 and should be paid for by trimming — the
`## Research loop` and `## Budgets` sections overlap and are the obvious candidates.

New probe assertions to add alongside (§8.1):

| Assertion | Why |
| :-- | :-- |
| A run that calls `planner-agent` calls it as the **first** delegation | 4.6.1 |
| No `run_research_batch` or `task` call appears **after** a `writer-agent` call | 4.6.1 |
| A `writer-agent` delegation's `description` names `/shared/plan.json`, the research notes, and the completion marker | 4.6.3 |

The first two need a probe that runs past turn 1 — the current `_first_turn()` helper cuts off
deliberately, so this needs a second, more expensive driver that records the full delegation
sequence. Worth building: ordering bugs are invisible to a turn-1 probe by construction.

---

## 5. Companion description edits (routing text = routing logic)

In this design the descriptions are load-bearing (`factory.py:110–118` says so explicitly). The
prompt rewrite is incoherent unless they move with it. Full replacement strings in **Appendix B**.

| String | File / line | Change |
| :-- | :-- | :-- |
| `RESEARCHER_SUBAGENT_DESCRIPTION` | `factory.py:119` | Lead with the **dependency chain** trigger, not "context isolation". Add "use this instead of searching yourself when a lookup has already failed twice." |
| `PLANNER_SUBAGENT_DESCRIPTION` | `factory.py:128` | Replace the stylistic trigger with the three structural triggers from §4.2(c). State plainly that it supersedes `write_todos` for multi-part *research* requests. |
| `WRITER_SUBAGENT_DESCRIPTION` | `factory.py:136` | Keep the `plan.json` precondition; sharpen "report-shaped" into "the deliverable has named sections the user asked for". |
| `_RESEARCH_BATCH_DESCRIPTION` | `tools/research.py:59` | Add: single-query batches are legitimate, so one unknown still goes through a worker; `depth:"high"` is expensive and at most one per request; the second batch is for consuming a resolved prerequisite, not for re-asking. |
| Retrieval-tools preamble | `orchestrator.j2` context block | Change "You hold all of these directly and may call them yourself" → the ≤2, verification-only framing. |

`GENERAL_PURPOSE_STUB_DESCRIPTION` stays exactly as is. It is doing its job (0 calls).

---

## 6. What the prompt cannot fix

Stated separately and honestly: a prompt states budgets; it does not enforce them. Three of the
measured failures survived a prompt that already discouraged them, so the prompt-only arm should be
expected to reduce, not eliminate, them. These are **sequenced after** the prompt A/B so the prompt
effect is measurable in isolation.

| # | Change | File | Why the prompt can't do it | Effort |
| :-- | :-- | :-- | :-- | :-- |
| C1 | Extend the identical-call guard to the **orchestrator's own** direct source calls (normalized args, not just `run_research_batch` signatures) | `custom_middleware.py` (`AutonomousOrchestratorLoopGuardMiddleware`) | `researcher_loop_guard.max_identical_source_calls` binds workers only; `awrap_tool_call` does not dedupe direct source args. This is the hole `deepsearchqa-0060` fell through (3.86 M tokens, no answer). | S |
| C2 | Hard cap orchestrator direct source calls, then withdraw the tools (the pattern `single_shot_search_budget` already uses in adaptive) | `custom_middleware.py` + config key | A stated cap is advisory | S |
| C3 | Verify + fix the Nemotron harness-profile model spec (D-obs) | `configs/config_autonomous_frag.yml` | Config, not prompt. Brings back progress-budget + final-answer guards for free if it is currently missing. | XS |
| C4 | Turn on provider prompt caching | config / provider | `n_cache_tokens = 0` makes cost quadratic in turn count. Largest pure-cost lever, zero behavior change. | S–M |
| C5 | Add a URL/CSV **fetch** tool | `sources/` | The thrash pattern is the agent correctly identifying a URL with no way to fetch it. Only two Tavily search variants are configured. No prompt can create a capability. | M |
| C6 | Consider `exclude_tools: [web_search_tool, advanced_web_search_tool]` on the orchestrator as a **separate arm** | `configs/config_autonomous_frag.yml` | Tests the "no direct search on turn 1" rule at its limit: does the orchestrator need *any* direct search? | XS (config only) |
| C7 | Make the uncommitted-run corrective turn land. Restrict the tools offered on that turn to the two exits, or synthesize the finalizer call directly from the assistant text already produced. | `custom_middleware.py` (`AutonomousFinalizationMiddleware`) | **Found by the probe (§8.1).** The nudge already names `submit_final_report`, yet an observed corrective turn answered it with `ls('/shared')`. A prompt cannot fix a turn whose whole purpose is to recover from the prompt being ignored. Cheap, and it closes the residual on the no-research bucket. | S |

**Recommended sequencing:** prompt-only arm first (this plan), then C3 + C1 + C2 as a
"guards" arm, then C6 as a variant, then C4/C5 as infrastructure. Changing them together makes the
prompt effect unattributable.

---

## 7. Token budget accounting

Principle P8 requires the rewrite to be net-neutral on the static prefix.

> **Corrected after implementation (2026-08-17).** The projection below was wrong: it sized the
> prefix from the raw file's 13,545 bytes, which counts the Jinja comment header (never rendered)
> and the post-cache-boundary Context block (excluded from a prefix comparison). Measured on the
> *rendered* prefix, no-sandbox/no-delta branch:
>
> | | Rendered static prefix |
> | :-- | --: |
> | Before | 8,944 B ≈ **2.24 k tokens** |
> | Revision 1, with the A–E ladder | 11,373 B ≈ 2.84 k tokens — **+610 tokens (+27 %)** |
> | Revision 2, two properties | 9,776 B ≈ 2.44 k tokens — **+208 tokens (+9 %)** |
> | **Final, after the §8.1 probe fixes** | **10,474 B ≈ 2.62 k tokens — +382 tokens (+17 %)** |
>
> Revision 1 missed P8 by a wide margin: the ladder's five entries each needed a definition, an
> opening move, worked examples, and a climb test, which is expensive prose. Replacing it with two
> properties recovered ~400 tokens as a side effect of dropping the taxonomy. The probe then bought
> ~175 tokens back to fix the no-research bucket (naming the exit, plus the tool-only-reply
> invariant in the role paragraph).
>
> **Final position: +382 tokens on the static prefix, ~3 % of the measured 11,499-token first
> orchestrator turn.** P8 is missed in the letter and met in the spirit: the design's target was
> never the prefix but the *peak* parent prompt (34,433 tokens, 3.0× growth), which the direct-search
> budget and the one-batch default attack. Revisit only if S4 misses.

Original (superseded) projection:

| | Current | Projected |
| :-- | --: | --: |
| `orchestrator.j2` static prefix | 13,545 bytes ≈ **3.4 k tokens** | ≈ 3.3–3.5 k tokens |
| Removed: `Choosing a research path` + `Calibrating effort` + `Workflow` | — | −~950 tokens |
| Removed: `Research depth` compression | — | −~180 tokens |
| Added: `Pick your opening move` (ladder + examples) | — | +~750 tokens |
| Added: `What goes in the answer` | — | +~230 tokens |
| Added: `When a lookup keeps failing` | — | +~170 tokens |

The real token win is not the prefix — it is the **peak** parent prompt. Current mean peak
orchestrator prompt is **34,433** tokens (3.0× growth from 11,499); adaptive's is 20,812 (2.1×).
The ≤2 direct-call rule and the one-batch default target that growth factor, not the prefix.

**Projected effect if the prompt behaves as designed** (arithmetic, not a promise):

- Direct source calls 480 → ≲180 (2/trial ceiling) removes the bulk of parent-context accumulation.
- Two-and-three-plus-batch runs (32 trials, avg 1.63 M tokens) shifting toward one-batch (644 k)
  is worth roughly **−0.35 M tokens/trial on average** on its own.
- Combined, ~1.0 M → **~0.6–0.7 M** input tokens/trial. Still above chat researcher's 295 k; that
  gap is structural (289 worker contexts) and is C4's problem, not the prompt's.

---

## 8. Validation plan

### 8.1 The routing probe set — run this *before* any Harbor job

DeepSearchQA cannot validate criteria 1 and 3. Almost no DSQA item is a greeting, and almost none
is report-shaped, so **planner usage on DSQA should stay near zero even after a correct fix.** The
Harbor report's recommendation *"get the planner invoked"* is, on DSQA specifically, chasing a
metric that ought to be near zero. Adaptive — the arm that *wins* — called the planner 9 times and
the writer once in 90 trials.

So: build a small local probe set (~24 items, 6 per bucket) and assert the **route taken on turn 1**,
not the answer. Cheap, deterministic-ish, and it directly tests all three user criteria.

| Bucket | n | Assertion on turn 1 |
| :-- | --: | :-- |
| Meta / chit-chat / capability | 6 | **zero** tool calls other than `submit_final_report(researched=false)` |
| Single-fact lookup | 6 | exactly one `run_research_batch`, one query, `depth != "high"`; **no** direct source call |
| Independent multi-part | 6 | exactly one `run_research_batch`, 2–5 queries; total ≤1 `depth:"high"` |
| Compound / report-shaped | 6 | `task(subagent_type="planner-agent")` |

Add a fifth assertion across all buckets: **≤2 orchestrator direct source calls per run**.

**Implemented 2026-08-17** at
`tests/aiq_agent/agents/autonomous_researcher/test_routing_probe.py` — the pytest option, not the
benchmark harness, because iterating on a prompt wants a fast local loop.

```bash
set -a; . deploy/.env; set +a
AIQ_ROUTING_PROBE=1 uv run pytest \
    tests/aiq_agent/agents/autonomous_researcher/test_routing_probe.py -v
```

Design notes worth carrying:

- **Real orchestrator LLM, stubbed source tools.** The thing under test is what the model decides,
  so the model must be real; retrieval quality is irrelevant, so Tavily is not. The stub tool
  *descriptions* are copied from `sources/tavily_web_search/src/register.py` because the
  description is rendered into the prompt and is part of what routes the model — a stub with a
  different description probes a different prompt.
- **Cut off after the first assistant turn.** `_first_turn()` streams the graph and returns on the
  first `AIMessage`, so no researcher, planner, or writer ever executes: one model call per item.
  The graph must be streamed rather than the model called directly, because `SubAgentMiddleware`
  injects the `task` tool and the subagent descriptions — the routing text — at graph build time.
- **Gated** on `AIQ_ROUTING_PROBE=1` plus `NVIDIA_API_KEY`, marked `integration`, mirroring the
  `test_openshell_live.py` convention. Skips cleanly (48 skipped) in a normal run.
- **Temperature 0 by default**, against the config's 0.7, so prompt iteration is not fighting
  sampling noise. Re-run at 0.7 before trusting a green board.
- **No tool call is itself a failure**, reported distinctly from "called the wrong tool" — see the
  first result below, where that distinction was the entire finding.

#### First results (2026-08-17, temperature 0, 24 items)

| Bucket | Assertion on turn 1 | Baseline |
| :-- | :-- | --: |
| One independent unknown | one `run_research_batch`, 1 query, `depth != high`, no direct search | **6/6** |
| Several independent unknowns | one `run_research_batch`, 2–5 queries, ≤1 `high` | **6/6** |
| Deliverable structure fixed | `task(subagent_type="planner-agent")` | **6/6** |
| Nothing to find out | `submit_final_report(researched=false)` alone | **0/6** |

The planner result is the headline: **6/6, against 0 calls in 90 eval trials.** D2's diagnosis —
that the planner lost every routing decision because it was absent from the routing menu and
triggered on style rather than structure — is confirmed, and the fix holds. Since `writer-agent` is
gated on `/shared/plan.json`, this is also what makes the writer reachable at all.

The no-research bucket failed 6/6 in one identical mode: **`NO TOOL CALL — answered in prose`.**
The cause was wording in this plan's own §4.2R text — "If that set is empty, answer" reads as
*reply*, not as *call the finalizer*, and deepagents' base prompt ("Be concise and direct", "NEVER
add unnecessary preamble") pushes the same way. The branch never named its exit.

Iterations against that bucket:

| Change | Result | Which items failed |
| :-- | --: | :-- |
| Baseline ("If that set is empty, answer.") | 0/6 | all six |
| Name the exit in the branch: `submit_final_report(…, researched=false)` as the only call | 3/6 | social turns ("Good morning!", "Thanks…"), "Who are you?" |
| Also state the invariant in the role paragraph: assistant prose is never delivered, every reply is a tool call | 4/6 | "Who are you?", "what kinds of questions…" |

**Final board: 22/24, with no regression in the three research buckets** (6/6, 6/6, 6/6 held after
both prompt edits).

The residual is **reproducible, not sampling noise** — an earlier reading of these iterations
called it noise on the strength of the failing set rotating between rounds, but two consecutive
runs at 4/6 failed the *same two items*. The rotation was the second edit doing real work: it fixed
pure social turns and left capability questions untouched.

The two survivors are both "describe yourself" questions. That is the narrowest possible residual
and it is oddly stubborn: the prompt's no-research branch **explicitly enumerates** "a question
about what you are or what you can do", and the model reads it and replies in prose anyway. The
pull is structural — our prefix sits *in front of* deepagents' base prompt ("Be concise and
direct", "NEVER add unnecessary preamble"), and on a question about the assistant itself that
instruction is both closer to the decision and, in ordinary terms, correct.

Prompt iteration was stopped at this point. A third wording round might convert these two, but C7
fixes the whole class — including the cases no probe item covers — and the failure is already
recoverable: `AutonomousFinalizationMiddleware` catches the uncommitted run. The reason C7 is
required rather than optional is that the *corrective* turn is itself unreliable (one trace
answered the nudge with `ls('/shared')`), which is a defect no prompt edit can reach.

This condition is *recoverable, not fatal*: `AutonomousFinalizationMiddleware` detects an
uncommitted run and spends a corrective turn. But the corrective turn is not reliable either — in
one trace it answered the nudge with `ls('/shared')` rather than the finalizer — so the residual
belongs in §6 as a guard fix, not in further prompt prose.

### 8.2 Harbor eval

Same 90-item DeepSearchQA set, same image-pin discipline (eval code is frozen in the Docker image;
a code change needs a rebuild + pin bump — config alone is live from the host).

Compare against `jobs/2026-08-13__19-24-13` (autonomous baseline) and
`jobs/2026-08-01__19-18-21` (adaptive control).

**Success criteria — the prompt-only arm.** Ordered by confidence.

| # | Metric | Baseline | Target | Confidence |
| :-- | :-- | --: | --: | :-- |
| S1 | Excessive answer items | 153 | **≤ 100** | high — §4.4 is a direct instruction against a directly-measured behavior |
| S2 | Precision | 0.5191 | **≥ 0.56** | high — S1 mechanically implies it |
| S3 | Orchestrator direct source calls | 480 | **≤ 200** | medium — advisory only until C1/C2 |
| S4 | Avg input tokens | 999,938 | **≤ 750,000** | medium |
| S5 | Trials with ≥2 batches | 32 | **≤ 18** | medium |
| S6 | Queries marked `depth:"high"` | 120/286 (42 %) | **≤ 25 %** | medium |
| S7 | Mean F1 | 0.5098 | **≥ 0.55** (parity with adaptive) | low-medium — F1 is noisy at n=90 |
| S8 | Trials that never reach `submit_final_report` | 2 | **0** | medium |
| S9 | Planner calls on DSQA | 0 | **no target** — measured on the probe set instead | — |

**Do not** treat S7 alone as the verdict. §2.8 of the token/F1 analysis shows 36 of 90 tasks tie and
the median paired difference is 0; the arm-level mean is carried by tails. S1/S3/S4/S5 are the
behavioral signals that the prompt actually changed what the model does.

### 8.3 Reuse the existing analysis scripts

No new tooling needed for the Harbor side:

```bash
python3 analysis/autonomous_agent_deep_dive.py       jobs/<new-job>
python3 analysis/autonomous_agent_patterns.py        jobs/<new-job>
python3 analysis/autonomous_agent_token_mechanics.py jobs/<new-job>
python3 analysis/compare_research_arms.py jobs/<new-job> jobs/2026-08-13__19-24-13 jobs/2026-08-01__19-18-21
```

### 8.4 Judge-noise control

`deepsearchqa-0588` returned `grader_valid = 0` (malformed judge JSON) and cost the baseline 0.0111
of its 0.0400 F1 gap. Capture judge responses this time and report both raw and
valid-graders-only means, as the token/F1 analysis did.

---

## 9. Risks

| Risk | Likelihood | Mitigation |
| :-- | :-- | :-- |
| ~~The ladder re-creates tiers by the back door and the reviewer rejects it on design grounds~~ | ~~Medium~~ | **This risk materialized and the mitigation was wrong.** "No enforcement, crossable mid-run" answered the wrong question: the objection was to the taxonomy itself, not to how it was enforced. Resolved by Revision 2 (§4.2R) — routing is now stated as two properties of the unknowns, with nothing to classify into. |
| The two properties are read as a 2×2 grid, i.e. a taxonomy with four cells | Low-medium | They compose rather than partition, neither is ordered, and both are facts about the tools (what parallel workers can exploit; what `plan.json` records) rather than judgments about the request. A regression test asserts the ladder vocabulary stays out of the rendered prompt. Watch for it in probe-set transcripts. |
| Killing direct search hurts recall on tasks where a quick parent lookup genuinely helped | Medium | Autonomous won 24/90 tasks outright; some of those may be direct-search wins. The probe set will not catch this — watch S7 and the per-task paired diff, and keep C6 as the separate, more aggressive arm. |
| ≤2 direct calls is too tight | Medium | It is a prompt number; one-line change. Report #3 suggested 5–10. Starting tight makes the effect visible. |
| "Commit to one candidate" trades recall for precision and nets zero | Medium | Recall gap is only 0.0091 while precision gap is 0.0714 — the trade is favorable at the margin. Watch S1 and recall together, not S2 alone. |
| Prompt grows and the base-prompt saving evaporates | Low | §7 budget; check rendered length in the implementation step. |
| Planner still never fires | Medium | Expected on DSQA (§8.1). The probe set is the real test. If it fails there too, the next lever is the description (Appendix B), then a structural trigger in middleware — not more prompt prose. |
| Behavior differs on FreshQA (`config_autonomous_frag_freshqa.yml`) | Medium | FreshQA is single-fact-dominant, so the ladder should push it toward shape B — a cheap secondary confirmation that criterion 2 works. Run it. |

---

## 10. Sequencing

1. ~~**Review this plan**~~ — done; the ladder was rejected, see Revision 2 / §4.2R.
2. ~~**Verify D-obs**~~ — done, **confirmed broken**: no harness profile resolves for this arm.
3. ~~**Build the probe set**~~ — done: `tests/aiq_agent/agents/autonomous_researcher/test_routing_probe.py`.
4. ~~**Implement** Appendix A + Appendix B~~ — done.
5. ~~**Re-run the probe set and iterate**~~ — done, and **stopped deliberately at 22/24**. The
   no-research bucket's residual 2 items rotate between runs, which is noise, not a fixable
   wording problem. Escalated to C7 rather than iterated further.
6. **Implement Revision 3** (§4.6): planner-first / writer-last ordering, the concrete identity,
   the writer delegation template, and scoping the answer-set contract to the inline path. Trim to
   hold the token budget. ← **next**
7. **Extend the probe** with the three ordering/parity assertions in §4.6.4, which needs a driver
   that records the full delegation sequence rather than only turn 1.
8. **Harbor eval**, prompt-only arm, DSQA 90 (§8.2). Then FreshQA.
9. **Then** the guards arm, separately measured. Ordered **C3 → C7 → C1 → C2**: C3 is a confirmed
   one-line config fix that restores three upstream guards, and C7 closes the probe's residual.

The probe loop was where the value was, exactly as predicted — it caught a routing defect in the
first item and a `AutonomousFinalizationMiddleware` weakness that no static test would have found,
for the cost of ~50 model calls rather than a Harbor job.

---

## Appendix A — Draft replacement for `prompts/orchestrator.j2`

> **Reflects the shipped Revision 2 prompt. It does NOT include Revision 3 (§4.6)** — the
> planner-first / writer-last ordering rules, the concrete identity paragraph, the writer
> delegation template, or the inline-path scoping of `## What goes in the answer`. Read §4.6
> alongside this appendix; implementing this block alone reproduces today's prompt, not the
> intended one.

Complete draft of the static prefix. The Jinja header comment, the `{% if execution_enabled %}`
filesystem branch, the `parent_report_context_available` block, the KV-cache boundary marker, and
the whole post-boundary Context block are carried over from the current file; only the changed
region is reproduced in full here, with unchanged blocks marked.

```jinja
{#-
  ============================================================================
  Autonomous Research orchestrator prompt.
  ----------------------------------------------------------------------------
  No effort classification, no per-tier section gating, no tool hiding. The
  Jinja conditionals switch only on optional *capabilities*.

  This prompt is a PREFIX. deepagents places it before its own base agent
  prompt, and SubAgentMiddleware appends TASK_SYSTEM_PROMPT, TASK_TOOL_DESCRIPTION,
  and the subagent list with their descriptions. Do not restate any of that here.

  The opening-move ladder below is the decision procedure; the subagent and tool
  *descriptions* (factory.py, tools/research.py) are the routing detail. They are
  written to agree with each other — change them together.
-#}
You are a research orchestrator. Answer the user's request completely and correctly, spending the effort the question actually warrants and no more. Every capability allowed by this configuration is already available to you; you never need to restart, rebuild, or escalate into a different mode to reach one.


{#- REVISION 2: the A–E ladder that stood here was rejected as a tier system and replaced. -#}
## Deciding what to do

Work out what you would need to know that you cannot already verify. If that set is empty, answer. Otherwise two properties of those unknowns decide everything:

**Do they depend on each other?** Unknowns you can state now are independent — send them together in one `run_research_batch`; workers run in parallel and cannot see each other. An unknown you cannot even phrase until another is resolved is a chain — hand the whole chain to `task(subagent_type="researcher-agent", …)`, which can carry one answer into the next search. Never fan out two queries aimed at the same unresolved fact.

**Is the deliverable's structure part of the request?** When the user has fixed the sections, the comparison axes, or the format, settle that before researching: `task(subagent_type="planner-agent", …)` writes the contract to `/shared/plan.json`, which is also what `writer-agent` reads. When the shape follows from the evidence, just write the answer yourself.

These compose. A request can have dependent unknowns and a fixed structure, and gets both.

`write_todos` tracks bookkeeping; it is not a substitute for a plan. When the deliverable's structure is part of the request, delegate to `planner-agent` rather than writing a todo list and researching it yourself.


## Budgets

These are the numbers this configuration expects. Staying inside them is part of answering well.

- **Batches: one.** One well-formed `run_research_batch` then answer is the normal shape. Issue a second batch only to **consume a prerequisite you have now resolved** — never to ask the same question again with different wording, and never for "more depth". If the first batch came back thin, the answer is a *differently targeted* query or a `researcher-agent` chain, not a rerun.
- **Queries per batch: 1–5** (hard limit {{ max_research_concurrency }} per call).
- **`depth: "high"`: at most one per request**, and only for a genuine multi-hop chain. `high` costs many times what `low` costs and buys nothing on a question a single search can answer. Default `low`; use `medium` when one corroborating check is warranted.
- **Your own direct source calls: at most 2 per request**, and only to *verify* or *disambiguate* something a researcher already returned. They are not the cheap path: their raw results stay in this conversation and are re-sent on every later turn. All primary research goes through `run_research_batch` or `researcher-agent`.
- **Never issue the same query twice**, in any path. If you already ran it, you already have its answer.


## Never answer from memory

You are a RESEARCH agent. When a fact could be time-sensitive, or you are not fully certain of it, research it. Anything involving current events, recent releases, prices, standings, holders of a position, "latest", "now", or a date inside the last few years is time-sensitive by default.

Shape A is the only exception: chit-chat, questions about your own capabilities, and restating what is already in this conversation. A truly timeless fact may also be answered directly, but if you find yourself justifying why a fact is timeless, it is not — run shape B.


## When a lookup keeps failing

- Search tools take **keywords, not URLs**. Pasting a URL or a database table id into a search tool does not fetch it, and it will not start working on the third attempt.
- If the same target has failed twice, change the **target**, not the wording: search for the publishing organization plus the figure, for a page that quotes it, or for a mirror or summary.
- If it fails again, stop searching and hand the whole chain to `task(subagent_type="researcher-agent", …)` **once**, listing what you already tried so it does not repeat you.
- If that also fails, finish under "Stopping" with an honest partial answer. Always finish. A run that stops mid-search delivers nothing.


## Research loop

- Each `ResearchQuery` must carry full standalone context. Workers cannot see this conversation, and query IDs are meaningless to them.
- Keep a ledger of the queries you have submitted, and check it before every call.
- Batch results come back as `ResearchNotes` JSON, persisted under `/shared/`; each note carries its own `evidence_judgment`.
- If a batch returns a tool error, revise only the failed queries and call again. Do not resubmit successful ones.
- Call `get_verified_sources` before writing anything cited. It returns the whitelist of source locators captured this run, across every research path you used.

{% if execution_enabled -%}
## Filesystem and sandbox
{#- unchanged from the current prompt -#}
{%- else -%}
## Filesystem
{#- unchanged from the current prompt -#}
{%- endif %}


## Delegation sequencing

Dependent steps must be serialized. Call only one dependent subagent per assistant turn, wait for its result, and read any file it wrote before the next step. **A prerequisite chain is a dependent step**: never fan out two queries that ask about the same unresolved prerequisite, and never fan out a query whose text you cannot write until another query has answered.

`writer-agent` reads its output contract from `/shared/plan.json`, so it is reachable only after `planner-agent` has produced one; a call before then is rejected by the runtime. If an answer does not warrant a plan, write it yourself.

{% if parent_report_context_available %}
## Parent-report delta
{#- unchanged from the current prompt -#}
{% endif %}

## Stopping and evidence failure

- If verified evidence is sufficient, stop researching and synthesize. More searching after that point does not improve the answer.
- If verified evidence is partial, answer only what it supports and state the important gaps plainly.
- For a comparison or superlative — "most", "fewest", "highest", "the minimum" — confirm before selecting a winner that **every candidate has a value on the same year, the same definition, and the same source family**. If one candidate is missing a comparable value, go and get it; do not drop the candidate and do not compare across definitions.
- If research was required but the verified source registry is still empty after reasonable distinct retries, do not fabricate an answer and do not downgrade it to an unresearched one. Submit an honest inability-to-verify explanation with `researched=true`.


## What goes in the answer

The answer is a commitment, not a survey of what you considered.

- Identify the exact set the question asks for: one entity, an exhaustive list, a number, a date.
- Include **only qualifying members**. Do not name rejected candidates, close alternatives, historical synonyms, deprecated or invalid names, or "commonly confused with" entries — **not even to say they do not qualify.** A name that appears in your answer is read as one of your answers.
- If two candidates survive and the evidence favors one, commit to that one and put the residual uncertainty in a single clause. Do not hedge by listing both.
- No process commentary: no "I searched", no "sources disagreed", no tool names, no description of how you decided.
- Match the requested form. "Which X" gets one X. "List all X" gets exactly the qualifying X's. A question that asks for a number gets the number.


## Finishing a run

Every run ends exactly one of two ways. Decide by **how you produced the answer**, not by how much work it took:

- **You delegated to `writer-agent`** (it wrote `/shared/output.md`): return only its short completion marker. Do NOT call `submit_final_report`.
- **You wrote the answer yourself**: call `submit_final_report(markdown, researched=…)` exactly once.
  - `markdown` is the complete final answer — never an acknowledgment, a plan, or a marker.
  - `researched=true` when **any** research ran this turn. The markdown must then carry numeric citations drawn only from `get_verified_sources`.
  - `researched=false` **only** for a deliberate shape-A answer. This skips citation verification, so never use it to escape a failed research attempt.

A plain text reply with neither exit does not finish the run. For shape A the entire turn is one call, for example:

    submit_final_report(
        markdown="Hello — I'm the AI-Q research assistant. Ask me a question and I'll research it for you.",
        researched=false,
    )


## Inline citation contract
{#- unchanged from the current prompt -#}


**Important**:
You MUST use the same language as the user's request throughout.
NEVER assume files exist. Paths are VIRTUAL.



{#- === KV CACHE BOUNDARY — dynamic content below === #}

{#- Context block unchanged EXCEPT the Retrieval Tools preamble: -#}

{% if retrieval_tools %}
## Retrieval Tools
Name these (exact names) in a `ResearchQuery.preferred_tools` to steer `run_research_batch` or a `researcher-agent` delegation — that is their normal use. You may also call one directly, but only within the 2-call verification budget above; raw results from a direct call stay in this conversation and are re-sent on every later turn.
{% for tool in (retrieval_tools or []) %}- **{{ tool.name }}**: {{ tool.description }}
{% endfor %}
{% endif %}
```

### Notes on the draft

- `## Workflow` is gone. Its content is distributed: step 1 (assess) is the ladder, step 2
  (`write_todos`) is a clause in E, step 3 (research) is the ladder, step 4
  (`get_verified_sources`) moved into `## Research loop`, step 5 (publish) is
  `## Finishing a run`. Nothing is lost and the pointer-to-a-pointer structure is removed.
- The ladder is written so **A is reachable in one turn** and **E has a checkable trigger**. Those
  are criteria 1 and 3 from the request.
- Criterion 2 ("shallow research ⇒ single sub-agent call") is shape B, and it is deliberately a
  one-query `run_research_batch` rather than a direct source call. See §4.2(a) — this is the
  decision most worth challenging in review.

---

## Appendix B — Replacement description strings

`src/aiq_agent/agents/autonomous_researcher/factory.py`

```python
RESEARCHER_SUBAGENT_DESCRIPTION = (
    "Investigate ONE topic end-to-end in an isolated context and return structured, cited findings. "
    "Choose this when the question is a PREREQUISITE CHAIN — you must resolve one fact before you can "
    "even write the next query — because parallel workers cannot pass results to each other. Also "
    "choose it when a lookup has already failed twice and you want a fresh, isolated attempt: give it "
    "the whole chain plus what you already tried, so it does not repeat you. Give it exactly one topic, "
    "stated with full standalone context. For several INDEPENDENT questions use run_research_batch."
)

PLANNER_SUBAGENT_DESCRIPTION = (
    "Turn a compound request into an explicit answer strategy plus a set of ResearchQuery objects, "
    "persisted to /shared/plan.json. Choose this when ANY of these is true: (1) the request contains "
    "three or more distinct deliverables; (2) the answer's structure must be fixed before research — a "
    "sectioned report, a comparison matrix, a briefing — which also means you intend to publish through "
    "writer-agent, since writer-agent reads its output contract from the plan; (3) a parent report is "
    "mounted for this request. For a multi-part RESEARCH request this supersedes write_todos: delegate "
    "here rather than writing a todo list and researching it yourself. Skip it when one batch of queries "
    "and an inline answer would fully satisfy the request."
)

WRITER_SUBAGENT_DESCRIPTION = (
    "Synthesize a long-form cited report from /shared/plan.json and the research notes under /shared/, "
    "writing the result to /shared/output.md. Choose this only when the deliverable has named sections "
    "the user asked for and is long enough that composing it inline would degrade it. Requires "
    "/shared/plan.json to exist first; the call is rejected otherwise. For anything you can write "
    "yourself, do that and call submit_final_report."
)
```

`src/aiq_agent/agents/autonomous_researcher/tools/research.py`

```python
_RESEARCH_BATCH_DESCRIPTION = """Run one or more independent research questions in parallel isolated contexts.

This is the normal way to research. A batch of ONE query is valid and is the right call for a single
self-contained fact — prefer it over searching yourself, because a worker's search trail is digested
before it reaches you instead of accumulating in your context.

Each query runs as its own worker, so nothing one worker learns can inform another. If one question
cannot be written until another is answered, that is a prerequisite chain: use
`task(subagent_type="researcher-agent", ...)` for the whole chain instead of fanning out.

Issue ONE batch per request as the default. A second batch is for consuming a prerequisite you have
now resolved — not for re-asking a question that came back thin.

Each `ResearchQuery` needs: `query` (full standalone context — workers cannot see your conversation),
`preferred_tools` (exact source-tool names), `target_components`, a `rationale`, and a `depth`:
  - `low`    — one quick self-contained lookup (the default choice);
  - `medium` — a few corroborating searches;
  - `high`   — iterative multi-hop, where each result informs the next search. Expensive: at most one
               per request, and only for a genuine chain.

Returns a JSON array of `ResearchNotes` and persists each note as a JSON file under `/shared/`;
every source the workers cited is added to the verified-source set for `get_verified_sources`."""
```

---

## Appendix C — Open questions for review

1. **Shape B as a one-query batch vs a direct source call** (§4.2a). This is the plan's biggest
   behavioral bet and the one that most directly implements criterion 2. It trades one sub-run of
   latency for keeping raw results out of the parent context. Confirm or reverse.
2. **Direct-call budget of 2.** The Harbor report suggested 5–10. Two makes the effect visible;
   five is safer. Pick one.
3. **Should the planner ever be forced?** This plan says no — adaptive wins while calling it 9
   times in 90 trials, and the token/F1 analysis explicitly recommends against making it mandatory.
   The alternative (a structural trigger in middleware for ≥3 deliverables) is available if the
   probe set shows the description-only trigger still fails.
4. **Probe set home** — `frontends/benchmarks/` vs a stubbed pytest module under `tests/`. The
   pytest option is faster to iterate on and costs nothing per run.
5. **Do we want the C6 arm** (orchestrator with no direct source tools at all) run alongside, or
   only if the prompt-only arm underperforms on S3/S4?
