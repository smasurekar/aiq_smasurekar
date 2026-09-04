<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Autonomous Researcher: Input-Token and DeepSearchQA F1 Analysis

## Scope and absolute paths

This report compares the same 90 DeepSearchQA tasks in these saved Harbor jobs:

- Autonomous researcher job:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13`
- Adaptive researcher job:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21`

The source trees used to explain the observed behavior are:

- Autonomous researcher:
  `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher`
- Adaptive researcher:
  `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/adaptive_researcher`
- Autonomous configuration:
  `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/config_autonomous_frag.yml`
- Adaptive configuration:
  `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/config_adaptive_frag.yml`

The image tag and commit-like image label are intentionally ignored. This analysis uses the saved
job artifacts for measurements and the source directories above for the architectural explanation.

## Executive summary

The autonomous job consumed **21.1% more input tokens** while achieving **0.0400 lower mean F1**:

| Metric | Autonomous | Adaptive | Difference |
| :-- | --: | --: | --: |
| Mean F1 | 0.5098 | 0.5499 | -0.0400 |
| Precision | 0.5191 | 0.5905 | -0.0714 |
| Recall | 0.5501 | 0.5592 | -0.0091 |
| Fully correct | 30.0% | 31.1% | -1.1 pp |
| Fully incorrect | 32.2% | 28.9% | +3.3 pp |
| Correct with excessive answers | 13.3% | 11.1% | +2.2 pp |
| Average input tokens | 999,938 | 825,695 | +174,243 (+21.1%) |
| Average output tokens | 31,787 | 20,990 | +10,797 (+51.4%) |
| Average recorded LLM calls | 45.58 | 46.31 | -0.73 |
| Input tokens per observed LLM call | 21,886 | 17,731 | +23.4% |

The two outcomes have a common architectural cause: the autonomous orchestrator receives the full
research menu and is allowed to search directly. Raw search results and reasoning accumulate in the
long-lived parent context, increasing token cost and exposing the final decision to more noisy or
conflicting evidence. At the same time, description-only routing rarely selects the isolated
multi-hop researcher and never selects the planner or writer in this job.

The lower F1 is primarily a **precision and convergence problem**, not a simple lack-of-research
problem. Autonomous found almost as many expected answers in aggregate, but emitted substantially
more incorrect or out-of-scope candidates and sometimes committed to the wrong intermediate entity
in a multi-stage question.

## Important evaluation qualification

This is an observational comparison of two stochastic runs, not a deterministic causal experiment.
The model temperature, live search results, and judge response can vary between runs.

One autonomous trial, `deepsearchqa-0588`, has `grader_valid = 0` because the judge returned malformed
JSON (`Expecting ',' delimiter`). Harbor assigned that trial F1 = 0, while adaptive received F1 = 1.
That single judge failure accounts for **0.0111** of the reported 0.0400 mean-F1 gap.

On the 89 tasks where both graders were valid:

| Metric | Autonomous | Adaptive | Difference |
| :-- | --: | --: | --: |
| Mean F1 | 0.5156 | 0.5448 | -0.0292 |

The unadjusted scores remain the official saved-job results. The adjusted view is included only to
separate agent behavior from a verifier parse failure. Leaderboard compatibility would require the
same saved outputs to be scored by Harbor and the official benchmark scripts, using captured judge
responses for deterministic comparison.

## 1. Why autonomous input-token usage is higher

### 1.1 The extra tokens come from the orchestrator, not the researcher workers

| Input-token location | Autonomous total | Adaptive total | Per trial: autonomous | Per trial: adaptive |
| :-- | --: | --: | --: | --: |
| Top-level orchestrator | 31,914,203 | 7,740,030 | 354,602 | 86,000 |
| Research workers | 58,080,254 | 66,572,516 | 645,336 | 739,695 |
| Total | 89,994,457 | 74,312,546 | 999,938 | 825,695 |

Worker prompt cost per call is effectively identical: **17,843 autonomous versus 17,930 adaptive**.
Adaptive actually spends more tokens inside workers. Autonomous's extra 24.2 million orchestrator
tokens more than explain the entire 15.7 million-token net job difference.

### 1.2 Autonomous exposes source tools to the parent loop

The autonomous factory builds one unconditional tool menu containing helper tools,
`run_research_batch`, `submit_final_report`, and every source tool. The autonomous prompt explicitly
offers a direct source call as the cheap path for a quick lookup.

Relevant source files:

- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/factory.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/prompts/orchestrator.j2`

The adaptive standard/deep path instead hides direct source tools and requires research through
`run_research_batch`. Only its single-shot path exposes direct retrieval, with a separate search
budget.

Relevant adaptive files:

- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/adaptive_researcher/factory.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/adaptive_researcher/custom_middleware.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2`

Observed source calls confirm the different context boundary:

| Source calls | Autonomous | Adaptive |
| :-- | --: | --: |
| Orchestrator direct source calls | 480 | 34 |
| Worker source calls | 1,635 | 1,901 |
| Total | 2,115 | 1,935 |

Autonomous made 393 direct `advanced_web_search_tool` calls and 87 direct `web_search_tool` calls.
Those results enter the parent conversation. Each later parent-model call includes the accumulated
conversation again. Adaptive keeps most search histories inside isolated workers and returns
digested `ResearchNotes` to the parent.

### 1.3 The autonomous parent prompt starts larger and grows much further

| Parent prompt measurement | Autonomous | Adaptive |
| :-- | --: | --: |
| Mean first orchestrator prompt | 11,499 | 9,716 |
| Mean peak orchestrator prompt | 34,433 | 20,812 |
| Growth factor | 3.0x | 2.1x |
| Mean orchestrator input per call | 37,239 | 16,193 |

Adaptive enables `dynamic_orchestrator_sections`, which swaps in a prompt trimmed to the selected
tier after routing. Autonomous always carries the undifferentiated workflow, full routing guidance,
tool descriptions, and subagent descriptions.

Relevant files:

- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/adaptive_researcher/tiers.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/config_adaptive_frag.yml`

The larger base prompt contributes, but context growth from direct retrieval is the larger effect.

### 1.4 Both jobs report zero cached prompt tokens

Both saved jobs record `n_cache_tokens = 0`. Under the saved accounting, every model turn is billed
for the complete prompt presented on that turn. A search result or long completion therefore costs
tokens once when produced and again whenever it remains in the context of later turns.

This is an amplifier shared by both arms, not the original architectural difference. It hurts
autonomous more because its parent context is larger and lives for more turns.

### 1.5 Direct-search repetition has only a broad turn cap

The autonomous request guard enforces duplicate signatures for `run_research_batch` queries, but
its `awrap_tool_call` path does not apply the same duplicate guard to direct source-tool arguments.
Direct tools remain available until the request enters finalization, including through the broad
`max_orchestrator_turns: 100` envelope.

Relevant files:

- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/src/aiq_agent/agents/autonomous_researcher/custom_middleware.py`
- `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/config_autonomous_frag.yml`

Two concrete outliers:

| Trial | Autonomous input tokens | Direct source calls | Orchestrator calls | Result |
| :-- | --: | --: | --: | :-- |
| `deepsearchqa-0545` | 4,612,249 | 61 | 63 | F1 = 0; no normal final report |
| `deepsearchqa-0060` | 3,862,976 | 62 | 63 | F1 = 0; no normal final report |

In `deepsearchqa-0545`, the parent prompt grows from about 11.5k to 97k tokens while alternating
repeated Statistics Canada keyword searches and raw CSV URLs. It eventually returns a partial
research fallback instead of the requested province list.

Absolute evidence paths:

- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0545__ygXhU4q/agent/trajectory.json`
- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0060__BMgrgFY/agent/trajectory.json`

### 1.6 The average difference is tail-heavy

Across paired tasks, autonomous used more tokens on 46 tasks and adaptive used more on 44. The
median paired difference is only **+9,022 tokens**, while the mean is **+174,243**. The largest
autonomous runaway loops pull the average upward.

Even after removing the five heaviest and five lightest tasks in each arm, however, the trimmed
means remain 865k autonomous versus 729k adaptive. The difference is therefore both a tail problem
and a systematic parent-context problem.

## 2. Why autonomous F1 is lower

### 2.1 The dominant aggregate deficit is precision

Recall differs by only 0.0091, but precision differs by 0.0714. The verifier identified:

| Grader item count | Autonomous | Adaptive |
| :-- | --: | --: |
| Correct answer items found | 153 | 159 |
| Excessive answer items | 153 | 84 |

The expected-item totals cannot be compared directly because the malformed autonomous grader lacks
the normal count fields. The correct and excessive counts still show the main pattern: autonomous
finds nearly as many expected items but emits **82% more excessive items**.

This is not simply final-answer verbosity. Autonomous final artifacts average about 2,625 characters
and adaptive artifacts average about 2,871. Autonomous answers are shorter on average, but their
candidate selection is less precise.

Among the 30 tasks where adaptive F1 is higher:

- Autonomous precision is lower on 26.
- Autonomous recall is lower on 24.
- Both are lower on 20.
- Autonomous emits more excessive items on 17.
- Autonomous finds fewer expected items on 24.

### 2.2 Description-only routing underuses the multi-hop and synthesis paths

The autonomous architecture supplies three intended research paths:

- direct source tools for quick lookups;
- `task(subagent_type="researcher-agent")` for one dependent, iterative topic;
- `run_research_batch` for several independent questions.

Observed autonomous routing across 90 tasks:

| Route | Observed use |
| :-- | --: |
| `run_research_batch` | 120 calls, 286 worker contexts |
| `task(... researcher-agent ...)` | 3 calls, all in one trial |
| `task(... planner-agent ...)` | 0 |
| `task(... writer-agent ...)` | 0 |

DeepSearchQA contains many dependent questions: identify an entity from one source, then use that
entity to retrieve and filter a second dataset. The intended isolated multi-hop route was nearly
unused. Instead, the model usually fanned out an initial batch and then searched directly in its
parent context.

Adaptive performed more staged delegated research:

| Delegation shape | Autonomous | Adaptive |
| :-- | --: | --: |
| Batches per trial | 1.33 | 1.66 |
| Delegated queries per trial | 3.18 | 3.90 |
| Research workers | 286 | 351 |

Adaptive's second batch can incorporate the first batch's result at the parent boundary. That is
useful for prerequisite chains. Autonomous often substitutes a long sequence of parent searches,
which is both more expensive and less protected from noisy intermediate conclusions.

Planner usage is not by itself the explanation: adaptive used planner-agent only nine times and
writer-agent once. The more direct issue is failure to enforce dependency-aware research and a
clean final answer selection step.

### 2.3 Cascading error example: wrong prerequisite entity

`deepsearchqa-0543` asks the agent to identify the shuttle with the second-fewest construction
milestones, then list that shuttle's EAFB missions.

Autonomous:

- submitted two parallel, nearly duplicate `depth="high"` queries about milestone counts;
- preferred the basic `web_search_tool` for both;
- then made 21 direct source calls in the parent context;
- selected Atlantis instead of Endeavour;
- returned 12 Atlantis missions, all excessive;
- received precision = 0, recall = 0, F1 = 0;
- consumed 2.62 million input tokens.

Adaptive used two staged batches, selected Endeavour, returned six of the seven expected missions
with no excessive answers, and received F1 = 0.9231 using 348k input tokens.

Absolute evidence paths:

- Autonomous trajectory:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0543__SLWJS6x/agent/trajectory.json`
- Autonomous grading:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0543__SLWJS6x/verifier/grading.json`
- Adaptive trajectory:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21/deepsearchqa-0543__3fQJF3j/agent/trajectory.json`
- Adaptive grading:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21/deepsearchqa-0543__3fQJF3j/verifier/grading.json`

This is the clearest accuracy failure produced by incorrect decomposition: the first-stage mistake
determines every second-stage answer.

### 2.4 Excessive-answer example: evidence is present but the output set is not controlled

`deepsearchqa-0314` asks for the exhaustive list of Tyrannosaurus types as of May 2024. Autonomous
includes both expected answers, so recall is 1.0, but also names 13 invalid species, synonyms, or
nomina nuda. The DeepSearchQA set grader counts those names as excessive even though the prose labels
them invalid.

| Arm | Precision | Recall | F1 | Excessive items |
| :-- | --: | --: | --: | --: |
| Autonomous | 0.1333 | 1.0000 | 0.2353 | 13 |
| Adaptive | 1.0000 | 0.5000 | 0.6667 | 0 |

Absolute evidence paths:

- Autonomous answer:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0314__tyF4v27/artifacts/answer.txt`
- Autonomous grading:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0314__tyF4v27/verifier/grading.json`
- Adaptive answer:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21/deepsearchqa-0314__NCLGdWU/artifacts/answer.txt`
- Adaptive grading:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21/deepsearchqa-0314__NCLGdWU/verifier/grading.json`

For this benchmark, rejected candidates should not be enumerated in the final answer when their
names can be interpreted as answer candidates. A final set-extraction step would preserve the
evidence while emitting only the qualifying entities.

### 2.5 Insufficient corroboration example

`deepsearchqa-0309` requires identifying the five countries with the most internally displaced
people and then comparing their agriculture-employment percentages.

Autonomous used one batch with two delegated queries, accepted a secondary-source Syria value of
16.6%, and answered Colombia. Adaptive used two batches with three queries, found a lower Syria
value, and answered Syria. The scores were F1 = 0 versus F1 = 1.

This example is not proof that two batches are always better. It shows that a multi-source numeric
comparison needs explicit same-year/source validation before selecting the minimum. The autonomous
workflow has a stopping rule for sufficient evidence but no deterministic completeness check that
every candidate is compared on a consistent definition and year.

Absolute evidence paths:

- Autonomous grading:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-13__19-24-13/deepsearchqa-0309__8HEDtne/verifier/grading.json`
- Adaptive grading:
  `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/jobs/2026-08-01__19-18-21/deepsearchqa-0309__js5GvrW/verifier/grading.json`

### 2.6 Research effort does not convert reliably into accuracy

Within the autonomous job:

| Outcome | Trials | Average input tokens | Average LLM calls | Average source calls |
| :-- | --: | --: | --: | --: |
| F1 = 0 | 30 | 1,435,245 | 55.3 | 30.8 |
| 0 < F1 < 1 | 33 | 936,850 | 46.4 | 23.5 |
| F1 = 1 | 27 | 593,373 | 34.1 | 15.4 |

Fully wrong tasks consume 2.4 times the input tokens of fully correct tasks. The extra work is a
symptom of uncertainty or a failed retrieval strategy, not evidence of better coverage.

The best observed autonomous run shape is one research batch:

| Autonomous run shape | Trials | Mean F1 | Average input tokens |
| :-- | --: | --: | --: |
| No delegated batch | 15 | 0.5039 | 667,089 |
| One batch | 43 | 0.5709 | 644,062 |
| Two batches | 21 | 0.3744 | 1,463,692 |
| Three or more batches | 11 | 0.5379 | 1,959,632 |

These buckets are observational and task difficulty is a confounder. They nevertheless show that
unbounded additional search is not an accuracy strategy.

### 2.7 Tool-depth choices are not well calibrated

Autonomous marks 120 of 286 delegated queries as high depth (42.0%), compared with 88 of 351
adaptive queries (25.1%). Only 43 autonomous queries are low depth (15.0%).

Within autonomous, trials using at least one high-depth query have essentially the same mean F1 as
trials without one (0.5116 versus 0.5076), while using substantially more tokens (1.15M versus
812k). This does not prove high depth is harmful—the tasks may be harder—but it shows that the
model's frequent high-depth selection is not producing a visible aggregate accuracy return.

Worker tool choice also differs:

| Worker source tool | Autonomous | Adaptive |
| :-- | --: | --: |
| `advanced_web_search_tool` | 790 | 1,394 |
| `web_search_tool` | 845 | 507 |

The configured basic web tool returns up to five results truncated to 1,000 content characters,
while the advanced tool returns two advanced-search results. Autonomous workers favor the basic
tool even while labeling many queries high depth. For exact tables, PDFs, and database records,
that combination can provide breadth without enough extractable detail. This is a plausible
contributor, not a proven independent causal effect.

### 2.8 The result is variable, not a uniform autonomous regression

Paired F1 outcomes across the 90 tasks:

- Autonomous higher: 24 tasks.
- Adaptive higher: 30 tasks.
- Equal: 36 tasks.
- Median paired F1 difference: 0.
- Adaptive F1 = 1 and autonomous F1 = 0: 7 tasks.
- Autonomous F1 = 1 and adaptive F1 = 0: 9 tasks.

Autonomous therefore retains meaningful capability and sometimes wins decisively. Its lower mean
comes from somewhat more losses, precision-heavy partial regressions, a few costly convergence
failures, and the one invalid grader—not from every task becoming worse.

## 3. Prioritized changes

| Priority | Change | Expected effect |
| :-- | :-- | :-- |
| 1 | Add a final answer-set extraction rule/pass: identify the exact requested entity set and omit rejected alternatives, historical synonyms, nearby candidates, and research process commentary. | Directly addresses the 0.071 precision gap and 153 excessive items. |
| 2 | Enforce dependency-aware routing. A question with a prerequisite chain should use one iterative `researcher-agent` or explicitly staged batches; do not parallelize two near-duplicate queries for the same prerequisite. | Prevents cascading wrong-entity failures such as `deepsearchqa-0543`. |
| 3 | Add a hard orchestrator direct-search budget and normalized duplicate guard, separate from `run_research_batch` budgets. A small default such as 5–10 calls is consistent with the intended quick-lookup path. | Removes the worst token outliers and forced partial-result exits. |
| 4 | When direct searching becomes sequential or results conflict, force delegation to `researcher-agent` instead of allowing the parent to continue accumulating raw results. | Improves context isolation, convergence, and token usage together. |
| 5 | For numeric comparison questions, require a completeness check before final selection: same year, same definition, same source family, and one value for every candidate. | Addresses wrong minima/maxima caused by inconsistent secondary-source values. |
| 6 | Bias default query depth to low/medium and reserve high for demonstrably chained retrieval. Avoid multiple parallel high-depth queries with the same target component. | Reduces worker loops that show no aggregate F1 return. |
| 7 | Prefer advanced extraction or a purpose-built URL/CSV fetch tool for exact database pages; do not repeatedly submit raw URLs to a keyword search tool. | Improves evidence quality for Statistics Canada, NYSED, NASA tables, and similar tasks. |
| 8 | Reduce the stable autonomous prompt/tool-schema footprint and enable provider prompt caching where supported. | Lowers cost without changing research behavior. |
| 9 | Capture judge responses and rescore the same saved answers in both Harbor and the official benchmark scorer. | Separates agent changes from judge nondeterminism and is required before leaderboard-compatibility claims. |

Making planner/writer mandatory for every task is not recommended from this result alone. Adaptive
used planner only nine times and writer once. The higher-value intervention is a small,
benchmark-compatible answer-selection contract plus deterministic routing for dependent research.

## 4. Reproduction and methodology

Repository analysis scripts used:

- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/analysis/autonomous_agent_deep_dive.py`
- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/analysis/autonomous_agent_token_mechanics.py`
- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/analysis/autonomous_agent_patterns.py`
- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals/analysis/compare_research_arms.py`

Example commands from the eval repository root:

```bash
python3 analysis/compare_research_arms.py \
  jobs/2026-08-13__19-24-13 \
  jobs/2026-08-01__19-18-21

python3 analysis/autonomous_agent_deep_dive.py jobs/2026-08-13__19-24-13
python3 analysis/autonomous_agent_deep_dive.py jobs/2026-08-01__19-18-21

python3 analysis/autonomous_agent_token_mechanics.py jobs/2026-08-13__19-24-13
python3 analysis/autonomous_agent_token_mechanics.py jobs/2026-08-01__19-18-21

python3 analysis/autonomous_agent_patterns.py jobs/2026-08-13__19-24-13
python3 analysis/autonomous_agent_patterns.py jobs/2026-08-01__19-18-21
```

Attribution details:

- `agent/aiq_events.jsonl` supplies ordered LLM and tool events. Calls inside awaited
  `run_research_batch` or `task` windows are attributed to delegated workers; calls outside are
  attributed to the orchestrator.
- `agent/trajectory.json` supplies per-step prompt/completion tokens and tool arguments.
- `result.json` supplies Harbor token totals and verifier rewards.
- `verifier/grading.json` supplies expected/correct/excessive item counts and the judge explanation.
- Trial identity is paired by the stable `deepsearchqa-NNNN` prefix, ignoring each run's random
  suffix.

The per-step token attribution reconciles to the saved job totals:

- Autonomous: 31,914,203 orchestrator + 58,080,254 worker = 89,994,457 input tokens.
- Adaptive: 7,740,030 orchestrator + 66,572,516 worker = 74,312,546 input tokens.
