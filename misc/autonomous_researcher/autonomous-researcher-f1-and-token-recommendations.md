# Autonomous researcher: raising F1 and cutting token spend with Fetch disabled

Measurement notes and change proposal.

- **Date:** 2026-08-31
- **Goal:** reach the accuracy and token cost of the Chat Researcher arm using the
  autonomous researcher on Nemotron 3 Ultra, with `fetch_url_tool` disabled.
- **Subject job:** `ai-q-harbor-evals/jobs/2026-08-27__10-57-20` — autonomous arm,
  `configs/config_autonomous_frag.yml` at HEAD of
  `dev/smasurekar/aiq-auto-agent-shallow-subagent`, dataset `dsqa90`, 90 trials, 0 errors.
  (The `AIQ_RUNTIME_IMAGE`/`AIQ_REVISION` values in `config.json` are hardcoded and do
  not describe this run.)
- **Reference job:** `ai-q-harbor-evals/jobs/2026-07-31__18-57-45` — Chat Researcher arm,
  `configs/shallow_deep_nemotron_ultra.yml`, same model, same dataset, same 90 tasks.
- **Follow-up job:** `ai-q-harbor-evals/jobs/2026-08-31__06-05-55` — autonomous arm with
  R1 applied (`max_content_length: 1000 → 10000`), same config otherwise, same 90 tasks.
- **Pooled re-analysis:** all 14 Fetch-disabled autonomous `dsqa90` jobs run in August 2026
  (1,259 trials over the same 90 questions). This is the evidence base for §8 and supersedes
  every single-run comparison in §1–§5.
- **Status:** R1 and R7 both tested and **refuted** (§7, §8). R2's accuracy claim is refuted too
  (§8.2a); it survives only as a cost control. R3–R5 stand, with R3 restated as a request-wide
  ceiling. **R6 is fixed and shipped** (`bb74e57`, 2026-08-31), not yet measured on an eval
  job. The load-bearing recommendations are now **N1–N4 in §8.7**.

> **Corrections (2026-08-31). Read this before acting on anything in §1–§5.**
>
> **R1 is refuted.** `max_content_length` truncates Tavily's *snippet*, not the page body, so
> it cannot recover a table that was never in the API response. Applying it left the median
> search unchanged, inflated the long tail, and raised cost 2.6× on a route-matched subset
> with no accuracy gain. Measurements in §7.
>
> **R7 is refuted, and so is the §2.4 argument behind it.** "Shallow beats deep at matched
> difficulty" was **selection bias**: difficulty is far too coarse a control, and the
> orchestrator picks the route. Controlling for the *same question* across 1,259 trials,
> route explains **0.6% of F1 variance and 28.9% of token variance**. Routing is a cost
> lever, not an accuracy lever. Measurements in §8.
>
> **What replaced them.** One real routing signal survives — it keys on the *shape of the
> answer* (enumerate-all vs pick-one-winner), not on difficulty or staging — plus a much
> larger token lever that has nothing to do with routing. See §8.7 for the revised actions.
>
> **A pattern worth naming.** R1, R2 and R7 failed the same way: a strong raw gradient that
> vanishes once the *same question* is held fixed. Difficulty-matching is not enough, because
> the agent chooses its own route and batch count. Any future claim of the form "shape X
> scores better than shape Y" must be identified within question (§8.10).
>
> §1–§5 are retained unedited apart from the supersede banners added in place, so the
> reasoning errors stay legible. Do not act on them.

---

## Summary in plain English

*No metrics vocabulary in this section. Everything here is restated with numbers in
sections 1-8 below. This section was rewritten on 2026-08-31 after two of the original
findings turned out to be wrong; §8 has the evidence.*

### What is going on

We have two versions of the research agent answering the same 90 questions with the same
model. The older one (Chat Researcher) gives better answers and is three times cheaper. The
new one (Autonomous Researcher) works much harder — twice as many thinking steps, nearly
three times as many web searches — and ends up **less** accurate.

So the problem is not that the new agent is lazy or underpowered. It is that it is
overworking, and the extra work is not buying anything.

### Two things we believed in the first draft that turned out to be false

Worth stating plainly, because both of them shaped the original recommendations.

**We thought the agent was only reading the first 1,000 characters of each search result,
and that widening it would be the biggest accuracy fix.** It was not. That setting only
trims the short preview text the search service already sends back; it cannot make the
service send back more. We tried it. Accuracy did not improve and cost went up 2.6× on the
questions we could compare like for like.

**We thought the quick route was more accurate than the expensive route, and that sending
more questions to it would raise the score.** It does not. That result came from comparing
questions of the same *difficulty label*, which is far too rough — and the agent itself
chooses which questions go where, so it was quietly sending the easy ones down the quick
route and making it look good.

The right comparison is the **same question**, answered both ways, many times. We had never
done that. There are 14 past runs of this same 90-question set with the page-opening tool
switched off, which is 1,259 answers in total, and most questions have been answered both
ways across those runs. When you compare that way, the answer is clear:

**Which route a question takes barely affects the score at all. It affects the cost enormously.**

Route choice accounts for about half a percent of the variation in accuracy, and about
thirty percent of the variation in cost. Sending everything down the quick route would save
roughly three-quarters of the budget and cost somewhere between nothing and five points of
accuracy — we cannot measure a difference with confidence even with 1,259 answers.

### What is actually going on, restated

**1. The routing is close to random, but not because the agent is confused.**
The agent does read the question and does have a consistent opinion about it — we checked,
and its preference is stable when measured across many runs. The problem is that it never
*acts* on that preference decisively. Its opinion comes out as roughly "a 30% chance of
choosing the quick route", and then the model rolls the dice. Across all 90 questions there
is not one where it reliably picks the quick route. Only 3 of 90 questions get the same
treatment in more than 80% of runs, and 59 of 90 are effectively coin flips.

The cause is mundane: the routing choice is made by the same creative-writing-temperature
model that does the research, in passing, as one option among five. The old agent that works
better has a **separate, calmer decision step** just for this.

**2. There is one real rule about which questions suit which route — and it is the opposite
of the one currently written in the instructions.**

Questions in this set come in two shapes:

- *"List all the states where X, Y and Z are true"* — the answer is a set. You get partial
  credit: finding 4 of the 6 correct items still scores something.
- *"Which company had the greatest reduction in expenses?"* — the answer is one winner. You
  get no partial credit. To answer at all you must find a value for **every** candidate, on
  the same definition and the same year. Miss one and you score zero.

The quick route is a single bounded pass. On the first shape it does fine — it gathers what
it can and banks the partial credit, and it does this **as well as the expensive route at
one-fifth the cost**. On the second shape it runs out of budget, writes up a confident
partial answer, and scores zero. The expensive route is worth paying for there.

That is a scoring rule, not a difficulty rule, and it is a large effect — about 22 points of
accuracy separating the two shapes. It replicates across two independent halves of the run
history.

**Here is the problem.** The written instructions tell the agent to *avoid* the quick route
when a request "narrows a set in steps… two number conditions stacked". But stacked
conditions are exactly the *first* shape — the enumeration questions the quick route handles
best. Meanwhile "which company had the greatest reduction" reads like a simple one-off
lookup and gets sent to the quick route, where it scores near zero. **The rule is pointing
the wrong way.**

**3. Roughly one run in ten runs away, and those runs cost more than a third of everything.**

About 11% of runs perform 61 or more web searches. Those runs consume **37% of the entire
token budget** and score 0.30, against 0.60 for everything else. Searching more has never
helped: comparing the same question with itself, accuracy is flat from zero to sixty
searches and measurably *worse* beyond sixty.

The reason is a gap in the safety limits. There are caps on how many research questions can
be sent out, how many rounds, and how many searches the top-level agent may run itself — but
**no cap on the total number of searches the whole request may perform.** Each delegated
worker gets its own fresh allowance, so twenty workers with twenty searches each is four
hundred searches, and nothing stops it. The old agent's worst run did 131 searches. The new
agent's worst did over 3,000.

**4. Two runs got stuck in a loop and burned a ninth of one job's budget. — NOW FIXED.**
A guard meant to stop a researcher repeating itself was shutting the researcher down
entirely, and then failing to actually take the search tool away. So the researcher tried
again, got blocked, and repeated. One run did this 131 times, another 122.

This was fixed on 2026-08-31. Two things changed: repeating one search no longer shuts the
researcher down (it just refuses that one search and tells it to ask differently), and there
is now a hard stop after three refusals in a row that makes the researcher write up what it
has. The reason the old "take the tool away" approach never worked is worth recording: taking
a tool away only changes what the model is *offered*. If the model copies an old tool name out
of its own conversation history, the system still runs it. So the hard stop is the only thing
that actually enforces anything.

We do not yet know how much this saves — no evaluation has been run against the fix. It may
be worth more than the two disaster runs suggest, because the same overrun shows up mildly
across the whole history (workers averaging 44 searches against an allowance of 20).

**5. The prompts still tell the agent to use a tool that is switched off.**
Also still stands, still untested. Four places in the instructions reference the
page-opening tool, which is disabled. One of them says: do not search a third time for the
same fact — open the page instead. The agent cannot open the page, so it is left with no
approved next move. Worth noting the old agent **never opened a single page** in all 90
questions and still scored higher, so keeping the tool off is fine. We just need to stop
telling the agent to reach for it.

### What to change

1. **Put a ceiling on the total number of searches one request may make** (around 45).
   Config only. This is the biggest measured saving: it removes a tenth of runs that consume
   a third of the budget and score half as well as everything else.
2. **Flip the quick-route rule to key on the shape of the answer.** Send "list everything
   that qualifies" questions to the quick route; keep "pick the single winner" questions off
   it. Instructions only, no code.
3. **Make the routing decision separate and calm**, the way the old agent does it. Without
   this, change 2 cannot hold — a rule the agent follows a third of the time is not a rule.
4. **Cut the quick researcher's allowance to match the old agent's** (5 searches, 10 turns
   instead of 10 and 20). Cheap to try, and it is a known-good setting on the arm that
   scores better.
5. ~~**Fix the stuck-loop bug.**~~ **Done** — shipped 2026-08-31. Needs an evaluation run to
   size the benefit, and that run should come **first**, because it may shrink the problem
   change 1 is meant to solve.
6. **Stop mentioning the page-opening tool while it is switched off.**

### What we expect — and the part we cannot promise

Changes 1-4 should cut cost by roughly half while leaving accuracy about where it is. That
is a real result and it gets us to the old agent's *cost*.

**It does not get us to the old agent's accuracy, and we should stop expecting it to.** We
simulated every possible way of assigning these 90 questions to routes. The best any of them
reaches is 0.58, and the expensive-everything option is what reaches it. The old agent scores
0.61. **No amount of routing gets there.**

So the accuracy gap is a separate problem from the cost problem, and it is not in the
routing. The shape of it is this: the old agent reaches 0.61 using about 12 searches and 22
thinking steps. The new agent reaches 0.56 using 34 searches and 51 thinking steps. Three
times the work, worse answers. Something about how the new agent turns evidence into an
answer is losing information that the old one keeps — and that is where the next
investigation should go, not into routing.

### The one-sentence version

Routing is a cost dial, not an accuracy dial: turning it down saves about half the budget
for free, but the missing accuracy is somewhere else entirely — in what the agent does with
the evidence after it has gathered it.

---

## 1. Where the two arms stand

| Metric | Reference (chat researcher) | Autonomous (current) | Ratio |
| :-- | --: | --: | --: |
| Mean F1 | **0.6078** | 0.5563 | −0.052 |
| Fully correct | 42.22% | 32.22% | −10.0 pp |
| Fully incorrect | 25.56% | 28.89% | +3.3 pp |
| Precision / Recall | 0.616 / 0.623 | 0.583 / 0.571 | — |
| Avg input tokens | **295,080** | 912,133 | **3.09×** |
| Avg output tokens | 8,394 | 19,709 | 2.35× |
| Avg LLM calls | 21.67 | 50.59 | 2.33× |
| Avg source calls | 11.93 | 33.10 | 2.77× |
| Avg latency (s) | 273.9 | 204.0 | 0.75× |
| Routed to shallow | **78%** (70/90) | 42% (38/90) | — |
| `fetch_url_tool` calls | **0** | 0 | — |

The reference arm made **zero page opens across all 90 trials** — its config exposes
`fetch_url_tool` but the model never selected it. The 0.6078 target is therefore fully
reachable with Fetch disabled; nothing in the gap is attributable to the missing tool.

Reproduce with:

```bash
cd ~/smasurekar/ai-q-harbor-evals
python3 analysis/autonomous_agent_deep_dive.py jobs/2026-08-27__10-57-20
python3 analysis/autonomous_case_breakdown.py  jobs/2026-08-27__10-57-20
```

---

## 2. Diagnosis

### 2.1 Cost and error are concentrated in the deep paths

`analysis/autonomous_case_breakdown.py` splits the job by delegation shape:

| Case | n | F1 | Fully wrong | Input tokens | LLM calls | Searches |
| :-- | --: | --: | --: | --: | --: | --: |
| 1: Shallow-Researcher | 38 | 0.603 | 18% | 101,456 | 9.1 | 6.3 |
| 2.1: No-Planner-Shallow | 6 | **0.952** | 0% | 381,594 | 18.3 | 10.3 |
| 2.2: No-Planner-Deep | 20 | **0.396** | **50%** | 1,515,625 | 86.3 | 63.2 |
| 3: Planner | 26 | 0.521 | 35% | 1,755,175 | 91.2 | 54.4 |

Cases 2.2 and 3 are 46 of 90 questions but **92.5% of all input tokens
(75.9M of 82.1M)**, and they average F1 0.466 — *below* the cheap shallow path's 0.603.

### 2.2 This is a turn-count problem, not a context-bloat problem

```
prompt_tokens_subagent_total   71,879,731   (87.6% of all input)
prompt_tokens_orchestrator_total 10,212,224
llm_subagent_mean                    46.46
llm_orchestrator_mean                 4.13
orchestrator_prompt_peak_mean        24,885
```

Peak context is ~25K for the orchestrator and ~43K for a deep trial. Tokens per LLM call
are 18.0K here versus 13.6K in the reference — a 1.3× difference. The 3.1× total gap comes
almost entirely from making **2.3× as many LLM calls**, nearly all of them inside research
sub-agents. Any fix must remove turns, not shrink prompts.

### 2.3 More effort correlates with *worse* accuracy

> **Partly superseded (§8.5).** The direction is right but the effect is much narrower than
> stated here. With question *and* job fixed effects absorbed, F1 is **flat from 0 to 60
> searches** and only turns significantly negative past 61 (−0.051, CI [−0.101, −0.001]).
> The graded "25 searches → 0.65, 99 searches → 0.45" comparison below is confounded by
> question difficulty. The ≥61 tail is the real, and still large, finding.


```
corr(F1, source calls) = -0.203   (all 90 trials)
corr(F1, LLM calls)    = -0.194
corr(F1, input tokens) = -0.195
corr(F1, batch calls)  = -0.171   (deep trials only)
```

Deep and planner trials bucketed by search volume:

```
low   n=15  searches=25.2  F1=0.645  in=  941,971
mid   n=15  searches=47.2  F1=0.310  in=1,372,834
high  n=16  searches=99.4  F1=0.446  in=2,576,561
```

This is not just hard questions causing more searching. At **matched difficulty** the
shallow path still beats the deep path:

| Difficulty | auto-shallow F1 | auto-deep F1 | auto tokens | ref F1 | ref tokens |
| :-- | --: | --: | --: | --: | --: |
| easy (2) | 0.929 | — | 98,786 | 0.938 | 23,402 |
| medium (68) | 0.617 (n=30) | 0.555 (n=32) | 775,448 | 0.657 | 327,216 |
| hard (20) | **0.420** (n=6) | **0.263** (n=14) | 1,458,197 | 0.408 | **212,985** |

On hard questions the autonomous arm spends 1.46M tokens to score 0.310; the reference
spends 213K to score 0.408.

### 2.4 The routing decision is the dominant lever — **REFUTED for accuracy, upheld for cost**

> **Superseded by §8.2.** The table below compares what the *reference arm* did with
> questions the *autonomous arm* sent deep. Both arms choose their own routes, so this is a
> comparison of two selection policies, not of two routes. Controlling for the same question
> across 1,259 trials, route explains **0.6% of F1 variance**. The "19× the tokens for
> statistically identical accuracy" claim survives — that is the cost half, and it is real
> and large. The implication that *re-routing raises F1* does not survive.


Restricting to the 46 questions the autonomous agent sent down deep/planner paths, and
asking what the reference did with those same questions:

| | n | ref F1 | ref tokens | auto F1 | auto tokens |
| :-- | --: | --: | --: | --: | --: |
| Reference sent these **shallow** | 28 | 0.431 | **75,153** | 0.420 | **1,458,360** |
| Reference sent these **deep** | 18 | 0.589 | 1,115,759 | 0.539 | 1,950,721 |

On 28 of 90 questions the autonomous agent spends **19× the tokens for statistically
identical accuracy**. This single row is the strongest argument in the whole analysis.

---

## 3. Recommendations

Ordered by implementation cost. Items 1–4 are config-only and can ship as one eval arm.

### R1 — Restore search-result width (config, 1 line) — **REFUTED, do not apply**

> **Tested 2026-08-31 and refuted.** See §7. The argument below is left unedited because
> the way it fails is instructive: it reasons correctly from two real failures to a cause
> that turns out not to be the cause. R2 below is also weakened (§8.2a); R3–R6 stand.


`configs/config_autonomous_frag.yml`:

```yaml
web_search_tool:
  _type: tavily_web_search
  include_answer: advanced
  max_results: 5
  max_content_length: 1000     # -> 10000
  advanced_search: true
```

The block's own comment states it is matching a profile of "5 results x 10000 chars", but
the value is `1000`. `TavilyWebSearchToolConfig.max_content_length` defaults to `None`
(no truncation), so this is an explicit 10× cut, and
`sources/tavily_web_search/src/register.py:113` applies it per result. The shallow
sub-agent is pinned to this tool (`shallow_subagent_tools: [web_search_tool]`), so 5×1000
characters is *all* the evidence that path will ever see.

Two failures traced end to end:

- **`deepsearchqa-0249`** (FDIC — highest final-dividend "Total Paid" among 2011 Wisconsin
  bank failures). The autonomous shallow run found all seven correct FDIC pages, then
  wrote *"the FDIC sources provided do not disclose the final 'Total Paid' dividend
  percentage for any of the three banks."* F1 **0.000**, after 10 searches and 157,287
  tokens. The reference read the same pages at 10,000 chars, produced the comparison
  table, and scored **1.000**.
- **`deepsearchqa-0242`** (ICILS 2023 Table 2.2). The autonomous agent inferred the table
  from snippets *about* the table and named Germany and Uruguay — F1 **0.000**. The
  reference counted the actual symbols in the table and named Netherlands and United
  States — F1 **1.000**.

Truncation is also *why* workers search more: a thin snippet never settles the question,
so the loop continues. Expect this to be roughly token-neutral or better despite returning
more per call.

**Expected:** +0.05 to +0.08 F1; tokens neutral to slightly down.

**Actual:** +0.004 F1 on a route-matched subset (95% CI [−0.12, +0.13]) and **2.6× tokens**.
Both trials cited above still scored 0.000. See §7.

### R2 — Cap the orchestrator at one research batch (config, 2 lines) — **accuracy claim not upheld**

> **Re-tested 2026-08-31 (§8.2a).** The one-batch/two-batch/three-batch F1 gradient that
> motivated this is **the same selection artifact as R7**: it disappears entirely under
> question and job fixed effects. The *token* gradient is real and steep (623K → 2,180K), so
> R2 survives as a cost control with no expected accuracy gain. It replicated across two runs
> because selection into batch count is stable, not because batching harms accuracy.


```yaml
request_termination:
  max_batch_calls: 6              # -> 1
  max_total_research_queries: 20  # -> 6
```

Non-shallow trials by number of `run_research_batch` calls:

```
batches=1   n=18  F1=0.674  in=  936,974  llm= 48.4  src= 24.8
batches=2   n=15  F1=0.454  in=1,447,568  llm= 74.4  src= 53.8
batches=3   n= 8  F1=0.385  in=1,551,754  llm= 79.2  src= 52.6
batches=6   n= 5  F1=0.267  in=2,474,471  llm=152.2  src=112.6
```

And by total queries dispatched:

```
queries 2-3   n=13  F1=0.597  in=  905,402
queries 4-5   n=14  F1=0.476  in=1,219,995
queries 8-9   n=13  F1=0.497  in=2,746,503
```

Monotonic in both: more fan-out is worse *and* more expensive. The existing config comment
already records "one batch remains the best observed shape"; the data now supports making
that a ceiling rather than guidance.

**Expected:** +0.05 to +0.10 F1 on the deep bucket; ≈−400K tokens/trial on that bucket.

### R3 — Cut per-worker search budgets (config, 4 lines)

```yaml
researcher_loop_guard:
  source_call_budgets:
    low: 5        # -> 3
    medium: 10    # -> 5
    high: 20      # -> 8
request_termination:
  max_high_depth_queries: 3   # -> 1
```

Sub-agents made 2,979 source calls across 293 dispatched queries — **10.2 searches per
worker**, i.e. workers run to their budget essentially every time. Declared depths were
`medium: 166, high: 73, low: 54`, a budget-weighted ceiling of 3,390 searches. Combined
with §2.3, that ceiling is buying noise.

Sub-agents also burned **403 `think` calls** (10.6/trial in the planner case, 5.9 in the
deep case). Each is a full LLM turn that gathers no evidence.

**Expected:** ≈−200K tokens/trial; small positive F1.

### R4 — Tighten the shallow sub-run to the reference bounds (config, 2 lines)

```yaml
shallow_subagent_max_tool_iterations: 10   # -> 5
shallow_subagent_max_llm_turns: 20         # -> 10
```

Head-to-head on the 38 questions the autonomous agent routed to shallow — the same
questions the reference also routed to shallow (37 of 38):

| | Reference | Autonomous |
| :-- | --: | --: |
| `max_tool_iterations` / `max_llm_turns` | 5 / 10 | 10 / 20 |
| Observed searches | 3.51 | 6.32 |
| Observed LLM calls | 5.46 | 9.11 |
| Input tokens | 86,104 | 101,456 |
| **F1** | **0.685** | **0.603** |

Autonomous better on 3, worse on 18, tied on 17.

Bucketed by search count, the degeneration is visible:

```
searches 1-3   n=11  F1=0.77   in= 35K
searches 4-9   n=15  F1=0.63   in= 95K
searches 10    n=12  F1=0.468  in=170K   <- hit the cap; ref scored 0.650 on these
```

Twelve trials logged `Max iterations (10) reached` / `Forcing synthesis`. A cap of 5 forces
the decision before the run degenerates into snippet-collecting.

**Expected:** +0.03 to +0.05 F1 on the shallow bucket; ≈−80K tokens on the capped trials.

### R5 — Gate the Fetch prompt text on tool availability (code)

With `fetch_url_tool` in `exclude_tools`, four prompt sites still instruct the model to use
it. There is no gating variable anywhere in `src/aiq_agent/`:

- `prompts/orchestrator.j2:22` — the entire `fetch_url_tool` bullet under Tool Instructions.
- `prompts/orchestrator.j2:36` — *"when you already have the URL, stop searching and open
  it with `fetch_url_tool`. A third search for the same fact is always worse than one page
  open."*
- `prompts/researcher.j2:10` — *"call `fetch_url_tool` on the best URL from your results and
  read the page itself. Do not infer a table's contents from a snippet about the table."*
- `prompts/researcher.j2:60` — budget accounting for a tool that cannot be called.

Line 36 is actively harmful in this arm: it forbids the third search and points at a door
that does not exist, leaving the model with no sanctioned next move. `researcher.j2:10`
names exactly the failure mode of `deepsearchqa-0242` and then prescribes an unavailable
remedy.

Add a `fetch_enabled` flag rendered into both templates, following the existing
`research_batch_enabled` / `researcher_subagent_enabled` pattern in
`factory.py`, and gate all four sites. When Fetch is off, the fallback guidance should be
"prefer the source organisation's own page in your next search" rather than silence.

**Expected:** small F1 gain; small token reduction from fewer dead-end turns.

### R6 — Fix the sticky-exhaustion block loop (code) — **FIXED, shipped in `bb74e57`**

> **Shipped 2026-08-31**, commit `bb74e57` *"Stop repeated-query trips from exhausting
> researchers and add a blocked-call ceiling"*. Both proposed fixes landed, plus a third the
> analysis below missed. 75 tests pass in
> `tests/aiq_agent/agents/adaptive_researcher/test_custom_middleware.py`.
>
> **Correction to this section's mechanism.** The paragraph below guesses that
> `CURRENT_RESEARCHER_GUARD_STATE` may not be set in the model-call path. That was wrong. The
> real cause is that **tool withdrawal is advisory**: `request.tools` controls only what the
> model is *bound* to, but LangChain routes a pending tool call to the tools node by
> *registered* name (`_make_model_to_tools_edge`). A model replaying a withdrawn-but-registered
> name out of its own message history still gets that call executed, is blocked again, and
> repeats. `_filter_tools` was working; it just cannot stop a replay. That is why the ceiling
> in fix 2 is the only actual enforcement, not a belt-and-braces addition.


`src/aiq_agent/agents/adaptive_researcher/custom_middleware.py:1180`:

```python
if state.exhausted or state.source_call_count >= budget:
    self._mark_exhausted(state, "total source-call budget")
    logger.warning(
        "... invocation=%s depth=%s tool=%s calls=%d/%d reason=total_budget",
        state.invocation_id, state.depth, tool_name, state.source_call_count, budget,
    )
    return self._blocked_result(tool_call, "the total source-call budget")
```

Once *any* rule sets `state.exhausted` — including the repeated-signature rule at line 1196
— every later call takes this branch and is logged as `reason=total_budget` with a stale,
misleading count. `_filter_tools` is meant to withdraw the tool at that point
(`wrap_model_call` / `awrap_model_call`), but it demonstrably is not reaching the model.

Observed in `jobs/2026-08-27__10-57-20/*/agent/aiq-agent-console-stdout.txt`:

```
12:23:34  blocked repeated source call | invocation=48e92c58... depth=high repeats=2/2 reason=repeated_signature
12:23:35  blocked source call          | invocation=48e92c58... depth=high calls=7/20  reason=total_budget
          ... the identical line 116 more times, same invocation ...
```

The worker was killed by a duplicate-query trip at **7 of 20** calls, then spun.

| Trial | Blocked calls | Dup trips | Input tokens | LLM calls | F1 |
| :-- | --: | --: | --: | --: | --: |
| `deepsearchqa-0459__nNYyJ9E` | 122 | 8 | 6,061,650 | 332 | 0.000 |
| `deepsearchqa-0189__xxK8aBD` | 116 | 3 | 3,556,500 | 194 | 0.571 |

Those two trials alone consume **9,618,150 tokens — 11.7% of the entire job's input** — at
mean F1 0.286. Job-wide there were **252 blocked source calls** and **28 of 90 trials
tripped the duplicate guard at least once**. Each trip permanently exhausts a worker that
still has budget left, so it costs accuracy too: thin notes come back and the orchestrator
launches another batch.

#### What shipped (`bb74e57`)

The guard now separates three rules that were previously conflated into one `exhausted` flag:

| Rule | Trigger | Effect on the worker |
| :-- | :-- | :-- |
| **Repeat** | same normalized request beyond `max_identical_source_calls` | rejects **that signature only**; worker keeps its remaining depth budget and is told to vary the query |
| **Budget** | `source_call_budgets.for_depth(depth)` calls executed | exhausted; `_filter_tools` withdraws source tools and `think` |
| **Ceiling** | `max_consecutive_blocked_source_calls` rejections with no call executing between | exhausted **and** the next model call is forced to emit `ResearchNotes` |

Implementation notes worth carrying forward:

- A `_BlockReason` dataclass ties each rejection's log `code`, model-facing `phrase`, and
  `exhaustion_reason` together, with `exhaustion_reason=None` for repeats. This is what makes
  the misleading `reason=total_budget` logging structurally impossible to reintroduce.
- New config knob: **`researcher_loop_guard.max_consecutive_blocked_source_calls`**, default
  `3`. `config_autonomous_frag.yml` does not need to set it; the default is the intended value.
- Blocked-call logging drops to `DEBUG` once the ceiling latches, with a single per-worker
  aggregate on completion — the old code emitted the same `WARNING` 116 times for one worker.
- The rejection message now differs by rule: a repeat asks for a *different* query and states
  the remaining budget, rather than reusing the "stop searching" text. Reusing the stop text
  there would have recreated the bug in prompt form.
- Both `researcher.j2` prompts (adaptive and autonomous) now describe the three rules
  separately, so the model is told that a repeat costs it nothing but that turn.
- Forced structured returns are capped at 5 model calls
  (`_MAX_FORCED_RETURN_MODEL_CALLS`), deliberately above
  `StructuredOutputRetryGuardMiddleware`'s 3 so schema-validation failures surface as pydantic
  errors rather than being masked.

**Expected:** ≈−100K tokens/trial on the job average; removes a 6M-token tail risk.
**Not yet measured** — no eval job has run against `bb74e57`. See §8.5 for why the effect may
be larger than the two-trial framing suggests.

### R7 — Widen shallow routing and require the planner otherwise (prompt) — **REFUTED, do not apply as written**

> **Refuted 2026-08-31 by the pooled 14-job re-analysis (§8).** Widening shallow routing
> does not raise F1; the supporting §2.4 evidence was selection bias. Measured
> shallow-minus-deep with question and job fixed effects: **−0.021 F1, CI [−0.049, +0.007]**.
> Paired-bootstrap all-shallow vs all-deep: **−0.053 F1, CI [−0.142, +0.036]**, −800K tokens.
> Retained unedited so the error is legible.
>
> **What survives:** the *token* argument, which is if anything understated — and one real
> routing rule that this recommendation missed entirely, keyed on answer shape rather than
> on staging. The second half of R7 (require `planner-agent` on the non-shallow path) was
> never tested and is not refuted; it is folded into §8.7 as optional.
>
> **Note the specific inversion.** R7 proposes relaxing the "narrows a set in steps… two
> number conditions stacked" exclusion. §8.3 shows stacked-condition questions are exactly
> the ones shallow handles *best*. R7 was directionally right about that clause for the
> wrong reason, and wrong about the target (~75-80% shallow) — see §8.6.


Routing lives entirely in prompt text; the config has no knob for it. Two sites gate it:

- `prompts/orchestrator.j2:48` — *"If one lookup could settle it, hand it over… **This is
  available on your FIRST turn only.**"*
- `factory.py:218` `_SHALLOW_SUBAGENT_DESCRIPTION` — *"DO NOT CHOOSE IT when the request
  narrows a set in steps… 'of those', 'from among these', 'first … then', 'exclude any
  that …', two number conditions stacked, or two different publishers named for two
  different facts."*

That exclusion list is what pushes DSQA-style staged questions into the expensive path, and
§2.4 shows the exclusion is not earning its cost. Relax it so it excludes only requests
whose **deliverable structure** is part of the ask (fixed sections, comparison axes, report
format) — not requests that merely stack conditions. Target ~75–80% shallow, matching the
reference's intent classifier.

Second, remove the "No-Planner-Deep" shape entirely. It is the worst bucket in the job: 20
trials, **50% fully incorrect**, 1.5M tokens each. Compare the reference's deep agent
(planner → research → **writer**): F1 0.630, 55% fully correct, 20% fully wrong. Case 3
(with a planner) beats Case 2.2 by +0.125 F1, so the planner is clearly load-bearing.

Note that **`writer-agent` was never invoked in any of the 90 trials**
(`task_calls: {shallow-researcher: 38, planner-agent: 26}`). The orchestrator always
composes from digested notes itself. Worth testing as a separate arm.

When the orchestrator declines shallow, it should be required to go through
`planner-agent`. There should be no third path where it fans out workers and free-composes.

**Expected:** the largest single token saving. Moving the 20 Case-2.2 trials to shallow
alone saves 28.7M tokens (−319K on the per-trial average) and lifts that bucket from 0.396
toward ~0.49.

---

## 4. Projected combined effect — **SUPERSEDED**

> **Do not use these figures (§8.6).** The 0.647 projection assumes R1 recovers the shallow
> gap (refuted, §7) and that reference-like routing lifts F1 (refuted, §8.2). Measured
> policy simulation over 1,259 trials puts the ceiling for *any* fixed route assignment at
> **0.581**, reached by routing everything deep. The revised expectation is roughly flat F1
> at roughly half the tokens. §8.6 has the policy table.


Adopting reference-like routing (78% shallow) plus R1–R6:

| Bucket | n | Projected F1 | Projected input tokens |
| :-- | --: | --: | --: |
| Shallow | 70 | 0.66 | 85,000 |
| Deep (planner-gated, 1 batch) | 20 | 0.60 | 800,000 |
| **Overall** | **90** | **0.647** | **243,889** |

Against the reference's 0.6078 / 295,080 and the current 0.5563 / 912,133 — better than the
reference on both axes. The shallow figure assumes the R1 width fix recovers most of the
0.082 gap already measured on the 38 shallow-routed questions; the deep figure assumes the
one-batch cap moves that bucket to the level already observed for one-batch trials (0.674)
discounted for harder residual questions.

---

## 5. Suggested rollout — **SUPERSEDED by §8.7**

> Step 1 (R1) is refuted, step 7 (R7) is refuted as written, and step 4's
> `source_call_budgets` change is replaced by a request-wide ceiling — per-worker budgets
> are precisely what the runaway tail escapes (§8.5). Steps 3, 5 and 6 carry over unchanged.


| Step | Change | Surface | Ships with |
| :-- | :-- | :-- | :-- |
| 1 | `max_content_length: 1000 → 10000` | config | arm A |
| 2 | `max_batch_calls: 6 → 1`, `max_total_research_queries: 20 → 6` | config | arm A |
| 3 | `shallow_subagent_max_tool_iterations: 10 → 5`, `max_llm_turns: 20 → 10` | config | arm A |
| 4 | `source_call_budgets: {3, 5, 8}`, `max_high_depth_queries: 1` | config | arm A |
| 5 | Gate Fetch prompt text on `fetch_enabled` | code | arm B |
| 6 | Fix sticky-exhaustion loop + blocked-call ceiling | code | arm B |
| 7 | Widen shallow routing; require planner on the non-shallow path | prompt | arm C |

Arm A is four config lines in `configs/config_autonomous_frag.yml` and needs no code
change, so it is the cheapest way to size R1–R4 against this baseline. Run each arm on
`dsqa90` and compare with `analysis/autonomous_case_breakdown.py` so the case mix stays
visible — the headline mean can move for routing reasons alone.

---

## 6. Reproducing the numbers in this document

```bash
cd ~/smasurekar/ai-q-harbor-evals

# Headline metrics and case split
python3 analysis/autonomous_agent_deep_dive.py  jobs/2026-08-27__10-57-20
python3 analysis/autonomous_case_breakdown.py   jobs/2026-08-27__10-57-20

# Reference arm route split (no case_wise_summary.csv exists for it)
#   route = deep_research_agent present in agent/aiq_events.jsonl FUNCTION_START names
#   metrics = result.json -> verifier_result.rewards / agent_result.n_input_tokens

# Guard trips and blocked-call loops
cd jobs/2026-08-27__10-57-20
cat */agent/aiq-agent-console-stdout.txt \
  | grep -oiE "(blocked|budget reached|Max iterations \([0-9]+\) reached|Forcing synthesis)" \
  | sort | uniq -c | sort -rn
```

Per-trial fields used throughout: `result.json → verifier_result.rewards.{f1_score,
precision, recall, fully_correct, fully_incorrect}`, `result.json →
agent_result.{n_input_tokens, n_output_tokens}`, `agent/aiq_state.json → calls.llm.started`,
and `agent/aiq_events.jsonl` `TOOL_START` / `FUNCTION_START` names. Difficulty labels come
from `datasets/dsqa90/<task>/tests/metadata.json → codex_difficulty.label`.

---

## 7. Follow-up: R1 tested and refuted (job `2026-08-31__06-05-55`)

R1 was applied on its own — `web_search_tool.max_content_length: 1000 → 10000`, everything
else unchanged — and re-run on the same 90 tasks.

### 7.1 Headline

| Metric | 1,000 chars (08-27) | 10,000 chars (08-31) |
| :-- | --: | --: |
| Mean F1 | 0.5563 | **0.5278** |
| Fully correct | 32.22% | 28.89% |
| Avg input tokens | 912,133 | **1,013,864** |
| Avg LLM calls | 50.59 | 39.79 |
| Tokens per LLM call | 18,030 | **25,481** |
| Sub-agent source calls | 2,979 | 2,372 |
| Routed to shallow | 38 (42%) | 52 (58%) |

Accuracy did not improve and cost went up. The predicted +0.05 to +0.08 F1 is ruled out.

### 7.2 The headline F1 drop is itself within noise — but the *absence of gain* is not

Paired per question across the two runs:

```
mean F1 change = -0.0284   sd = 0.373   se = 0.039   95% CI = [-0.106, +0.049]
better on 24, worse on 19, unchanged on 47
mean |change| among the 43 questions that moved = 0.433
```

So "F1 decreased" overstates it — the drop is not distinguishable from run-to-run variance.
But the CI's upper bound (+0.049) sits below the predicted floor (+0.05), so the *predicted
gain* is excluded. The honest reading is: **no measurable accuracy effect, and a real cost
increase.**

### 7.3 Why it did nothing: the config knob does not do what R1 assumed

`sources/tavily_web_search/src/register.py:113` truncates `doc.get("content")` — Tavily's
**snippet**, not the page body. `include_raw_content` is never requested and is not
supported anywhere in `sources/tavily_web_search/`. So `max_content_length` caps what Tavily
already returned; it cannot make Tavily return more.

This is the error in R1. The FDIC and ICILS trials failed because the figure was **never in
the API response at any length**, not because a cap cut it off. Raising the cap could not
have fixed them, and did not:

| Trial | 1,000 chars | 10,000 chars |
| :-- | --: | --: |
| `deepsearchqa-0249` (FDIC dividends) | F1 0.000, 157,287 tok | F1 0.000, **361,387** tok |
| `deepsearchqa-0242` (ICILS Table 2.2) | F1 0.000, 1,197,436 tok | F1 0.000, **2,010,302** tok |

Observed evidence size per search step, on the 32 questions that took the shallow route in
**both** runs:

```
                 median growth   p90 growth   peak context
1,000 chars           1,751         3,087        17,329
10,000 chars          1,892         7,839        33,181
```

The median search barely moved (+8%) — most snippets were already under 1,000 characters,
so the cap was rarely binding. Only the tail grew (p90 up 2.5×). The change therefore had
no effect where accuracy is decided and a large effect on cost. Worst of both.

On that same route-matched subset:

```
              F1      searches   LLM calls   tokens
1,000 chars   0.576     6.12       8.91       95,740
10,000 chars  0.581     8.06      11.81      246,731
paired delta  +0.004                          +2.6x
```

Note the direction of the search count: R1 predicted **fewer** searches once each result
carried more evidence. Searches went **up** (6.12 → 8.06), as did turns. Longer snippets
gave the model more to react to, not less.

### 7.4 Run-to-run variance is large enough to have produced the original R1 signal

Two facts change how the rest of this document should be read:

1. **Per-question F1 has sd ≈ 0.373 across identical-config reruns.** For n=90 that is
   se ≈ 0.039, so overall differences below ~0.08 are not interpretable from single runs.
   For a 38-trial bucket, se ≈ 0.061 — a ±0.12 band.
2. **Routing is unstable: only 59 of 90 questions (66%) took the same path in both runs.**

```
    1-shallow      -> 1-shallow      32       2.2-np-deep  -> 2.2-np-deep    6
    3-planner      -> 3-planner      16       3-planner    -> 1-shallow      6
    2.2-np-deep    -> 1-shallow      13       1-shallow    -> 2.2-np-deep    5
```

The original R1 case rested on the reference arm scoring 0.685 against 0.603 on the 38
shallow-routed questions — a 0.082 gap on n=38, i.e. inside the ±0.12 noise band. It should
never have carried a point prediction. **Case-level tables cannot be compared across runs
at all**, because the case mix is itself a random variable: Case 1 grew from 38 to 52 trials
and absorbed twice as many hard questions, which is most of why its mean fell from 0.603 to
0.554.

### 7.5 What the second run *does* confirm

Two recommendations replicate independently across both jobs.

**R2 (cap batches).** Monotonic in both runs, with cost roughly doubling per step:

```
                  job A (08-27)              job B (08-31)
batches=1    n=18  F1 0.674  0.94M      n=14  F1 0.578  1.25M
batches=2    n=15  F1 0.454  1.45M      n= 6  F1 0.474  2.25M
batches=3+   n=18  F1 0.412  2.20M      n=17  F1 0.410  2.66M
```

**R7 (route to shallow).** Shallow beats deep at matched difficulty in both runs, at
7–13× lower cost:

```
              job A (08-27)                        job B (08-31)
medium   shallow 0.617 @ 98K | deep 0.555 @ 1.48M   shallow 0.600 @ 316K | deep 0.515 @ 2.14M
hard     shallow 0.420 @120K | deep 0.263 @ 2.03M   shallow 0.345 @ 232K | deep 0.176 @ 2.95M
```

The strongest evidence in either job is an accidental natural experiment. Thirteen questions
took the No-Planner-Deep route in run A and the shallow route in run B — same question, same
config except R1, route changed on its own:

```
as No-Planner-Deep :  F1 0.377   1,634,339 tokens
as Shallow         :  F1 0.383     321,269 tokens     (-80% cost, accuracy unchanged)
```

This is a within-question, route-only comparison, and it is exactly R7's claim.

### 7.6 Revised guidance

| Item | Status |
| :-- | :-- |
| R1 (search-result width) | **Refuted.** Revert to `1000`. It costs 2.6× on the shallow route for no gain. |
| R2 (cap batches at 1) | Replicated across both runs, but **the F1 half is selection** (§8.2a). Keep as a cost control only. |
| R7 (route to shallow; require planner otherwise) | **Confirmed** across both runs plus the natural experiment. |
| R3, R4 (cut search budgets) | Untested. Directionally supported, but the R1 result shows "fewer searches" and "better answers" do not follow from each other automatically. |
| R5 (gate Fetch prompt text) | Untested. Correctness fix regardless — the prompts should not name absent tools. |
| R6 (sticky-exhaustion loop) | Untested as an F1 fix. Still a real defect. Blocked source calls fell 252 → 13 between runs, which shows the trigger is workload-dependent, not that the bug is gone. |

**The real gap R1 was aiming at remains open.** Questions whose answer sits in a table,
filing, or PDF need the page body, and this deployment has no way to get it: `fetch_url_tool`
is disabled by design, and `tavily_web_search` cannot request `include_raw_content`. Adding
that option to the Tavily source is the honest replacement for R1 — it would fetch the page
text through the existing search call rather than through a separate page-opening tool. That
is a code change in `sources/tavily_web_search/`, and it should be measured on its own.

### 7.7 Method note for the next run

Single-attempt n=90 cannot resolve the effect sizes in this document. For anything expected
to move F1 by less than ~0.08:

- Set `n_attempts: 3` and compare means, or
- Report the paired per-question delta with its CI rather than two bucket means, and
- Always report the case mix alongside the headline, since routing drift alone moves it.

---

## 8. Pooled re-analysis: routing stability, route effects, and the runaway tail

Everything in §1–§5 rests on single-run comparisons between two arms that each choose their
own routes. That design cannot separate *route* from *the kind of question the agent chooses
to send down that route*. This section replaces it with a within-question design.

### 8.1 Design

**Sample.** Every `dsqa90` job in August 2026 running `configs/config_autonomous_frag.yml`
with `fetch_url_tool` disabled: **14 jobs, 1,259 trials, the same 90 questions**. Jobs where
the agent made any `fetch_url_tool` call were excluded — that check matters, because the two
highest-scoring autonomous jobs on record (0.832 and 0.830) both had Fetch enabled and are
not comparable to a Fetch-disabled target.

**Identification.** Route is chosen by the model, so raw route means are confounded by
question difficulty. Two controls are applied throughout:

- **Question fixed effects** — compare the same question routed differently across runs.
- **Job fixed effects** — absorb prompt/config drift between jobs, which is large (job means
  range 0.447 to 0.701 within this Fetch-disabled set alone).

Estimated by iterative two-way demeaning. `shallow` = a `task(shallow-researcher)` call;
`DEEP` = everything else, matching §2's case scheme.

**Why this matters here.** Most questions have been answered *both* ways across 14 runs, so
the route contrast is identified within question. That is the comparison §1–§5 never made.

### 8.2 Route barely moves F1; it dominates cost

Two-way fixed effects, n = 1,259:

| Route | n | F1 residual | 95% CI | raw F1 | tokens |
| :-- | --: | --: | :-- | --: | --: |
| `shallow` | 385 | −0.021 | [−0.049, +0.007] | 0.550 | 221K |
| `np_shallow` | 349 | +0.002 | [−0.032, +0.037] | 0.611 | 525K |
| `np_deep` | 412 | +0.017 | [−0.010, +0.044] | 0.559 | 1,431K |
| `planner` | 113 | +0.003 | [−0.049, +0.055] | 0.501 | 1,754K |

Every interval crosses zero. Raw shallow (0.550) is *below* raw deep (0.559) — the opposite
of §2.4's claim, before any control is applied at all.

Paired bootstrap over whole policies (4,000 resamples, same resampled questions in both arms):

| Contrast | ΔF1 | 95% CI | Δtokens |
| :-- | --: | :-- | --: |
| all-shallow − all-deep | −0.053 | [−0.142, +0.036] | −800K |
| all-shallow − by-answer-shape | −0.040 | [−0.120, +0.041] | −241K |
| all-deep − by-answer-shape | +0.013 | [−0.078, +0.104] | +555K |

Variance decomposition makes the point most directly:

```
variance in F1     explained by route:   0.6%
variance in TOKENS explained by route:  28.9%
```

**Routing is a cost lever, not an accuracy lever.** Sending everything to shallow costs at
most ~0.05 F1 — plausibly zero — and saves roughly three-quarters of the token budget.

### 8.2a R2 re-tested: the batch gradient is also selection

Trials with at least one `run_research_batch` call, question and job fixed effects applied:

| Batches | n | F1 residual | 95% CI | raw F1 | tokens |
| :-- | --: | --: | :-- | --: | --: |
| 1 | 341 | +0.004 | [−0.023, +0.031] | 0.666 | 623K |
| 2 | 192 | +0.011 | [−0.034, +0.057] | 0.545 | 1,176K |
| 3+ | 206 | −0.017 | [−0.059, +0.025] | 0.433 | 2,180K |

The raw column falls by 0.233 across the range; the controlled column is flat and every
interval crosses zero. The agent sends harder questions round again — batching does not make
answers worse. §3 R2 replicated across both runs (§7.5) for exactly this reason: the
*selection* is stable, not the causal effect.

The token column is a different story: **3.5× from one batch to three**. Cap batches to
control cost, not to raise F1.

### 8.3 The one real routing signal: answer shape, not staging

A per-question router is impossible. Split-half reliability of the per-question route effect
is **r = −0.076**, and the observed spread of per-question effects (sd 0.150) is *smaller*
than sampling noise alone predicts (sd 0.184) — implied true between-question variance is
negative. There is nothing there to route on.

One **group-level** contrast does survive, keyed on the dataset's `answer_type`:

| Answer shape | questions | shallow F1 | deep F1 | shallow − deep |
| :-- | --: | --: | --: | --: |
| **Set Answer** (enumerate/filter) | 59 | **0.646** | 0.605 | **+0.053** |
| **Single Answer** (pick one winner) | 31 | **0.325** | 0.515 | **−0.169** |

- Interaction **+0.222**, permutation test **p = 0.0037** (3,000 shuffles of the answer-shape
  label across the 90 questions).
- Replicates across chronological halves of the job history: era 1 gives +0.025 / −0.158,
  era 2 gives +0.074 / −0.176.
- Concentrated in `medium` difficulty (Single Answer −0.224, CI ±0.118). `hard` shows nothing,
  because `hard` is bad on every route.

**The mechanism is scoring geometry, not capability.** F1 distribution by shape and route:

| | %F1 = 0 | %F1 = 1 | %partial |
| :-- | --: | --: | --: |
| Single Answer / shallow | 67.0 | 31.3 | **1.7** |
| Single Answer / deep | 45.5 | 46.7 | 7.8 |
| Set Answer / shallow | 18.1 | 33.3 | **48.5** |
| Set Answer / deep | 22.0 | 31.7 | 46.3 |

Single-answer questions are all-or-nothing: 98% of trials score exactly 0 or exactly 1. To
score at all, the agent needs a value for *every* candidate on the same definition, year and
source kind; a bounded pass that runs out of budget publishes a confident partial answer and
takes the zero. Set-answer questions are graded, so shallow's partial coverage still earns
credit — and deep's extra ~30 searches buy nothing there (0.605 against shallow's 0.646).

### 8.4 The current shallow-routing rule points the wrong way

`factory.py:218` `_SHALLOW_SUBAGENT_DESCRIPTION`:

> *"DO NOT CHOOSE IT when the request narrows a set in steps… 'of those', 'from among
> these', 'first … then', 'exclude any that …', **two number conditions stacked**…"*

Stacked numeric conditions are the defining feature of the **Set Answer** questions — the
ones shallow handles best (§8.3). Meanwhile `WHEN TO CHOOSE IT`'s *"the request asks for one
thing, once"* reads as a match for *"which company had the greatest reduction in operating
expenses?"*, which is a Single Answer question where shallow scores **0.325**.

The routing data confirms the rule is not discriminating in the useful direction:

```
shallow-rate on Set Answer questions:    32.7%   (n=825)
shallow-rate on Single Answer questions: 26.5%   (n=434)
```

A rule that worked would show a wide gap here. This is a 6-point gap, in a policy whose
overall shallow rate is ~31%.

**Replacement test.** Key on the shape of the *answer*, not on staging:

- **Choose shallow** when the request asks for **every member meeting stated conditions** —
  enumerate/filter, however many conditions are stacked.
- **Do not choose shallow** when the request asks you to **select one winner by comparison
  across candidates** (most / greatest / highest / largest / fewest / best over a group).
  Every candidate needs a value before the question can be answered at all, and a partial
  answer scores zero.

A text-only heuristic (grammatical number of the interrogative target) recovers the
Single-Answer class at 0.97 recall but only 0.56 precision, and still reproduces the
contrast — singular target −0.085, plural target +0.032. Precision 0.56 is the regex's
limit, not the signal's; an LLM asked the binary question directly should do better. That is
why this belongs in the routing prompt rather than in a hand-written classifier.

### 8.5 The token lever that dwarfs routing: no request-wide search ceiling

```
trials with >= 61 searches:   135 / 1,259  =  10.7% of trials
                              36.7% of ALL input tokens
                              F1 0.297   vs 0.598 for everything else
                              2,869K tokens/trial vs 595K
```

Within question and job, F1 is flat from 0–60 searches and significantly negative past 61:

| Searches | n | F1 residual | 95% CI | raw F1 | tokens |
| :-- | --: | --: | :-- | --: | --: |
| 0–4 | 289 | +0.009 | [−0.022, +0.040] | 0.720 | 102K |
| 5–12 | 380 | −0.019 | [−0.048, +0.010] | 0.567 | 289K |
| 13–35 | 304 | +0.020 | [−0.015, +0.056] | 0.568 | 877K |
| 36–60 | 151 | +0.035 | [−0.015, +0.086] | 0.501 | 1,741K |
| **61+** | **135** | **−0.051** | **[−0.101, −0.001]** | **0.297** | **2,869K** |

**Cause: there is no request-wide source-call ceiling.** `request_termination` caps research
queries (20), batch calls (6), the orchestrator's own direct calls (8) and model turns (100)
— but each delegated worker gets a *fresh* `researcher_loop_guard.source_call_budgets`
allowance (`high: 20`). Twenty queries × twenty calls is 400 searches, and nothing in the
envelope stops it.

| | p50 | p75 | p90 | p95 | max |
| :-- | --: | --: | --: | --: | --: |
| Autonomous (Fetch off) | 11 | 32 | 62 | 91 | **3,147** |
| Reference arm | 5 | 5 | 41 | 57 | 131 |

**These are real searches, not blocked retries.** Guard-block counts across the two most
recent jobs are 252 + 69 (job A) and 13 + 38 (job B); only two trials show the sticky-guard
loop (131 and 122 blocks). The tail is legitimate unbounded searching, so R6 does not fix it
and a ceiling is required.

Tail composition: `np_deep` 84, `planner` 30, `np_shallow` 17, `shallow` 4 — with 8.3
research queries dispatched on average against 2.3 for the rest, and **43.8 searches per
research query** against 6.6. The per-worker budget of 20 is being exceeded threefold.

> **Update: R6 shipped in `bb74e57`.** That 43.8-against-20 overrun is exactly what the new
> consecutive-blocked-call ceiling is built to stop, and it is measured over the whole
> Fetch-disabled history — a far wider footprint than the two catastrophic trials R6 was
> written from. **The tail must therefore be re-measured before N1's ceiling is sized.** If
> `bb74e57` alone pulls per-worker calls back toward the configured 20, the request-wide
> ceiling of ~45 may be conservative, or in the limit unnecessary. Run a `dsqa90` job on
> `bb74e57` with no other change and re-run §8.5 before shipping N1.

### 8.6 Policy simulation: the F1 target is not reachable through routing

Each policy scored by drawing observed trials for the route it assigns, bootstrapped over
questions:

| Policy | F1 | tokens | % shallow |
| :-- | --: | --: | --: |
| all-deep | **0.581** | 1,043K | 0% |
| by-answer-shape (§8.3) | 0.567 | **485K** | 66% |
| current (job `2026-08-27`) | 0.556 | 912K | 42% |
| all-shallow | 0.528 | 243K | 100% |
| **Reference (chat arm)** | **0.6078** | **295K** | — |

Two conclusions:

1. **The current policy is dominated.** By-answer-shape gets +0.011 F1 at −47% tokens.
2. **No route assignment reaches 0.608.** The ceiling over all fixed policies is 0.581, and
   it is the expensive one. Routing delivers the token target; it cannot deliver the F1
   target.

> An "oracle per-question" policy scores 0.638, but it selects the winning route using the
> same data it is scored on. With split-half reliability r ≈ 0 (§8.3) that number is pure
> overfitting and is **not achievable**. It is recorded here only so nobody re-derives it.

**Where the F1 gap actually lives.** The reference reaches 0.608 from **11.9 searches and
21.7 LLM calls**; the autonomous arm reaches 0.556 from **33.6 searches and 50.6 LLM calls**
— three times the work for a worse answer, with zero page opens on both sides. The reference
also bounds its shallow agent at `max_tool_iterations: 5` / `max_llm_turns: 10`, against our
`10` / `20`. The gap is in evidence-to-answer conversion, not in retrieval volume and not in
routing.

### 8.7 Revised recommendations, ranked

Replaces §5. Ranked by expected value = (evidence strength) × (effect size) ÷ (cost + risk).
Each item states what is **measured** and what is only **predicted**.

| # | Change | Surface | Evidence | Effect | Ship |
| --: | :-- | :-- | :-- | :-- | :-- |
| **0** | Revert `max_content_length` → `1000` | config, 1 line | Measured (§7) | −2.6× tokens on the shallow route, no F1 cost | now |
| **1** | **N1** request-wide source-call ceiling (~45) | config | Measured tail; **F1 effect unmeasured** | **−29% job tokens** | arm A |
| **2** | **N3** separate low-temperature router | config | Measured instability (3/90 stable) | enables #3; no direct F1 | arm A |
| **3** | **N2** re-key shallow test on answer shape | prompt | Measured, p=0.0037, replicated | +0.011 F1, **−47% tokens** | arm B |
| **4** | **N4** shallow caps → reference bounds 5/10 | config, 2 lines | Known-good on the 0.608 arm | untested here | arm A |
| **5** | **R5** gate Fetch prompt text | code | Untested | removes a dead-end instruction | arm C |
| **6** | **R2** cap batches | config | Cost only (§8.2a) | −3.5× tokens at 1 vs 3 | arm B |
| ~~—~~ | ~~**R6** sticky-exhaustion loop~~ | code | **DONE** — `bb74e57` | pending measurement | shipped |

**Ranking rationale.** #1 is first because it is the largest measured quantity in the whole
analysis and needs no code. #2 outranks #3 despite having no direct effect of its own: a
routing rule that the model applies on 31% of eligible questions is not a rule, so #3's value
is gated on it. #4 is cheap and borrowed from the arm that actually hits the target. #5–#6
are real but small.

**Sequencing changed by `bb74e57`.** R6 shipped before any of the above, and it targets the
same per-worker overrun that produces the §8.5 tail. **Run a `dsqa90` job on `bb74e57` with no
other change first**, and re-derive §8.5 from it. That job is both the measurement R6 still
owes and the correct baseline for sizing N1 — sizing a request-wide ceiling against
pre-`bb74e57` data risks setting it against a tail that no longer exists.

**What none of this does.** Items 0–7 are a *cost* programme. They hold F1 roughly flat while
roughly halving spend. The F1 gap to 0.608 is not addressed by any of them, because §8.6 shows
no route assignment reaches it. That is a separate investigation (§8.6, final paragraph).

**N1 — Add a request-wide source-call ceiling (~45). Config.**
Removes a 10.7% tail that consumes 36.7% of tokens at half the F1 of everything else.
*Measured:* the tail's size, cost and F1, and that F1 declines past 61 searches.
*Predicted:* −29% job tokens. **The F1 effect cannot be predicted from observational data** —
a capped run is not the same as a run that stopped on its own. It must be measured. The
downside is bounded by the fact that these runs already score 0.297.

**N2 — Re-key the shallow routing test on answer shape. Prompt only (`factory.py:218`).**
Enumerate-all → shallow; pick-one-winner → deep. *Measured:* the +0.222 interaction,
p = 0.0037, replicated across both job eras. *Predicted:* +0.011 F1 and −47% tokens against
job `2026-08-27`.

**N3 — Make the route decision separate and low-temperature.**
`config_autonomous_frag.yml:58` runs the router at `temperature: 0.7 / top_p: 0.7` as one
option among five `task` targets. The reference gives routing its own `intent_llm` at
`temperature: 0.5` (`shallow_deep_nemotron_ultra.yml:44`). **N2 cannot hold without this** —
see §8.8. *Predicted:* routing stability, not F1.

**N4 — Cut the shallow sub-run to the reference bounds.**
`shallow_subagent_max_tool_iterations: 10 → 5`, `shallow_subagent_max_llm_turns: 20 → 10`.
This is R4 unchanged, now additionally motivated as a known-good setting on the arm that
reaches 0.608.

**Done:** R6 (sticky-exhaustion loop) shipped in `bb74e57` — repeated-signature trips no
longer exhaust a worker, and a `max_consecutive_blocked_source_calls` ceiling (default 3)
forces the structured return when tool withdrawal is replayed around. Effect not yet measured.

**Carried over:** R5 (gate Fetch prompt text) stands, untested.

**Also:** revert `max_content_length` to `1000`. It is still at `10000` from the R1 test and
is costing 2.6× on the shallow route for no measured gain (§7).

**Explicitly not recommended:** widening shallow routing to a 75–80% target (R7), or any
routing change justified by expected F1 gain.

### 8.8 Why routing is unstable — and why it is not a comprehension failure

The router *is* reading the question. Split-half reliability of P(shallow | question) across
the 14 jobs is **r = +0.577**, and the spread of P(shallow) across questions is well above
binomial noise (observed variance 0.0472 against 0.0118 expected).

What it never does is **act** on that preference:

```
P(shallow) per question:  mean 0.31,  question-driven sd 0.188,  max over all 90 questions 0.71
questions routed the same way in >= 80% of runs:      3 / 90
questions with no majority route at all:             31 / 90
questions in the 0.2-0.8 coin-flip zone:             59 / 90
mean modal-route share:                              0.530
```

There is not one question in the dataset the agent reliably sends to shallow.

The diagnosis is precise: **the routing signal exists but is sampled rather than
thresholded.** A stable preference of "0.31 in favour of shallow" becomes a coin flip at
`temperature: 0.7`. This is why N2 cannot ship alone — a prompt rule the model applies on
31% of eligible questions is not a rule, and it is also why every case-level table in §2 is
uninterpretable across runs: the case mix is itself a random variable.

`factory.py:210` states the design directly — *"in this agent descriptions are the routing
logic"*. There is no classifier and no threshold. Making the first-turn route a separate
low-temperature binary classification, rather than a free choice among five `task` targets,
is the structural fix.

### 8.9 Reproducing §8

```bash
cd ai-q-harbor-evals

# 1. Pool every dsqa90 job, tagging config file and Fetch usage per trial.
#    analyse_trial() supplies token split, tool counts, task_calls and batch shape.
python3 - <<'PY'
import json, sys, glob
from pathlib import Path
sys.path.insert(0, 'analysis')
from autonomous_agent_deep_dive import analyse_trial
# jobs with >=80 trials and source == 'dsqa90'; case derived exactly as in
# analysis/autonomous_case_breakdown.py (shallow > planner > nres>2 > else)
PY

# 2. Restrict to Fetch-disabled autonomous jobs:
#    keep jobs where mean(fetch_url_tool calls per trial) == 0  -> 14 jobs, 1,259 trials

# 3. Two-way fixed effects: iterative demeaning of f1 on (qid, job), 60 passes.

# 4. Answer shape comes from the dataset card, not from the run:
#    datasets/dsqa90/<id>/tests/metadata.json -> answer_type  ("Set Answer" | "Single Answer")
#    difficulty likewise -> codex_difficulty.label

# 5. Guard blocks are logged to agent/aiq-agent-stdout.txt (NOT stderr, which is empty):
grep -oE 'reason=[a-z_]+' jobs/<job>/*/agent/aiq-agent-stdout.txt | sort | uniq -c
```

Significance claims in §8 use: paired bootstrap over questions (policy contrasts), a
permutation test shuffling the answer-shape label across the 90 questions (the §8.3
interaction), and split-half reliability over repeated runs of the same question (§8.3
heterogeneity, §8.8 router signal).

### 8.10 Method note, superseding §7.7

- Per-question F1 sd across identical-config reruns is **0.373**, so a single 90-question run
  resolves nothing below ~0.08. Run `n_attempts: 3`.
- **Never compare case-level tables across runs.** Only 3 of 90 questions hold their route in
  ≥80% of runs; the case mix is a random variable (§8.8).
- Report **paired per-question deltas with CIs**, never two independent means.
- When pooling historical jobs, **check Fetch usage per job first**. The two best autonomous
  jobs on record had Fetch enabled and would silently poison a Fetch-disabled comparison.
- Prefer within-question contrasts over difficulty-matched ones. Difficulty matching is what
  produced the refuted R7.
