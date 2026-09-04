# Autonomous researcher: unbounded `ResearchNotes` structured-output retry loop

**Status:** root cause identified and confirmed against captured payloads. The retry guard and the
diagnostics that closed the unknown are implemented and verified in two rerun jobs (§4.1-§4.4).
Still open: the schema/serialization mismatch that triggers the rejections in the first place
(§4.5), the workflow deadline (§4.6), and confirming whether `48626d1` raised the trigger rate
(§4.7).

**The defect, in one line:** the model emits a field typed as a *single nested Pydantic model* as a
**JSON-encoded string** rather than as an object, pydantic rejects it, and — before §4.1 —
`ToolStrategy(handle_errors=True)` retried that rejection without any attempt cap until the harness
killed the trial.

**Originally observed in:** eval job `2026-08-21__11-12-50`, dataset `deepsearchqa-smoke`,
config `configs/config_autonomous_frag.yml`, model `nvidia/nvidia/nemotron-3-ultra`.
2 of 5 trials died with `AgentTimeoutError`; all 5 trials showed the defect.

### The three jobs this document rests on

All three ran the same 5 `deepsearchqa-smoke` tasks against the same config and model.

| Job | Guard (§4.1) | `AIQ_LOG_PAYLOADS` | Worst identical-payload streak | Timeouts | `grader_valid` | `reward` |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-08-21__11-12-50` | no | off | 5, 115, 3, 152, 6 | **2** | 0.6 | 0.4 |
| `2026-08-21__13-59-47` | yes | off | 2, 1, 3, 0, 1 | 0 | 1.0 | 0.2 |
| `2026-08-21__14-11-50` | yes | **on** | 2, 2, 3, 3, 2 | 0 | 1.0 | 0.27 |

Read this table for the timeout column, not the reward column. The guard converts a hung trial into
a completed one, which is why `grader_valid` goes to 1.0 and stays there. `reward` moves within
noise: one trial per task, no repeats, n=5 — 0.4 → 0.2 → 0.27 is two tasks flipping, and nothing
in the artifacts supports reading a quality trend out of it. What the reward column *does* show is
that removing the hang did not by itself recover the lost research, because the abandoned sub-runs
still cost coverage (§3.5).

## Path conventions used in this document

| Prefix | Absolute root |
| --- | --- |
| `AIQ/` | `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar` (this repo) |
| `EVALS/` | `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals` |

From this file's directory, `EVALS/` is `../../../../gitlab_repos/ai-q-harbor-evals`.
Copies of the decisive artifacts are checked in under
`artifacts/structured-output-retry-loop/` so this analysis stays readable after the eval job
directories are rotated.

---

## 1. The problem

### 1.1 Symptom

A `run_research_batch` worker finishes its research in ~75 seconds, then emits the same
`ResearchNotes` structured-output tool call over and over until the harness kills the trial 28
minutes later.

```
05:51:21  TOOL_START  run_research_batch          <- never gets a TOOL_END
05:52:36  Researcher source-call budget reached | depth=medium tool=fetch_url_tool calls=10/10
05:52:54  → ResearchNotes  Args: chars=13731 ref=sha256:8ae1ed823ece   prompt=26192  completion=3430
05:53:10  → ResearchNotes  Args: chars=13731 ref=sha256:8ae1ed823ece   prompt=28266  completion=3430
   ...    (113 more, byte-identical)
06:20:59  → ResearchNotes  Args: chars=13731 ref=sha256:8ae1ed823ece   prompt=121193 completion=3430
06:21:07  AgentTimeoutError: Agent execution timed out after 1800.0 seconds
```

- `ResearchNotes` never appears as a `TOOL_START` in `aiq_events.jsonl` — it is the
  `response_format` pseudo-tool, not a real tool, so no tool-level guard can see it.
- Prompt grows by exactly **+3552 tokens per round** while `completion` stays at **3430**. The
  ~122-token remainder is the error `ToolMessage` being appended each round.
- The payload digest is identical on **every** attempt, starting from the *first* retry.
- Cost: 10.5M prompt tokens on `deepsearchqa-0002`, 15.9M on `deepsearchqa-0004`, for zero output.
- No `/workspace/answer.txt` is ever written, so the trial errors instead of being graded:
  `grader_valid` drops to 0.6 and job `reward` to 0.4.

### 1.2 Root cause: three conditions, all of which must hold

**(a) The retry primitive has no counter.**
`ResearchNotes` is handed to `create_agent` as a bare Pydantic class, so LangChain wraps it in
`ToolStrategy(schema, handle_errors=True)` — the default.

- `AIQ/src/aiq_agent/agents/deep_researcher/factory.py:403` — `response_format=ResearchNotes`
  in `build_researcher_runnable`. **This is the site that loops**: the autonomous agent's batch
  workers are built here (`AIQ/src/aiq_agent/agents/autonomous_researcher/factory.py:787`).
- `AIQ/src/aiq_agent/agents/autonomous_researcher/factory.py:578` — same binding for the
  `task`-reachable `researcher-agent` sub-agent. Same exposure.

On a validation failure, `langchain/agents/factory.py:1249` appends
`ToolMessage("Error: {error}\n Please fix your mistakes.")` and re-invokes the model —
unbounded, by design (`handle_errors: True` → "Catch all errors with default error template",
`langchain/agents/structured_output.py:210-227`). langchain 1.3.11.

No repo-side guard covers it either. `ResearcherLoopGuardMiddleware` counts source-tool calls,
`ConsecutiveThinkGuardMiddleware` counts `think` calls; both hook `awrap_tool_call`, and a
structured-output pseudo-tool never reaches the tool node.

**(b) The retry carries no information the model can act on.**
The error names the failing field, but the system prompt, the research request and the gathered
evidence are byte-identical between attempts. The only thing that changes is that the model's own
rejected output is now in its context — one more copy per round.

**(c) The model is not reasoning on the retries.**
`[Reasoning] chars=1 ref=sha256:01ba4719c80b` is `sha256("\n")` — the reasoning block is a bare
newline. That is **138 of 141 completions** in `deepsearchqa-0002`; `enable_thinking: true`
(`AIQ/configs/config_autonomous_frag.yml:63`) produced real reasoning exactly 3 times, all during
the search phase. On every retry the model emits a 3430-token JSON with zero deliberation and
reproduces it byte-for-byte, at `temperature: 0.7`. It is not reconsidering; it is replaying a
finished answer.

### 1.3 Why nothing stopped it

| Guard | Why it missed |
| --- | --- |
| `researcher_loop_guard` (`config_autonomous_frag.yml:147`) | Counts source-tool calls. Fired correctly (`calls=10/10`) and is irrelevant here. |
| `recursion_limit: 250` (`config_autonomous_frag.yml:195`) | Would eventually fire, but at ~15 s/step that is ~60 min — well past the harness kill. |
| `workflow_timeout_seconds: 2400` (`config_autonomous_frag.yml:193`) | **Above** the harness `timeout_sec = 1800.0` (`EVALS/datasets/deepsearchqa-smoke/deepsearchqa-0002/task.toml:38`). The agent's own deadline and its `_build_partial_result` fallback (`AIQ/src/aiq_agent/agents/autonomous_researcher/agent.py:584-596`) can never run. This is why the trial produced no artifact at all instead of a partial answer. |
| `StructuredResponseTextFallbackMiddleware` (`AIQ/src/aiq_agent/agents/deep_researcher/custom_middleware.py:107`) | Recovers the case where the model returns JSON *as text instead of* a tool call. Here the model does call the tool; the args just fail validation. Different failure. |

---

## 2. Is commit `48626d1` ("Add a query-driven answer contract to autonomous-researcher exits") responsible?

**Two separate questions. The answers differ.**

### 2.1 Did it create the loop? No.

The retry loop is a property of `ToolStrategy(handle_errors=True)` and predates the commit. Longest
run of byte-identical `ResearchNotes` payloads per trial, across the 90-task jobs:

| Job | Trials | Trials with a run ≥ 2 | Longest run |
| --- | --- | --- | --- |
| `2026-08-20__12-58-09` | 90 | 1 | 4 |
| `2026-08-20__16-47-37` | 90 | 2 | 3 |
| `2026-08-20__21-44-00` | 90 | 2 | 4 |

So the mechanism existed, fired rarely, and self-corrected by the 3rd or 4th attempt — costing
tokens rather than the run. The two timeouts in those jobs were slow-work timeouts (e.g.
`2026-08-20__12-58-09/deepsearchqa-0755`: 564 real `web_search_tool` calls), not this.

**Measurement correction.** An earlier revision of `collect-evidence.sh` reported
`sort | uniq -c | head -1` — the *total* occurrences of the most common digest — and labelled it a
streak. That overcounts whenever a payload recurs after an interruption, and it also counted every
trial that emitted any `ResearchNotes` at all as "a trial with a repeat", which is how this table
previously read 64 / 63 / 55. Both §6 and §8 of the script now compute the longest consecutive run;
the figures above and everywhere else in this document are the corrected ones.

### 2.2 Did it make the loop far more likely to fire? Probably yes — it is the prime suspect.

Same 5 smoke tasks, same `config_autonomous_frag.yml`, before and after:

| Job | Worst identical streak per trial |
| --- | --- |
| `2026-08-19__17-06-25` | 1, 1, 1, 0, 1 |
| `2026-08-20__21-25-36` (after `0099c80`, before `48626d1`) | 1, 1, 1, 1, 1 |
| `2026-08-21__11-12-50` (with `48626d1`) | **5, 115, 3, 152, 6** |

0 of 5 trials retried before; 5 of 5 after, two of them fatally. `48626d1` is the only committed
change in that window (`0099c80` landed 08-20 21:09, before the 21-25-36 run).

**Narrowing which part of the commit is in play.** The commit touches four files:

| File | Reached by the failing trials? |
| --- | --- |
| `AIQ/src/aiq_agent/agents/autonomous_researcher/prompts/orchestrator.j2` | **Yes** — it is the orchestrator system prompt, present from turn 0 of every trial. |
| `AIQ/src/aiq_agent/agents/autonomous_researcher/prompts/writer.j2` | No — neither failing trial delegated to `writer-agent`. |
| `AIQ/src/aiq_agent/agents/autonomous_researcher/subagents/shallow.py` | No — neither failing trial called `task`, so no shallow sub-run. |
| `AIQ/src/aiq_agent/agents/shallow_researcher/agent.py` | No — same reason. |

Tool-call census for the two failing trials confirms it: `deepsearchqa-0002` used only
`run_research_batch`, `web_search_tool`, `fetch_url_tool`, `ResearchNotes`; `deepsearchqa-0004`
adds `advanced_web_search_tool` and `think`. No `task`, no `submit_final_report`.

So **`orchestrator.j2` is the only edited surface the failing trials executed.** Note that the
researcher worker's own prompt (`autonomous_researcher/prompts/researcher.j2`, last changed by
`0099c80`) and the `ResearchNotes` schema (`deep_researcher/models/subagent_contracts.py`, last
changed 08-13) are both untouched by `48626d1`.

**Hypothesised mechanism — since refuted.** The original guess was that the commit's new
`## Answer` / `### Considered and excluded` structural vocabulary propagated from the
orchestrator's system prompt into the `ResearchQuery` it authors (`format_research_request`,
`deep_researcher/tools/research.py:48`), and that a worker mirroring a *section structure* into a
schema with `extra="forbid"` would fail permanently. The payload run disproves this: the rejections
are stringified nested objects, not invented keys, and `extra_forbidden` never fires on
`ResearchNotes` at all (§3.4, §3.7).

**What that leaves.** The A/B above is unexplained, not explained away. `48626d1` still correlates
with a jump from ~0 rejections across 5 trials to 5-of-5 trials rejecting, and it is still the only
committed change in the window — but there is now no proposed causal path from it to a
double-encoded `evidence_judgment`. `orchestrator.j2` is the orchestrator's prompt; the field that
fails belongs to the *researcher worker's* schema, and neither `researcher.j2` nor
`subagent_contracts.py` was touched by the commit. Longer or differently-shaped queries changing the
worker's output style is a possible but entirely unevidenced link. §4.7 is the experiment that
settles it; treat §2.2 as an open correlation until then.

**Correction to an earlier reading of this incident:** an initial pass treated the repeated
`Agent input: chars=437` line as the retry error message and concluded the commit was fully ruled
out. `current_input` is set only in `on_chain_start` and is sticky
(`AIQ/src/aiq_agent/common/callbacks.py:147`), so 437 is the worker's original research request,
not the error. The token-delta evidence in §1.1 is what actually establishes the retry
`ToolMessage`, and the A/B in §2.2 is what puts the commit back on the suspect list.

---

## 3. The diagnosis

**The open question was which field fails validation.** Nothing in the original job artifacts could
answer it: every tool-call argument was recorded as `chars=<n> ref=sha256:<digest>` by
`log_content_metadata` (`AIQ/src/aiq_agent/common/logging_utils.py`), and the pydantic error itself
was written only into the message history, never to a logger.

Both gaps are now instrumented (§4.1, §4.4), and two rerun jobs have answered the question.

### 3.1 What the guard logs, with no flags set

`StructuredOutputRetryGuardMiddleware` writes the pydantic error to `WARNING` on every rejection.
Job `2026-08-21__13-59-47`, `AIQ_LOG_PAYLOADS` unset:

```
WARNING aiq_agent.agents.deep_researcher.custom_middleware:268 -
  ResearchNotes failed schema validation on attempt 1/3: Error: Failed to parse structured output
  for tool 'ResearchNotes': Failed to parse data to ResearchNotes: 1 validation error for ResearchNotes
evidence_judgment
  Input should be a valid dictionary or instance of EvidenceJudgment
  [type=model_type, input_value=<redacted>, input_type=str]
 Please fix your mistakes. | rejected arguments: chars=4489 ref=sha256:f10e92f51084
```

The field path and the error type are logged unconditionally, because they are what identify the
defect. `input_type=str` is already decisive here: the model sent a **string** where a nested model
was expected. Pydantic echoes the offending value in `input_value=`; that is model-generated content
which can carry retrieved customer data, so it is redacted unless `AIQ_LOG_PAYLOADS` is set (§4.4).

### 3.2 What the payload run shows

Job `2026-08-21__14-11-50`, same tasks, `AIQ_LOG_PAYLOADS=1`. The same line now carries the value:

```json
"evidence_judgment": "{\"relevance_score\": 95, \"confidence\": \"high\", \"rationale\": \"Comprehensive coverage of all 38 OECD countries…\"}"
```

The content is correct. `relevance_score` is in `[0, 100]`, `confidence` is a valid `Literal`
member, `rationale` is present and substantive. **The only defect is one extra layer of JSON
encoding.**

The contrast inside the very same payload is what makes the cause specific — every
list-of-objects field is emitted correctly:

```
   ResearchNotes.evidence_judgment was emitted as a JSON *string*
   sibling list-of-object fields in the same payload:
     findings:        list[7 objects]   <- correctly typed
     gaps:            list[2 objects]   <- correctly typed
     sources:         list[3 objects]   <- correctly typed
```

So the model handles arrays of objects fine and stringifies the bare object. The failure tracks
**scalar nested-model fields**, not nesting in general.

### 3.3 Every stringified field is a scalar nested-model field

Walking `model_fields` on every `_StrictContract` subclass inside the runtime image — the
authoritative view, since it reflects what was actually running — turns up exactly **three** fields
typed as a single nested `BaseModel`:

| Field | Type | Required | Observed stringified |
| --- | --- | --- | --- |
| `ResearchNotes.evidence_judgment` (`subagent_contracts.py:212`) | `EvidenceJudgment \| None` | no | yes — 67 times |
| `ResearchPlan.answer_strategy` (`subagent_contracts.py:147`) | `AnswerStrategy` | yes | yes — twice |
| `ResearchPlan.task_analysis` (`subagent_contracts.py:146`) | `TaskAnalysis` | yes | no — only ever seen `missing`, in the relocation case in §3.4 |

Every other nested field on every contract is a `list[...]` — `ResearchNotes.findings`, `.gaps`,
`.sources`, `ResearchPlan.constraints`, `.queries`, `AnswerStrategy.required_components`,
`SourceRoutingPlan.recommendations`, `.fallback_sources`. **Not one of them has been observed
stringified, in 74 rejections.** Every stringified field is a scalar nested model; two of the three
scalar nested models have been caught stringified, and the third has only ever failed in a
different way. The containment holds in the direction that matters for the fix.

Note that `answer_strategy` is **required** and carries no `| None`. An earlier reading of this
incident attributed the defect to the `| None` union rendering as `anyOf` in the emitted tool
schema; `answer_strategy` refutes that. The common factor is the scalar-nested-object shape itself,
not optionality.

Contributing factor, unchanged: `ChatNVIDIA` is not a `BaseChatOpenAI` subclass, so LangChain skips
the `strict=True` tool-schema enforcement it applies to OpenAI-compatible clients
(`langchain/agents/factory.py:576-594`). Nothing constrains the arguments server-side. The
`langchain_nvidia_ai_endpoints` warning `Model 'nvidia/nvidia/nemotron-3-ultra' is not known to
support tools`, repeated throughout `aiq-agent-console-stderr.txt`, is the same signal.

### 3.4 Census across both rerun jobs

74 rejections over 10 trials. Payloads were captured for the 44 in `2026-08-21__14-11-50`.
Classified by rejection, not by field, since one rejection can carry several field errors:

| Shape | Rejections | Detail |
| --- | --- | --- |
| A scalar nested model arrived as a JSON string, and nothing else was wrong | **67** | 66 `evidence_judgment`, 1 `answer_strategy` |
| The same, mixed with a second error | 2 | see below |
| `queries[1].subqueries` exceeds `max_length=8` (`subagent_contracts.py:114-116`) | 3 | one trial |
| All four `AdaptiveResearchPlan` fields reported `missing` | 1 | one trial |
| `constraints` / `queries` nested inside `answer_strategy` (`extra_forbidden`) and absent at top level | 1 | one trial |

**93% of all rejections are the one defect** (69 of 74 field-level `model_type` errors on a scalar
nested field: 67 `evidence_judgment`, 2 `answer_strategy`). Of the 40 stringified values captured
with payloads, **39 re-parse cleanly into a valid dict** — the model's JSON is well-formed, just
double-encoded.

The residual rows break down as follows. One mixed rejection is on `2026-08-21__13-59-47`
`deepsearchqa-0001__gjuremo` — `findings` sent as a non-list alongside a stringified
`evidence_judgment`. The all-four-fields-`missing` rejection is on that job's
`deepsearchqa-0005__VmtFoh3`. The other mixed rejection and the relocation rejection are two
consecutive attempts by the same trial, `2026-08-21__14-11-50/deepsearchqa-0001__AN3ukZa`, and
together they tell a coherent story about `AdaptiveResearchPlan`:

```
attempt 1: answer_strategy      model_type       (a string containing the whole plan)
           constraints, queries missing
attempt 2: answer_strategy.constraints, answer_strategy.queries  extra_forbidden
           constraints, queries                                  missing
```

On attempt 2 the model dropped the string encoding but kept the nesting — it believes `constraints`
and `queries` belong *inside* `answer_strategy`. That is why the attempt-1 string breaks at char
5137 and ends with `"depth": "high"}]`, the tail of `queries` rather than of `answer_strategy`, and
it is the one captured value that does not re-parse.

### 3.5 The cost that remains after the guard

The guard removed the hang and exposed what the hang was hiding. On
`2026-08-21__13-59-47/deepsearchqa-0003__NNxWvxC`, 8 rejections hit the cap and the trial answered:

> I could not verify the answer to this question. The research budget was exhausted before I could
> identify […] Only one source was captured during research […]

Three separate guards produced that outcome, and only one of them is new:

| Guard | Where | Effect on this trial |
| --- | --- | --- |
| Researcher source-call budget | `adaptive_researcher/custom_middleware.py:1215-1224`, `config_autonomous_frag.yml:149-152` (`low: 5`) | Both workers were dispatched at `depth=low` and hit `calls=5/5`. Appends a nudge, sets `exhausted=True`; does not kill the worker. Pre-existing. |
| `StructuredOutputRetryGuardMiddleware` | §4.1 | Killed worker `7d441f2d` after 3 `evidence_judgment` rejections → `Researcher worker … failed`. **New.** |
| Autonomous loop guard | `autonomous_researcher/custom_middleware.py:880` | Refused re-research three times: `reason=duplicate_query`, then `already_finalizing` ×2. Pre-existing. |

One of two workers was thrown away over a serialization detail, the surviving one had already spent
its source budget on a single source, and the loop guard correctly declined to start over. The
writer then had one source and said so. This is the right failure mode and still the wrong outcome,
which is why §4.5 — not §4.1 — is the fix that matters now.

**This is the dominant failure mode across the reruns, not a one-off.** Grading `2026-08-21__14-11-50`
by what the writer actually produced:

| Trial | `reward` | Cap hits | Verifier's explanation |
| --- | --- | --- | --- |
| `deepsearchqa-0003__UXB2VjA` | 1.0 | 5 | correct |
| `deepsearchqa-0002__YCdB2MT` | 0.33 | 1 | 1 of 3 set items; the other two invented, cited to `instagram.com/popular/…` URLs |
| `deepsearchqa-0001__AN3ukZa` | 0.0 | 5 | "could not verify the Organised Crime Index scores … due to research budget exhaustion" |
| `deepsearchqa-0005__ChBLchV` | 0.0 | 3 | "Unable to determine with verified evidence. The research budget was exhausted…" |
| `deepsearchqa-0004__zZrUQiE` | 0.0 | 2 | `_build_partial_result` fired: "No fully synthesized findings were available … 147 source(s) were consulted" |

**Three of the four non-perfect trials are explicit non-answers, not wrong answers.** The agent
gathered sources — 147 of them on `0004` — and then had no surviving `ResearchNotes` to synthesise
from, because the workers that would have produced them were abandoned at the retry cap. `0004`
also logged 48 `loop guard blocked research` lines: having lost its notes, the orchestrator kept
trying to re-research until the request-wide envelope was spent.

Only `deepsearchqa-0002__YCdB2MT` is a genuine wrong answer, and its cause is unrelated — it
answered from `instagram.com/popular/…` pages, which is a retrieval-quality problem, not a schema
one.

Two caveats against over-reading this. `deepsearchqa-0003__UXB2VjA` took 5 cap hits and still scored
1.0, so cap hits do not determine the outcome on their own; and n=5 with one trial per task cannot
separate this effect from task difficulty. What the table does establish is the *shape* of the
failure — "I could not determine" rather than "here is the wrong answer" — and that shape points
squarely at lost notes.

### 3.6 The three standing hypotheses were all wrong

For the record, since they shaped the earlier investigation. All three are refuted by the captured
payloads in §3.2:

1. **An extra top-level key**, from `researcher.j2:44` ("**Analysis**: provide in-depth
   analysis…") landing in a schema with `extra="forbid"`. **No.** Every key in every rejected
   `ResearchNotes` payload is a declared field; `extra_forbidden` never fires on `ResearchNotes` in
   either rerun.
2. **`confidence` outside its `Literal`** (`"High"`, `"very high"`). **No.** It is `"high"` — and
   it sits *inside* the stringified blob, so pydantic never evaluates it.
3. **`relevance_score` outside `ge=0, le=100`.** **No.** 85 and 95 in the captured examples.

Hypotheses 2 and 3 were unreachable by construction: both name fields of `EvidenceJudgment`, and
the whole submodel is rejected as a string before any of its contents are looked at.

### 3.7 The `48626d1` query-vocabulary hypothesis is also refuted

§2.2 proposed that the commit's `## Answer` / `### Considered and excluded` vocabulary propagated
into `query` or `target_components` and caused an `extra="forbid"` failure. The payload run covers
those fields in full (§4.3) and shows no such thing: the rejections are stringified nested objects,
not invented keys. **The proposed mechanism is dead.** The A/B correlation in §2.2 still stands
unexplained — see §4.7.

---

## 4. The fix

§4.1-§4.4 are implemented on this branch and verified in jobs `2026-08-21__13-59-47` and
`2026-08-21__14-11-50`. §4.5-§4.7 are open; **§4.5 is the one that removes the defect** —
§4.1 only bounds its cost.

### 4.1 `StructuredOutputRetryGuardMiddleware` — bound the retry, log the error

`AIQ/src/aiq_agent/agents/deep_researcher/custom_middleware.py`

A `handle_errors` callable cannot signal "stop" — `_handle_structured_output_error` returns
`True` unconditionally for a callable (`langchain/agents/factory.py:623`) — and the researcher
runnable is shared across concurrent batch workers, so instance state would race. The guard is
therefore a `wrap_model_call` middleware that derives its attempt count from the request's own
message list, and raises `StructuredOutputRetryExhausted` once the budget is spent.

Two design points worth keeping:

- **Detection keys off LangChain's error text, not a schema name.** One instance covers every
  structured schema on the agent, including the ones the autonomous and adaptive factories retype
  after the spec is built (`spec["response_format"] = AutonomousResearchPlan`,
  `autonomous_researcher/factory.py:627`). A name-matching guard silently misses those.
- **It counts rejection messages, not rejection/`AIMessage` pairs.** Both tail shapes occur in
  practice: a provider that stamps a fresh id on each response leaves an alternating tail, while
  one that reuses an id has its `AIMessage` replaced in place by `add_messages`, leaving
  consecutive `ToolMessage`s with nothing between them. The first implementation paired them off,
  passed its unit tests, and still looped forever under real `create_agent` wiring —
  `test_stops_a_real_create_agent_structured_output_loop` is the regression test for that.

Registered next to the existing `StructuredResponseTextFallbackMiddleware` at all four
structured-output sites:

| Site | Schema |
| --- | --- |
| `deep_researcher/factory.py:393` (`build_researcher_runnable`) | `ResearchNotes` — the site that looped |
| `deep_researcher/factory.py:466` | `SourceRoutingPlan` |
| `deep_researcher/factory.py:498` | `ResearchPlan` |
| `autonomous_researcher/factory.py:575` (`_researcher_subagent_spec`) | `ResearchNotes` |

Default `max_attempts=3`. Failure is already graceful: the raise propagates out of
`researcher_runnable.ainvoke` into `_run_research_query`'s `except Exception`
(`deep_researcher/tools/research.py`), which wraps it as a per-query `RuntimeError`;
`asyncio.gather(..., return_exceptions=True)` collects it into the batch's `errors` list and the
remaining queries continue. One lost note instead of a lost trial.

**Verified.** Across the two rerun jobs the worst identical-payload streak is 3, by construction,
against 152 before; 74 rejections were capped and logged, 24 of them hit the cap and abandoned
their sub-run, and neither job produced an `AgentTimeoutError`. See the header table and
`artifacts/structured-output-retry-loop/measurements-rerun.txt` §1-§3.

### 4.2 Worker-scoped logging in `_run_research_query`

`AIQ/src/aiq_agent/agents/deep_researcher/tools/research.py`

Batch workers run concurrently and interleave line-by-line in one console log, which is why the
original investigation could not attribute a repeated tool call to a worker. Every worker now logs
start and finish against the `invocation_id` the loop guards already key their state on:

```
Researcher worker 8df601ef… starting | depth=medium tools=web_search_tool query chars=437 ref=sha256:d8f231685e8a
Researcher worker 8df601ef… returned ResearchNotes | findings=6 gaps=1 sources=9 source_calls=10 exhausted=True
```

The `structured_response` that fails `ResearchNotes.model_validate` is also logged before it is
discarded — that path is the last place the rejected payload still exists.

### 4.3 Full batch payload logging

`AIQ/src/aiq_agent/agents/autonomous_researcher/tools/research.py` (and the deep-researcher
equivalent). The existing 80-character preview covers `query` only; the dispatched batch is now
serialized in full, so `target_components`, `rationale`, `subqueries` and `preferred_tools` — the
rest of a worker's entire instruction — are recoverable. This is what §4.6 needs.

### 4.4 The `AIQ_LOG_PAYLOADS` switch

`AIQ/src/aiq_agent/common/logging_utils.py`

`log_content_metadata` is called from ~60 places — every LangChain callback in
`common/callbacks.py`, every guard, every error path. Adding a payload branch there rather than at
the call sites means one switch enriches all of them at once, with no call-site churn:

| Variable | Default | Effect |
| --- | --- | --- |
| `AIQ_LOG_PAYLOADS` | unset | `1`/`true`/`yes`/`on` appends the raw content to every `log_content_metadata` line |
| `AIQ_LOG_PAYLOAD_MAX_CHARS` | `20000` | Per-payload cap. `0` removes it. Oversized payloads are trimmed from the middle, keeping both ends, since malformed JSON usually goes wrong at the tail |

Off by default, and the `chars=… ref=sha256:…` prefix is emitted either way, so existing log
parsing and correlation-by-digest keep working with the switch on. The docstring on
`log_content_metadata` is explicit that prompts and tool payloads can carry credentials or private
customer data; this is a switch for one diagnostic run, not a production setting.

**Turning it on for an eval run** — add to the `env` block of the agent in the Harbor job config
(`EVALS/configs/…`), which `_runtime_env` forwards into the container
(`EVALS/src/aiq_harbor_evals/agents/aiq_harbor.py`):

```yaml
    env:
      NVIDIA_API_KEY: ${NVIDIA_API_KEY}
      TAVILY_API_KEY: ${TAVILY_API_KEY}
      AIQ_LOG_PAYLOADS: "1"
      AIQ_LOG_PAYLOAD_MAX_CHARS: "0"     # optional: no cap
```

Locally, `AIQ_LOG_PAYLOADS=1` in front of the command is enough.

### 4.5 (open) Coerce a stringified nested model — the fix that removes the defect

The field is named and the value is captured (§3.2), so this is no longer conditional. The defect is
one layer of JSON encoding on a scalar nested-model field, the payload underneath is well-formed in
39 of 40 captured cases, and the fix belongs on the base contract where it covers all three affected
fields at once.

`AIQ/src/aiq_agent/agents/deep_researcher/models/subagent_contracts.py`, on `_StrictContract`
(`:35`): a `model_validator(mode="before")` that, for each field whose annotation is a `BaseModel`,
`json.loads` a `str` value and substitutes the decoded object when it decodes to a `dict`.

Design constraints worth honouring:

- **Only for fields annotated as a nested `BaseModel`.** A blanket "parse any string that looks like
  JSON" would corrupt genuine string fields — `rationale`, `narrative_notes` and `summary` routinely
  contain braces and quoted JSON fragments from fetched pages.
- **Only when it decodes to a `dict`.** Anything else must fall through to pydantic unchanged, so
  the error the model sees stays truthful.
- **Never swallow the failure.** If `json.loads` raises, leave the value alone; the existing
  `model_type` error is the correct outcome. This is what keeps the one non-re-parsing case in §3.4
  — where the model packed the whole plan into the `answer_strategy` string — a visible, capped
  failure rather than a silent partial parse.
- **`extra="forbid"` stays.** It is what stops silent field invention, it never fired on
  `ResearchNotes` in either rerun (§3.6), and it is not implicated in this defect.

Expected effect, measured against the captured data rather than estimated: **39 of 44 rejections in
`2026-08-21__14-11-50` become first-attempt successes**, along with the 28 `evidence_judgment`
rejections in `2026-08-21__13-59-47`. The four residual rejections are separate defects and stay
visible — the `subqueries` `max_length=8` overrun (`subagent_contracts.py:114-116`), the
`findings` `list_type`, and the plan-packed-into-`answer_strategy` case.

Two complements, neither a substitute:

- **Prompt reinforcement.** `researcher.j2:46` already describes `evidence_judgment` as "a 0-100
  usefulness score, confidence, and short rationale" without stating it is an object. Saying so
  explicitly is cheap. It is not the fix: the model is being asked to do the right thing and is
  emitting the right content, so this only shifts the rate, and the endpoint offers no `strict=True`
  enforcement to fall back on (§3.3).
- **Escalate before abandoning.** On the first failure, inject the schema-carrying, tools-disabled
  correction that `StructuredResponseTextFallbackMiddleware._correction_request` already builds
  (`custom_middleware.py:139-145`). Far more actionable than "Please fix your mistakes.", and it may
  recover the note instead of dropping it. Worth doing for the residual cases once §4.5 lands.

**Why this ordering matters.** §4.1 caps the cost of a rejection at three model calls; it does not
prevent one. Every capped rejection still throws away a research worker, and §3.5 traces exactly how
that turns into an unanswerable question. §4.5 is what stops the rejection happening.

### 4.6 (open) Make the agent's own deadline reachable

Set `workflow_timeout_seconds` below the harness agent timeout, e.g. `1500` against the `1800.0`
in `task.toml`:

```yaml
# AIQ/configs/config_autonomous_frag.yml:193
      workflow_timeout_seconds: 1500
```

This does not address the loop — §4.1 does — but it converts *any* future runaway from a bare
`AgentTimeoutError` with no artifact into a graded partial answer via `_build_partial_result`. It is
the reason both timed-out trials scored `grader_valid: 0.0` rather than being scored on the evidence
they had already gathered. Still worth doing after §4.1: the guard covers structured-output retries
specifically, and nothing stops a future runaway of a different shape.

### 4.7 (open) Settle §2.2

Re-run the 5 smoke tasks with `orchestrator.j2` reverted to its `0099c80` content, §4.1-§4.4 in
place and `AIQ_LOG_PAYLOADS=1`, everything else held constant.

The metric to compare is **rejection count**, not streak length: §4.1 caps every streak at 3, so
streaks are no longer a usable signal. The baselines are 30 rejections over 5 trials
(`2026-08-21__13-59-47`) and 44 over 5 (`2026-08-21__14-11-50`), against effectively zero in
`2026-08-20__21-25-36`.

- Rejections drop to near zero → `48626d1`'s orchestrator section is confirmed as the trigger, and
  §4.3's batch payloads should show what about the dispatched queries changed.
- Rejections stay in the 30-44 range → the trigger is elsewhere (task mix, endpoint behaviour,
  model version) and §4.5 is the whole answer.

Either way this is now a question about *rate*, not about the defect: §3 establishes what the
failure is independently of what makes it more or less frequent, and §4.5 fixes it in both branches
of the experiment.

---

## 5. Reference artifacts

### 5.1 Checked in with this document

Path relative to `AIQ/misc/autonomous_researcher/artifacts/structured-output-retry-loop/`.

**The original loop (`2026-08-21__11-12-50`, no guard):**

| Artifact | What it shows |
| --- | --- |
| `collect-evidence.sh` | Regenerates every §1/§2 measurement. `EVALS=<path> ./collect-evidence.sh > measurements.txt` |
| `measurements.txt` | Output of the above, as captured on 2026-08-21 |
| `dsqa-0002-exception.txt` | The `AgentTimeoutError` traceback |
| `dsqa-0002-aiq_state.json` | `status: running`, 142 LLM / 141 done, `tool.in_flight: 1` at kill time |
| `dsqa-0002-loop-excerpt.log` | Last real research call, the first non-validating `ResearchNotes`, and the tail 28 min later |
| `dsqa-0004-aiq_state.json` | Same shape, 246 LLM calls |
| `dsqa-0004-loop-excerpt.log` | Tail of the second timed-out trial |
| `job-result.json` | Job metrics and `exception_stats` |

**The verification reruns (`2026-08-21__13-59-47` guard-only, `2026-08-21__14-11-50` guard + payloads):**

| Artifact | What it shows |
| --- | --- |
| `collect-rerun-evidence.sh` | Regenerates every §3 measurement across all three jobs. `EVALS=<path> ./collect-rerun-evidence.sh > measurements-rerun.txt`; `LOOP_JOB` / `GUARD_JOB` / `PAYLOAD_JOB` are individually overridable |
| `measurements-rerun.txt` | Output of the above. §1 streaks, §2-§3 rejection census, §4 the diagnosis, §5 a verbatim instance, §6 outcomes, §7 log-size cost |
| `stringified-nested-model-example.log` | **The decisive artifact.** One complete rejection with `AIQ_LOG_PAYLOADS=1`: `evidence_judgment` as a JSON string, `findings`/`gaps`/`sources` as proper arrays in the same payload |
| `dsqa-0003-guard-cascade.log` | The §3.5 cascade on one trial — source-call budget, the retry guard killing a worker, then the loop guard refusing to re-research |
| `dsqa-0003-degraded-answer.txt` | What that trial wrote instead: "The research budget was exhausted…" |
| `job-result-guard-run.json` | `2026-08-21__13-59-47` metrics — 0 errors, `grader_valid: 1.0` |
| `job-result-payload-run.json` | `2026-08-21__14-11-50` metrics — 0 errors, `grader_valid: 1.0` |

### 5.2 Source artifacts in the eval repo

Path relative to `EVALS/`:

| Path | Contents |
| --- | --- |
| `jobs/2026-08-21__11-12-50/config.json` | Harbor job config: agent import path, `config_autonomous_frag.yml`, retry policy |
| `jobs/2026-08-21__11-12-50/job.log` | `AgentTimeoutError is in exclude_exceptions, not retrying`; missing `/workspace/answer.txt` |
| `jobs/2026-08-21__11-12-50/result.json` | 5 trials, 2 errors, reward 0.4 |
| `jobs/2026-08-21__11-12-50/deepsearchqa-0002__DsPCtn4/exception.txt` | Timeout traceback |
| `jobs/2026-08-21__11-12-50/deepsearchqa-0002__DsPCtn4/agent/aiq-agent-console-stdout.txt` | 172 KB; the loop, 118 `→ ResearchNotes` lines |
| `jobs/2026-08-21__11-12-50/deepsearchqa-0002__DsPCtn4/agent/aiq-agent-console-stderr.txt` | Repeated `Model 'nvidia/nvidia/nemotron-3-ultra' is not known to support tools` |
| `jobs/2026-08-21__11-12-50/deepsearchqa-0002__DsPCtn4/agent/aiq_events.jsonl` | `run_research_batch` `TOOL_START` with no `TOOL_END` |
| `jobs/2026-08-21__11-12-50/deepsearchqa-0004__gcnVpr2/` | Second timed-out trial, same shape |
| `datasets/deepsearchqa-smoke/deepsearchqa-0002/task.toml` | `timeout_sec = 1800.0` (agent), `300.0` (verifier) |
| `jobs/2026-08-20__21-25-36/` | Same 5 tasks, same config, **before** `48626d1` — the A/B baseline |
| `jobs/2026-08-20__21-44-00/` | 90-task job showing 3–4× streaks pre-`48626d1` |
| `jobs/2026-08-21__13-59-47/` | Rerun with §4.1-§4.3 in place, `AIQ_LOG_PAYLOADS` unset — 0 timeouts, 30 rejections, field named |
| `jobs/2026-08-21__14-11-50/` | Rerun with `AIQ_LOG_PAYLOADS=1` — 0 timeouts, 44 rejections, all 44 payloads captured |
| `configs/deepsearchqa_autonomous_frag.yaml:35-44` | Where `AIQ_LOG_PAYLOADS` / `AIQ_LOG_PAYLOAD_MAX_CHARS` are declared for eval runs |

### 5.3 Code, this repo

Path relative to `AIQ/`:

| Path | Relevance |
| --- | --- |
| `src/aiq_agent/agents/deep_researcher/factory.py:372-403` | `build_researcher_runnable` — the looping `response_format=ResearchNotes` binding |
| `src/aiq_agent/agents/autonomous_researcher/factory.py:787` | Where the autonomous agent builds that runnable |
| `src/aiq_agent/agents/autonomous_researcher/factory.py:537-580` | `_researcher_subagent_spec` — the second `ResearchNotes` binding |
| `src/aiq_agent/agents/deep_researcher/models/subagent_contracts.py:35-38` | `_StrictContract` / `extra="forbid"` |
| `src/aiq_agent/agents/deep_researcher/models/subagent_contracts.py:155-215` | `ResearchSource`, `ResearchFinding`, `ResearchGap`, `EvidenceJudgment`, `ResearchNotes` |
| `src/aiq_agent/agents/deep_researcher/tools/research.py:84-152` | `_run_research_query` — where a raise degrades to a per-query error |
| `src/aiq_agent/agents/deep_researcher/tools/research.py:206-240` | Per-query error aggregation |
| `src/aiq_agent/agents/deep_researcher/custom_middleware.py:107-161` | `StructuredResponseTextFallbackMiddleware` — adjacent, different failure |
| `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py:1229` | `ConsecutiveThinkGuardMiddleware` — the pattern Fix 1 mirrors |
| `src/aiq_agent/agents/autonomous_researcher/agent.py:584-596` | Workflow deadline + `_build_partial_result` fallback |
| `src/aiq_agent/agents/deep_researcher/models/subagent_contracts.py:146-147` | `task_analysis`, `answer_strategy` — two of the three scalar nested-model fields (§3.3) |
| `src/aiq_agent/agents/deep_researcher/models/subagent_contracts.py:212` | `evidence_judgment` — the third, and 67 of 74 observed rejections |
| `src/aiq_agent/agents/deep_researcher/models/subagent_contracts.py:114-116` | `subqueries` `max_length=8` — the unrelated residual defect in §3.4 |
| `src/aiq_agent/agents/autonomous_researcher/prompts/researcher.j2:46` | Describes `evidence_judgment` without stating it is an object — §4.5 prompt complement |
| `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py:1215-1224` | Researcher source-call budget — the guard the degraded answer refers to (§3.5) |
| `src/aiq_agent/agents/autonomous_researcher/custom_middleware.py:880` | Autonomous loop guard — `duplicate_query` / `already_finalizing` (§3.5) |
| `configs/config_autonomous_frag.yml:149-152` | `source_call_budgets` — `low: 5`, the ceiling both workers hit in §3.5 |
| `src/aiq_agent/agents/deep_researcher/custom_middleware.py:164-292` | `StructuredOutputRetryGuardMiddleware`, `StructuredOutputRetryExhausted` — §4.1 |
| `src/aiq_agent/common/logging_utils.py` | `log_content_metadata`, `payload_logging_enabled`, `truncate_payload` — §4.4 |
| `tests/aiq_agent/agents/deep_researcher/test_custom_middleware.py` | `TestStructuredOutputRetryGuardMiddleware`, incl. the real-`create_agent` regression test |
| `tests/aiq_agent/common/test_logging_utils.py` | Payload-gate tests |
| `src/aiq_agent/common/callbacks.py:147` | `current_input` set on chain start and sticky |
| `configs/config_autonomous_frag.yml:53-63` | Model, `temperature: 0.7`, `enable_thinking: true` |
| `configs/config_autonomous_frag.yml:193-195` | `workflow_timeout_seconds: 2400`, `recursion_limit: 250` |

### 5.4 Third-party code

In the runtime image, `/app/.venv/lib/python3.13/site-packages/` (langchain 1.3.11,
deepagents 0.6.8):

| Path | Relevance |
| --- | --- |
| `langchain/agents/factory.py:1195-1266` | Structured-output tool-call handling and the retry return |
| `langchain/agents/factory.py:596-623` | `_handle_structured_output_error` — a callable can never stop the retry |
| `langchain/agents/factory.py:113` | `STRUCTURED_OUTPUT_ERROR_TEMPLATE` |
| `langchain/agents/factory.py:576-594` | `_is_openai_compatible_model` — why `strict=True` is skipped for `ChatNVIDIA` |
| `langchain/agents/structured_output.py:196-242` | `ToolStrategy`, `handle_errors` default `True` |
| `langchain/agents/structured_output.py:60-75` | `StructuredOutputValidationError` message format |
| `deepagents/middleware/patch_tool_calls.py` | `PatchToolCallsMiddleware` — `before_agent` only, not involved |
