# Adaptive Researcher + GLM Intermediate-Response Failure Analysis

## Document status

- **Repository studied:** `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar`
- **Evaluation repository:** `/home/smasurekar/Desktop/Swapnil/gitlab_repos/rag`
- **Current evaluation:** `deepsearchqa-adaptive-web-only-glm-90-2`
- **Prior comparison evaluation:** `deepsearchqa-adaptive-web-only-glm-90-1`
- **Agent:** `adaptive_researcher`
- **Primary model:** `nvidia/zai-org/glm-5.2`
- **Analysis date:** 2026-07-30
- **Purpose:** Explain why GLM-generated intermediate narration is being returned as a successful final report, document the evidence, distinguish related but separate termination cases, and propose a reliable remediation.

## Executive summary

The observed failure is not simply that GLM occasionally ignores a prompt instruction. It is a contract mismatch between probabilistic model behavior and deterministic workflow termination.

The adaptive orchestrator is a ReAct-style agent. On any turn, the model may either:

1. issue one or more tool calls; or
2. return an assistant message with no tool calls.

The underlying agent graph treats case 2 as normal completion. GLM sometimes uses the assistant `content` field for narration such as:

- “Let me retry with a properly formatted query.”
- “Now I need Toronto's data. Let me run the Toronto query.”
- “I've hit the query budget limit. Let me use my final query strategically.”

These sentences describe an intended next action, but they do not contain a structured tool call. Consequently, the ReAct loop ends before that intended action occurs.

AI-Q then compounds the premature graph termination. When neither of the positive final-output files exists, `AdaptiveResearcherAgent._salvage_inline_report()` accepts **any non-empty last assistant message**, rejecting only an empty message and the exact writer completion marker. That broad fallback was introduced to preserve conversational greetings, but it cannot distinguish a greeting or real answer from an intermediate plan, retry narration, refusal, or transport-error string.

The resulting sequence is:

```text
GLM emits plain narration without a tool call
    → ReAct graph interprets the turn as terminal
    → no /shared/final_report.md and no /shared/output.md exists
    → broad inline salvage accepts the narration
    → citation/sanitization post-processing runs
    → callback emits the salvaged text as final_report
    → async server records job status=success
    → evaluation may score an intermediate sentence as an answer
```

This behavior is deterministic once GLM emits a no-tool message. Prompt improvements can reduce the probability of the first step, but they cannot enforce the completion contract.

The reliable solution is therefore:

1. define completion using a positive, machine-verifiable signal;
2. reject arbitrary no-tool assistant text as a final research report;
3. when a no-tool intermediate message appears, resume the same request with a bounded recovery turn;
4. if continuation is no longer appropriate, run a constrained finalize-only recovery;
5. if model-based finalization fails, generate a deterministic partial result;
6. emit one canonical terminal event that tells consumers whether the result is complete, partial, or failed.

Merely removing salvage would stop false successes but would convert the same GLM behavior into job failures. A recovery path is needed in addition to a stricter acceptance gate.

## Scope and evidence sources

This analysis used the following evidence:

### Live execution evidence

- The `eval_auto_run` tmux pane.
- `aiq-agent` Docker logs.
- Read-only queries against the `aiq_jobs` Postgres database:
  - `job_info`
  - `job_events`
- Per-job `tool.start`, `tool.end`, `llm.start`, `llm.end`, and `artifact.update` records.
- Current job output previews for jobs that reached server-side `success` without a `submit_final_report` event.

### Prior-run evidence

- `/home/smasurekar/Desktop/Swapnil/gitlab_repos/rag/results/aiq/deepsearchqa-adaptive-web-only-glm-90-1/tier_summary.md`
- Prior-run event classification already performed while updating the evaluation adapter.

### Source-code evidence

The principal files studied were:

- `src/aiq_agent/agents/adaptive_researcher/agent.py`
- `src/aiq_agent/agents/adaptive_researcher/factory.py`
- `src/aiq_agent/agents/adaptive_researcher/custom_middleware.py`
- `src/aiq_agent/agents/adaptive_researcher/models/request_termination.py`
- `src/aiq_agent/agents/adaptive_researcher/tools/finalize.py`
- `src/aiq_agent/agents/adaptive_researcher/prompts/orchestrator.j2`
- `configs/config_adaptive_frag.yml`
- `frontends/aiq_api/src/aiq_api/jobs/callbacks.py`
- Adaptive-researcher tests under `tests/aiq_agent/agents/adaptive_researcher/`
- Design documents under `misc/`, particularly:
  - `AI-Q-3.0-Unified-Research-POC-Plan.md`
  - `adaptive-researcher-request-termination-plan.md`
  - `adaptive-researcher-orchestrator-prompt-notes.md`
  - `single-shot-redundant-final-llm-call.md`

## Evaluation observations

### Prior run

The prior run attempted 90 samples:

- 89 were represented as completed.
- `dsqa_id_39` failed.
- 50 of the 89 server-completed records did not contain an explicit `submit_final_report` tool-start event.
- Only 39 records had the expected explicit inline-finalization signal.

The tier summary reported:

| Tier | Completed records | Average accuracy |
|---|---:|---:|
| `single_shot` | 22 | 0.4545 |
| `standard` | 57 | 0.2680 |
| `deep` | 10 | 0.0000 |
| Overall | 89 | 0.2840 |

Those accuracy values were misleading because the completed population included records whose “report” was actually intermediate orchestration text.

### Current run snapshot

At the principal evidence snapshot, the current run had:

| State | Count |
|---|---:|
| Total submitted jobs | 90 |
| Server-side `success` | 89 |
| Server-side `running` | 1 |
| Server-side `failed` | 0 |
| Successful jobs with `submit_final_report` | 50 |
| Successful jobs without `submit_final_report` | 39 |

The snapshot was taken while the tmux evaluator showed `89/90`, approximately 2 hours and 9 minutes into inference.

This means the server-level submission rate at that point was:

```text
50 / 89 = 56.18%
```

and the no-submit rate was:

```text
39 / 89 = 43.82%
```

The current run was therefore better than the prior run at the earlier 81-completed snapshot, but it was still substantially affected. The issue was not eliminated.

### Declared tiers among the 39 no-submit jobs

The most recent `declare_effort_tier` events showed:

| Declared tier | No-submit jobs |
|---|---:|
| `standard` | 28 |
| `deep` | 10 |
| Unknown/no recorded declaration | 1 |

This distribution is important:

- The failure is not isolated to one shallow path.
- It is particularly common on standard workflows that alternate between reading persisted notes and attempting additional batches.
- Deep jobs can also terminate before reaching the writer path.

### Last tool used before premature completion

An earlier snapshot of 37 no-submit jobs produced this distribution:

| Last tool | Jobs |
|---|---:|
| `read_file` | 20 |
| `advanced_web_search_tool` | 4 |
| `run_research_batch` | 3 |
| `get_verified_sources` | 3 |
| `think` | 2 |
| `write_todos` | 2 |
| `ls` | 1 |
| `web_search_tool` | 1 |
| No tool, model failure before tool use | 1 |

The dominance of `read_file` is revealing. A common sequence is:

1. the orchestrator reads a persisted research note;
2. GLM explains what evidence is present or missing;
3. GLM states what it intends to query next;
4. GLM fails to encode that intent as a tool call;
5. the graph terminates.

### Representative invalid terminal outputs

Observed no-submit outputs included:

```text
Let me retry with a properly formatted query.
```

```text
I have Vancouver's data. Now I need Toronto's data. Let me run the Toronto query.
```

```text
The queries were flagged as duplicates. Let me rephrase them to be distinct searches
targeting the specific gaps.
```

```text
I've used 6 of my 9 query budget. Let me run 2 more focused queries to cover the
remaining four artists, keeping 1 in reserve.
```

```text
I have notes for Zion and Arches, but Canyonlands is missing. Let me run the
Canyonlands query separately.
```

These are unambiguously intermediate. They contain future-tense intended actions and do not attempt to answer the user.

Other outputs were more ambiguous:

- A few contained partial synthesis mixed with “I still need...” language.
- One was a substantial long-form answer, suggesting that GLM completed synthesis but forgot to call the finalizer.
- One was the deterministic workflow-timeout partial report.
- One was a model transport failure string after retry exhaustion.

These cases must not all receive the same treatment.

## Expected completion contract

The adaptive researcher has two legitimate normal finalization mechanisms.

### Inline path

The orchestrator authors the answer itself and calls:

```text
submit_final_report(
    markdown=<complete final Markdown>,
    researched=<true or false>,
    tier=<selected tier>,
)
```

The tool writes:

- `/shared/final_report.md`
- `/shared/final_report_meta.json`

It uses `return_direct=True`, so a successful call ends the ReAct loop immediately without a redundant final LLM turn.

### Writer path

The orchestrator delegates to `writer-agent`. The writer persists:

- `/shared/output.md`

The orchestrator then returns only:

```text
Wrote /shared/output.md
```

The adaptive runtime reads the file. The writer path intentionally does **not** call `submit_final_report`.

### Forced-termination path

On workflow timeout or graph recursion exhaustion, `AdaptiveResearcherAgent.run()` returns a deterministic partial result assembled from:

- persisted research-note summaries;
- persisted research gaps;
- the verified source registry.

This path also does not currently call `submit_final_report`.

### Correct positive completion predicate

At the product-runtime level, a normal completion should therefore be:

```text
has_nonempty(/shared/final_report.md)
OR
has_nonempty(/shared/output.md)
```

A deterministic partial should carry its own explicit terminal metadata, such as:

```json
{
  "status": "partial",
  "reason": "workflow_timeout",
  "report_path": "/shared/final_report.md"
}
```

The last assistant message alone should not be authoritative for a research request.

## Important evaluation-contract mismatch

An evaluation rule that requires `submit_final_report` for **every** adaptive-researcher job conflicts with the current product design:

- valid inline branch → calls `submit_final_report`;
- valid writer branch → writes `/shared/output.md` and deliberately does not call `submit_final_report`;
- deterministic timeout fallback → currently returns a partial message without calling the tool.

Therefore, “no `submit_final_report` event” is an excellent detector for invalid inline paths, but it is not by itself a universally correct product-level completion rule.

The evaluator should ideally accept either:

1. an explicit `submit_final_report` event; or
2. verified writer completion, consisting of actual writer delegation plus a non-empty `/shared/output.md` completion artifact; or
3. explicit runtime-generated partial-result metadata.

If the desired future contract is instead “every path must emit one unified submission event,” then the product should be changed first:

- after loading `/shared/output.md`, the runtime can programmatically emit the canonical final-submission event;
- deterministic partial fallback can do the same;
- the event should distinguish `complete` from `partial`.

The evaluator can then safely require that unified event.

## Source-code root-cause analysis

### 1. The prompt requires finalization, but prompts are not enforcement

`orchestrator.j2` repeatedly instructs the model to call `submit_final_report`.

Examples include:

- meta/chit-chat must call it with `researched=false`;
- direct must call it with `researched=false`;
- single-shot must call it with `researched=true`;
- standard-inline must call it with `researched=true`;
- the finalization section says inline answers call it exactly once.

The prompt even states that a plain conversational reply does not finish the run.

Nevertheless, the model API is not configured so that every non-writer terminal turn must contain a tool call. GLM remains free to return ordinary assistant content.

### 2. ReAct interprets a no-tool assistant message as completion

The graph is built using `create_deep_agent(...)`.

The model receives a set of tools. When its `AIMessage` has tool calls, the graph executes them and continues. When it has none, the graph exits normally.

This is standard agent behavior. The graph cannot infer that:

```text
Let me run the Toronto query.
```

means the model intended to call a search tool but failed to encode the call.

From the graph's perspective, the model chose to answer in plain text.

### 3. Inline salvage accepts nearly everything

`AdaptiveResearcherAgent._salvage_inline_report()`:

1. takes the last message;
2. reads its string content;
3. strips whitespace;
4. rejects it only if it is empty or exactly equal to `Wrote /shared/output.md`;
5. returns every other string.

The comments explicitly say the path accepts a short conversational reply. The implementation extends that policy to all non-empty strings.

The related tests intentionally assert:

- a substantive inline message is accepted;
- a short heading-less greeting is accepted;
- only an empty message and writer marker are rejected;
- a greeting without `submit_final_report` succeeds.

Thus the invalid acceptance is not an accidental untested edge case. The current tests codify it.

### 4. Post-processing legitimizes the salvaged message

After salvage, the runtime:

- defaults `researched` to `True` when final-report metadata is absent;
- runs citation verification if sources exist;
- sanitizes the text;
- emits it using `emit_final_report`;
- replaces the graph's last message with the processed text;
- returns a valid `AdaptiveResearchAgentState`.

Because sources often exist by this point, even obvious narration can pass through the same post-processing path as a real report.

### 5. The async API records server success

The agent did not raise. It returned a valid state with a last message. The async job runner therefore records:

```text
status = success
```

Server success currently means “the workflow returned without an unhandled exception,” not “the finalization protocol was satisfied.”

### 6. Event/callback behavior makes the result look official

The async callback emits model outputs as draft/research/intermediate artifacts during execution. After adaptive-agent post-processing, `emit_final_report()` explicitly emits the salvaged content with:

```text
output_category = final_report
```

Therefore downstream clients see a final-report artifact even though the authoritative finalizer tool never ran.

## Why GLM emits no-tool intermediate messages

The exact internal reason for any individual model token is not observable, but the execution evidence supports several contributing factors.

### Natural-language planning leaks into the assistant content channel

GLM frequently explains the next intended action in natural language:

```text
Let me run...
I need to...
Now I should...
I will retry...
```

In a tool-calling agent, this is only safe when accompanied by a structured tool call. GLM sometimes emits the narration alone.

### The orchestrator prompt is complex

The orchestrator simultaneously manages:

- effort-tier declaration;
- direct, single-shot, standard, and deep paths;
- inline versus writer branches;
- query depth;
- research delegation;
- duplicate-query avoidance;
- source verification;
- citation formatting;
- filesystem rules;
- budget rules;
- escalation rules;
- parent-report delta behavior;
- finalization rules.

Dynamic prompt sections reduce irrelevant content, but the model still has a significant protocol to follow. Long histories containing plans, research notes, errors, and source metadata further increase the chance that the model follows the semantic intent but misses the structured tool-call requirement.

### Temperature is relatively high for orchestration

The active config uses:

```yaml
temperature: 0.7
top_p: 0.7
```

That can be reasonable for research synthesis but is unnecessarily stochastic for workflow control. The same LLM configuration is assigned to:

- orchestrator;
- source router;
- planner;
- researcher;
- writer.

The best decoding settings for prose synthesis are not necessarily the best settings for strict tool protocol compliance.

### Thinking mode does not guarantee tool emission

The config enables:

```yaml
chat_template_kwargs:
  enable_thinking: true
```

Thinking can improve reasoning, but it does not enforce that a stated next action becomes a structured tool call. Depending on model/template behavior, planning text can leak into visible `content`.

### Guard and tool errors create high-risk transition points

Many premature completions occur after:

- duplicate-query rejection;
- query-budget rejection;
- invalid `ResearchNotes`;
- partially successful research batches;
- reading existing notes and identifying a missing component.

The guard returns a `ToolMessage(status="error")` instructing the model to finalize or represent gaps. GLM sometimes instead narrates a revised plan without making the next tool call.

### Tool withdrawal is deterministic, finalization is not

The loop guard can remove:

- `run_research_batch`;
- `think`;
- direct search tools in single-shot mode.

This reliably prevents additional research. It does not force the remaining `submit_final_report` call. The next model response may still be plain text, which the graph treats as terminal.

### Transport errors can be converted into content

The current run contained repeated transient HTTP 429 capacity/rate-limit responses. Most were retried successfully.

At least one job exhausted retries on an HTTP 500 and returned text beginning:

```text
Model call failed after 3 attempts with Exception: [500] ...
```

That string was accepted by the same salvage path and emitted as a final report. A transport failure should be represented as an exception or explicit failed/partial terminal state, never as ordinary assistant report content.

## Failure taxonomy

### Type A: intended-next-tool narration

Example:

```text
I have Vancouver's data. Now I need Toronto's data. Let me run the Toronto query.
```

Desired handling:

- preserve state;
- append a runtime recovery instruction;
- resume the graph so GLM can make the intended tool call;
- limit retries.

### Type B: guard/budget narration

Example:

```text
I've hit the query budget limit. Let me use my final query strategically...
```

Desired handling depends on actual guard state:

- if budget remains and the proposed query is allowed, resume once;
- if research is closed, do not allow more research;
- switch to finalize-only recovery using gathered evidence.

### Type C: synthesis completed but finalizer omitted

Example:

- a long, answer-like Markdown report with citations;
- no `/shared/final_report.md`;
- no `/shared/output.md`;
- no `submit_final_report` event.

Desired handling:

- validate the content against the verified source registry;
- invoke the canonical finalization mechanism programmatically or through a forced finalize-only model turn;
- do not rerun research.

### Type D: writer branch

Expected:

- actual `writer-agent` delegation;
- non-empty `/shared/output.md`;
- short writer completion marker;
- no `submit_final_report` by design.

Desired handling:

- accept as valid product completion;
- preferably emit one canonical terminal event for evaluator/client consistency.

### Type E: deterministic timeout or recursion partial

Expected:

- explicit reason such as workflow timeout;
- conservative partial report;
- terminal status `partial`, not indistinguishable `success`;
- no model retry after the hard deadline unless a separately budgeted finalizer is intentionally allowed.

Desired handling:

- persist the partial report;
- emit explicit partial-result metadata;
- optionally route through the canonical submission event with `completion_status="partial"`.

### Type F: model or provider failure

Example:

```text
Model call failed after 3 attempts with Exception: [500] ...
```

Desired handling:

- do not treat as report content;
- raise/propagate a typed model-provider exception;
- optionally invoke deterministic partial fallback if durable evidence exists;
- otherwise mark the job failed.

## Design contradiction discovered

The original POC design explicitly warned against the exact broad-salvage behavior now present.

The design says to add a positive `submit_final_report` signal and avoid relaxing salvage to “any last message,” because that would accept:

- a short acknowledgment;
- a re-plan;
- a refusal.

The implementation later changed salvage so that it rejects only empty content and the writer marker. The current failure is therefore a realized version of a risk already identified in the design.

## Why simple fixes are insufficient

### Prompt-only fix

Adding more “MUST call `submit_final_report`” text may reduce frequency but cannot guarantee compliance.

The current prompt already contains strong repeated instructions and examples. The failure rate remains material.

### Lower-temperature-only fix

Lowering orchestrator temperature should improve consistency, but deterministic decoding does not mathematically force a tool call. If the model's most likely response is narration, temperature 0 will reproduce it reliably.

### Remove-salvage-only fix

Removing broad salvage is necessary for correctness, but by itself it changes:

```text
false success
```

into:

```text
job failure
```

That protects evaluation integrity but does not recover useful work.

### Force a tool on every orchestrator turn

Global `tool_choice="required"` is incompatible with the current graph:

- the writer branch legitimately returns a marker with no tool call;
- some turns may need ordinary model output that is consumed by a subagent protocol;
- forcing an arbitrary tool does not guarantee the correct next tool;
- forcing finalization too early may truncate valid research.

Tool forcing is most appropriate in a narrowly scoped recovery/finalization phase, not throughout the complete adaptive workflow.

### Require `submit_final_report` for every job without changing product behavior

This incorrectly rejects the documented writer branch and deterministic partial paths.

Either:

- the evaluator must recognize all legitimate completion mechanisms; or
- the product must emit one unified terminal event for all mechanisms.

## Recommended target invariants

### Invariant 1: positive completion

A research request is never considered complete solely because the last LLM message is non-empty.

### Invariant 2: authoritative report origin

The authoritative report must come from one of:

1. `/shared/final_report.md`;
2. `/shared/output.md`;
3. a deterministic runtime partial persisted to a canonical report path.

### Invariant 3: explicit terminal status

Every job ends as one of:

- `complete`;
- `partial`;
- `failed`;
- `cancelled`.

These statuses should not be inferred from prose.

### Invariant 4: bounded recovery

No-finalizer recovery has a strict retry and time budget.

### Invariant 5: provider failures are not content

Known transport/retry-exhaustion output is never eligible for report salvage.

### Invariant 6: one canonical terminal event

Every accepted terminal report emits a single machine-readable event containing:

```json
{
  "completion_status": "complete | partial",
  "origin": "inline_tool | writer_file | deterministic_fallback",
  "tier": "direct | single_shot | standard | deep | meta | unknown",
  "researched": true,
  "report_path": "/shared/final_report.md",
  "reason": null
}
```

This event is a stronger evaluation contract than checking for one model-chosen tool call.

## Recommended remediation architecture

## Layer 1: strict report acceptance

Change normal `run()` post-processing so it first checks only known output files:

```python
final_message = self._resolve_output_file_markdown(result, state.files)
if final_message is None:
    result = await self._recover_unfinalized_result(...)
    final_message = self._resolve_output_file_markdown(result, state.files)
if final_message is None:
    return self._build_partial_or_raise(...)
```

Do not immediately call `_salvage_inline_report()` for a research request.

For meta/direct requests, the prompt already requires `submit_final_report(researched=false)`. They can follow the same positive-signal rule. A forgotten finalizer should enter bounded recovery rather than broad salvage.

## Layer 2: classify the unfinalized turn

Before recovery, collect:

- declared tier;
- whether verified sources exist;
- whether research notes exist;
- whether a plan exists;
- whether writer-agent was invoked;
- whether `/shared/output.md` exists;
- request guard phase and exhaustion reason;
- last assistant content;
- last tool result/status;
- whether the content resembles a provider error;
- elapsed time and remaining recovery budget.

Possible classifications:

```text
CONTINUE_RESEARCH
FINALIZE_FROM_EVIDENCE
ACCEPT_WRITER_FILE
DETERMINISTIC_PARTIAL
PROVIDER_FAILURE
INVALID_TERMINATION
```

Classification should use state and artifacts, not an LLM guess.

## Layer 3: bounded continuation recovery

Use continuation recovery when:

- no output file exists;
- the last response has no tool calls;
- research is still active;
- request budgets are not exhausted;
- the last content describes a plausible next research action.

Recovery should:

1. preserve the existing messages and `/shared/` files;
2. preserve source registry contents;
3. preserve request-wide guard counters;
4. append an internal runtime instruction stating that the previous assistant message was non-terminal;
5. invoke the same graph again;
6. allow at most one or two such recoveries;
7. remain inside an independent short timeout.

Example internal instruction:

```text
[SYSTEM RECOVERY]
Your previous assistant message was not a valid terminal response because it neither
called a tool nor produced an accepted output file. Do not repeat or explain the plan.
Continue by making the concrete tool call you intended. When the answer is ready, finish
through the required protocol: submit_final_report for an inline answer, or writer-agent
plus /shared/output.md for the writer path.
```

This specifically addresses GLM's narration/tool-call mismatch.

The recovery message should be marked as internal metadata so it is not confused with new user input.

## Layer 4: finalize-only recovery

Use finalize-only recovery when:

- research budget is exhausted;
- duplicate-query guard has moved the request into finalizing;
- enough evidence exists;
- the last message is already answer-like but lacks a finalizer;
- continuation recovery has already failed.

The recovery model call should have a much smaller action surface.

Preferred sequence:

1. make verified sources available;
2. expose `get_verified_sources` if it has not been called;
3. expose `submit_final_report`;
4. hide all search, planning, filesystem exploration, think, todo, and delegation tools;
5. instruct the model to represent missing evidence as explicit gaps;
6. set a low temperature;
7. where supported, require tool use;
8. after sources are available, force `submit_final_report` specifically.

This is where `tool_choice="required"` or:

```text
tool_choice = {"type": "function", "function": {"name": "submit_final_report"}}
```

is appropriate, subject to ChatNVIDIA/GLM compatibility testing.

## Layer 5: dedicated fallback finalizer

The request-termination design already reserves:

```yaml
fallback_finalizer_timeout_seconds: 60
```

but the current config model documents it as unused by the deterministic first-slice fallback.

Implement the originally planned bounded finalizer:

- do not re-enter the adaptive graph;
- provide only:
  - original question;
  - persisted `ResearchNotes`;
  - compact verified-source metadata;
  - declared tier;
  - termination/recovery reason;
- expose no research tools;
- require structured finalizer output;
- prohibit unsupported claims;
- require an Evidence Gaps/Limitations section when incomplete;
- enforce the independent timeout.

A structured result could be:

```python
class FallbackFinalReport(BaseModel):
    markdown: str
    researched: bool
    completion_status: Literal["complete", "partial"]
    reason: str | None
```

After validating that structure, the runtime—not the model—should persist the report and emit the canonical terminal event.

If it is important that evaluation observes an actual `submit_final_report` tool event, invoke the tool programmatically with the normal callbacks so `tool.start` and `tool.end` are recorded. Do not fabricate an event row directly.

## Layer 6: deterministic partial

If the fallback finalizer fails or times out:

- use `_render_deterministic_partial()`;
- persist the result to the canonical final report path;
- mark it `completion_status="partial"`;
- attach the forced-termination reason;
- emit the canonical terminal event.

This preserves useful work without pretending the answer is complete.

## Layer 7: remove or severely narrow salvage

Recommended:

- remove `_salvage_inline_report()` from research completion entirely;
- make meta/direct requests use the same required finalizer;
- let a missing finalizer enter recovery.

If backward compatibility requires temporary salvage, restrict it to a deterministic pre-research condition:

- no declared research tier, or explicitly declared `meta`/`direct`;
- no source tools called;
- no sources registered;
- no research notes;
- no plan;
- no provider-error signature;
- content passes a narrow conversational-answer classifier.

Even this should be transitional. The prompt already teaches meta/direct to call the finalizer.

## GLM-specific configuration recommendations

### Split role configurations

Do not use one decoding configuration for every role.

Example:

```yaml
llms:
  glm_orchestrator:
    _type: nim
    model_name: nvidia/zai-org/glm-5.2
    base_url: https://inference-api.nvidia.com/v1
    temperature: 0.0
    top_p: 1.0
    max_tokens: 65536
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: false

  glm_researcher_writer:
    _type: nim
    model_name: nvidia/zai-org/glm-5.2
    base_url: https://inference-api.nvidia.com/v1
    temperature: 0.3
    top_p: 0.7
    max_tokens: 65536
    num_retries: 5
    chat_template_kwargs:
      enable_thinking: true
```

Then assign:

```yaml
adaptive_research_agent:
  orchestrator_llm: glm_orchestrator
  source_router_llm: glm_orchestrator
  planner_llm: glm_researcher_writer
  researcher_llm: glm_researcher_writer
  writer_llm: glm_researcher_writer
```

This is a mitigation, not the correctness boundary.

### Test thinking mode independently

Run an A/B evaluation:

- orchestrator `enable_thinking=true`;
- orchestrator `enable_thinking=false`;

Measure:

- no-tool intermediate-turn rate;
- correct tool-call rate;
- valid completion rate;
- total orchestrator tokens;
- report accuracy.

Do not assume that disabling thinking will necessarily improve overall quality.

### Use forced tool choice only during recovery

Test the exact ChatNVIDIA payload and model behavior for:

- `tool_choice="required"`;
- named `submit_final_report` tool choice;
- parallel tool calls disabled during finalization.

Do not rely on prompt wording as a substitute if named tool choice is supported.

### Keep recovery prompts short

The recovery request should not resend a second large static protocol. It should append a concise instruction to the existing state or use a separate small finalizer context.

## Prompt recommendations

Prompt changes should support, not replace, runtime enforcement.

### Make the per-turn contract explicit

Add near the top of the orchestrator prompt:

```text
Every assistant turn must do exactly one of:
1. issue the concrete tool call(s) for the next action;
2. return the writer completion marker after /shared/output.md exists.

Never describe a future action in ordinary assistant text. If you write “I will,”
“let me,” “next I need,” or “I should,” make the corresponding tool call in the
same turn instead.
```

### Add negative examples

Invalid:

```text
Let me run the Toronto query.
```

Valid:

```text
<structured run_research_batch tool call>
```

Invalid:

```text
I now have enough evidence. Here is the answer...
```

Valid:

```text
<structured submit_final_report tool call containing the answer>
```

### State what a guard error means

After a guard error:

```text
Do not narrate a revised plan. If research remains allowed, make the revised tool call
immediately. If research is closed, call get_verified_sources and submit_final_report.
```

### Avoid duplicating finalization rules

Consolidate the authoritative end-of-run contract into one prominent section and refer to it from tier-specific sections. Repetition can help, but inconsistent or widely separated instructions increase protocol complexity.

## Error-handling recommendations

### Provider errors

Detect typed retry exhaustion before it becomes assistant content.

Expected behavior:

```text
provider exception
    → retry policy
    → if exhausted and evidence exists: partial fallback
    → otherwise: failed job
```

Never:

```text
provider exception string
    → AIMessage.content
    → salvage
    → final_report
```

### HTTP 429

The current logs show many transient capacity/rate-limit responses. They are not necessarily fatal because automatic retry often succeeds.

Track:

- total 429 attempts;
- requests affected;
- retry-success rate;
- retry-exhaustion rate;
- added latency.

### HTTP 500

At least one current-run request exhausted retries and returned the failure string as content. This should become a typed failure or partial fallback.

### Workflow timeout

The 2400-second deadline worked as intended in that it bounded the job. However, terminal semantics should explicitly say `partial`, not generic `success`.

## Proposed implementation sequence

### Phase 1: correctness gate

1. Stop accepting arbitrary last assistant content for research requests.
2. Add tests that intermediate narration is rejected.
3. Preserve writer-file completion.
4. Add explicit partial metadata for timeout/recursion fallback.
5. Update evaluation to recognize the intended completion contract.

Outcome:

- no more false successful reports;
- some jobs may fail until recovery is added.

### Phase 2: bounded continuation recovery

1. Detect no-tool/no-output termination.
2. Resume once with an internal “make the intended tool call” instruction.
3. Preserve request-wide counters and state.
4. Prevent more than one or two continuation recoveries.

Outcome:

- recovers the common “Let me run...” GLM failure.

### Phase 3: finalize-only recovery

1. Implement constrained tool visibility.
2. use low-temperature GLM.
3. test forced named tool choice.
4. persist the result via the canonical finalization mechanism.

Outcome:

- recovers completed synthesis and budget-exhaustion cases.

### Phase 4: bounded fallback finalizer

1. Implement the reserved timeout.
2. use compact durable evidence.
3. validate structured output.
4. fall back deterministically on failure.

Outcome:

- produces honest partial results even when GLM remains non-compliant.

### Phase 5: observability and evaluation alignment

1. Emit one canonical terminal event.
2. distinguish complete/partial/failed.
3. update evaluator acceptance logic.
4. report recovery counts and causes.

## Required tests

### Extraction and acceptance tests

- accepts non-empty `/shared/final_report.md`;
- accepts non-empty `/shared/output.md`;
- rejects arbitrary last assistant text;
- rejects “Let me retry...”;
- rejects a provider-error string;
- does not treat writer marker as report content;
- accepts deterministic partial only with explicit partial metadata.

### Continuation recovery tests

- no-tool narration triggers one recovery;
- recovery retains prior messages and files;
- recovery retains verified-source registry;
- recovery retains loop-guard counters;
- recovered tool call proceeds normally;
- recovered `submit_final_report` ends through `return_direct`;
- repeated no-tool response stops after configured retry count.

### Finalize-only tests

- search tools are hidden;
- planning/delegation tools are hidden;
- source whitelist is available;
- finalizer is available;
- forced tool choice is passed when enabled;
- valid report is persisted;
- incomplete evidence produces explicit gaps;
- timeout routes to deterministic partial.

### Writer-path tests

- actual writer delegation plus `/shared/output.md` is accepted;
- writer marker without `/shared/output.md` fails;
- `/shared/output.md` without expected writer provenance is logged and policy-tested;
- canonical terminal event is emitted for writer output.

### Provider-error tests

- retry exhaustion raises a typed exception;
- error text is never accepted as a report;
- evidence-present provider failure routes to partial fallback;
- evidence-absent provider failure marks the job failed.

### Timeout and recursion tests

- workflow timeout cancels the graph;
- recursion exhaustion cancels the graph;
- deterministic partial is persisted;
- terminal status is `partial`;
- termination reason is recorded;
- canonical terminal event is emitted exactly once.

### Evaluation adapter tests

- accepts inline finalizer completion;
- accepts writer completion if that remains a supported product contract;
- accepts explicit partial only when evaluation policy permits it;
- rejects intermediate narration;
- rejects empty reports;
- rejects provider-error reports;
- reports error reason separately from accuracy.

## Metrics for the next evaluation

Record at least:

| Metric | Purpose |
|---|---|
| Total attempted | Denominator |
| Server complete | Runtime availability |
| Server partial | Bounded but incomplete work |
| Server failed | Hard failures |
| Inline finalizer count | Normal inline completion |
| Writer-file completion count | Normal writer completion |
| No-output first-pass count | Raw GLM protocol failure rate |
| Continuation recovery attempted | Recovery load |
| Continuation recovery succeeded | Recovery effectiveness |
| Finalize-only recovery attempted | Finalization pressure |
| Finalize-only recovery succeeded | Forced finalization effectiveness |
| Deterministic fallback count | Residual failure rate |
| Provider retry exhaustion | Infrastructure/model reliability |
| Workflow timeout count | Budget adequacy |
| Invalid terminal content rejected | Correctness protection |
| Mean added recovery latency | Operational cost |
| Accuracy by origin | Quality impact |
| Accuracy by tier | Tier effectiveness |

The primary success metric should be:

```text
valid_terminal_reports / attempted_requests
```

not:

```text
server_status_success / attempted_requests
```

## Suggested acceptance criteria

A remediation is ready for a 90-sample rerun when:

- zero intermediate narration strings are accepted as final reports;
- zero provider-error strings are accepted as final reports;
- every accepted normal completion has a positive output signal;
- every timeout/recursion result is explicitly marked partial;
- writer-path behavior and evaluator expectations agree;
- recovery is bounded;
- all new unit tests pass;
- valid completion rate materially exceeds the current approximately 56% explicit-inline-submission rate;
- accuracy is calculated only over valid, policy-accepted reports.

## Immediate operational recommendations

Until the runtime fix lands:

1. Keep the evaluation-side validation that prevents obvious no-submit inline outputs from being scored.
2. Do not interpret server `success` as report success.
3. Review writer-path compatibility before requiring `submit_final_report` universally.
4. Report:
   - valid completions;
   - invalid/unsubmitted outputs;
   - partial timeouts;
   - provider failures;
   separately.
5. Consider a low-temperature, non-thinking orchestrator configuration for the next controlled experiment.
6. Do not rely on that configuration change as the permanent correctness boundary.

## Final conclusion

GLM is exposing a latent weakness in the adaptive workflow's termination contract. The model occasionally narrates an intended action without encoding it as a tool call. A standard ReAct graph then terminates, and AI-Q's permissive salvage logic converts the narration into a successful final report.

The core defect is therefore:

```text
absence of a tool call is treated as completion
AND
absence of a positive final report is repaired by accepting arbitrary text
```

The durable correction is:

```text
positive completion signal
    + strict acceptance
    + bounded continuation recovery
    + constrained finalization recovery
    + deterministic partial fallback
    + explicit terminal status
```

Prompt and decoding changes should be used to reduce how often recovery is needed. They must not be the mechanism that decides whether a report is real.
