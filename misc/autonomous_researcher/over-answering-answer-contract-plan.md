# Pattern 2 — An answer contract for the autonomous researcher

**Problem:** the agent over-answers. Nine trials in DSQA-90 job `2026-08-20__21-44-00` had
*perfect recall and still scored zero*, purely on precision.

**Constraint that shapes the whole design:** long-report capability is a first-class AI-Q
capability and must not be removed, narrowed, or capped. Answer shape must be **query-driven,
not harness-enforced**. Citations stay exactly as they are.

**Source of the measurement:** `ai-q-harbor-evals` job `jobs/2026-08-20__21-44-00/`
(DSQA-90, 90 trials, 1 error) and `pattern_recheck_recommendations.md` §4 rec 1.
Throughout, `aiq:` means this repo; other paths are relative to the `ai-q-harbor-evals` root.

**Status: IMPLEMENTED** (prompt phases + the repair-pass fix). This document has been updated to
describe what actually shipped, not what was proposed. Where the implementation diverged from the
original design the divergence is called out inline and the reasoning is kept, so the record stays
useful. The one deferred item is the tool-schema change (§6, Phase 4), which remains conditional on
measurement.

---

## 0. TL;DR

- Over-answering is **not** verbosity. Answer length does not predict correctness in this run, so
  nothing here shortens a report.
- The defect is that **there is no boundary between "the answer" and "the report"**. Report
  furniture — excluded-candidate rows, "Key Takeaways", even the `## Sources` heading — gets read
  as answer items.
- The rule that prevents this **already exists** at `orchestrator.j2:64`. It is simply absent from
  the shallow path (32/90 trials, the worst offender) and contradicted on the writer path.
- Fixed on a rising risk ladder: prompt-only on the inline path → scoped prompt injection on the
  shallow path → writer reconciliation. The *optional*, gated, additive tool-schema change was
  **not** needed and remains deferred.
- **Tracing every exit during implementation found eight answering points, not three** (§2.10). One
  of them — the shallow citation-repair pass — is a second LLM call that rewrites the finished
  answer and would have silently dissolved the contract. That became a fourth change.
- Shipped as **4 files, 170 insertions, 4 deletions**: three Jinja prompts, one Python constant plus
  helper, and one line in the citation-repair instruction. It matches AI-Q's stated vision
  ("agent behavior is driven by workflow YAML, Jinja2 prompts … not by hard-coded logic").
- Realistic ceiling is **+9 tasks (~+10 FC points)**, not the `+12.2` claimed in the source
  recommendation. §1.2 explains why that number is inflated. **Not yet measured** — the eval gates
  in §9 are still outstanding.

---

## 1. The problem, restated from measurement

### 1.1 Length is not the lever — so long reports are not at risk

Across the 89 graded trials (`artifacts/answer.txt` × `verifier/grading.json`):

| answer length (chars) | n | fully-correct | mean excessive items |
|---|---|---|---|
| 0 – 1,500 | 41 | 0.488 | 0.71 |
| 1,500 – 3,000 | 33 | 0.303 | 1.15 |
| 3,000 – 6,000 | 13 | **0.538** | 1.00 |
| > 6,000 | 3 | 0.000 | 3.33 |

Median answer is 1,557 chars; mean 1,976; max 10,278. The relationship is **non-monotonic** —
3–6k-char answers are the *best* performing band. Only the three answers over 6k score zero, at
n=3 that is noise.

**Conclusion: there is no evidence in this run that shortening answers helps.** Any proposal that
caps length is unsupported by the data *and* violates the stated constraint. This plan caps
nothing.

### 1.2 Most "excessive answers" are wrong answers, not extra ones

32 trials carry at least one `excessive_answers` entry. They are three different failures:

| bucket | n | what it is | does an answer contract help? |
|---|---|---|---|
| `recall == 1.0`, `precision < 1.0` | **9** | true over-answering — every gold item found, extras added | **Yes — this is the target** |
| `recall == 0.0` | 16 | plain wrong; the grader labels the single incorrect answer "excessive" | **No** |
| `0 < recall < 1.0` | 7 | partly wrong *and* padded | Partially |

The 9 true cases: **0208, 0212, 0216, 0230, 0256, 0314, 0398, 0755, 0824** — identical to the
"perfect recall but zero" list in the source recommendation.

The 16 recall-zero cases — 0084, 0103, 0206, 0242, 0300, 0309, 0316, 0396, 0474, 0519, 0529, 0532,
0588, 0592, 0793, 0813 — are retrieval and reasoning failures. **No answer contract can recover
them.** Of the 9 regressions the source recommendation attributes to extras (0212, 0230, 0316,
0529, 0532, 0588, 0644, 0793, 0813), only **three** (0212, 0230, 0644) have non-zero recall.
That is why the `+12.2 pts` projection is inflated and this plan targets **+9 tasks ≈ +10 points**.

### 1.3 The actual mechanism: report furniture is read as answers

- **0256** is decisive. Grader-flagged excessive items include `"The chart visualization of stats"`,
  `"The 'Difference' column values"`, `"The 'Key Takeaways' summary"` and **`"The 'Sources'
  section"`**. Report *scaffolding* was parsed as answer content. The answer also opens with leaked
  scratch text: *"Now I have all the information needed. Let me compile the comparison."*
- **0212** (Albacore / lifespan ≤ 20y) and **0230** (O'Keeffe / NM landscapes) both lost on rows
  the agent had *explicitly marked as excluded* — `Bluefin … ✗ (excluded)`, `Cow's Skull … is a
  still-life composition rather than a landscape`. Good research writing, parsed as answers.
- **0314** lost on an exhaustive taxonomic-synonym table: 17 excessive items, precision 0.11,
  recall 1.0.
- **0824** is the one genuine set-discipline failure: it asserted all five states qualified where
  gold has three. This is the case a self-check catches.

**There is no boundary between the answer and the report.** That is the defect.

---

## 2. What exists today (facts this plan builds on)

1. **The rule already exists, on one path only.**
   `aiq:src/aiq_agent/agents/autonomous_researcher/prompts/orchestrator.j2:64`:
   > Include **only qualifying members**. Do not name rejected candidates, close alternatives,
   > historical synonyms, deprecated or invalid names, or "commonly confused with" entries —
   > **not even to say that they do not qualify**.

   Pinned by `tests/aiq_agent/agents/autonomous_researcher/test_factory.py:561-562`.

2. **Per-exit behaviour in this run** (derived from `agent/trajectory.json` × `grading.json`):

   | exit | n | FC | % trials with extras | mean extras |
   |---|---|---|---|---|
   | inline (`submit_final_report`) | 57 | 0.456 | 33.3 % | 0.70 |
   | `shallow-researcher` | 32 | **0.344** | **40.6 %** | **1.56** |
   | `writer-agent` | **0** | — | — | — |

   The path **with** the rule performs better on both axes. The path **without** it is worst.
   `writer-agent` never ran, so any writer-side change is unvalidated by this run.

3. **The shallow path has no answer-shape discipline at all.**
   `aiq:src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2` (99 lines) contains no
   qualifying-set rule. It *does* actively push furniture: "present them as a chart, not just prose
   or a table" (`:46`), "At most 3 charts per answer" (`:51`) — which is precisely what produced
   0256's grader-confusing output.

4. **That shallow prompt file is shared by 9+ shipped configs** — `config_cli_default.yml`,
   `config_mcp.yml`, `config_web_default_guardrails.yml`, `config_frontier_models.yml`,
   `config_web_azure_ai_search.yml`, `config_chat_researcher_temp.yml`, `shallow_nemotron_ultra.yml`,
   `shallow_deep_nemotron_ultra.yml`. **Editing it globally is a product-wide change.**
   This plan does not touch it. See §4.3.

5. **A scoped injection point already exists and is unused.**
   `ShallowResearcherAgent.__init__` accepts `system_prompt: str | None = None`
   (`aiq:src/aiq_agent/agents/shallow_researcher/agent.py:246`), resolved as
   `self.system_prompt = system_prompt or self._load_system_prompt()` (`:275`) and still
   Jinja-rendered at `:390`. The autonomous sub-run at
   `aiq:src/aiq_agent/agents/autonomous_researcher/subagents/shallow.py:244-250` **does not pass
   it**. This is the whole lever for Phase 2.

6. **The writer contract says the opposite.**
   `aiq:src/aiq_agent/agents/autonomous_researcher/prompts/writer.j2:76`:
   > Err on the side of more useful information rather than less, while staying focused on the
   > requested answer shape.

   Correct for the body, wrong for the answer set.

7. **A query-driven shape declaration already exists — on the wrong path.**
   `aiq:src/aiq_agent/agents/autonomous_researcher/prompts/planner.j2:55-59` defines
   `answer_type ∈ {long_form_report, brief_answer, table}`. It reaches the writer through
   `/shared/plan.json` (`writer.j2:6, :49`). But the planner route ran only 10/90 trials and scored
   worst (FC 0.200). The inline and shallow paths — 89/90 answers — have no declared shape.

8. **Citation machinery is independent of body structure.**
   `aiq:src/aiq_agent/agents/autonomous_researcher/agent.py:468` appends `## Sources` at the end;
   `get_verified_sources` is the sole whitelist (`orchestrator.j2:121-126`, `writer.j2:92-96`).
   Nothing in this plan touches either.

9. **Prompt-size discipline is enforced.** `orchestrator.j2:127` marks a
   `KV CACHE BOUNDARY — dynamic content below`. **All new static prompt text must go above it** or
   it breaks prefix caching on every turn. `test_factory.py:354` caps the research-loop section at
   6 content lines / 1,600 chars; the section this plan edits is not under that cap but the same
   terseness norm applies.

### 2.10 Every answering point — traced during implementation

The three-exit model in items 1–3 above is what the design was written against. Tracing the code
during implementation found **eight** paths that can produce user-facing answer text. The extra
ones matter because a contract that only covers three of them is not a contract.

| # | Answering point | Governed by | Outcome |
|---|---|---|---|
| 1 | Inline — `submit_final_report` (`tools/finalize.py:137` → `commit_final_report:86-109`) | `orchestrator.j2` | Change 1 |
| 2 | Writer — `/shared/output.md` via `FinalReportCommitMiddleware._commit_write` | `writer.j2` | Change 3 |
| 3 | Shallow — zero orchestrator turns, committed by `ShallowFinalizationMiddleware` (`custom_middleware.py:587-613`) | shared shallow template | Change 2 (injected) |
| 3a | **Citation-repair rewrite** — `_repair_missing_citations` (`shallow_researcher/agent.py:312-362`), a 2nd LLM call that fully rewrites the answer; prompt hardcoded in Python | hardcoded Python string | **Change 4 — new** |
| 4 | Salvage — `_salvage_inline_report` (`agent.py:324-344`) when neither exit committed | `orchestrator.j2`, but unenforced free text | Inherits Change 1; best-effort by nature |
| 5 | Reused report on timeout/recursion — `_build_partial_result` (`agent.py:517`) | whichever prompt authored it | Inherits 1/2/3 |
| 6 | Deterministic partial — `_render_deterministic_partial` (`agent.py:414-475`) | **none — Python** | **No change, deliberately.** A partial-result body has no discrete answer set, so it correctly gets no `## Answer` section. |
| 7 | Tool-unavailability notice (`register.py:339-343`) | none — Python | No change; not an answer |
| 8 | Post-commit rewrites — `verify_citations` → `sanitize_report` → `append_artifact_index` → `_replace_last_message_content` (`agent.py:619-727`) | none — Python | **Verified safe.** `sanitize_report` (`common/citation_verification.py:1344-1384`) only normalises the source section and strips body URLs; it does not touch body headings, so `## Answer` survives. |

Confirmed **non**-answer paths, checked and excluded: `_failure_notice` (`subagents/shallow.py:88-110`)
never reaches the user because `capture.status="failed"` makes `has_report` False; `run_research_batch`
and `task(researcher-agent)` are blocked from writing the report by `FinalReportOwnershipGuardMiddleware`.

**Change 4 — the citation-repair rewrite.** This is the hazard the three-exit view misses. The
repair pass fires whenever a shallow answer fails the citation contract, and its instruction was
*"Preserve the answer's meaning, remove unsupported claims"* — **saying nothing about structure**,
leaving it free to dissolve the `## Answer` section it was handed. One clause was added:

> Preserve the answer's meaning, **and preserve the draft's existing headings and section order
> verbatim.** Remove unsupported claims, …

This is the only place the change touches the shared `shallow_researcher` package. It is strictly
conservative — a repair pass should never restructure — and is an improvement for every consumer,
not just this one.

---

## 3. Goals and non-goals

### Goals

1. Raise precision on the 9 true over-answering tasks without losing their recall.
2. Keep long-report capability **fully intact** — no length cap, no forced brevity, no truncation.
3. Make answer shape follow from the **query**, decided by the answering agent, never imposed by
   the harness or a config switch.
4. Leave citations byte-for-byte unchanged in mechanism and coverage.
5. Every phase independently shippable and independently revertable.
6. Stay prompt-driven, consistent with AI-Q's vision; keep code changes minimal, additive and
   backward-compatible.

### Non-goals (this plan)

- Fixing the 16 recall-zero failures — that is retrieval/routing work (source recommendation
  rec 2: fetch budget + named-source trigger). Explicitly out of scope.
- Hedging (rec 3), spawn-route consolidation (rec 4), planner investigation (rec 5), stagnation
  guard (rec 6), timeout fallback (rec 7). Independent workstreams.
- Touching `sources/`, the data-source registry, or any retrieval tool.
- Changing the shared `shallow_researcher` package used by shipped product configs.
- Any change to `enable_citation_verification` (source recommendation rec 8: leave alone).

---

## 4. Design — the answer contract

One idea in three parts: **declare the shape, separate the answer from the report, relocate the
exclusions.**

### 4.1 Declare the shape — query-driven, asymmetric default

The answering agent decides, on its first turn, from the question's own wording:

- **Discrete-target question** — names one thing to return: `which`, `list all`, `how many`,
  `identify`, `what is the`, "exhaustive list". → emit an **answer block** (§4.2), body length
  still free.
- **Open/exploratory question** — "write me a report on X", "compare and analyse", "assess". →
  **no answer block.** The report *is* the answer. Unchanged behaviour.

**The default is asymmetric and this is the single most important safety property in the plan:**
when the shape is ambiguous, **resolve to the richer form**. Narrow only on explicit
discrete-target syntax. Never infer "this looks simple" and downgrade.

> Rationale to carry into the prompt comment: a wrongly-long report costs tokens; a wrongly-short
> one costs the deliverable. These are not symmetric errors.

Note this is *recognition*, not classification-by-a-model-call — no new LLM turn, no new tool, no
config flag. It reuses the vocabulary already in `planner.j2:55-59` so the three paths converge on
one taxonomy rather than inventing a second.

### 4.2 Separate the answer from the report

When a discrete target is present, the report opens with a short, delimited block:

```markdown
## Answer

- Skipjack tuna [1]
- Yellowfin tuna [2]

<the full report follows, at whatever length the question warrants>
```

Rules for the block, and **only** the block:
- Exactly the entities that pass every stated filter. Nothing else.
- No excluded candidates, no status columns, no "it depends on interpretation", no working table.
- Inline `[N]` citations exactly as everywhere else.

Rules for the body: **unchanged.** All of `writer.j2:68-74` — cross-synthesis, developed
paragraphs, conflict surfacing, nuance, tables — continues to apply verbatim.

This is what fixes 0256 (furniture no longer sits where an answer is looked for), 0230 and 0212
(excluded rows leave the answer list), and 0314 (the synonym table stays in the body).

### 4.3 Relocate the exclusions — a report-quality *gain*

Today's `orchestrator.j2:64` forbids naming rejected candidates **"not even to say that they do not
qualify"**. On the inline path — 57 of 90 answers this run — that is a live tax on report quality:
it deletes exactly the analysis a good research report should contain.

Replace the ban with **relocation**:

> Qualifying entities appear in the `## Answer` block and nowhere else in list or table form.
> Rejected candidates, close alternatives, historical synonyms and near-misses belong in the body
> under an explicit `### Considered and excluded` heading, with the reason each one fails. Never
> mix them into the answer block.

0212's *"Bluefin lives 32–40 years, beyond the 20-year threshold"* and 0230's *"Cow's Skull is a
still-life, not a landscape"* are good research writing. Today the rule deletes them. Under this
plan they are **restored** to the body.

**This is why the change is net-positive for reporting quality, not merely neutral.**

### 4.4 The self-check

One line, at submission, no extra retrieval and no extra turn:

> Before submitting: for each item in your answer block, name the filter it satisfies. If you
> cannot, remove it.

0824 is exactly this failure — it asserted all five states qualified where gold has three.

---

### 4.5 Position — decided: top of the report

The block leads. Open question 2 in the original draft left this undecided; it was settled in favour
of the top for readability and instruction-following, and because lead-with-the-verdict is already
house style (`shallow_researcher/prompts/researcher.j2:46`). The anchoring risk this creates is
Risk 1 in §10, and its fallback — move the block to the end — remains available without a redesign.

## 5. What this deliberately does not change

| Untouched | Why |
|---|---|
| Report length, anywhere | §1.1 — length does not predict correctness |
| `writer.j2:68-74` depth/synthesis contract | The body contract is not the defect |
| `get_verified_sources`, `## Sources`, `agent.py:468` | Citations are out of scope by constraint |
| `enable_citation_verification` (`config_autonomous_frag.yml:123`) | Source rec 8: no evidence it predicts correctness |
| `shallow_researcher/prompts/researcher.j2` | Shared by 9+ shipped configs — §2.4. Byte-identical; the contract is injected per sub-run via `system_prompt=` instead. |
| Rest of `shallow_researcher/**` | Only `agent.py:333` changed — the one-clause repair fix (§2.10, Change 4) |
| `autonomous_researcher/factory.py` | **Not touched at all** — see the §6 Phase 2 divergence note |
| `sources/`, data-source registry, retrieval tools | Different workstream |
| Budgets, loop guards, routing, `max_direct_source_calls` | Source recs 2 and 6 |
| `submit_final_report` signature | Schema change deferred; Phase 4 was not needed and remains conditional |

---

## 6. Phased rollout — the risk ladder

Each phase is independently shippable, independently measurable, independently revertable, and
strictly larger in code surface than the one before.

### Phase 1 — Inline path, prompt-only (**zero code**) — ✅ SHIPPED

- Edit `orchestrator.j2` §"What goes in the answer you write yourself" (lines 59-67), **above** the
  KV-cache boundary at `:127`.
- Add: the discrete-target recognition rule (§4.1), the answer-block contract (§4.2), the
  self-check (§4.4). Replace the blanket exclusion ban with relocation (§4.3).
- **Keep the literal phrase `only qualifying members`** so `test_factory.py:562` continues to pass;
  extend around it rather than rewriting it.
- Covers 57/90 trials. Zero blast radius outside the autonomous agent. Revert = one file.

**As shipped:** the section now states both arms of the shape rule, relocates rejected candidates to
`### Considered and excluded`, adds the per-entity filter self-check, and closes with an explicit
*"no length target, and nothing here asking you to be brief"* — the sentence that makes the
non-brevity constraint assertable. A Jinja comment above it records the measurement, at zero token
cost.

### Phase 2 — Shallow sub-run, scoped injection — ✅ SHIPPED (**smaller than planned**)

The highest-value phase: 40.6 % of shallow trials over-answer, mean 1.56 extras.

- The contract is passed through the **already-existing, currently-unused** `system_prompt`
  parameter (§2.5).

> **Divergence from the design — scope reduced.** The plan threaded a new optional
> `system_prompt` kwarg on `build_shallow_researcher_subagent` and rendered the prompt in
> `factory.py`. That turned out to be unnecessary: the contract is a fixed constant owned by the
> autonomous agent and is not configurable, so it is built *inside* `subagents/shallow.py`.
> **`factory.py` is untouched**, `build_shallow_researcher_subagent` keeps its exact signature, and
> the returned spec dict is unchanged — which removed the risk of breaking the ~30 tests that call
> that builder with a fixed explicit kwarg set.

As shipped, in `subagents/shallow.py` only:
- `SHALLOW_ANSWER_CONTRACT` — a module-level, deliberately Jinja-inert string (mirrors the
  `FORCE_SEARCH_GUIDANCE` precedent at `clarifier/agent.py:96-102`).
- `_shallow_system_prompt()` — loads the shared template via `load_prompt(SHALLOW_AGENT_DIR /
  "prompts", "researcher")` and appends the contract. Returns `None` on any load failure, which
  hands construction back to `ShallowResearcherAgent._load_system_prompt()` and its own inline
  fallback, so a missing template degrades instead of failing the request.
- One argument at the `ShallowResearcherAgent(...)` construction.

- **The shared template stays byte-identical.** The 9+ shipped configs are unaffected because they
  construct without `system_prompt` — proven by `test_default_model_profiles.py:155-164`
  (`test_shallow_profiles_use_the_shared_citation_prompt`), which still passes.
- The contract must stay Jinja-inert: `render_prompt_template` uses `StrictUndefined` and the
  shallow render site passes only `tools`/`user_info`/`current_datetime`/`available_documents`, so a
  stray `{{ }}` would raise at agent-node time, not at build time. A test pins this.
- Revert = drop one argument.

### Phase 3 — Writer path reconciliation (**zero code**) — ✅ SHIPPED

- `writer.j2`: scope line 76 ("err on the side of more useful information") explicitly to the
  **body**, and add the answer-block contract keyed off `answer_strategy.answer_type`, which the
  writer already reads (`:6`, `:49`).
- Ranked last among the prompt phases because `writer-agent` ran **0/90** trials — lowest
  confidence, and therefore lowest priority, but cheap and it removes a standing contradiction.
- **Corrected during implementation:** the 0/90 figure is not evidence the writer is unimportant.
  It is configured and reachable (`config_autonomous_frag.yml:127` sets `writer_llm`); it simply
  never fires because **DSQA-90 contains no report-shaped questions**. The writer *is* the
  long-report path, which makes it the exit where the "don't take away long reports" constraint
  actually bites — and the reason its body-depth contract now has explicit test coverage.

### Phase 4 — Structured field — ⏸ NOT SHIPPED, still conditional

Only if precision on the recall-1.0 subset has not moved after Phases 1–3.

- Add `final_answer: list[str] | None = None` to `submit_final_report`
  (`tools/finalize.py:137`), defaulting to `None`.
- When present, `commit_final_report` renders it as the `## Answer` block ahead of `markdown`.
  When absent, behaviour is **byte-identical to today**.
- Backward compatible by construction: every existing caller and every existing test passes an
  unchanged signature.

> Deliberately last. AI-Q's vision is that behavior is driven by prompts and YAML, not hard-coded
> logic — a tool-schema change is the least vision-aligned instrument available here, so it is used
> only if the prompt-level instruments demonstrably fail.

---

## 7. File-by-file change list — as built

`4 files changed, 170 insertions(+), 4 deletions(-)` in `src/`, plus three test files.

### Source
| File | Change |
|---|---|
| `autonomous_researcher/prompts/orchestrator.j2` | `## What goes in the answer you write yourself`: shape rule, `## Answer` contract, exclusion relocation, self-check, non-brevity closer. All above the KV boundary at `:127`. |
| `autonomous_researcher/subagents/shallow.py` | `SHALLOW_ANSWER_CONTRACT` constant + `_shallow_system_prompt()` helper + one constructor argument (~55 lines, mostly rationale comments). |
| `autonomous_researcher/prompts/writer.j2` | Answer-block bullet in `## Synthesis Contract` keyed on `answer_strategy.answer_type`; `:76` scoped to the body rather than deleted. |
| `shallow_researcher/agent.py` | **1 line** — structure-preserving clause in the citation-repair `repair_request` (`:333`). |

### Tests
| File | Change |
|---|---|
| `tests/…/autonomous_researcher/test_factory.py` | Extended `test_states_the_answer_set_contract`; added `test_answer_shape_is_query_driven_not_length_capped`, `test_the_shallow_sub_run_carries_the_answer_contract`, `test_the_shallow_contract_is_jinja_inert`. |
| `tests/…/autonomous_researcher/test_writer_prompt.py` | **New file.** The writer prompt had zero content tests, which is how the contradiction at `:76` survived unnoticed. |
| `tests/…/shallow_researcher/test_agent.py` | `test_citation_repair_preserves_the_draft_structure`. |

### Not modified — as promised
`autonomous_researcher/factory.py` · `autonomous_researcher/tools/finalize.py` ·
`shallow_researcher/prompts/researcher.j2` · `configs/**` · `sources/**` · `frontends/**`.

> Per repo convention (well-commented code): every prompt block carries a Jinja comment naming the
> measurement that motivated it, and every code change carries a docstring or comment explaining the
> *why*, not the *what*. Jinja comments (`{# … #}`) cost zero tokens — see the precedent at
> `orchestrator.j2:27-29`.

---

## 8. Test plan — as built, with results

### What the tests pin

**The contract is present** — `test_states_the_answer_set_contract` keeps the two pre-existing
pinned strings (`What goes in the answer`, `only qualifying members`) and adds
`### Considered and excluded` and the self-check.

**The constraint is present, as an assertion** — `test_answer_shape_is_query_driven_not_length_capped`
asserts both arms of the shape rule, that ambiguity resolves to *"treat it as NOT discrete and write
the fuller answer"*, and that the section contains no `be brief` / `be concise` / `word limit` /
`no more than` / `maximum length` artifact. This is the guard against a future edit quietly turning
the contract into a brevity mechanism. (The scan skips the section's own disclaimer sentence, which
necessarily quotes the phrasing it forbids.)

**The injection works and cannot explode at runtime** —
`test_the_shallow_sub_run_carries_the_answer_contract` asserts the shared template is still the base
and the contract is appended on top; `test_the_shallow_contract_is_jinja_inert` catches a `{{ }}`
that would otherwise raise at agent-node time rather than build time.

**Report depth survives** — `test_writer_prompt.py` pins six body-depth rules
(cross-synthesis, analytical narrative, conflict surfacing, nuance, tables, cited-vs-inferred) plus
`long_form_report` getting *"no `## Answer` section and no length target"*.

**Repair cannot restructure** — `test_citation_repair_preserves_the_draft_structure` captures the
messages sent to the repair LLM and asserts the structure-preserving clause is in them.

### Regression guard — held

The design set a stop condition: *"if any shallow_researcher test needs touching, Phase 2 has leaked
out of scope."* Phase 2 did not touch them. The one shallow_researcher test added is for Change 4,
a separately-agreed fix on a different code path, and
`test_shallow_profiles_use_the_shared_citation_prompt` — which asserts a default-constructed shallow
agent still uses the shared template verbatim — **passes unchanged**. That is the direct proof the
shipped configs are unaffected.

### Results

```
uv run pytest -q
2831 passed, 79 skipped, 4 failed
```

The 4 failures are **pre-existing and unrelated**: `test_default_model_profiles` on three scratch
configs (`config_chat_researcher_temp.yml`, `config_deep_research_only_temp.yml`,
`config_shallow_frag.yml`) and the deprecated `inference-api.nvidia.com` endpoint reference. None
touches a file in this change.

`ruff check` and `ruff format --check` are clean on every modified file. The two repo-level ruff
errors are in untracked `misc/` scratch scripts.

### Commands

```bash
uv run pytest tests/aiq_agent/agents/autonomous_researcher -q
uv run pytest tests/aiq_agent/agents/shallow_researcher -q
uv run pytest tests/aiq_agent/test_default_model_profiles.py -q -k shared_citation_prompt
uv run ruff check . && uv run ruff format --check .
```

### Still outstanding

- **End-to-end smoke has not been run** (needs a live backend and API keys). Two probes:
  a discrete question should produce a leading `## Answer` block; *"Write me a report on X"* should
  produce **no** answer block and a full-length report. The second is the regression that matters.
- The §9 eval gates are unmeasured.

---

## 9. Validation and gates — OUTSTANDING

> None of the gates below has been run. The code is in place and unit-tested; the behavioural
> claims remain unmeasured. A DSQA-90 re-run needs an image rebuild and pin bump first.

**DSQA-90 cannot measure report quality.** All 90 prompts are factoid/set lookups — 13 begin
"According to…", 12 "Which…", plus "list all", "identify", "provide an exhaustive list". Zero are
report-shaped. It measures the *benefit* here, never the *risk*.

| metric | harness | role | gate |
|---|---|---|---|
| Precision on the `recall == 1.0` subset (n=9) | DSQA-90 | **primary benefit** | must improve |
| FC on 0208, 0212, 0216, 0230, 0256, 0314, 0398, 0755, 0824 | DSQA-90 | benefit detail | target ≥ +5 of 9 |
| Overall FC | DSQA-90 | context | must not regress |
| Answer-length distribution (median ≈ 1,557 chars) | DSQA-90 | **constraint check** | must **not** collapse |
| RACE **Comprehensiveness** | DeepResearch Bench | **primary risk gate** | must not regress |
| RACE Instruction Following / Readability | DeepResearch Bench | expected to improve | — |
| FACT citation accuracy + effective citations | DeepResearch Bench | citation guard | must not regress |

The length-distribution row is the constraint expressed as a number: **if median answer length
drops materially, the change has become a brevity mechanism and must be reverted**, regardless of
what F1 does.

```bash
dotenv -f deploy/.env run nat eval \
  --config_file frontends/benchmarks/deepresearch_bench/configs/config_deep_research_bench.yml
```

> **Operational note:** harbor eval runs execute frozen agent code from a pinned Docker image while
> reading config live from the host. These are `src/` changes, so each phase needs an **image
> rebuild and pin bump** before its DSQA-90 run — a config-only re-run will silently measure the
> old code.

---

## 10. Risks

| # | Risk | Likelihood | Mitigation | Fallback |
|---|---|---|---|---|
| 1 | **Anchoring** — having stated the answer up front, the agent writes a thinner body. Hits RACE Comprehensiveness. | Medium | Body contract untouched; block is a lead summary, and lead-with-the-verdict is already house style (`shallow_researcher/prompts/researcher.j2:46`) | **Move the answer block to the end of the report.** Same precision benefit, no lead-anchoring. Cheap, not a redesign. |
| 2 | Shape misrecognised — a broad query gets an answer block and reads as a factoid | Medium | Asymmetric default (§4.1): ambiguity always resolves to the richer form | Tighten the trigger to explicit `which` / `list all` / `how many` syntax only |
| 3 | ~~Phase 2 leaks into the shared shallow package~~ **RESOLVED** | — | Shipped without touching the shared template or `factory.py`; `test_shallow_profiles_use_the_shared_citation_prompt` passes unchanged, proving the shipped configs are unaffected | n/a |
| 4 | Prompt growth breaks KV caching or the terseness norm | Low | All static text above `orchestrator.j2:127`; asserted in tests | Move block below the section, trim |
| 5 | Grader still counts body items as answers despite the block | Medium | This is the core hypothesis being tested — Phase 1 measures it before any further investment | If Phase 1 shows no precision movement, stop; the defect is in the grader's parse, not the agent, and no further phase is justified |
| 6 | Gains are inside run-to-run noise | **High** | 26/90 tasks flipped between the two prior runs; effects under ~10 points are not separable | Target is +9 tasks, at the edge of separability — **plan for two runs**, or accept the result as directional only |
| 7 | ~~Citation repair silently dissolves the `## Answer` section~~ **RESOLVED** | — | Found by tracing every exit (§2.10); fixed with the one-clause Change 4 and pinned by `test_citation_repair_preserves_the_draft_structure` | n/a |
| 8 | Salvage path (`_salvage_inline_report`) emits free text with no tool call, so nothing enforces the contract there | Low | Same prompt governs it, so the model is instructed; but it is unenforceable by construction | Accept — it is a last-resort path, and greetings/capability replies are deliberately routed through it |

**Honest limitation:** the RACE Comprehensiveness prediction is reasoning from the benchmark's
documented dimensions, **not a measurement**. It has not been run. Risk 1's gate exists precisely
because that claim is unverified.

---

## 11. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Cap answer length / force brevity | Violates the stated constraint, and §1.1 shows length does not predict correctness. 3–6k-char answers are the best-performing band. |
| Route more traffic to `shallow-researcher` for "simple" queries | Harness-enforced shape — exactly what the constraint forbids. Also shallow is the *worse* path here (FC 0.344 vs 0.456). |
| Edit `shallow_researcher/prompts/researcher.j2` directly | Product-wide blast radius across 9+ shipped configs (§2.4) for a POC-driven fix. |
| Lead with the tool-schema change (Phase 4 first) | Least vision-aligned instrument; hard-codes what prompts should express. Deferred and made conditional. |
| Force everything through `planner-agent` to get `answer_type` | The planner route is the worst performer in the run (FC 0.200, n=10) and the most token-expensive (79,655 output tokens/task). |
| Strip tables/charts from answers | Would remove genuine reporting capability to satisfy a grader artifact. Relocation (§4.3) achieves the precision goal without the loss. |
| Do nothing, treat it as grader strictness | Partly true — 0212's Bigeye (18 y ≤ 20 y) and 0230's *Red and Pink Rocks and Teeth* are defensible answers. But 0256 and 0314 are unambiguous agent-side defects, and 0824 is a real filter failure. |

---

## 12. Open questions

1. **Is 0212 actually wrong?** Bigeye tuna at 18 years satisfies "≤ 20 years" on the agent's own
   cited data. If several of the 9 are gold-truth disagreements rather than agent defects, the
   realistic ceiling is below +9. Worth auditing all 9 against gold **before** Phase 2.
2. ~~**Block at the top or the bottom?**~~ **Decided: top** (§4.5). Chosen for readability and
   instruction-following, and because lead-with-the-verdict is already house style. If DRB RACE
   Comprehensiveness regresses, the fallback is to move it to the end — same precision benefit,
   no lead-anchoring, no redesign.
3. **Should `answer_type` become a shared, first-class concept** across planner/inline/shallow
   rather than three parallel recognitions? Cleaner, but a larger refactor than this plan's risk
   budget allows — deferred.
4. **Two runs or one?** Risk 6 says +9 tasks sits at the edge of separability. Confirm whether the
   eval budget supports a repeat run before treating any single result as decisive.
5. **Does the repair pass fire often enough to matter?** Change 4 was made on correctness grounds,
   but the eval artifacts do not record whether citation repair ran on a given trial. Worth
   instrumenting before drawing conclusions about its contribution either way.
6. **Should the deterministic partial (`agent.py:414-475`) ever carry an answer block?** Left alone
   on the reasoning that a partial-result body has no discrete answer set. If timeout trials turn
   out to hold a usable answer, that assumption is worth revisiting.
