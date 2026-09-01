# FreshQA eval — shallow vs. shallow+deep on the local Nemotron Ultra NIM

Runbook for evaluating two AI-Q workflow configs against the FreshQA benchmark on
this server, at three scales each (10-question smoke, 500-question TEST split, full
600-question dataset).

The point of running both is the A/B: does routing some questions to the deep
researcher buy accuracy on FreshQA, and what does it cost in latency and tokens?

| | Config 1 — shallow only | Config 2 — intent-routed shallow + deep |
| :-- | :-- | :-- |
| Source config | `configs/shallow_nemotron_ultra.yml` | `configs/shallow_deep_nemotron_ultra.yml` |
| Env file | `deploy/.env_shallow` | `deploy/.env_shallow_deep` |
| Workflow | `shallow_research_workflow` | `chat_deepresearcher_agent` |
| Routing | none — every question goes to `shallow_research_agent` | `intent_classifier` → shallow *or* `deep_research_agent` |
| Search tools | `web_search_tool` | `web_search_tool` (shallow) / `advanced_web_search_tool` (deep) |

- **Dataset:** `/localhome/local-smasurekar/smasurekar/blueprint-pipeline/evaluation/rag-eval/scripts/datasets/freshqa/FreshQA_v042126.xlsx`
- **Harness:** `frontends/benchmarks/freshqa`
- **Agent model:** `nvidia/nemotron-3-ultra-550b-a55b` on the local NIM at
  `http://10.86.10.114:8999/v1` (verified reachable).
- **Judge:** `azure/openai/gpt-5.2` on `https://inference-api.nvidia.com/v1`, i.e. a
  different model and endpoint from the agent. Authenticates with the
  `NVIDIA_API_KEY` already present in both env files — see [Caveats](#caveats).

All commands run from the repo root:
`/localhome/local-smasurekar/smasurekar/aiq`

> **Status:** the environment, converter, and all six eval configs are in place and
> were checked against the installed NAT 1.8.0 schemas. Run 4
> (`shallow_deep_ultra_smoke10`) has completed; its routing behaviour is documented
> under [Step 4](#where-the-shallowdeep-routing-is-recorded) and in the Caveats. The
> other five runs have not been executed, and no accuracy or latency numbers are
> reproduced in this document.

---

## Files created for this run

| File | Purpose |
| :-- | :-- |
| `frontends/benchmarks/freshqa/src/convert_xlsx_to_json.py` | Converts the official FreshQA `.xlsx` release to the JSON format NAT eval expects |
| `frontends/benchmarks/freshqa/configs/config_shallow_ultra_freshqa_{smoke10,500,full}.yml` | Config 1 + judge LLM + `eval:` section + file logging + profiler |
| `frontends/benchmarks/freshqa/configs/config_shallow_deep_ultra_freshqa_{smoke10,500,full}.yml` | Config 2 + the same additions |
| `misc/eval_metrics_summary.py` | Prints avg latency / input tokens / output tokens / avg LLM calls from the profiler trace (already present) |

---

## Step 1 — Environment

Already done on this server — recorded here for reproducibility. The existing
`.venv` (Python 3.12, NAT 1.8.0) was healthy and only missing the benchmark plugin,
so it was patched in place rather than rebuilt:

```bash
uv pip install -e ./frontends/benchmarks/freshqa openpyxl
```

> Do **not** run `./scripts/setup.sh` unless you actually want a rebuild: it deletes
> and recreates `.venv` at Python 3.13 and runs `npm ci` for the UI. Neither is needed
> for FreshQA.

**Activate the venv in every shell you run evals from.** Both `nat` and `dotenv` live
in `.venv/bin` and are not on the system PATH — without this you get
`Command 'dotenv' not found, but can be installed with: apt install dotenv-cli`.
Ignore that suggestion; the `dotenv` you want is python-dotenv from the venv, not the
unrelated Debian package.

```bash
source .venv/bin/activate
```

Verify:

```bash
python -c "import freshqa_eval, openpyxl, nat; print('ok')"
nat --version                    # nat, version 1.8.0
dotenv --version                 # dotenv, version 1.2.2
```

If you would rather not activate, prefix every command with the venv path instead
(`.venv/bin/dotenv ...`, `.venv/bin/nat ...`).

Confirm the NIM is up before launching anything long:

```bash
curl -s http://10.86.10.114:8999/v1/models
```

## Step 2 — Convert the dataset

The release is an `.xlsx` workbook whose first sheet opens with a warning banner, so
the header is not row 1 and the repo's existing `convert_csv_to_json.py` (CSV, row-0
header) cannot read it. `convert_xlsx_to_json.py` locates the header row and handles
the subsetting.

It also carries `num_hops` and `effective_year` into the output, which
`convert_csv_to_json.py` drops. The evaluator needs both for its one-hop/multi-hop
and old/new accuracy breakdowns — without them those tables come out empty.

```bash
XLSX=/localhome/local-smasurekar/smasurekar/blueprint-pipeline/evaluation/rag-eval/scripts/datasets/freshqa/FreshQA_v042126.xlsx
CONV=frontends/benchmarks/freshqa/src/convert_xlsx_to_json.py

# Full dataset — 600 rows (500 TEST + 100 DEV)
.venv/bin/python $CONV $XLSX \
  frontends/benchmarks/freshqa/data/FreshQA_v042126.json

# 500-question run — this is exactly the TEST split, no arbitrary slicing needed
.venv/bin/python $CONV $XLSX \
  frontends/benchmarks/freshqa/data/FreshQA_v042126_500.json \
  --split TEST

# 10-question smoke subset
.venv/bin/python $CONV $XLSX \
  frontends/benchmarks/freshqa/data/FreshQA_v042126_smoke10.json \
  --split TEST --limit 10
```

Expect `Wrote 600 / 500 / 10 records to ...` respectively.

Each record carries `question` and `expected_output` (the populated `answer_0..9`
columns) plus the `split` / `fact_type` / `num_hops` / `false_premise` /
`effective_year` metadata the evaluator uses for its breakdowns.

### Why the 500-question run is `--split TEST`

The workbook holds 600 rows: 500 `TEST` and 100 `DEV`. So "500 questions" and "the
TEST split" are the same set — you get a principled, upstream-comparable subset for
free, instead of a first-500 slice.

This matters because the workbook groups rows by category. `--limit 500` would take
the first 500 rows in workbook order, which is *not* a proportional slice, and the
per-dimension breakdowns would reflect whatever mix landed in those rows.

The converter's flags, for reference:

- `--split TEST` — filter by split. Used for the 500 run.
- `--limit N` — first N rows in workbook order. Deterministic and easy to
  cross-reference against the spreadsheet; fine for the 10-question smoke set,
  misleading for anything you report accuracy on.
- `--sample N --seed 42` — random N-row subset drawn from the whole filtered set, so
  the fact-type / hop-count / false-premise mix stays representative. Use this if you
  ever want an intermediate size (say 200) rather than a prefix.
- `--limit` and `--sample` are mutually exclusive.

## Step 3 — Run the evals

Six runs: two configs × three scales. Each config writes to its own `output_dir`, so
nothing overwrites anything and all six can be compared afterwards.

| # | Config file (under `frontends/benchmarks/freshqa/configs/`) | Env file | Dataset | Results dir | Concurrency |
| :-- | :-- | :-- | :-- | :-- | --: |
| 1 | `config_shallow_ultra_freshqa_smoke10.yml` | `.env_shallow` | smoke10 | `results/shallow_ultra_smoke10` | 2 |
| 2 | `config_shallow_ultra_freshqa_500.yml` | `.env_shallow` | 500 | `results/shallow_ultra_500` | 4 |
| 3 | `config_shallow_ultra_freshqa_full.yml` | `.env_shallow` | full | `results/shallow_ultra_full` | 4 |
| 4 | `config_shallow_deep_ultra_freshqa_smoke10.yml` | `.env_shallow_deep` | smoke10 | `results/shallow_deep_ultra_smoke10` | 2 |
| 5 | `config_shallow_deep_ultra_freshqa_500.yml` | `.env_shallow_deep` | 500 | `results/shallow_deep_ultra_500` | 2 |
| 6 | `config_shallow_deep_ultra_freshqa_full.yml` | `.env_shallow_deep` | full | `results/shallow_deep_ultra_full` | 2 |

NAT's `FileHandler` does not create parent directories, so the results directory has
to exist before the run or the config's file logger fails on startup.

**Start with the two smoke runs** and only scale up once both are clean.

Every command below assumes the venv is active (Step 1):

```bash
source .venv/bin/activate
```

Runs 1 and 4 are short and run in the foreground. Runs 2, 3, 5, and 6 are long, so
they use `nohup ... &` — a dropped terminal will not kill them. Check on a
backgrounded run with `tail -f <results dir>/console.log`.

### Run 1 — Config 1 (shallow), smoke 10

```bash
RUN=shallow_ultra_smoke10
mkdir -p frontends/benchmarks/freshqa/results/$RUN

dotenv -f deploy/.env_shallow run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_ultra_freshqa_smoke10.yml \
  2>&1 | tee frontends/benchmarks/freshqa/results/$RUN/console.log
```

### Run 2 — Config 1 (shallow), 500 / TEST split

```bash
RUN=shallow_ultra_500
mkdir -p frontends/benchmarks/freshqa/results/$RUN

dotenv -f deploy/.env_shallow run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_ultra_freshqa_500.yml \
  > frontends/benchmarks/freshqa/results/$RUN/console.log 2>&1 &
```

### Run 3 — Config 1 (shallow), full 600

```bash
RUN=shallow_ultra_full
mkdir -p frontends/benchmarks/freshqa/results/$RUN

dotenv -f deploy/.env_shallow run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_ultra_freshqa_full.yml \
  > frontends/benchmarks/freshqa/results/$RUN/console.log 2>&1
```

### Run 4 — Config 2 (shallow + deep), smoke 10

```bash
RUN=shallow_deep_ultra_smoke10
mkdir -p frontends/benchmarks/freshqa/results/$RUN

dotenv -f deploy/.env_shallow_deep run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_deep_ultra_freshqa_smoke10.yml \
  2>&1 | tee frontends/benchmarks/freshqa/results/$RUN/console.log
```

### Run 5 — Config 2 (shallow + deep), 500 / TEST split

```bash
RUN=shallow_deep_ultra_500
mkdir -p frontends/benchmarks/freshqa/results/$RUN

dotenv -f deploy/.env_shallow_deep run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_deep_ultra_freshqa_500.yml \
  > frontends/benchmarks/freshqa/results/$RUN/console.log 2>&1
```

### Run 6 — Config 2 (shallow + deep), full 600

```bash
RUN=shallow_deep_ultra_full
mkdir -p frontends/benchmarks/freshqa/results/$RUN

dotenv -f deploy/.env_shallow_deep run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_deep_ultra_freshqa_full.yml \
  > frontends/benchmarks/freshqa/results/$RUN/console.log 2>&1
```

Run 6 is the most expensive of the six by a wide margin: 600 questions, and every
question the intent classifier routes to `deep_research_agent` runs a full
plan → research → write → citation-verify cycle. Do it last, after the smoke and 500
runs have told you what the deep-routing rate actually is.

**Do not run two evals concurrently** unless you have checked the NIM has headroom.
All six share the single endpoint at `10.86.10.114:8999`, and overlapping runs will
contaminate the latency numbers you are trying to compare.

If you get `Command 'dotenv' not found`, the venv is not active — see Step 1. Do not
`apt install dotenv-cli`; that is a different tool with different flags.

To skip `dotenv` entirely, source the env file into the shell instead. This still
needs the venv active for `nat`:

```bash
set -a; source deploy/.env_shallow; set +a
nat eval --config_file frontends/benchmarks/freshqa/configs/config_shallow_ultra_freshqa_smoke10.yml
```

Note this leaks the env file's variables into your shell for the rest of the session,
which matters if you then run the *other* config — `deploy/.env_shallow` and
`deploy/.env_shallow_deep` set overlapping names. Use a subshell, or prefer `dotenv`,
when switching between the two.

### Logging

Each config carries two loggers under `general.telemetry.logging`:

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: WARNING
      file:
        _type: file
        path: frontends/benchmarks/freshqa/results/shallow_ultra_smoke10/eval.log
        level: DEBUG
        mode: overwrite
```

The key name (`file`) is just a label — `_type: file` selects NAT's file logging
provider. Note this is `general.telemetry.logging`, distinct from the
`eval.general.profiler` block further down the same file.

The file logger captures NAT's own logger output at `DEBUG` — considerably more than
the console, which the source configs pin at `WARNING`. The `| tee ... console.log`
is complementary: it also captures eval progress bars and anything third-party
libraries write straight to stdout/stderr, which never reaches the Python logging
handlers. Use both. Switch `mode: overwrite` to `append` to accumulate across runs.

---

## Step 4 — Collect the performance metrics

`nat eval` on its own reports **accuracy only**. Latency, token usage, and LLM call
counts come from the NAT **profiler**, which is off by default and is enabled in all
six configs under `eval.general.profiler`:

```yaml
eval:
  general:
    profiler:
      compute_llm_metrics: true       # per-call ISL/OSL/latency — carries the token counts
      workflow_runtime_forecast: true
      token_uniqueness_forecast: true
      csv_exclude_io_text: true
      bottleneck_analysis:
        enable_nested_stack: true
```

This adds two files to each `output_dir`, alongside the evaluator's own output:

| File | Contents |
| :-- | :-- |
| `all_requests_profiler_traces.json` | One entry per query: every LLM call, tool call, token count, and timestamp |
| `standardized_data_all.csv` | Flat CSV of the same events with NAT-computed metrics |
| `freshqa_output.json` | Written by the evaluator, not the profiler: `average_score` plus a per-item `score` and judge `reasoning` |

Run this after every eval, against that run's `output_dir`:

```bash
# Config 1 (shallow-only) — headline numbers, no routing to break out
PYTHONPATH=src .venv/bin/python misc/eval_metrics_summary.py \
  frontends/benchmarks/freshqa/results/shallow_ultra_smoke10/all_requests_profiler_traces.json

# Config 2 — same numbers, plus a shallow/deep breakdown
PYTHONPATH=src .venv/bin/python misc/eval_metrics_summary.py \
  frontends/benchmarks/freshqa/results/shallow_deep_ultra_smoke10/all_requests_profiler_traces.json \
  --by-intent

# ... plus a per-query table with a route column
PYTHONPATH=src .venv/bin/python misc/eval_metrics_summary.py \
  frontends/benchmarks/freshqa/results/shallow_deep_ultra_smoke10/all_requests_profiler_traces.json \
  --by-intent --per-query
```

`PYTHONPATH=src` is required because `aiq_agent` is imported from the source tree;
use the venv interpreter (`.venv/bin/python`, or plain `python` with the venv
activated) — the system `python3` lacks `pydantic`.

Output shape:

```
ALL QUERIES  (n=10)
  Avg. Score (accuracy):             <value>   (<k>/<n> correct)
  Avg. Latency (Total) (s):          <value>
    median / max:                    <value> / <value>
  Avg. Token Usage Input:            <value>   (total <value>)
  Avg. Token Usage Output:           <value>   (total <value>)
  Avg. No. of LLM calls:             <value>   (total <value>)
```

The same block repeats per group under `--by-tier` / `--by-intent`, so accuracy is
broken out by route alongside cost.

Every metric is a per-query mean, with the group total in parentheses:

| Requested metric | Source |
| :-- | :-- |
| Avg. Score (accuracy) | `score` per item in `freshqa_output.json` (1.0 = credited, 0.0 = not) |
| Avg. Latency (Total) (s) | `RequestProfile.duration_s`, averaged over queries |
| Avg. Token Usage Input | `RequestProfile.total_prompt_tokens`, averaged over queries |
| Avg. Token Usage Output | `RequestProfile.total_completion_tokens`, averaged over queries |
| Avg. No. of LLM calls | `RequestProfile.total_llm_calls`, averaged over queries |

The script uses a zero-rate `PricingRegistry`, so it needs no pricing YAML — it
reports counts and durations, never dollars.

### Where accuracy comes from

Accuracy is **not** in the profiler trace. It is read from `freshqa_output.json`, the
evaluator's own output, which `nat eval` writes into the same `output_dir`. The script
picks it up automatically from the trace's directory; point elsewhere with `--scores`:

```bash
PYTHONPATH=src .venv/bin/python misc/eval_metrics_summary.py \
  frontends/benchmarks/freshqa/results/shallow_deep_ultra_smoke10/all_requests_profiler_traces.json \
  --by-intent \
  --scores frontends/benchmarks/freshqa/results/shallow_deep_ultra_smoke10/freshqa_output.json
```

If the file is absent the script prints a note to stderr and reports performance
metrics only, so it still works on a trace collected without an evaluator.

Scores are joined to trace requests **by question text**, taken from the trace's
`WORKFLOW_START` input and the evaluator item's `reasoning.question`. Joining on the
evaluator's `id` would be wrong in general: that is the *dataset's* id, which only
coincides with the request index for a prefix subset like `smoke10`. A `--sample`d or
DEV-only subset would silently mis-join on position. Requests the question join misses
fall back to an `id == request_index` match.

Two self-checks are printed when they trigger:

- `NOTE: evaluator reported average_score=X over all items; the N joined to this trace
  average Y` — the join dropped or duplicated items. Trust `freshqa_output.json`'s own
  `average_score` as the headline accuracy and treat the per-group split as suspect.
- `NOTE: n item(s) were flagged as errored by the evaluator and score 0` — these drag
  the mean down and usually mean a generation or judge failure, not a wrong answer.
  Cross-check against `error` fields in `freshqa_output.json` before reporting.

Latency is wall-clock under the config's `max_concurrency`, so it includes queuing
against other in-flight queries, not pure per-query service time. Both configs drive
the same single local NIM, so raising concurrency shifts that queuing around rather
than removing it — keep `max_concurrency` identical between two runs you intend to
compare on latency.

This is a live problem for the Config 1 vs Config 2 comparison: the table above sets
Config 1's 500/full runs to `max_concurrency: 4` and Config 2's to `2`, chosen so the
deep researcher does not swamp the NIM. Accuracy is unaffected, but the per-query
latency figures are **not** comparable across that pair. Either drop Config 1 to `2`
for the runs you plan to compare, or compare token counts and LLM-call counts, which
are concurrency-independent.

### `--by-tier` does not apply here — read this before using it

`misc/eval_metrics_summary.py --by-tier` groups queries by the *adaptive researcher's*
effort tier, which it recovers from a `declare_effort_tier` tool call in the trace.
**Neither config in this runbook uses the adaptive researcher**, so the flag will
report every query as `unknown`:

- **Config 1** has no routing at all — every question takes the same shallow path.
  There is nothing to break out.
- **Config 2** does route, but via `intent_classifier` (shallow vs. deep), which is a
  different mechanism and does not emit `declare_effort_tier`.

Use `--by-intent` instead — see the next section. `--by-tier` is retained in the
script for adaptive-researcher traces from other benchmarks.

### Where the shallow/deep routing *is* recorded

For Config 2 the shallow/deep split is the single most interesting number in the
whole comparison — it tells you what fraction of FreshQA the classifier thought
warranted deep research, and lets you attribute the cost difference.

Confirmed against the `shallow_deep_ultra_smoke10` trace: the decision is recorded in
two independent places, and they agreed on all 10 queries.

**1. The classifier's declared intent** — an `LLM_END` event whose
`function_ancestry.function_name` is `intent_classifier`. Its `data.output` is a JSON
document:

```json
{
  "intent": "research",
  "route": "standalone_research",
  "meta_response": null,
  "research_depth": "shallow",
  "route_reasoning": "The user asks a single factual question ... no active report exists, so standalone shallow research is appropriate."
}
```

`research_depth` is the `shallow` / `deep` field. `route_reasoning` is useful for
spot-checking *why* a question was routed the way it was.

**2. The route actually executed** — a `FUNCTION_START` event named
`shallow_research_agent` or `deep_research_agent`.

`misc/eval_metrics_summary.py --by-intent` groups on signal 2, because that is what
actually ran and it survives a classifier response whose JSON fails to parse. It
reads signal 1 as well, purely to warn when the two disagree:

```
WARNING: declared research_depth != executed route for requests: [3, 7]
```

A mismatch means the classifier asked for one depth and another ran, which would
invalidate reading the breakdown as "what the classifier chose". None occurred in the
smoke run.

`--by-intent` output shape:

```
ALL QUERIES  (n=10)
  ...

ROUTE: shallow  [100% of queries]
  Avg. Score (accuracy):             <value>   (<k>/<n> correct)
  Avg. Latency (Total) (s):          <value>
  ...

ROUTEs not exercised in this run: deep
```

This is the table that answers the Step 5 question directly: accuracy *and* cost, side
by side, for the queries the classifier sent shallow versus deep.

Running it on a **Config 1** trace is harmless but pointless — `shallow_research_workflow`
emits no `*_research_agent` spans, so every request lands in a single `unknown` group.
Use the bare command for Config 1 and `--by-intent` for Config 2.

With `--per-query`, a `route` column is added automatically when routing resolved for
at least one request, so Config 1 tables stay uncluttered.

### Optional: full tokenomics HTML report

For cost on top of these metrics, plus per-model and per-query breakdowns, latency
percentiles (p50/p90/p99), and token-distribution charts:

```bash
PYTHONPATH=src .venv/bin/python -m aiq_agent.tokenomics.report \
  --trace  frontends/benchmarks/freshqa/results/shallow_ultra_smoke10/all_requests_profiler_traces.json \
  --config configs/config_tokenomics_pricing.yml
```

This needs a pricing YAML declaring `tokenomics.pricing.models` / `.tools` rates —
see `docs/source/profiling/index.md` and the example under
`frontends/benchmarks/deepresearch_bench/configs/`. The output is a self-contained
`tokenomics_report.html`.

Note the doc's caveat: the Orchestrator / Planner / Researcher phase split is
best-effort. Totals for latency, tokens, and call counts are reliable; the per-role
attribution is not, because the adapter cannot see a role on every LLM call.

---

## Step 5 — Compare the two configs

Accuracy for each run lands in that run's `output_dir` as the evaluator's output,
broken down by split (`all` / `test` / `dev`), fact type (fast / slow / never-
changing), premise validity, hop count, and time period.

Compare like-for-like — same dataset, same scale:

| Comparison | Config 1 run | Config 2 run |
| :-- | :-- | :-- |
| Smoke | `results/shallow_ultra_smoke10` | `results/shallow_deep_ultra_smoke10` |
| 500 (TEST split) | `results/shallow_ultra_500` | `results/shallow_deep_ultra_500` |
| Full (600) | `results/shallow_ultra_full` | `results/shallow_deep_ultra_full` |

The 500-question TEST-split pair is the one to report. The smoke pair is a
plumbing check — at n=10 the accuracy difference between the two configs is noise.

Questions worth answering from the pair:

1. **Accuracy delta**, overall and on `fast-changing` valid-premise questions — the
   subset where extra research should help most if it helps anywhere.
2. **Cost delta** — avg latency, input tokens, and LLM calls per query from Step 4.
3. **Deep-routing rate** for Config 2, via `--by-intent` (see Step 4). If the
   classifier routes almost nothing to deep, the two configs will score the same and
   the interesting finding is about the classifier, not the researcher.
4. **False-premise accuracy.** FreshQA's false-premise questions punish agents that
   confabulate support for a bad premise; more research is not automatically better
   here.

---

## How the eval configs differ from the source configs

Neither `configs/shallow_nemotron_ultra.yml` nor
`configs/shallow_deep_nemotron_ultra.yml` has an `eval:` section, so `nat eval`
cannot consume either one directly. The benchmark copies add:

- **`judge_llm`** — `azure/openai/gpt-5.2` on `https://inference-api.nvidia.com/v1`,
  a different model and endpoint from the agent, so grading is independent of the
  system under test. It authenticates with the `NVIDIA_API_KEY` already in both env
  files, so no new secret is needed. Temperature `0.1` keeps the judge output in the
  strict `comment: ... / evaluation: ...` shape the evaluator parses.
- **`eval:` section** — dataset, `output_dir`, `max_concurrency`, `workflow_alias`,
  and the `freshqa_evaluator` wired to `judge_llm`.
- **`file` logging method** under `general.telemetry.logging`, so runtime logs are
  persisted rather than only streamed to a terminal pinned at `WARNING`.
- **`profiler:` block** under `eval.general`, so latency / token / LLM-call-count data
  is emitted alongside accuracy.
- **Config 2 only: `checkpoint_db` scoped to the run's `output_dir`** instead of the
  source config's `./checkpoints.db`. Eval drives many concurrent requests through
  `chat_deepresearcher_agent`; pointing every run at one repo-root SQLite file
  invites cross-run contention. `${AIQ_CHECKPOINT_DB:-...}` still lets you override it.

Everything else — LLM parameters, tools, agent settings, `enable_escalation`,
`enable_clarifier: false` — is copied verbatim. There is no `front_end` block in
either source config, so nothing had to be removed.

---

## Caveats

**The judge is independent of the agent, but is not the upstream judge.** Grading
uses `azure/openai/gpt-5.2` while the agent runs Nemotron Ultra, so these numbers do
not carry the self-judging bias they would if both were the same model. They are
still not directly comparable to published FreshQA results, which use GPT-4o under
the original FreshEval prompt. For the Config 1 vs Config 2 A/B this does not matter
— both are judged identically.

Two things to watch with this judge:

- The configs comment out `max_tokens`, `num_retries`, and `chat_template_kwargs`.
  Dropping `num_retries` means a transient error on `inference-api.nvidia.com` is no
  longer retried, and the evaluator falls back to its own `max_retries: 3` on an
  unparseable verdict. If you see items scored as errors, re-add `num_retries: 5`.
- `enable_thinking: false` was a Nemotron chat-template flag and does not apply to a
  GPT-family model, so commenting it out is correct. If verdicts come back wrapped in
  reasoning prose the evaluator cannot parse, that is the thing to revisit.

**The evaluator's old/new split is miscalibrated for this dataset release.** It
classifies a question as "new" only when `effective_year` is `2022` or `2023`:

```python
def is_new(item_id: str) -> bool:
    year = self.dataset_metadata.get(item_id, {}).get("effective_year", "")
    return year in ["2022", "2023"]
```

`FreshQA_v042126` contains 258 `before 2022`, 98 `2022`, 39 `2023`, 23 `2024`,
92 `2025`, and 90 `2026`. So the 205 questions from 2024–2026 — the freshest ones,
and the most discriminating — are bucketed as **old**. Read `accuracy_vp_old` /
`accuracy_vp_new` with that in mind, or widen the year list in
`frontends/benchmarks/freshqa/src/evaluator.py` before reporting those two rows.
Every other breakdown (fact type, premise, hops, split) is unaffected.

**The judge no longer shares the agent's NIM**, so judging adds no load to
`10.86.10.114:8999` and does not perturb the latency figures. It does mean the eval
now depends on outbound access to `inference-api.nvidia.com` and on a valid
`NVIDIA_API_KEY` — a run can generate all 600 answers successfully and still fail at
the grading step. The agent's own latency numbers remain sensitive to anything else
sharing the local NIM.

**FreshQA did not exercise the deep researcher at smoke scale — confirmed, not
predicted.** In `shallow_deep_ultra_smoke10`, the intent classifier returned
`research_depth: "shallow"` on **10 of 10** questions, and `deep_research_agent` never
ran. At smoke scale Config 2 is therefore measuring almost exactly what Config 1
measures, plus one extra classifier LLM call per query.

Two consequences before you spend the 500 and full runs:

- If the 500-run routing rate is also ~0%, the accuracy A/B is answering a question
  about the *classifier*, not about the deep researcher. Check the routing tally
  early in the run rather than at the end.
- The deep path needs a benchmark with genuinely multi-part questions (e.g. Deep
  Research Bench) to show up at all. That is a property of FreshQA, not a
  misconfiguration.

The original expectation below is retained for context:

**FreshQA may not exercise the deep researcher.** It is short-form factoid QA. On a
comparable adaptive-researcher run, the orchestrator never escalated past its
second-cheapest tier. If Config 2's classifier likewise routes everything to shallow,
the two configs are measuring nearly the same thing and the deep path needs a harder
benchmark (e.g. Deep Research Bench) to show up at all. The smoke run will tell you
this in about ten questions — check it before committing to the full 600.

**No knowledge/RAG layer is in play.** Both configs search the public web via Tavily
only; neither registers a `knowledge_search` tool. `TAVILY_API_KEY` must be set in
whichever env file you use.
