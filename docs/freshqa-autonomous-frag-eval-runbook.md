# FreshQA evaluation runbook for the autonomous researcher

This runbook evaluates the autonomous researcher defined by
`configs/config_autonomous_frag.yml` against FreshQA at four scales: a 10-query
smoke test, a seeded 100-query sample, the 500-query TEST split, and the complete
600-query dataset.

Run every command from the repository root:

```bash
cd /localhome/local-smasurekar/smasurekar/aiq_smasurekar
```

The evaluation uses `deploy/.env.auto`. Do not print or commit its values.

## Files used by the evaluation

| Queries | Dataset | Eval config | Output directory |
| :-- | :-- | :-- | :-- |
| 10 | `frontends/benchmarks/freshqa/data/FreshQA_v042126_smoke10.json` | `config_autonomous_frag_freshqa_smoke10.yml` | `autonomous_frag_smoke10` |
| 100 | `frontends/benchmarks/freshqa/data/FreshQA_v042126_100.json` | `config_autonomous_frag_freshqa_100.yml` | `autonomous_frag_100` |
| 500 | `frontends/benchmarks/freshqa/data/FreshQA_v042126_500.json` | `config_autonomous_frag_freshqa_500.yml` | `autonomous_frag_500` |
| 600 | `frontends/benchmarks/freshqa/data/FreshQA_v042126.json` | `config_autonomous_frag_freshqa_600.yml` | `autonomous_frag_600` |

All config paths in the table are relative to
`frontends/benchmarks/freshqa/configs/`; all output directories are relative to
`frontends/benchmarks/freshqa/results/`.

The 100-query file is a deterministic sample of the full dataset, generated with
Python's `random.Random(42)`. It contains 84 TEST and 16 DEV questions and retains
the source workbook order after sampling. The 500-query file is the canonical TEST
split; the 600-query file contains 500 TEST and 100 DEV questions.

The 10-, 500-, and 600-query files were copied from:

```text
/localhome/local-smasurekar/smasurekar/aiq/frontends/benchmarks/freshqa/data
```

FreshQA data directories are intentionally ignored by this repository's
`.gitignore`. These files are local evaluation inputs, so copy or regenerate them
after a fresh clone rather than expecting Git to restore them.

## 1. Prepare and verify the environment

The FreshQA evaluator must be installed in the active project environment. If the
repository environment has already been set up, verify it first:

```bash
source .venv/bin/activate
python -c "import freshqa_eval, nat; print('FreshQA and NAT imports: OK')"
nat --version
dotenv --version
```

If `freshqa_eval` is missing, install the local benchmark package into the existing
environment:

```bash
uv pip install -e ./frontends/benchmarks/freshqa
```

Confirm that the env file declares the required variables without displaying their
values:

```bash
for name in NVIDIA_API_KEY TAVILY_API_KEY RAG_SERVER_URL RAG_INGEST_URL COLLECTION_NAME; do
  grep -q "^${name}=" deploy/.env.auto || echo "Missing: ${name}"
done
```

`RAG_SERVER_URL`, `RAG_INGEST_URL`, and `COLLECTION_NAME` are present for parity
with the source FRAG workflow. `knowledge_search` is excluded from the autonomous
agent in this config, so FreshQA answers are produced from web search.

## 2. Verify the datasets and configs

Check the expected record counts:

```bash
for dataset in \
  FreshQA_v042126_smoke10.json \
  FreshQA_v042126_100.json \
  FreshQA_v042126_500.json \
  FreshQA_v042126.json; do
  printf '%s: ' "$dataset"
  jq 'length' "frontends/benchmarks/freshqa/data/$dataset"
done
```

Expected counts are `10`, `100`, `500`, and `600`, respectively.

Validate all four NAT configurations before running an evaluation:

```bash
for config in \
  frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_smoke10.yml \
  frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_100.yml \
  frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_500.yml \
  frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_600.yml; do
  .venv/bin/dotenv -f deploy/.env.auto run -- \
    .venv/bin/nat validate --config_file "$config"
done
```

Use NAT's validator here. The maintainer helper at
`.agents/skills/aiq-configure-workflow/scripts/validate_config.py` currently only
accepts the legacy `chat_deepresearcher_agent` topology and incorrectly rejects the
supported `autonomous_research_workflow` type.

The eval configs preserve the autonomous agent, models, tools, and termination
budgets from `configs/config_autonomous_frag.yml`. They omit `general.front_end`
because `nat eval` invokes the workflow in-process, and add:

- `judge_llm` (`azure/openai/gpt-5.2` through NVIDIA's inference endpoint), using
  the existing `NVIDIA_API_KEY`;
- the FreshQA dataset and evaluator configuration;
- profiler output for latency, token, and call-count analysis;
- separate result and log paths for every dataset size.

## 3. Run the evaluations

Start with the 10-query smoke test. The results directory must exist before NAT
starts because its file log handler does not create parent directories.

```bash
RUN=autonomous_frag_smoke10
mkdir -p "frontends/benchmarks/freshqa/results/$RUN"

dotenv -f deploy/.env.auto run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_smoke10.yml \
  2>&1 | tee "frontends/benchmarks/freshqa/results/$RUN/console.log"
```

Only continue to the longer runs after the smoke test completes successfully.

### Seeded 100-query sample

```bash
RUN=autonomous_frag_100
mkdir -p "frontends/benchmarks/freshqa/results/$RUN"

dotenv -f deploy/.env.auto run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_100.yml \
  2>&1 | tee "frontends/benchmarks/freshqa/results/$RUN/console.log"
```

### 500-query TEST split

```bash
RUN=autonomous_frag_500
mkdir -p "frontends/benchmarks/freshqa/results/$RUN"

nohup dotenv -f deploy/.env.auto run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_500.yml \
  > "frontends/benchmarks/freshqa/results/$RUN/console.log" 2>&1 &
echo $!
```

### Full 600-query dataset

```bash
RUN=autonomous_frag_600
mkdir -p "frontends/benchmarks/freshqa/results/$RUN"

nohup dotenv -f deploy/.env.auto run -- \
  nat eval --config_file frontends/benchmarks/freshqa/configs/config_autonomous_frag_freshqa_600.yml \
  > "frontends/benchmarks/freshqa/results/$RUN/console.log" 2>&1 &
echo $!
```

Do not overlap these runs unless the model and search endpoints have enough
headroom. Concurrent evaluations make latency comparisons unreliable and may
trigger rate limits. The autonomous workflow has a request timeout of 2,400 seconds,
so the 500- and 600-query evaluations can take a long time.

## 4. Monitor and inspect results

For a background run, follow the console log:

```bash
tail -f frontends/benchmarks/freshqa/results/autonomous_frag_100/console.log
```

Check whether its recorded process is still active with `ps -p <PID>`. A completed
run writes its evaluator output and profiler artifacts under its configured output
directory. Important files include:

- `freshqa_output.json`: per-question judge decisions and aggregate accuracy;
- `all_requests_profiler_traces.json`: per-query model/tool calls, token counts,
  timestamps, and latency;
- `standardized_data_all.csv`: tabular profiler data;
- `eval.log` and `console.log`: runtime diagnostics.

Read the headline accuracy after the run with:

```bash
jq '{average_score, total_correct, total_evaluated}' \
  frontends/benchmarks/freshqa/results/autonomous_frag_100/freshqa_output.json
```

Report at least the dataset size, FreshQA accuracy and correct count, average
end-to-end latency, average input/output tokens, average LLM calls, failed or timed
out queries, config filename, and git revision. Treat the 10-query result as a
pipeline check rather than a meaningful accuracy estimate.

## Troubleshooting

- `dotenv: command not found`: activate `.venv`, or use
  `.venv/bin/dotenv -f deploy/.env.auto run -- .venv/bin/nat eval ...`.
- `No module named freshqa_eval`: run
  `uv pip install -e ./frontends/benchmarks/freshqa` in the active environment.
- File-handler startup failure: create the config's exact output directory before
  running `nat eval`.
- Authentication or search failures: confirm the required variable names exist in
  `deploy/.env.auto`; never paste their values into logs or issues.
- Judge parse failures: inspect `eval.log`. The judge temperature is intentionally
  low to keep its response compatible with the FreshQA evaluator.
