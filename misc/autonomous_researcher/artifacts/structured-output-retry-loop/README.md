# Artifacts: `ResearchNotes` structured-output retry loop

Supporting evidence for
[`../../structured-output-retry-loop-analysis.md`](../../structured-output-retry-loop-analysis.md).

Captured 2026-08-21 from three eval jobs in the `ai-q-harbor-evals` checkout at
`/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals`. Copied here because job
directories are rotated and the eval repo is a separate checkout.

| Job | What it is |
| --- | --- |
| `2026-08-21__11-12-50` | The original failure: no retry guard, 2 trials lost to `AgentTimeoutError` |
| `2026-08-21__13-59-47` | Retry guard in place, `AIQ_LOG_PAYLOADS` unset — loop bounded, failing field named |
| `2026-08-21__14-11-50` | Retry guard plus `AIQ_LOG_PAYLOADS=1` — the rejected values themselves captured |

| File | Source (relative to `ai-q-harbor-evals/`) |
| --- | --- |
| `dsqa-0002-exception.txt` | `jobs/2026-08-21__11-12-50/deepsearchqa-0002__DsPCtn4/exception.txt` |
| `dsqa-0002-aiq_state.json` | `jobs/2026-08-21__11-12-50/deepsearchqa-0002__DsPCtn4/agent/aiq_state.json` |
| `dsqa-0002-loop-excerpt.log` | `.../deepsearchqa-0002__DsPCtn4/agent/aiq-agent-console-stdout.txt`, lines 440-545 plus `tail -30`, ANSI stripped |
| `dsqa-0004-aiq_state.json` | `jobs/2026-08-21__11-12-50/deepsearchqa-0004__gcnVpr2/agent/aiq_state.json` |
| `dsqa-0004-loop-excerpt.log` | `.../deepsearchqa-0004__gcnVpr2/agent/aiq-agent-console-stdout.txt`, `tail -40`, ANSI stripped |
| `job-result.json` | `jobs/2026-08-21__11-12-50/result.json` |
| `measurements.txt` | Output of `collect-evidence.sh` |
| `collect-evidence.sh` | Regenerates `measurements.txt` from the eval repo |
| `job-result-guard-run.json` | `jobs/2026-08-21__13-59-47/result.json` |
| `job-result-payload-run.json` | `jobs/2026-08-21__14-11-50/result.json` |
| `stringified-nested-model-example.log` | One full rejection with payload, from `jobs/2026-08-21__14-11-50/deepsearchqa-0001__AN3ukZa/agent/aiq-agent-console-stdout.txt` |
| `dsqa-0003-guard-cascade.log` | Guard lines from `jobs/2026-08-21__13-59-47/deepsearchqa-0003__NNxWvxC/agent/aiq-agent-console-stdout.txt` |
| `dsqa-0003-degraded-answer.txt` | `jobs/2026-08-21__13-59-47/deepsearchqa-0003__NNxWvxC/artifacts/answer.txt` |
| `measurements-rerun.txt` | Output of `collect-rerun-evidence.sh` |
| `collect-rerun-evidence.sh` | Regenerates `measurements-rerun.txt` across all three jobs |

## Regenerating

```bash
cd misc/autonomous_researcher/artifacts/structured-output-retry-loop
./collect-evidence.sh       > measurements.txt          # the original loop
./collect-rerun-evidence.sh > measurements-rerun.txt    # the two verification runs

EVALS=/path/to/ai-q-harbor-evals ./collect-evidence.sh  # if your checkout is elsewhere
```

Both scripts need only `python3`. `collect-rerun-evidence.sh` additionally accepts `LOOP_JOB`,
`GUARD_JOB` and `PAYLOAD_JOB` to point at different job directories. Sections whose jobs are
missing print nothing rather than failing.

## What each `measurements.txt` section establishes

1. `run_research_batch` gets a `TOOL_START` and never a `TOOL_END`; `ResearchNotes` never appears
   as a tool at all (it is the `response_format` pseudo-tool).
2. 118 / 187 `ResearchNotes` requests versus ~20 / ~55 real tool calls.
3. 115 and 154 byte-identical payloads — the model never varies its answer.
4. `+3552` prompt tokens per round against a constant `completion=3430`: the ~122-token
   difference is the error `ToolMessage`. Two interleaved workers are visible as the
   discontinuities at calls 111-112.
5. `reasoning_content` is `"\n"` (`sha256` `01ba4719c80b`) on 138 of 141 completions.
6. The loop existed pre-`48626d1` at 3-4 repeats, in every prior job.
7. 2 of 5 trials lost to `AgentTimeoutError`; job reward 0.4.
8. The A/B on the same 5 smoke tasks: 0/5 trials retried before `48626d1`, 5/5 after.
9. 10.5M and 15.9M prompt tokens burned.

## What each `measurements-rerun.txt` section establishes

1. The loop is bounded. Worst identical-payload streak per trial goes from `5, 115, 3, 152, 6`
   to `2, 1, 3, 0, 1` and `2, 2, 3, 3, 2` — capped at 3 by construction.
2. 30 rejections across 5 trials with payloads off; the field is named regardless.
3. 44 rejections across 5 trials with payloads on; all 44 payloads captured.
4. **The diagnosis.** The offending field is a `str` in 40 of 44 rejections, and 39 of those
   strings re-parse into a valid `dict`. The model double-encodes a nested object.
5. A verbatim instance, shown next to the correctly-typed `list[...]` siblings in the same payload.
6. Job outcomes: 2 timeouts before, 0 after; `grader_valid` 0.6 → 1.0.
7. The cost of `AIQ_LOG_PAYLOADS=1` uncapped: console logs grow from 36-173 KB per trial to
   0.9-7.4 MB. Set `AIQ_LOG_PAYLOAD_MAX_CHARS` for anything larger than a smoke run.
