# `AgentSetupTimeoutError` with `chat_deepresearcher_agent` configs

**Job:** `jobs/2026-08-12__14-30-17` — 5/5 trials failed, zero answers produced, and no
benchmark reward was computed (`mean: 0.0` is only Harbor's empty-result placeholder).
**Date diagnosed:** 2026-08-12
**Verdict:** Not a config or scoring error. A leaked non-daemon thread in `aiq_agent`
prevents Harbor's one-shot runner process from exiting. Harbor must observe subprocess
termination before setup is complete. A successful AIQ CLI invocation may use a different
lifecycle or termination path, so CLI success does not disprove this teardown defect.

---

## 0. Repository context

This report is filed in the **AIQ repo**, because the defect is in `aiq_agent`. It was
diagnosed from an eval run in a **separate** repo. Two repos are referenced throughout:

| Short name | Absolute path | Role |
|---|---|---|
| **AIQ repo** (this one) | `/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar` | Contains the defect (`src/aiq_agent/...`) and the AI-Q workflow configs |
| **Evals repo** | `/home/smasurekar/Desktop/Swapnil/gitlab_repos/ai-q-harbor-evals` | Harbor harness that surfaced it; holds `jobs/`, `configs/*.yaml`, and `src/aiq_harbor_evals/...` |

Unqualified relative paths below follow that split: `src/aiq_agent/...` and `misc/...` are in
this repo; `jobs/...`, `configs/*.yaml`, and `src/aiq_harbor_evals/...` are in the evals repo.

---

## 1. Reproduction

Run from the **evals repo** root:

```bash
uv run harbor run \
  --env-file /home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/deploy/.env.shallow_deep \
  --config configs/deepresearchqa_shallow_deep_nemotron_ultra.yaml \
  --path datasets/deepsearchqa-smoke/ \
  --n-concurrent 2
```

The AI-Q config under test is
`/home/smasurekar/Desktop/Swapnil/github_repos/aiq_smasurekar/configs/shallow_deep_nemotron_ultra.yml`
(sha256 `b9b33625855d40987d40bb2874d2f0dc66b3224a81f05586f6b5489fe25c7feb`), whose workflow is:

```yaml
workflow:
  _type: chat_deepresearcher_agent
  enable_escalation: true
  enable_clarifier: false
  checkpoint_db: ${AIQ_CHECKPOINT_DB:-./checkpoints.db}
```

---

## 2. Job artifacts

### 2.1 `jobs/2026-08-12__14-30-17/result.json`

```json
"n_total_trials": 5,
"stats": {
  "n_completed_trials": 5,
  "n_errored_trials": 5,
  "evals": {
    "aiq-agent__deepsearchqa-smoke": {
      "n_trials": 0,
      "n_errors": 5,
      "metrics": [{ "mean": 0.0 }],
      "exception_stats": {
        "AgentSetupTimeoutError": [
          "deepsearchqa-0005__UQEiPBB", "deepsearchqa-0004__TVWWaJH",
          "deepsearchqa-0001__igPd8Qq", "deepsearchqa-0003__nKkpTsf",
          "deepsearchqa-0002__vqJukdp"
        ]
      }
    }
  }
}
```

`"mean": 0.0` is **not a score** — `n_trials: 0` means nothing was graded. Do not read this
as "the agent answered badly"; the agent never ran at all.

### 2.2 Timeline — every trial burns exactly 6m17s, two at a time

`--n-concurrent 2` plus the 360s setup timeout gives a perfectly regular staircase:

| Trial | started_at (UTC) | finished_at (UTC) | Wall |
|---|---|---|---|
| `deepsearchqa-0005__UQEiPBB` | 09:00:18.430 | 09:06:35.385 | 6m17s |
| `deepsearchqa-0004__TVWWaJH` | 09:00:18.457 | 09:06:35.235 | 6m17s |
| `deepsearchqa-0001__igPd8Qq` | 09:06:35.238 | 09:12:52.082 | 6m17s |
| `deepsearchqa-0003__nKkpTsf` | 09:06:35.390 | 09:12:52.207 | 6m17s |
| `deepsearchqa-0002__vqJukdp` | 09:12:52.085 | 09:19:08.918 | 6m17s |

360s timeout + ~17s container build/teardown. No trial ever reached the `run` phase.

### 2.3 The decisive artifact — `*/agent/aiq_setup.json`

Every trial wrote a **successful** validation record:

```json
{
  "status": "valid",
  "config_sha256": "b9b3362585...",
  "expected_config_sha256": "b9b3362585...",
  "workflow_type": "chat_deepresearcher_agent",
  "aiq_version": "2.2.0",
  "nat_version": "1.8.0",
  "functions": ["advanced_web_search_tool", "data_sources", "deep_research_agent",
                "intent_classifier", "shallow_research_agent", "web_search_tool"],
  "tool_preflight": { "requested": [...], "resolved": [...], "unavailable": [] },
  "duration_sec": 3.006485939025879
}
```

Per-trial `duration_sec`: `3.006`, `3.006`, `3.005`, `2.998`, `2.955`.

**The work finished in 3 seconds and passed. Then the process sat for another 357 seconds
doing nothing.** Config digest matched, all tools resolved, `unavailable: []`. There is
nothing wrong with the config.

### 2.4 `*/exception.txt` and `*/trial.log`

```
File ".../src/aiq_harbor_evals/agents/aiq_harbor.py", line 150, in setup
    result = await environment.exec(
File ".../harbor/environments/docker/docker.py", line 441, in _collect_buffered_output
    stdout_bytes, stderr_bytes = await process.communicate()
  ...
asyncio.exceptions.CancelledError
    -> TimeoutError
    -> harbor.trial.errors.AgentSetupTimeoutError: Agent setup timed out after 360.0 seconds
```

`process.communicate()` reads stdout/stderr **to EOF**. A process that never exits never
closes its pipes, so `communicate()` never returns. This is the mechanism by which a
harmless "slow exit" becomes a hard eval failure.

### 2.5 Cascading (cosmetic) artifact failure

```
RuntimeError: Docker compose command failed ... cp main:/workspace/answer.txt ...
Error response from daemon: Could not find the file /workspace/answer.txt in container
```

And `*/artifacts/manifest.json`:

```json
[ { "source": "/logs/artifacts", "status": "empty" },
  { "source": "/workspace/answer.txt", "status": "failed" } ]
```

This is downstream noise, not a second bug. Setup died before `run()`, so
`/workspace/answer.txt` was never created. Harbor's best-effort artifact collection then
logs a failure. **Ignore it** — chasing this leads away from the real cause.

### 2.6 Leaked processes and containers

The hung processes outlive the job. Observed on the host ~2 hours after their jobs ended:

```
$ pgrep -af "aiq_runner.py validate"
318896  01:59:01  /app/.venv/bin/python /installed-agent/aiq_runner.py validate ...
318908  01:59:01  /app/.venv/bin/python /installed-agent/aiq_runner.py validate ...

$ docker ps
deepsearchqa-0003__eicvvh2-main-1   Up 2 hours
deepsearchqa-0001__qrwvdyd-main-1   Up 2 hours
deepsearchqa-0115__ud2fcjh-main-1   Up 3 hours
deepsearchqa-0644__lm43usw-main-1   Up 3 hours
```

Inside a live stuck container, the process is present with 33 threads, all parked:

```
$ docker exec <trial-container> ps -ef
root  46  0  2 09:12 ?  00:00:02 /app/.venv/bin/python /installed-agent/aiq_runner.py validate ...

$ ... /proc/46/task/*/wchan
futex_do_wait   (x33)
```

Every repeated failed run adds more of these. They must be cleaned up manually (§6).

---

## 3. Root cause

`py-spy` could not attach (host needs `sudo`; in-container ptrace is blocked by the default
seccomp profile). The stack was instead obtained by re-running the exact validate command
under `faulthandler.dump_traceback_later(75, exit=True)`:

```
Timeout (0:01:15)!
Thread 0x...6c0 (most recent call first):
  File "/app/.venv/lib/python3.13/site-packages/aiosqlite/core.py", line 59 in _connection_worker_thread
  File ".../threading.py", line 995 in run
  File ".../threading.py", line 1044 in _bootstrap_inner
  File ".../threading.py", line 1015 in _bootstrap

Thread 0x...740 (most recent call first):
  File ".../threading.py", line 1543 in _shutdown
```

The main thread is in `threading._shutdown`, blocked joining a live `aiosqlite` worker
thread. The chain:

1. **`chat_deepresearcher_agent` always creates a checkpointer.**
   `aiq_agent/agents/chat_researcher/register.py:555`
   ```python
   checkpointer = await get_checkpointer(config.checkpoint_db)
   ```

2. **The connection is cached forever and never closed.**
   `aiq_agent/common/__init__.py:183-188`
   ```python
   conn = await aiosqlite.connect(checkpoint_db)
   checkpointer = AsyncSqliteSaver(conn)
   await checkpointer.setup()
   ...
   _checkpointers[checkpoint_db] = checkpointer
   ```
   `grep -rn "_checkpointers"` over the whole `aiq_agent` source returns exactly three
   hits — the declaration (`:68`), the read (`:165`), and the write (`:188`). **No close
   path exists anywhere in the codebase.**

3. **The aiosqlite worker thread is non-daemon.**
   `aiosqlite/core.py:90`
   ```python
   self._thread = Thread(target=_connection_worker_thread, args=(self._tx,))
   ```
   No `daemon=True`. The thread exits only on the `_STOP_RUNNING_SENTINEL` that
   `conn.close()` posts — and `conn.close()` is never called.

4. **CPython hangs at interpreter shutdown.** `threading._shutdown` joins all non-daemon
   threads before exiting. That join never returns.

5. **Harbor times out.** `environment.exec` → `process.communicate()` → waits for EOF that
   never comes → `AgentSetupTimeoutError` after 360s.

Note step 1 runs during **validation**, not during inference. Merely *loading* the workflow
is enough to poison process exit — which is why setup fails and the agent never runs.

---

## 4. Why the other configs work

Four hypotheses were tested by re-running the validate path in a throwaway container from
image `aiq-harbor:c075362751ce` against edited copies of the config.

| Variant | `front_end: aiq_api` | `checkpoint_db:` key | Workflow | Result |
|---|---|---|---|---|
| `shallow_deep_nemotron_ultra.yml` (as-is) | yes | yes | `chat_deepresearcher_agent` | **hangs** |
| same, `front_end:` block stripped | no | yes | `chat_deepresearcher_agent` | **hangs** |
| same, `checkpoint_db:` line stripped | yes | no | `chat_deepresearcher_agent` | **hangs** |
| `config_cli_default.yml` | no | yes | `chat_deepresearcher_agent` | **hangs** |
| `shallow_nemotron_ultra.yml` | no | no | `shallow_research_workflow` | **exits cleanly** |

Conclusions:

- **Not the `front_end: aiq_api` block.** Removing it changes nothing. (This was the
  intuitive first suspect, since it is the only structural difference between the working
  and failing harbor configs — it is a red herring.)
- **Not the `checkpoint_db:` key.** Removing it only falls back to the `./checkpoints.db`
  default; `get_checkpointer()` is still called.
- **It is the workflow type.** `shallow_research_workflow` never touches
  `get_checkpointer`, so no aiosqlite thread is created and the process exits normally.
  That is the sole reason `jobs/2026-08-12__14-24-06` succeeded (5/5 completed,
  reward `0.6`).

### Why the AIQ CLI appears to work

`config_cli_default.yml` **does not exit cleanly either** — it hangs on an identical
aiosqlite stack when driven through the same load-and-teardown path used by the Harbor
runner. That proves the configuration is not the differentiator, but it does **not** prove
that every AIQ CLI command leaks an orphan: the exact CLI command may use another lifecycle,
may remain intentionally interactive, or may have its own termination handling.

Harbor blocks on process termination to collect the exit code and buffered output, so the
leak is fatal in this adapter. If the CLI behavior itself needs to be characterized, record
the exact command, confirm whether the shell prompt returns, and inspect the PID afterward.

> **Proven scope:** with the pinned AIQ image and SQLite checkpoint backend, loading
> `workflow: _type: chat_deepresearcher_agent` through this runner leaves the aiosqlite
> worker alive and prevents normal process exit. PostgreSQL and other lifecycle paths have
> not been established by this reproduction.

### Branch scope — `develop` is affected as well

The defect lives in this repo, not in the evals harness, so it is worth asking whether
`develop` already carries a fix. It does not.

**Branch identity.** Checked 2026-08-12 after `git fetch upstream develop`:

| Ref | Commit | Note |
|---|---|---|
| `develop` (local) | `e4406e8` *Merge pull request #435 from NVIDIA-AI-Blueprints/release/2.2* | up to date with upstream |
| `upstream/develop` (`NVIDIA-AI-Blueprints/aiq`) | `e4406e8` | identical to local `develop` |
| `origin/develop` (personal fork) | `bb35df3` | **8 commits stale**; do not use as the reference |

**Static evidence.** The two files that carry the defect are byte-identical between
`develop` and the working branch `dev/smasurekar/research-guard`:

```bash
git diff --stat develop dev/smasurekar/research-guard -- \
  src/aiq_agent/common/__init__.py \
  src/aiq_agent/agents/chat_researcher/register.py
# (no output — identical)
```

On `develop`, `get_checkpointer` still ends in the same unclosed connection:

```python
# develop:src/aiq_agent/common/__init__.py:183-189
conn = await aiosqlite.connect(checkpoint_db)
checkpointer = AsyncSqliteSaver(conn)
await checkpointer.setup()
logger.info("SQLite checkpointer initialized: %s", checkpoint_db)

_checkpointers[checkpoint_db] = checkpointer
return checkpointer
```

A search for any release path — `_checkpointers[...].close`, `await conn.close()`,
`close_checkpointer`, `checkpointer.close` — returns **0 hits** on every branch checked:
`develop`, `origin/develop`, `origin/main`, and `dev/smasurekar/research-guard`. The
`chat_researcher` registration still calls `get_checkpointer` on all of them.

The pinned eval image revision `c075362751ce` is itself an **ancestor of `develop`**
(`git merge-base --is-ancestor` → true), so the original job failure was already running
develop-lineage code. The only change to `common/__init__.py` between that revision and
`develop` tip is unrelated:

```diff
+from .tool_validation import validate_research_source_configuration
+    "validate_research_source_configuration",
```

an import and an `__all__` entry. `get_checkpointer` itself was untouched.

**Empirical confirmation.** `develop`'s source was extracted with `git archive` (leaving the
working tree alone) and bind-mounted over the image's editable install path:

```bash
git archive develop src/aiq_agent | tar -x -C /tmp/aiqdev

docker run --rm --env-file <env-file> \
  -e PROBE_CONFIG=/probe/with_frontend.yml -e PROBE_SHA=b9b3362585... \
  -v /tmp/aiqhang:/probe \
  -v /tmp/aiqdev/src/aiq_agent:/app/src/aiq_agent:ro \
  aiq-harbor:c075362751ce /app/.venv/bin/python /probe/probe.py
```

That `develop` code was genuinely in effect was verified independently, using a symbol that
does not exist at the pinned revision:

```
aiq_agent path: /app/src/aiq_agent/__init__.py
develop-only symbol present: True     # validate_research_source_configuration
```

Result — the identical hang:

```
Timeout (0:01:15)!
Thread 0x...6c0: aiosqlite/core.py line 59 in _connection_worker_thread
Thread 0x...740: threading.py line 1543 in _shutdown
```

The hard-exit variant against the same `develop` source returned `REACHED_END rc=0`,
`EXIT=0` immediately, so the §5.1 workaround remains effective there.

**Caveat on this test.** It substitutes `develop`'s `aiq_agent` **source** into the pinned
image's dependency set. It therefore does not exercise `develop`'s own pinned dependency
versions — if `develop` moved to an `aiosqlite` release that made the worker thread
daemonic, this method would not reveal it. That is unlikely (the leak is the missing
`conn.close()`, which is `aiq_agent`'s own code and demonstrably still absent), but a
definitive check requires building an image from `develop` and re-running the probe.

**Consequence for the fix plan:** upgrading or rebasing onto `develop` will not resolve this
incident. The §5.1 Harbor-side workaround is required regardless of branch, and the §5.2
upstream lifecycle fix still needs to be written and landed.

---

## 5. Recommended fix

### Recommendation

Use a two-stage fix:

1. **Unblock Harbor now:** use the hard-exit workaround only in the dedicated, one-shot
   `aiq_runner.py` container process, after `main()` and all async workflow teardown have
   returned. Protect the final exit with `finally` so a flush error cannot recreate the hang.
2. **Fix ownership upstream:** make checkpointer acquisition a managed lease. Protect the
   global cache with a lock, count users per database key, and close/remove a resource only
   when the last lease is released (or at application shutdown). Once the pinned AIQ runtime
   contains that fix, remove the hard-exit workaround and restore normal `SystemExit`.

Do not close the entire global cache whenever an arbitrary workflow exits: a long-lived
service can have multiple workflow contexts using the same saver or pool concurrently.

### 5.1 Immediate — unblock the eval (evals repo)

Bypass interpreter shutdown once the work is done, in the **evals repo** at
`src/aiq_harbor_evals/agents/aiq_runner.py`. Current tail (lines 862-863):

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Replace with:

```python
if __name__ == "__main__":
    code = main()
    # aiq_agent's chat_deepresearcher_agent leaks a non-daemon aiosqlite checkpointer
    # thread (aiq_agent/common/__init__.py:183, never closed), so normal interpreter
    # shutdown blocks forever in threading._shutdown. Harbor's exec waits for EOF on
    # our pipes, so a slow exit becomes AgentSetupTimeoutError. Exit hard instead.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # A flush error must not send execution back through threading._shutdown.
        os._exit(code)
```

`os` and `sys` are already imported (lines 17-18). The explicit flushes are **required** —
`os._exit` skips normal cleanup, so buffered stdout/stderr would otherwise be lost, and
Harbor captures and persists that output after the subprocess returns.

This placement covers **both** subcommands. Fixing only `validate` would just relocate the
hang to the `run` phase and trade `AgentSetupTimeoutError` for a run timeout. This is safe
only as a scoped adapter workaround: `os._exit` skips `atexit` handlers, queued logging,
tracing-exporter flushes, and other process-global cleanup. Do not copy it into the AIQ API
server or general CLI entrypoint.

**Validate-path verification.** Against the exact failing config
(`sha256 b9b3362585...`), the same command that previously hung past 75s:

```
REACHED_END rc=0
EXIT=0
```

Returns immediately with the correct exit code.

That test proves the setup hang is removed; it is not sufficient acceptance for the eval.
Before merging, also run the regression checks in §5.3.

### 5.2 Proper — fix the leak here (AIQ repo, `src/aiq_agent/common/__init__.py`)

`os._exit` is an adapter workaround; the defect belongs to `aiq_agent`. Add explicit,
ownership-aware lifecycle management around `_checkpointers` and `_postgres_pools`:

- Serialize cache creation and release with an async lock so concurrent callers cannot
  create duplicate resources or race a close.
- Return a lease/context manager from acquisition and maintain a reference count per
  database key. The `chat_deepresearcher_agent` registration should hold that lease for the
  lifetime of its yielded function.
- On the final SQLite release, call `await saver.conn.close()` and then remove the saver
  from `_checkpointers`.
- If PostgreSQL is supported, close its `AsyncConnectionPool` only after no saver still
  references it, then remove it from `_postgres_pools`. This branch needs a separate test;
  the current incident proves only SQLite behavior.
- Provide an application-shutdown `close_all_checkpointers()` as a defensive finalizer for
  long-lived server/CLI processes. It should be idempotent and tolerate partial failures.

Do **not** mark the aiosqlite worker daemonic. The thread is already running after connect,
and Python does not permit changing daemon status then; forcing daemon behavior earlier
would also allow process exit with queued database work or an unclean SQLite handle.

### 5.3 Required regression checks

Before calling the workaround complete:

1. Run the exact `validate` command in a subprocess with a short outer timeout and assert
   exit code `0` plus intact setup metadata.
2. Run at least one real `chat_deepresearcher_agent` Harbor task end to end. Assert that it
   reaches `run`, writes `answer.txt`, produces trajectory/state/event sidecars, and reaches
   the verifier.
3. Exercise an intentional runner failure and assert that its original nonzero exit code
   and stderr are preserved through the hard exit.
4. Re-run a shallow-workflow task to confirm the workaround does not change its output or
   verifier behavior.
5. Confirm that no trial containers or runner PIDs from those test jobs remain afterward.

The saved answer must be identical regardless of exit strategy; this change is process
lifecycle plumbing and must not alter benchmark prompts, answers, preprocessing, or scoring.

### 5.4 Not recommended

- **Raising the setup timeout.** The process hangs forever; no timeout is large enough.
- **Switching to a Postgres `checkpoint_db`.** The Postgres branch uses
  `AsyncConnectionPool`, which is also never closed. That is a lifecycle defect, but this
  incident did not establish that PostgreSQL produces the same interpreter-shutdown hang.
- **Removing `checkpoint_db` from the config.** Proven ineffective (§4) — it just uses the
  default path.

---

## 6. Cleanup after a failed run

Orphaned processes and containers can accumulate across runs and exhaust the machine. First
list candidates and verify that each belongs to a finished job. Do not use a broad name
filter as the target of a force-remove or kill command.

```bash
# Inspect candidates, including exact names and creation times.
docker ps --filter "name=deepsearchqa-" --format '{{.Names}}\t{{.Status}}'

# Inspect runner processes; host pgrep may also show processes inside containers.
pgrep -af "aiq_runner.py"

# After matching a container to a finished trial, remove only that exact container.
docker rm -f <confirmed-stale-container-name>

# For a genuinely host-native orphan, signal only the exact verified PID.
kill -TERM <confirmed-stale-host-pid>
```

Removing a stale container also terminates its contained runner; normally no separate host
`kill` is needed. Do not blanket-kill by runner pattern or image name—the active evals,
long-lived `aiq-agent`, `aiq-blueprint-ui`, `aiq-postgres`, and RAG stack are unrelated.

---

## 7. Diagnostic recipe (for the next hang)

`py-spy` is the natural tool but fails in both obvious spots: on the host it needs `sudo`
(`Permission Denied: Try running again with elevated permissions`), and inside the
container ptrace is blocked by the default seccomp profile (`Failed to copy Py_Version
symbol: Permission denied`). Use `faulthandler` instead — it needs no privileges, because
the process dumps its own stack:

```python
# probe.py — mount alongside a copy of aiq_runner.py
import faulthandler, runpy, sys, os
faulthandler.dump_traceback_later(75, exit=True)   # dump all thread stacks, then die
sys.argv = ["aiq_runner.py", "validate", "--config-file", os.environ["PROBE_CONFIG"], ...]
runpy.run_path("/probe/aiq_runner.py", run_name="__main__")
```

```bash
docker run --rm \
  --env-file <env-file> \
  -e PROBE_CONFIG=/probe/<config>.yml \
  -e PROBE_SHA=<sha256 of that config> \
  -v /tmp/aiqhang:/probe -w /app \
  aiq-harbor:c075362751ce /app/.venv/bin/python /probe/probe.py
```

Cheap triage before reaching for stacks:

```bash
# Is the process alive after its work reported done?
docker exec <trial-container> ps -ef

# Compare aiq_setup.json duration_sec against the trial's wall time.
# 3s of work inside a 360s timeout means a shutdown hang, not slow work.
```

---

## 8. Summary

| | |
|---|---|
| **Symptom** | `AgentSetupTimeoutError: Agent setup timed out after 360.0 seconds`, 5/5 trials |
| **Misleading signals** | `"mean": 0.0` (nothing graded); missing `/workspace/answer.txt` (downstream) |
| **Actual cause** | Non-daemon `aiosqlite` thread leaked by `get_checkpointer()`, never closed |
| **Proven trigger** | SQLite-backed `workflow: _type: chat_deepresearcher_agent` through the pinned runner path |
| **Affected branches** | `develop` (`e4406e8`, = `upstream/develop`), `origin/main`, and `dev/smasurekar/research-guard` — no close path on any; rebasing will not fix it |
| **Why CLI success differs** | Potentially different or long-lived lifecycle; exact command must be traced before claiming an orphan |
| **Fix (now)** | Harbor-only `os._exit(code)` after teardown, with stdio flush protected by `finally` |
| **Fix (right)** | Ownership-aware/ref-counted saver and pool teardown in `aiq_agent`, plus application shutdown cleanup |
