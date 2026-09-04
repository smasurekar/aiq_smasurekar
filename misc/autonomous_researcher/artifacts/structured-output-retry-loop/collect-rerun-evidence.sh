#!/usr/bin/env bash
# Regenerate the verification-run measurements quoted in
#   misc/autonomous_researcher/structured-output-retry-loop-analysis.md  (sections 3 and 6)
#
# Three jobs on datasets/deepsearchqa-smoke:
#   LOOP_JOB     no guard                          -> the unbounded loop, 2 timeouts
#   GUARD_JOB    guard on, AIQ_LOG_PAYLOADS unset  -> loop bounded, field named
#   PAYLOAD_JOB  guard on, AIQ_LOG_PAYLOADS=1      -> field named AND value shown
#
#   ./collect-rerun-evidence.sh > measurements-rerun.txt
set -uo pipefail

EVALS="${EVALS:-../../../../../../gitlab_repos/ai-q-harbor-evals}"
LOOP_JOB="${LOOP_JOB:-$EVALS/jobs/2026-08-21__11-12-50}"
GUARD_JOB="${GUARD_JOB:-$EVALS/jobs/2026-08-21__13-59-47}"
PAYLOAD_JOB="${PAYLOAD_JOB:-$EVALS/jobs/2026-08-21__14-11-50}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/parse.py" <<'PY'
"""Emit one JSON object per structured-output rejection found in a job's console logs."""
import re, sys, glob, os, json, signal

# Downstream consumers may stop reading early (see example.py); die quietly if so.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

HEAD = re.compile(r"custom_middleware:\d+ - (\S+) failed schema validation on attempt (\d+)/(\d+)")
TAIL = re.compile(r"rejected arguments: chars=(\d+) ref=sha256:(\w+)(?: payload=(.*))?$")

for d in sorted(glob.glob(os.path.join(sys.argv[1], "deepsearchqa-*"))):
    path = os.path.join(d, "agent", "aiq-agent-console-stdout.txt")
    if not os.path.exists(path):
        continue
    lines = open(path, errors="replace").read().splitlines()
    i = 0
    while i < len(lines):
        m = HEAD.search(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and "rejected arguments:" not in lines[j]:
            j += 1
        block = "\n".join(lines[i:j + 1])
        # A pydantic error block is "<field path>\n  <message> [type=<errtype>, ...]".
        fields = [(fm.group(1), fm.group(2))
                  for fm in re.finditer(r"^([\w.\[\]0-9]+)\n\s+.*?\[type=(\w+)", block, re.M)]
        tm = TAIL.search(lines[j]) if j < len(lines) else None
        print(json.dumps({
            "trial": os.path.basename(d), "schema": m.group(1),
            "attempt": int(m.group(2)), "cap": int(m.group(3)),
            "fields": fields, "payload": tm.group(3) if tm else None,
        }))
        i = j + 1
PY

cat > "$TMP/census.py" <<'PY'
import sys, json, collections
rows = [json.loads(l) for l in sys.stdin]
print("   total rejections:      ", len(rows))
print("   hit the cap:           ", sum(1 for r in rows if r["attempt"] == r["cap"]))
print("   payload captured:      ", sum(1 for r in rows if r["payload"]), "/", len(rows))
print("   per trial:             ", dict(collections.Counter(r["trial"] for r in rows)))
counts = collections.Counter((r["schema"], f[0], f[1]) for r in rows for f in r["fields"])
for (schema, field, errtype), n in counts.most_common():
    print("     %3d  %s.%s  type=%s" % (n, schema, field, errtype))
PY

cat > "$TMP/diagnose.py" <<'PY'
"""For every rejection that carries a payload, report the offending field's real JSON type."""
import sys, json, collections
rows = [json.loads(l) for l in sys.stdin]
types = collections.Counter()
reparses = malformed = 0
for r in rows:
    if not r["payload"]:
        continue
    try:
        obj = json.loads(r["payload"])
    except Exception:
        types[("<payload unparseable>", "")] += 1
        continue
    missing = object()  # a sentinel, so an absent field is never mistaken for a str
    for field, _ in r["fields"]:
        value = obj.get(field.split(".")[0], missing)
        kind = "ABSENT" if value is missing else type(value).__name__
        types[("%s.%s" % (r["schema"], field), kind)] += 1
        if isinstance(value, str):
            try:
                reparses += isinstance(json.loads(value), dict)
            except Exception:
                malformed += 1
for (name, kind), n in types.most_common():
    print("     %3d  %s -> %s" % (n, name, kind))
print("   stringified value re-parses to a dict: %d   malformed: %d" % (reparses, malformed))
PY

cat > "$TMP/example.py" <<'PY'
"""Print one verbatim instance of the defect, next to its correctly-typed siblings."""
import sys, json
# Prefer an example that also carries correctly-typed list-of-object siblings, since
# the contrast is the whole point: the model gets arrays of objects right and only
# stringifies the bare nested object.
best = None
for line in sys.stdin:
    r = json.loads(line)
    if not r["payload"]:
        continue
    try:
        obj = json.loads(r["payload"])
    except Exception:
        continue
    for field, _ in r["fields"]:
        value = obj.get(field.split(".")[0])
        if not isinstance(value, str):
            continue
        siblings = [(k, v) for k, v in obj.items()
                    if isinstance(v, list) and v and isinstance(v[0], dict)]
        if best is None or len(siblings) > len(best[3]):
            best = (r["schema"], field, value, siblings)

if best is None:
    print("   (no stringified nested field found)")
else:
    schema, field, value, siblings = best
    print("   %s.%s was emitted as a JSON *string*:" % (schema, field))
    print("     " + value[:240] + ("..." if len(value) > 240 else ""))
    print("   sibling list-of-object fields in the same payload:")
    for k, v in siblings:
        print("     %-16s list[%d objects]   <- correctly typed" % (k + ":", len(v)))
PY

cat > "$TMP/streaks.py" <<'PY'
import re, sys, glob, os
for d in sorted(glob.glob(os.path.join(sys.argv[1], "deepsearchqa-*"))):
    path = os.path.join(d, "agent", "aiq-agent-console-stdout.txt")
    if not os.path.exists(path):
        continue
    # The tool name and its Args digest are logged on consecutive lines, with ANSI
    # colour codes in between, so match across the newline rather than within a line.
    digests = re.findall(r"\u2192 ResearchNotes.*?\n.*?Args: chars=\d+ ref=sha256:(\w+)",
                         open(path, errors="replace").read())
    worst = current = 0
    previous = None
    for digest in digests:
        current = current + 1 if digest == previous else 1
        worst = max(worst, current)
        previous = digest
    print("   %-28s worst_identical_streak=%-5d total_ResearchNotes=%d"
          % (os.path.basename(d), worst, len(digests)))
PY

cat > "$TMP/outcome.py" <<'PY'
import json, sys, os
r = json.load(open(os.path.join(sys.argv[1], "result.json")))
stats = r["stats"]
ev = list(stats["evals"].values())[0]
metrics = ev["metrics"][0]
print("-- %s  finished=%s" % (os.path.basename(sys.argv[1]), r["finished_at"]))
print("   trials=%s errors=%s exceptions=%s" % (ev["n_trials"], ev["n_errors"], ev["exception_stats"]))
print("   reward=%s grader_valid=%s" % (metrics["reward"], metrics["grader_valid"]))
print("   tokens in=%s out=%s" % (stats["n_input_tokens"], stats["n_output_tokens"]))
PY

echo "=== 1. the loop is bounded: worst identical-payload streak per trial ==="
for job in "$LOOP_JOB" "$GUARD_JOB" "$PAYLOAD_JOB"; do
  echo "-- $(basename "$job")"
  python3 "$TMP/streaks.py" "$job"
done

echo
echo "=== 2. rejection census, guard job (AIQ_LOG_PAYLOADS unset) ==="
python3 "$TMP/parse.py" "$GUARD_JOB" | python3 "$TMP/census.py"

echo
echo "=== 3. rejection census, payload job (AIQ_LOG_PAYLOADS=1) ==="
python3 "$TMP/parse.py" "$PAYLOAD_JOB" | python3 "$TMP/census.py"

echo
echo "=== 4. THE DIAGNOSIS: real JSON type of the offending field in each rejected payload ==="
python3 "$TMP/parse.py" "$PAYLOAD_JOB" | python3 "$TMP/diagnose.py"

echo
echo "=== 5. one verbatim instance ==="
python3 "$TMP/parse.py" "$PAYLOAD_JOB" | python3 "$TMP/example.py"

echo
echo "=== 6. job outcomes ==="
for job in "$LOOP_JOB" "$GUARD_JOB" "$PAYLOAD_JOB"; do
  python3 "$TMP/outcome.py" "$job"
done

echo
echo "=== 7. console log size (cost of AIQ_LOG_PAYLOADS=1, uncapped) ==="
for job in "$GUARD_JOB" "$PAYLOAD_JOB"; do
  echo "-- $(basename "$job")"
  for f in "$job"/deepsearchqa-*/agent/aiq-agent-console-stdout.txt; do
    [ -f "$f" ] || continue
    printf '   %-30s %9d bytes  payload_lines=%s\n' \
      "$(basename "$(dirname "$(dirname "$f")")")" "$(wc -c < "$f")" "$(grep -c 'payload=' "$f")"
  done
done
