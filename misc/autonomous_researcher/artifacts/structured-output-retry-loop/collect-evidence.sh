#!/usr/bin/env bash
# Regenerate every measurement quoted in
#   misc/autonomous_researcher/structured-output-retry-loop-analysis.md
#
# Run from this directory. EVALS points at the ai-q-harbor-evals checkout that
# holds the job artifacts; override it if yours lives elsewhere.
#
#   ./collect-evidence.sh > measurements.txt
set -uo pipefail

EVALS="${EVALS:-../../../../../../gitlab_repos/ai-q-harbor-evals}"
JOB="$EVALS/jobs/2026-08-21__11-12-50"
T0002="$JOB/deepsearchqa-0002__DsPCtn4"
T0004="$JOB/deepsearchqa-0004__gcnVpr2"

# The tool name and its Args digest are logged on consecutive lines with ANSI colour
# codes in between, so the digest must be matched across the newline. Earlier revisions
# of this script reported `sort | uniq -c | head -1`, i.e. the TOTAL occurrences of the
# most common digest, and labelled it a streak; that overcounts whenever the same payload
# recurs after an interruption. This computes the longest CONSECUTIVE run.
STREAK_PY="$(mktemp)"
trap 'rm -f "$STREAK_PY"' EXIT
cat > "$STREAK_PY" <<'PYEOF'
import re, sys, glob, os
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    path = os.path.join(d, "agent", "aiq-agent-console-stdout.txt")
    if not os.path.exists(path):
        continue
    digests = re.findall(r"\u2192 ResearchNotes.*?\n.*?Args: chars=\d+ ref=sha256:(\w+)",
                         open(path, errors="replace").read())
    worst = current = 0
    previous = None
    for digest in digests:
        current = current + 1 if digest == previous else 1
        worst = max(worst, current)
        previous = digest
    print("%s\t%d\t%d" % (os.path.basename(d), worst, len(digests)))
PYEOF

echo "=== 1. lifecycle events: run_research_batch starts, never ends ==="
python3 - "$T0002/agent/aiq_events.jsonl" <<'PY'
import json, sys, collections
evs = [json.loads(l) for l in open(sys.argv[1])]
print("event totals:", dict(collections.Counter(e["event_type"] for e in evs)))
names = collections.Counter((e["event_type"], e.get("name")) for e in evs
                            if e["event_type"].startswith(("TOOL", "FUNCTION")))
for (et, n), c in sorted(names.items()):
    print(f"  {et:<14} {n:<28} {c}")
PY

echo
echo "=== 2. what the model asked for, by tool name ==="
for t in "$T0002" "$T0004"; do
  echo "-- $(basename "$t")"
  grep -oP '→ \S+' "$t/agent/aiq-agent-console-stdout.txt" | sed 's/\x1b\[[0-9;]*m//g' \
    | sort | uniq -c | sort -rn
done

echo
echo "=== 3. identical ResearchNotes payloads (args digest histogram) ==="
for t in "$T0002" "$T0004"; do
  echo "-- $(basename "$t")"
  grep -A2 '→ ResearchNotes' "$t/agent/aiq-agent-console-stdout.txt" \
    | grep -oP 'chars=\d+ ref=sha256:\w+' | sort | uniq -c | sort -rn | head -5
done

echo
echo "=== 4. prompt-token growth per retry (completion is constant) ==="
grep -oP 'prompt=\d+, completion=\d+' "$T0002/agent/aiq-agent-console-stdout.txt" \
  | awk -F'[=,]' '{d=$2-p; p=$2; printf "call=%-4d prompt=%-8d delta=%-8d completion=%d\n", NR, $2, (NR>1?d:0), $4}' \
  | tail -35

echo
echo "=== 5. reasoning_content on every completion (sha256 01ba4719c80b == \"\\n\") ==="
grep -oP '\[Reasoning\] chars=\d+ ref=sha256:\w+' "$T0002/agent/aiq-agent-console-stdout.txt" \
  | sort | uniq -c | sort -rn

echo
echo "=== 6. the loop predates 48626d1: worst identical-payload streak per trial ==="
for d in 2026-08-20__12-58-09 2026-08-20__16-47-37 2026-08-20__21-44-00 2026-08-21__11-12-50; do
  printf '%-24s ' "$d"
  python3 "$STREAK_PY" "$EVALS/jobs/$d" | awk -F'\t' '{print $2}' | sort -rn | uniq -c \
    | awk '{printf "%sx streak=%s  ", $1, $2} END {print ""}'
done

echo
echo "=== 7. job outcome ==="
python3 - "$JOB/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ev = r["stats"]["evals"]["aiq-agent__deepsearchqa-smoke"]
print("trials:", ev["n_trials"], "errors:", ev["n_errors"])
print("metrics:", ev["metrics"][0])
print("exceptions:", ev["exception_stats"])
PY

echo
echo "=== 8. same 5 smoke tasks, same config, before vs after 48626d1 ==="
for d in 2026-08-19__17-06-25 2026-08-20__21-25-36 2026-08-21__11-12-50; do
  echo "-- $d"
  python3 "$STREAK_PY" "$EVALS/jobs/$d" \
    | awk -F'\t' '{printf "   %-28s worst_identical_streak=%-5s total_ResearchNotes=%s\n", $1, $2, $3}'
done

echo
echo "=== 9. tokens burned inside the loop ==="
for t in "$T0002" "$T0004"; do
  echo -n "-- $(basename "$t"): "
  grep -oP 'prompt=\d+, completion=\d+' "$t/agent/aiq-agent-console-stdout.txt" \
    | awk -F'[=,]' '{p+=$2; c+=$4} END {printf "total prompt=%d completion=%d tokens\n", p, c}'
done
