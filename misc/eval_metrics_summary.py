#!/usr/bin/env python3
"""Summarize NAT eval accuracy, latency, tokens, calls, routes, and cases.

The profiler trace supplies latency/token/call metrics. FreshQA scores and errors
come from ``freshqa_output.json`` beside the trace. In addition to the original
adaptive tier and intent-route summaries, ``--by-autonomous-case`` recognizes the
autonomous researcher's execution patterns and ``--csv`` writes the requested
aggregate table.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from collections import deque
from pathlib import Path

from aiq_agent.tokenomics import PricingRegistry
from aiq_agent.tokenomics import parse_trace

TIER_TOOL = "declare_effort_tier"
TIER_ORDER = ["direct", "single_shot", "standard", "deep"]
INTENT_AGENTS = {
    "shallow_research_agent": "shallow",
    "deep_research_agent": "deep",
}
SCORE_FILENAME = "freshqa_output.json"
CORRECT_THRESHOLD = 0.5
UNKNOWN = "unknown"

CASE_SHALLOW = "Shallow Researcher"
CASE_NO_PLANNER = "No-Planner"
CASE_NO_PLANNER_SHALLOW = "No-Planner-Shallow"
CASE_NO_PLANNER_DEEP = "No-Planner-Deep"
CASE_PLANNER = "Planner"
CASE_ORDER = [CASE_SHALLOW, CASE_NO_PLANNER_SHALLOW, CASE_NO_PLANNER_DEEP, CASE_PLANNER]

CSV_COLUMNS = [
    "Number of Queries",
    "Errors",
    "Case",
    "Accuracy/Score",
    "Avg. Latency (Total) (s)",
    "Avg. Token Usage Input",
    "Avg. Token Usage Output",
    "Avg. No. of LLM calls",
]


def _load_json(path: str | Path) -> object:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _ancestor(step: dict) -> str:
    ancestry = step.get("function_ancestry")
    if isinstance(ancestry, dict):
        return str(ancestry.get("function_name") or "")
    return str(ancestry or "")


def _tool_inputs(payload: dict) -> dict:
    """Return normalized inputs for a TOOL_START event."""
    inputs = (payload.get("metadata") or {}).get("tool_inputs")
    if isinstance(inputs, dict):
        return inputs
    inputs = (payload.get("data") or {}).get("input")
    return inputs if isinstance(inputs, dict) else {}


def extract_tiers(trace_path: str) -> dict[int, str]:
    """Map request index to the last adaptive effort tier declaration."""
    tiers: dict[int, str] = {}
    for index, request in enumerate(_load_json(trace_path)):
        request_index = request.get("request_number", index)
        for step in request.get("intermediate_steps", []):
            payload = step.get("payload", {})
            if payload.get("event_type") != "TOOL_START" or payload.get("name") != TIER_TOOL:
                continue
            tier = _tool_inputs(payload).get("tier")
            if tier:
                tiers[request_index] = str(tier)
    return tiers


def extract_intents(trace_path: str) -> tuple[dict[int, str], dict[int, str]]:
    """Map request index to executed and declared chat-researcher routes."""
    executed: dict[int, str] = {}
    declared: dict[int, str] = {}
    for index, request in enumerate(_load_json(trace_path)):
        request_index = request.get("request_number", index)
        for step in request.get("intermediate_steps", []):
            payload = step.get("payload", {})
            event_type = payload.get("event_type")
            name = str(payload.get("name") or "")
            if event_type == "FUNCTION_START" and name in INTENT_AGENTS:
                executed.setdefault(request_index, INTENT_AGENTS[name])
            elif event_type == "LLM_END" and _ancestor(step) == "intent_classifier":
                depth = _parse_research_depth((payload.get("data") or {}).get("output"))
                if depth:
                    declared[request_index] = depth
    return executed, declared


def _parse_research_depth(output: object) -> str | None:
    if not output:
        return None
    try:
        parsed = json.loads(str(output))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    depth = parsed.get("research_depth")
    return str(depth) if depth else None


def extract_autonomous_cases(trace_path: str) -> tuple[dict[int, str], dict[int, int]]:
    """Recover autonomous execution case and researcher count.

    Case precedence follows the requested definitions:

    1. any ``shallow-researcher`` delegation -> ``Shallow Researcher``;
    2. otherwise any planner delegation -> ``Planner``;
    3. otherwise researcher count <= 2 -> ``No-Planner-Shallow``;
    4. otherwise -> ``No-Planner-Deep``.

    ``run_research_batch`` spawns one researcher per entry in its ``queries`` list;
    explicit ``task(subagent_type="researcher")`` calls count individually.
    """
    cases: dict[int, str] = {}
    researcher_counts: dict[int, int] = {}

    for index, request in enumerate(_load_json(trace_path)):
        request_index = request.get("request_number", index)
        shallow_called = False
        planner_called = False
        researcher_count = 0

        for step in request.get("intermediate_steps", []):
            payload = step.get("payload", {})
            if payload.get("event_type") != "TOOL_START":
                continue
            name = str(payload.get("name") or "")
            inputs = _tool_inputs(payload)
            if name == "task":
                subagent_type = str(inputs.get("subagent_type") or "").lower()
                if subagent_type == "shallow-researcher":
                    shallow_called = True
                elif "planner" in subagent_type:
                    planner_called = True
                elif "researcher" in subagent_type:
                    researcher_count += 1
            elif name == "run_research_batch":
                queries = inputs.get("queries")
                researcher_count += len(queries) if isinstance(queries, list) else 1

        researcher_counts[request_index] = researcher_count
        if shallow_called:
            cases[request_index] = CASE_SHALLOW
        elif planner_called:
            cases[request_index] = CASE_PLANNER
        elif researcher_count > 2:
            cases[request_index] = CASE_NO_PLANNER_DEEP
        else:
            cases[request_index] = CASE_NO_PLANNER_SHALLOW

    return cases, researcher_counts


def extract_questions(trace_path: str) -> dict[int, str]:
    questions: dict[int, str] = {}
    for index, request in enumerate(_load_json(trace_path)):
        request_index = request.get("request_number", index)
        for step in request.get("intermediate_steps", []):
            payload = step.get("payload", {})
            if payload.get("event_type") == "WORKFLOW_START":
                question = (payload.get("data") or {}).get("input")
                if question:
                    questions.setdefault(request_index, str(question).strip())
    return questions


def resolve_scores_path(trace_path: str, override: str | None) -> Path | None:
    if override:
        return Path(override)
    candidate = Path(trace_path).parent / SCORE_FILENAME
    return candidate if candidate.is_file() else None


def load_scores(scores_path: Path, questions: dict[int, str]) -> tuple[dict[int, float], set[int], float | None, int]:
    """Join evaluator scores/errors to trace requests by normalized question text."""
    payload = _load_json(scores_path)
    if not isinstance(payload, dict):
        return {}, set(), None, 0

    by_question: dict[str, deque[tuple[float, bool]]] = defaultdict(deque)
    by_id: dict[str, tuple[float, bool]] = {}
    for item in payload.get("eval_output_items") or []:
        if not isinstance(item, dict) or item.get("score") is None:
            continue
        value = (float(item["score"]), bool(item.get("error")))
        reasoning = item.get("reasoning") if isinstance(item.get("reasoning"), dict) else {}
        question = reasoning.get("question")
        if question:
            by_question[str(question).strip()].append(value)
        if item.get("id") is not None:
            by_id[str(item["id"])] = value

    scores: dict[int, float] = {}
    error_requests: set[int] = set()
    for request_index, question in questions.items():
        pending = by_question.get(question)
        value = pending.popleft() if pending else by_id.get(str(request_index))
        if value is None:
            continue
        scores[request_index] = value[0]
        if value[1]:
            error_requests.add(request_index)

    file_average = payload.get("average_score")
    total_errors = int(payload.get("total_errors") or len(error_requests))
    return scores, error_requests, (float(file_average) if file_average is not None else None), total_errors


def metric_row(label: str, profiles: list, scores: dict[int, float], error_requests: set[int]) -> dict[str, object]:
    n = len(profiles)
    scored = [scores[p.request_index] for p in profiles if p.request_index in scores]
    return {
        "Number of Queries": n,
        "Errors": sum(p.request_index in error_requests for p in profiles),
        "Case": label,
        "Accuracy/Score": f"{statistics.mean(scored):.3f}" if scored else "n/a",
        "Avg. Latency (Total) (s)": f"{statistics.mean(p.duration_s for p in profiles):.1f}" if n else "n/a",
        "Avg. Token Usage Input": f"{statistics.mean(p.total_prompt_tokens for p in profiles):.1f}" if n else "n/a",
        "Avg. Token Usage Output": f"{statistics.mean(p.total_completion_tokens for p in profiles):.1f}"
        if n
        else "n/a",
        "Avg. No. of LLM calls": f"{statistics.mean(p.total_llm_calls for p in profiles):.1f}" if n else "n/a",
    }


def print_metrics(label: str, profiles: list, scores: dict[int, float], error_requests: set[int]) -> None:
    row = metric_row(label, profiles, scores, error_requests)
    print(f"{label}  (n={row['Number of Queries']}, errors={row['Errors']})")
    print(f"  Avg. Score (accuracy):      {row['Accuracy/Score']:>12}")
    print(f"  Avg. Latency (Total) (s):   {row['Avg. Latency (Total) (s)']:>12}")
    print(f"  Avg. Token Usage Input:     {float(row['Avg. Token Usage Input']):>12,.1f}")
    print(f"  Avg. Token Usage Output:    {float(row['Avg. Token Usage Output']):>12,.1f}")
    print(f"  Avg. No. of LLM calls:      {row['Avg. No. of LLM calls']:>12}")


def grouped_profiles(profiles: list, labels: dict[int, str], order: list[str]) -> list[tuple[str, list]]:
    grouped: dict[str, list] = defaultdict(list)
    for profile in profiles:
        grouped[labels.get(profile.request_index, UNKNOWN)].append(profile)
    names = [name for name in order if name in grouped] + sorted(set(grouped) - set(order))
    return [(name, grouped[name]) for name in names]


def write_csv(
    output_path: Path,
    profiles: list,
    cases: dict[int, str],
    scores: dict[int, float],
    error_requests: set[int],
) -> None:
    rows = [metric_row("Overall", profiles, scores, error_requests)]
    case_groups = dict(grouped_profiles(profiles, cases, CASE_ORDER))
    no_planner_profiles = [
        *case_groups.get(CASE_NO_PLANNER_SHALLOW, []),
        *case_groups.get(CASE_NO_PLANNER_DEEP, []),
    ]
    rows.extend(
        [
            metric_row(CASE_SHALLOW, case_groups.get(CASE_SHALLOW, []), scores, error_requests),
            metric_row(CASE_NO_PLANNER, no_planner_profiles, scores, error_requests),
            metric_row(
                CASE_NO_PLANNER_SHALLOW,
                case_groups.get(CASE_NO_PLANNER_SHALLOW, []),
                scores,
                error_requests,
            ),
            metric_row(CASE_NO_PLANNER_DEEP, case_groups.get(CASE_NO_PLANNER_DEEP, []), scores, error_requests),
            metric_row(CASE_PLANNER, case_groups.get(CASE_PLANNER, []), scores, error_requests),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize metrics from a NAT profiler trace.")
    parser.add_argument("trace", help="Path to all_requests_profiler_traces.json")
    parser.add_argument("--scores", help=f"Evaluator output (default: {SCORE_FILENAME} beside trace)")
    parser.add_argument("--by-tier", action="store_true", help="Print adaptive effort tiers")
    parser.add_argument("--by-intent", action="store_true", help="Print chat-researcher routes")
    parser.add_argument("--by-autonomous-case", action="store_true", help="Print autonomous execution cases")
    parser.add_argument("--per-query", action="store_true", help="Print per-query classifications and metrics")
    parser.add_argument("--csv", type=Path, help="Write overall and autonomous case metrics to CSV")
    args = parser.parse_args()

    zero_pricing = PricingRegistry.from_dict({"default": {"input_per_1m_tokens": 0.0, "output_per_1m_tokens": 0.0}})
    profiles = parse_trace(args.trace, zero_pricing)
    if not profiles:
        print(f"No requests found in {args.trace}", file=sys.stderr)
        return 1

    scores: dict[int, float] = {}
    error_requests: set[int] = set()
    file_average: float | None = None
    total_errors = 0
    scores_path = resolve_scores_path(args.trace, args.scores)
    if scores_path and scores_path.is_file():
        scores, error_requests, file_average, total_errors = load_scores(scores_path, extract_questions(args.trace))
    else:
        print(f"Note: no {SCORE_FILENAME} found; reporting performance only.", file=sys.stderr)

    tiers = extract_tiers(args.trace) if args.by_tier or args.per_query else {}
    executed, declared = extract_intents(args.trace) if args.by_intent or args.per_query else ({}, {})
    cases, researcher_counts = extract_autonomous_cases(args.trace)

    print_metrics("ALL QUERIES", profiles, scores, error_requests)
    if file_average is not None and scores and abs(statistics.mean(scores.values()) - file_average) > 1e-6:
        print(f"  NOTE: joined score average differs from evaluator average_score={file_average:.3f}.")
    if total_errors != len(error_requests):
        print(f"  NOTE: evaluator reports {total_errors} total errors; {len(error_requests)} joined to trace requests.")

    group_specs = []
    if args.by_tier:
        group_specs.append(("TIER", tiers, TIER_ORDER))
    if args.by_intent:
        group_specs.append(("ROUTE", executed, ["shallow", "deep"]))
    if args.by_autonomous_case:
        no_planner_profiles = [
            profile
            for profile in profiles
            if cases[profile.request_index] in {CASE_NO_PLANNER_SHALLOW, CASE_NO_PLANNER_DEEP}
        ]
        print()
        print_metrics(f"CASE: {CASE_NO_PLANNER}", no_planner_profiles, scores, error_requests)
        group_specs.append(("CASE", cases, CASE_ORDER))
    for heading, labels, order in group_specs:
        for name, group in grouped_profiles(profiles, labels, order):
            print()
            print_metrics(f"{heading}: {name}", group, scores, error_requests)

    mismatched = sorted(i for i, value in declared.items() if i in executed and executed[i] != value)
    if mismatched:
        print(f"\nWARNING: declared research_depth != executed route for requests: {mismatched}")

    if args.per_query:
        print("\nquery  case                 researchers  score  latency_s  in_tokens  out_tokens  llm_calls")
        for profile in profiles:
            index = profile.request_index
            score = scores.get(index)
            print(
                f"{index:>5}  {cases[index]:<20} {researcher_counts[index]:>11}  "
                f"{score if score is not None else '-':>5}  {profile.duration_s:>9.1f}  "
                f"{profile.total_prompt_tokens:>9}  {profile.total_completion_tokens:>10}  {profile.total_llm_calls:>9}"
            )

    if args.csv:
        write_csv(
            args.csv,
            profiles,
            cases,
            scores,
            error_requests,
        )
        print(f"\nWrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
