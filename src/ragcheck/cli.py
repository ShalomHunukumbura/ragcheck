"""Command line interface for ragcheck."""

import argparse
import json
import os
import sys
from pathlib import Path

from ragcheck.evaluate import EvaluationError, evaluate

# (summary key, label) in report order.
METRICS = [
    ("answer_pass_rate", "Answer pass rate"),
    ("grounded_pass_rate", "Grounded pass rate"),
    ("retrieval_recall_at_k", "Retrieval recall@{k}"),
]


def _rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= rate <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return rate


def _failed_cases(report: dict) -> list[dict]:
    return [
        case for case in report["cases"]
        if not case["answer_pass"] or not case["grounded_pass"] or case["missing_relevant_doc_ids"]
    ]


def _problems(case: dict) -> list[str]:
    problems = []
    if case["missing_answer_phrases"]:
        problems.append("missing answer phrase: " + ", ".join(f'"{p}"' for p in case["missing_answer_phrases"]))
    if case["missing_evidence_phrases"]:
        problems.append("missing evidence phrase: " + ", ".join(f'"{p}"' for p in case["missing_evidence_phrases"]))
    if case["missing_relevant_doc_ids"]:
        problems.append("missing relevant docs: " + ", ".join(case["missing_relevant_doc_ids"]))
    return problems


def _below(report: dict, thresholds: dict[str, float]) -> list[str]:
    return [key for key, _ in METRICS if report["summary"][key] < thresholds[key]]


def _text(report: dict, thresholds: dict[str, float]) -> str:
    """The report for a terminal: aligned metrics, then each case that needs attention."""
    summary = report["summary"]
    below = _below(report, thresholds)
    labels = [label.format(k=summary["k"]) for _, label in METRICS]
    width = max(len(label) for label in labels)
    cases = f"{summary['count']} case{'s' if summary['count'] != 1 else ''}"

    lines = [f"ragcheck: {cases}, retrieval cutoff k={summary['k']}", ""]
    for (key, _), label in zip(METRICS, labels):
        status = "below" if key in below else "ok"
        lines.append(f"  {label:<{width}}  {summary[key]:>6.1%}  needs {thresholds[key]:>6.1%}  {status}")

    failed = _failed_cases(report)
    if failed:
        lines += ["", "Cases needing attention:"]
        for case in failed:
            lines.append(f"  {case['id']}")
            lines += [f"    {problem}" for problem in _problems(case)]

    lines.append("")
    if below:
        lines.append(f"RAG checks failed: {len(below)} of {len(METRICS)} metrics below their threshold.")
    else:
        lines.append(f"RAG checks passed: all {len(METRICS)} metrics meet their threshold.")
    return "\n".join(lines)


def _markdown(report: dict, thresholds: dict[str, float]) -> str:
    """The same report as Markdown, for the GitHub Actions job summary."""
    summary = report["summary"]
    rows = ["| Metric | Result | Required |", "|---|---:|---:|"]
    for key, label in METRICS:
        rows.append(f"| {label.format(k=summary['k'])} | {summary[key]:.1%} | {thresholds[key]:.1%} |")
    rows.append("")
    failed = _failed_cases(report)
    if failed:
        rows += ["<details><summary>Cases needing attention</summary>", "", "| Case | Details |", "|---|---|"]
        rows += [f"| `{case['id']}` | {'; '.join(_problems(case))} |" for case in failed]
        rows += ["", "</details>", ""]
    rows.append("**RAG checks passed.**" if report["passed"] else "**RAG checks failed.**")
    return "\n".join(rows) + "\n"


def _annotations(report: dict, thresholds: dict[str, float]) -> str:
    """Workflow commands GitHub shows on the pull request's checks."""
    summary = report["summary"]
    labels = dict(METRICS)
    return "\n".join(
        f"::error title=ragcheck::{labels[key].format(k=summary['k'])} {summary[key]:.1%} "
        f"is below the required {thresholds[key]:.1%}"
        for key in _below(report, thresholds)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RAG answers and retrieval against a baseline.")
    parser.add_argument("--dataset", required=True, help="JSONL questions with expected answer phrases and document IDs")
    parser.add_argument("--predictions", required=True, help="JSONL pipeline outputs with answers and retrieved text")
    parser.add_argument("--k", type=int, default=5, help="retrieval cutoff (default: 5)")
    parser.add_argument("--min-answer-pass-rate", type=_rate, default=1.0)
    parser.add_argument("--min-grounded-pass-rate", type=_rate, default=1.0)
    parser.add_argument("--min-retrieval-recall", type=_rate, default=1.0)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    thresholds = {
        "answer_pass_rate": args.min_answer_pass_rate,
        "grounded_pass_rate": args.min_grounded_pass_rate,
        "retrieval_recall_at_k": args.min_retrieval_recall,
    }
    try:
        report = evaluate(args.dataset, args.predictions, args.k)
    except EvaluationError as exc:
        parser.error(str(exc))
    report["passed"] = not _below(report, thresholds)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_text(report, thresholds))
        if os.environ.get("GITHUB_ACTIONS") == "true" and not report["passed"]:
            print(_annotations(report, thresholds))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as file:
            file.write(_markdown(report, thresholds))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
