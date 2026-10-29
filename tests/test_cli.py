import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import CASE, GOOD

WRONG = {**GOOD, "answer": "Refunds are available for 7 days."}


def run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if k not in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY")}
    return subprocess.run(
        [sys.executable, "-m", "ragcheck", *args],
        capture_output=True,
        text=True,
        env={**clean, **(env or {})},
    )


def test_passing_run(jsonl) -> None:
    result = run("--dataset", str(jsonl("d.jsonl", CASE)), "--predictions", str(jsonl("p.jsonl", GOOD)))
    assert result.returncode == 0
    assert "RAG checks passed: all 3 metrics meet their threshold." in result.stdout
    assert "  Answer pass rate    100.0%  needs 100.0%  ok" in result.stdout


def test_failing_run_lists_the_cases(jsonl) -> None:
    result = run("--dataset", str(jsonl("d.jsonl", CASE)), "--predictions", str(jsonl("p.jsonl", WRONG)))
    assert result.returncode == 1
    assert "Cases needing attention:\n  refund-window\n    missing answer phrase: \"30 days\"" in result.stdout
    assert "RAG checks failed: 2 of 3 metrics below their threshold." in result.stdout
    assert "::error" not in result.stdout  # only on GitHub Actions


def test_thresholds_decide_the_result(jsonl) -> None:
    second = {**CASE, "id": "other"}
    dataset = jsonl("d.jsonl", CASE, second)
    predictions = jsonl("p.jsonl", GOOD, {**WRONG, "id": "other"})
    common = ["--dataset", str(dataset), "--predictions", str(predictions)]
    assert run(*common).returncode == 1
    lenient = run(*common, "--min-answer-pass-rate", "0.5", "--min-grounded-pass-rate", "0.5")
    assert lenient.returncode == 0


def test_json_output(jsonl) -> None:
    result = run("--dataset", str(jsonl("d.jsonl", CASE)), "--predictions", str(jsonl("p.jsonl", WRONG)), "--json")
    report = json.loads(result.stdout)
    assert report["passed"] is False
    assert report["cases"][0]["missing_answer_phrases"] == ["30 days"]


def test_github_actions_summary_and_annotations(jsonl, tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    result = run(
        "--dataset", str(jsonl("d.jsonl", CASE)), "--predictions", str(jsonl("p.jsonl", WRONG)),
        env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary)},
    )
    assert "::error title=ragcheck::Answer pass rate 0.0% is below the required 100.0%" in result.stdout
    markdown = summary.read_text()
    assert "| Answer pass rate | 0.0% | 100.0% |" in markdown
    assert "| `refund-window` | missing answer phrase: \"30 days\"" in markdown
    assert markdown.endswith("**RAG checks failed.**\n")


def test_bad_input_is_a_usage_error(jsonl, tmp_path: Path) -> None:
    result = run("--dataset", str(jsonl("d.jsonl", CASE)), "--predictions", str(tmp_path / "missing.jsonl"))
    assert result.returncode == 2
    assert "cannot read" in result.stderr


def test_rates_must_be_between_0_and_1(jsonl) -> None:
    result = run("--dataset", "d", "--predictions", "p", "--min-answer-pass-rate", "95")
    assert result.returncode == 2
    assert "must be between 0 and 1" in result.stderr
