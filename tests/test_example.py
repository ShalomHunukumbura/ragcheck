"""The example pipeline is the README's story: it passes, and the 'harmless' change fails."""

import os
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(tmp_path: Path, degrade: bool) -> subprocess.CompletedProcess:
    predictions = tmp_path / "predictions.jsonl"
    env = {
        **os.environ,
        "RAGCHECK_DATASET": str(EXAMPLES / "questions.jsonl"),
        "RAGCHECK_PREDICTIONS": str(predictions),
        "RAGCHECK_DEGRADE": "1" if degrade else "0",
    }
    env.pop("GITHUB_ACTIONS", None)
    subprocess.run([sys.executable, str(EXAMPLES / "run_pipeline.py")], env=env, check=True)
    return subprocess.run(
        [sys.executable, "-m", "ragcheck", "--dataset", env["RAGCHECK_DATASET"],
         "--predictions", str(predictions), "--k", "3"],
        capture_output=True, text=True, env=env,
    )


def test_the_example_pipeline_passes(tmp_path: Path) -> None:
    result = run_example(tmp_path, degrade=False)
    assert result.returncode == 0, result.stdout


def test_trimming_documents_breaks_the_answers(tmp_path: Path) -> None:
    result = run_example(tmp_path, degrade=True)
    assert result.returncode == 1
    assert "Answer pass rate     20.0%" in result.stdout
    assert "Retrieval recall@3   60.0%" in result.stdout
    for case in ("refund-window", "colombo-delivery", "cash-on-delivery", "delete-account"):
        assert f"  {case}\n" in result.stdout
    assert "kettle-warranty" not in result.stdout  # its fact is in the first sentence
