"""Evaluate RAG answers against expected phrases and retrieved documents."""

import json
from pathlib import Path
from typing import Any


class EvaluationError(ValueError):
    """Raised when a dataset or prediction file is invalid."""


def _read_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        with Path(path).open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                    raise EvaluationError(f"{path}:{line_number}: each row needs a non-empty string 'id'")
                if row["id"] in rows:
                    raise EvaluationError(f"{path}:{line_number}: duplicate id {row['id']!r}")
                rows[row["id"]] = row
    except OSError as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    return rows


def _strings(value: Any, field: str, case_id: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise EvaluationError(f"case {case_id!r}: '{field}' must be a list of non-empty strings")
    if required and not value:
        raise EvaluationError(f"case {case_id!r}: '{field}' must contain at least one item")
    return value


def evaluate(dataset_path: str | Path, predictions_path: str | Path, k: int = 5) -> dict[str, Any]:
    """Evaluate answer phrases, lexical evidence support, and retrieval recall@k.

    ``grounded_pass`` is a deterministic lexical check, not a semantic entailment
    judgment: all configured evidence phrases must occur in retrieved text.
    """
    if k < 1:
        raise EvaluationError("k must be at least 1")
    cases = _read_jsonl(dataset_path)
    predictions = _read_jsonl(predictions_path)
    if not cases:
        raise EvaluationError("dataset has no cases")
    missing = cases.keys() - predictions.keys()
    extra = predictions.keys() - cases.keys()
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing predictions: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown prediction IDs: {', '.join(sorted(extra))}")
        raise EvaluationError("; ".join(details))

    results = []
    for case_id, case in cases.items():
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(f"case {case_id!r}: 'question' must be a non-empty string")
        answer_phrases = _strings(case.get("answer_contains"), "answer_contains", case_id, required=True)
        evidence_phrases = _strings(case.get("evidence_contains", answer_phrases), "evidence_contains", case_id, required=True)
        relevant = set(_strings(case.get("relevant_doc_ids"), "relevant_doc_ids", case_id, required=True))
        prediction = predictions[case_id]
        answer = prediction.get("answer")
        retrieved = prediction.get("retrieved")
        if not isinstance(answer, str):
            raise EvaluationError(f"prediction {case_id!r}: 'answer' must be a string")
        if not isinstance(retrieved, list) or any(not isinstance(doc, dict) for doc in retrieved):
            raise EvaluationError(f"prediction {case_id!r}: 'retrieved' must be a list of objects")
        for doc in retrieved:
            if not isinstance(doc.get("id"), str) or not isinstance(doc.get("text"), str):
                raise EvaluationError(f"prediction {case_id!r}: each retrieved item needs string 'id' and 'text'")

        answer_lower = answer.casefold()
        evidence_lower = " ".join(doc["text"] for doc in retrieved).casefold()
        missing_answer = [phrase for phrase in answer_phrases if phrase.casefold() not in answer_lower]
        missing_evidence = [phrase for phrase in evidence_phrases if phrase.casefold() not in evidence_lower]
        retrieved_ids = {doc["id"] for doc in retrieved[:k]}
        found_relevant = sorted(retrieved_ids & relevant)
        recall = len(found_relevant) / len(relevant)
        results.append({
            "id": case_id,
            "answer_pass": not missing_answer,
            "grounded_pass": not missing_answer and not missing_evidence,
            "retrieval_recall": recall,
            "missing_answer_phrases": missing_answer,
            "missing_evidence_phrases": missing_evidence,
            "missing_relevant_doc_ids": sorted(relevant - retrieved_ids),
        })

    count = len(results)
    return {
        "cases": results,
        "summary": {
            "count": count,
            "answer_pass_rate": sum(row["answer_pass"] for row in results) / count,
            "grounded_pass_rate": sum(row["grounded_pass"] for row in results) / count,
            "retrieval_recall_at_k": sum(row["retrieval_recall"] for row in results) / count,
            "k": k,
        },
    }
