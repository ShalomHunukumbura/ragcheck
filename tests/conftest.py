import json
from pathlib import Path

import pytest


@pytest.fixture
def jsonl(tmp_path: Path):
    """Write rows to a JSONL file and return its path."""

    def write(name: str, *rows: dict) -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    return write


CASE = {
    "id": "refund-window",
    "question": "How long do customers have to request a refund?",
    "answer_contains": ["30 days"],
    "relevant_doc_ids": ["policy-refunds"],
}

GOOD = {
    "id": "refund-window",
    "answer": "You can request a refund up to 30 days after delivery.",
    "retrieved": [{"id": "policy-refunds", "text": "You can request a refund up to 30 days after delivery."}],
}
