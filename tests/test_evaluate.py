import pytest

from conftest import CASE, GOOD
from ragcheck import EvaluationError, evaluate


def test_a_good_prediction_passes_everything(jsonl) -> None:
    report = evaluate(jsonl("d.jsonl", CASE), jsonl("p.jsonl", GOOD))
    assert report["summary"] == {
        "count": 1,
        "answer_pass_rate": 1.0,
        "grounded_pass_rate": 1.0,
        "retrieval_recall_at_k": 1.0,
        "k": 5,
    }
    assert report["cases"][0]["missing_answer_phrases"] == []


def test_a_wrong_answer_fails_answer_and_grounding(jsonl) -> None:
    wrong = {**GOOD, "answer": "Refunds are available for 7 days."}
    [case] = evaluate(jsonl("d.jsonl", CASE), jsonl("p.jsonl", wrong))["cases"]
    assert not case["answer_pass"]
    assert not case["grounded_pass"]
    assert case["missing_answer_phrases"] == ["30 days"]
    assert case["retrieval_recall"] == 1.0  # retrieval was fine


def test_a_right_answer_without_evidence_is_not_grounded(jsonl) -> None:
    # The model said "30 days", but nothing it retrieved says so.
    unsupported = {**GOOD, "retrieved": [{"id": "policy-refunds", "text": "Refunds are handled by support."}]}
    [case] = evaluate(jsonl("d.jsonl", CASE), jsonl("p.jsonl", unsupported))["cases"]
    assert case["answer_pass"]
    assert not case["grounded_pass"]
    assert case["missing_evidence_phrases"] == ["30 days"]


def test_evidence_phrases_can_differ_from_answer_phrases(jsonl) -> None:
    case = {**CASE, "evidence_contains": ["after delivery"]}
    [result] = evaluate(jsonl("d.jsonl", case), jsonl("p.jsonl", GOOD))["cases"]
    assert result["grounded_pass"]


def test_matching_ignores_case(jsonl) -> None:
    case = {**CASE, "answer_contains": ["30 DAYS"]}
    [result] = evaluate(jsonl("d.jsonl", case), jsonl("p.jsonl", GOOD))["cases"]
    assert result["answer_pass"]


def test_recall_counts_relevant_docs_in_the_top_k(jsonl) -> None:
    case = {**CASE, "relevant_doc_ids": ["policy-refunds", "returns-form"]}
    docs = [{"id": f"filler-{i}", "text": "30 days"} for i in range(2)]
    prediction = {**GOOD, "retrieved": docs + GOOD["retrieved"] + [{"id": "returns-form", "text": ""}]}
    # policy-refunds is 3rd, returns-form is 4th.
    assert evaluate(jsonl("d.jsonl", case), jsonl("p.jsonl", prediction), k=4)["cases"][0]["retrieval_recall"] == 1.0
    [result] = evaluate(jsonl("d.jsonl", case), jsonl("p.jsonl", prediction), k=3)["cases"]
    assert result["retrieval_recall"] == 0.5
    assert result["missing_relevant_doc_ids"] == ["returns-form"]


def test_rates_average_over_cases(jsonl) -> None:
    second = {**CASE, "id": "other"}
    wrong = {**GOOD, "id": "other", "answer": "No idea."}
    summary = evaluate(jsonl("d.jsonl", CASE, second), jsonl("p.jsonl", GOOD, wrong))["summary"]
    assert summary["answer_pass_rate"] == 0.5


@pytest.mark.parametrize(
    ("dataset", "predictions", "message"),
    [
        ([CASE], [], "missing predictions: refund-window"),
        ([CASE], [GOOD, {**GOOD, "id": "extra"}], "unknown prediction IDs: extra"),
        ([CASE, CASE], [GOOD], "duplicate id 'refund-window'"),
        ([], [], "dataset has no cases"),
        ([{**CASE, "question": ""}], [GOOD], "'question' must be a non-empty string"),
        ([{**CASE, "answer_contains": []}], [GOOD], "'answer_contains' must contain at least one item"),
        ([{**CASE, "relevant_doc_ids": "policy-refunds"}], [GOOD], "'relevant_doc_ids' must be a list"),
        ([CASE], [{**GOOD, "answer": None}], "'answer' must be a string"),
        ([CASE], [{**GOOD, "retrieved": [{"id": "x"}]}], "needs string 'id' and 'text'"),
        ([{"question": "no id"}], [GOOD], "non-empty string 'id'"),
    ],
)
def test_invalid_input_is_explained(jsonl, dataset, predictions, message) -> None:
    with pytest.raises(EvaluationError, match=message.replace("(", r"\(").replace(")", r"\)")):
        evaluate(jsonl("d.jsonl", *dataset), jsonl("p.jsonl", *predictions))


def test_bad_json_names_the_line(tmp_path, jsonl) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n{not json\n')
    with pytest.raises(EvaluationError, match=r"bad.jsonl:2: invalid JSON"):
        evaluate(bad, jsonl("p.jsonl", GOOD))


def test_missing_file(tmp_path, jsonl) -> None:
    with pytest.raises(EvaluationError, match="cannot read"):
        evaluate(tmp_path / "nope.jsonl", jsonl("p.jsonl", GOOD))


def test_k_must_be_positive(jsonl) -> None:
    with pytest.raises(EvaluationError, match="k must be at least 1"):
        evaluate(jsonl("d.jsonl", CASE), jsonl("p.jsonl", GOOD), k=0)
