# ragcheck

**Catch RAG regressions in pull requests.** Run your pipeline on a small set of questions
with known answers, and fail CI when the right answer, its supporting text, or the right
document stops coming back.

Changes that break a RAG pipeline rarely look dangerous. Here someone trims every document
to its first sentence "to save tokens". The code review is easy, the tests pass, and the
assistant stops answering four questions out of five:

```console
$ ragcheck-ci --dataset questions.jsonl --predictions predictions.jsonl --k 3
ragcheck: 5 cases, retrieval cutoff k=3

  Answer pass rate     20.0%  needs 100.0%  below
  Grounded pass rate   20.0%  needs 100.0%  below
  Retrieval recall@3   60.0%  needs 100.0%  below

Cases needing attention:
  refund-window
    missing answer phrase: "30 days"
    missing evidence phrase: "30 days"
  colombo-delivery
    missing answer phrase: "2 working days"
    missing evidence phrase: "2 working days"
    missing relevant docs: shipping-times
  cash-on-delivery
    missing answer phrase: "cash on delivery"
    missing evidence phrase: "cash on delivery"
    missing relevant docs: payment-methods
  delete-account
    missing answer phrase: "Settings"
    missing evidence phrase: "Settings"

RAG checks failed: 3 of 3 metrics below their threshold.
```

That's the pipeline in [`examples/`](examples/) with `RAGCHECK_DEGRADE=1`. See it block a
real pull request in [ragcheck-demo](https://github.com/ShalomHunukumbura/ragcheck-demo/pulls).

The checks are deterministic: no LLM judge, no API key, and the same inputs always give the
same result, so a failure in CI means something changed in your pipeline.

## Install

```bash
pip install ragcheck-ci
```

No dependencies beyond Python 3.10+. (`ragcheck` was taken on PyPI, so the package and
command are `ragcheck-ci`.)

## 1. Write down what good looks like

One JSON object per line, for questions your users really ask:

```json
{"id": "refund-window", "question": "How long do customers have to request a refund?", "answer_contains": ["30 days"], "relevant_doc_ids": ["policy-refunds"]}
```

| Field | Meaning |
|---|---|
| `answer_contains` | Phrases the answer must include. |
| `evidence_contains` | Phrases the retrieved text must include. Defaults to `answer_contains`. |
| `relevant_doc_ids` | Documents that should be in the top *k* results. |

Ten to thirty questions covering your most important answers is enough to start.

## 2. Have your pipeline write what it did

For each question, your pipeline writes one line with the same `id`, its answer, and the
chunks it retrieved, in order:

```json
{"id": "refund-window", "answer": "You can request a refund up to 30 days after delivery.", "retrieved": [{"id": "policy-refunds", "text": "Refunds are handled by our support team. You can request a refund up to 30 days after delivery."}]}
```

[`examples/run_pipeline.py`](examples/run_pipeline.py) is a small working pipeline that
does this. Replace its retrieval and answer steps with calls to your own.

## 3. Check it

```bash
ragcheck-ci --dataset questions.jsonl --predictions predictions.jsonl \
  --min-answer-pass-rate 0.95 --min-grounded-pass-rate 0.95 --min-retrieval-recall 0.9
```

| Metric | A case passes when |
|---|---|
| Answer pass rate | every `answer_contains` phrase is in the answer |
| Grounded pass rate | the answer passes **and** every `evidence_contains` phrase is in the retrieved text, so the answer is backed by what was retrieved |
| Retrieval recall@k | the share of `relevant_doc_ids` found in the first *k* retrieved items |

Exit code `0` = every metric meets its threshold, `1` = at least one is below, `2` = bad
input (the message names the file, line or case). `--json` prints the full report.

## In GitHub Actions

```yaml
- uses: ShalomHunukumbura/ragcheck@v0
  with:
    dataset: eval/questions.jsonl
    predictions: /tmp/predictions.jsonl
    command: python eval/run_pipeline.py
    min-answer-pass-rate: "0.95"
    min-grounded-pass-rate: "0.95"
    min-retrieval-recall: "0.90"
    k: "5"
```

The Action runs your `command` with `RAGCHECK_DATASET` and `RAGCHECK_PREDICTIONS` set, then
checks the result. Metrics below their threshold fail the check and show up as errors on the
pull request, and the full table goes in the job summary.

## Limits

- Phrase matching is a regression signal, not a judge of correctness. A paraphrase ("one
  month" for "30 days") fails, and a long answer can contain the phrase and still say
  something wrong. Pick phrases that any correct answer must contain: numbers, names, steps.
- It checks the questions you give it, nothing else.
- If your pipeline isn't deterministic (sampling temperature above 0), lower the thresholds
  a little or results will flicker.

## License

MIT
