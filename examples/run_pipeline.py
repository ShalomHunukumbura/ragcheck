"""A tiny but real RAG pipeline, to show what ragcheck-ci checks.

It retrieves the documents that share the most words with the question, then answers
with the best-matching sentence of the top document. Replace it with calls to your own
retriever and model; ragcheck only needs the output format at the bottom.

RAGCHECK_DEGRADE=1 makes a change that looks harmless: keep only the first sentence of
each document, "to save tokens". Retrieval mostly still works. The answers don't.
"""

import json
import os
import re
from pathlib import Path

TOP_K = 3
STOPWORDS = {"a", "an", "and", "are", "can", "do", "does", "for", "have", "how", "i", "in",
             "is", "my", "of", "on", "the", "to", "what", "when", "where", "with", "you"}


def terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w.removesuffix("s") for w in words if w not in STOPWORDS}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip()) if s]


def answer(question: str, docs: list[dict]) -> str:
    if not docs:
        return "I couldn't find that."
    wanted = terms(question)
    return max(sentences(docs[0]["text"]), key=lambda s: len(wanted & terms(s)))


def main() -> None:
    dataset = Path(os.environ["RAGCHECK_DATASET"])
    output = Path(os.environ["RAGCHECK_PREDICTIONS"])
    knowledge = [json.loads(line) for line in Path(__file__).with_name("knowledge.jsonl").open()]
    if os.environ.get("RAGCHECK_DEGRADE") == "1":
        knowledge = [{**doc, "text": sentences(doc["text"])[0]} for doc in knowledge]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for line in dataset.open():
            if not line.strip():
                continue
            case = json.loads(line)
            wanted = terms(case["question"])
            ranked = sorted(knowledge, key=lambda doc: len(wanted & terms(doc["text"])), reverse=True)
            retrieved = [doc for doc in ranked[:TOP_K] if wanted & terms(doc["text"])]
            row = {"id": case["id"], "answer": answer(case["question"], retrieved), "retrieved": retrieved}
            out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
