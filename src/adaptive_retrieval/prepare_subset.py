"""Build the local corpus from the document subset committed to Git."""

import argparse
import json
from pathlib import Path

from .data import clean_document_text
from .paths import PROCESSED_DIR, SUBSET_DOCUMENTS_DIR


def prepare_subset(root: Path, out: Path) -> tuple[int, int]:
    questions_path = root / "questions_100.jsonl"
    if not questions_path.is_file():
        raise FileNotFoundError(f"Question file not found: {questions_path}")

    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError(f"Question file is empty: {questions_path}")

    gold_ids = {doc_id for question in questions for doc_id in question["expected_doc_ids"]}
    mined_for: dict[str, list[str]] = {}
    for question in questions:
        for doc_id in question["hard_negative_ids"]:
            mined_for.setdefault(doc_id, []).append(question["question_id"])
    required_ids = gold_ids | mined_for.keys()

    documents: dict[str, Path] = {}
    for path in root.rglob("*.txt"):
        doc_id = path.name.split("__", 1)[0]
        if doc_id in documents:
            raise ValueError(f"Duplicate document ID {doc_id}: {documents[doc_id]} and {path}")
        documents[doc_id] = path
    missing = required_ids - documents.keys()
    if missing:
        raise ValueError(
            f"Missing {len(missing)} referenced documents in {root}: {sorted(missing)[:5]}"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "subset_questions.jsonl").write_bytes(questions_path.read_bytes())
    with (out / "subset_corpus.jsonl").open("w", encoding="utf-8") as stream:
        for doc_id in sorted(required_ids):
            path = documents[doc_id]
            relative_path = path.relative_to(root)
            stream.write(json.dumps({
                "doc_id": doc_id,
                "source": relative_path.parts[0],
                "path": relative_path.as_posix(),
                "is_gold": doc_id in gold_ids,
                "mined_for": mined_for.get(doc_id, []) if doc_id not in gold_ids else [],
                "text": clean_document_text(path.read_text(encoding="utf-8")),
            }) + "\n")
    return len(questions), len(required_ids)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the checked-in evaluation subset")
    parser.add_argument("--root", type=Path, default=SUBSET_DOCUMENTS_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)
    questions, documents = prepare_subset(args.root, args.out)
    print(f"Prepared {questions} questions and {documents} documents in {args.out}")
