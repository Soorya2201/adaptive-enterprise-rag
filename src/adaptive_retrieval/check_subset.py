"""Check that the subset corpus contains every document its questions reference.

For each question in subset_questions.jsonl, every expected (gold) doc id and
every hard negative id must be present both in subset_corpus.jsonl and as a
file under data/subset_documents/. Also reports documents nobody references.

Run through ``python src/runner.py check-subset`` from any directory.
"""

import json
import sys
from pathlib import Path

from .data import clean_document_text
from .paths import PROCESSED_DIR, SUBSET_DOCUMENTS_DIR


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=PROCESSED_DIR / "subset_questions.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROCESSED_DIR / "subset_corpus.jsonl")
    parser.add_argument("--docs-dir", type=Path, default=SUBSET_DOCUMENTS_DIR)
    args = parser.parse_args(argv)
    questions_path, corpus_path, docs_dir = args.questions, args.corpus, args.docs_dir

    with questions_path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    with corpus_path.open(encoding="utf-8") as f:
        corpus = {d["doc_id"]: d for d in map(json.loads, f)}
    files = {p.name.split("__", 1)[0]: p for p in docs_dir.rglob("*.txt")}

    errors = []
    referenced = set()
    for q in questions:
        for kind in ("expected_doc_ids", "hard_negative_ids"):
            for doc_id in q[kind]:
                referenced.add(doc_id)
                if doc_id not in corpus:
                    errors.append(
                        f"{q['question_id']}: {kind} {doc_id} missing from {corpus_path}"
                    )
                if doc_id not in files:
                    errors.append(
                        f"{q['question_id']}: {kind} {doc_id} missing from {docs_dir}"
                    )
        if q["question_type"] != "info_not_found" and not q["expected_doc_ids"]:
            errors.append(f"{q['question_id']}: no expected_doc_ids")

    for doc_id, doc in corpus.items():
        if (
            doc_id in files
            and clean_document_text(files[doc_id].read_text(encoding="utf-8", errors="ignore"))
            != doc["text"]
        ):
            errors.append(f"{doc_id}: text in {corpus_path} differs from {files[doc_id]}")

    gold = {d for q in questions for d in q["expected_doc_ids"]}
    print(f"questions:           {len(questions)}")
    print(f"gold docs referenced: {len(gold)}")
    print(f"all docs referenced:  {len(referenced)}")
    print(f"docs in corpus jsonl: {len(corpus)}")
    print(f"docs in {docs_dir}: {len(files)}")
    for label, extra in (
        ("corpus jsonl", corpus.keys() - referenced),
        (str(docs_dir), files.keys() - referenced),
    ):
        if extra:
            print(
                f"note: {len(extra)} unreferenced docs in {label}: {sorted(extra)[:5]}"
            )

    if errors:
        print(f"\nFAILED: {len(errors)} problems")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print("\nOK: every referenced document is present")


if __name__ == "__main__":
    main()
