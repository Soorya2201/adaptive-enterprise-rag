"""Freeze a 300-question benchmark using real EnterpriseRAG-Bench documents."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

from .answer_evaluation import keyed, read_records
from .build_subset import tokenize
from .vector_index import file_sha256

QUOTAS = {"basic": 100, "semantic": 65, "intra_document_reasoning": 25,
          "constrained": 15, "project_related": 30, "completeness": 20,
          "conflicting_info": 20, "info_not_found": 10, "miscellaneous": 15}
SEED = 13


def canonical_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_final(questions, legacy, *, quotas=None):
    """Exclude legacy IDs, then split connected gold-document groups together.

    Any group touching previously evaluated gold documents is development only.
    Grouping takes priority over exact split proportions.
    """
    keyed(questions)
    previous = keyed(legacy)
    exposed_docs = {doc for q in legacy for doc in q["expected_doc_ids"]}
    exposed_text = {q["question"].strip().casefold() for q in legacy}
    rng = random.Random(SEED)
    selected = []
    for kind, count in (QUOTAS if quotas is None else quotas).items():
        pool = sorted((q for q in questions if q["question_type"] == kind
                       and q["question_id"] not in previous
                       and q["question"].strip().casefold() not in exposed_text),
                      key=lambda q: q["question_id"])
        if len(pool) < count:
            raise ValueError(f"{kind}: need {count}, only {len(pool)} unused questions")
        rng.shuffle(pool)
        selected.extend(dict(q) for q in pool[:count])
    parent = list(range(len(selected)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, question in enumerate(selected):
        if not question["expected_doc_ids"] and question["question_type"] != "info_not_found":
            raise ValueError("Only explicit info_not_found records may have no gold documents")
        tokens = ["doc:" + doc for doc in question["expected_doc_ids"]]
        tokens.append("text:" + question["question"].strip().casefold())
        for token in tokens:
            if token in owners:
                parent[root(i)] = root(owners[token])
            owners[token] = i
    groups = {}
    for i, question in enumerate(selected):
        groups.setdefault(root(i), []).append(question)
    counts = Counter()
    targets = {"dev": .70 * len(selected), "val": .15 * len(selected), "test": .15 * len(selected)}
    ordered = sorted(groups.values(), key=lambda group: (-len(group), group[0]["question_id"]))
    for group in ordered:
        exposed = any(set(q["expected_doc_ids"]) & exposed_docs for q in group)
        if exposed:
            split = "dev"
        else:
            split = min(targets, key=lambda s: counts[s] / max(targets[s], 1))
        group_id = min(q["question_id"] for q in group)
        for question in group:
            question.update(split=split, evaluation_group=group_id,
                            legacy_document_exposure=exposed)
            question.pop("hard_negative_ids", None)
        counts[split] += len(group)
    return sorted(selected, key=lambda q: q["question_id"])


def build_benchmark(source: Path, legacy_path: Path, archive: Path, out: Path):
    questions, legacy = read_records(source), read_records(legacy_path)
    selected = select_final(questions, legacy)
    needed = {d for q in selected + legacy for d in q["expected_doc_ids"] + q.get("hard_negative_ids", [])}
    documents = {}
    with zipfile.ZipFile(archive) as stream:
        members = {}
        for entry in stream.infolist():
            path = PurePosixPath(entry.filename)
            if path.suffix != ".txt" or path.name.startswith("._"):
                continue
            doc_id = path.name.split("__", 1)[0]
            if doc_id not in needed:
                continue
            members.setdefault(doc_id, []).append(entry)
        if needed - members.keys():
            raise ValueError(f"Missing documents in archive: {sorted(needed - members.keys())[:5]}")
        for doc_id in sorted(needed):
            entries = sorted(members[doc_id], key=lambda entry: entry.filename)
            # Read members directly; no extraction of untrusted archive paths.
            variants = [{"path": entry.filename, "text": stream.read(entry).decode("utf-8")}
                        for entry in entries]
            text = variants[0]["text"] if len(variants) == 1 else "\n\n".join(
                f"SOURCE VARIANT: {v['path']}\n{v['text']}" for v in variants)
            # Offsets in a combined record refer to that record, not to the
            # first original file. Never attribute a later variant to its sibling.
            provenance = entries[0].filename if len(entries) == 1 else f"combined-source-variants/{doc_id}"
            documents[doc_id] = {"doc_id": doc_id, "path": provenance,
                                 "source": entries[0].filename.split("/")[0],
                                 "source_paths": [v["path"] for v in variants], "text": text}
    # Mine two deterministic lexical distractors within this documented corpus.
    # This is a bounded evaluation corpus, not full-archive hard-negative mining.
    terms = {key: Counter(tokenize(doc["text"])) for key, doc in documents.items()}
    for question in selected:
        words = set(tokenize(question["question"]))
        candidates = [key for key in documents if key not in question["expected_doc_ids"]]
        candidates.sort(key=lambda key: (-sum(min(terms[key][word], 3) for word in words), key))
        question["hard_negative_ids"] = candidates[:2]
    manifest = {
        "version": 2, "seed": SEED,
        "source": "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/tag/v1.0.0",
        "inputs": {"source_questions_canonical_sha256": canonical_hash(questions),
                   "legacy_questions_canonical_sha256": canonical_hash(legacy),
                   "archive_sha256": file_sha256(archive)},
        "questions": len(selected), "documents": len(documents),
        "multi_document_questions": sum(len(set(q["expected_doc_ids"])) > 1 for q in selected),
        "duplicate_id_policy": "retain all sorted source variants under the shared original document ID",
        "duplicate_id_documents": {key: doc["source_paths"] for key, doc in documents.items() if len(doc["source_paths"]) > 1},
        "by_split": dict(Counter(q["split"] for q in selected)),
        "by_type": dict(Counter(q["question_type"] for q in selected)),
        "split_policy": "gold-document connected groups; legacy gold exposure dev only",
        "corpus_policy": "selected gold documents plus legacy gold/hard-negative documents",
        "negative_policy": "top two capped term-frequency overlap documents in bounded corpus",
        "documents_sha256": {key: hashlib.sha256(doc["text"].encode()).hexdigest() for key, doc in documents.items()},
    }
    # A frozen output is never silently replaced.
    out.mkdir(parents=True, exist_ok=False)
    for name, records in (("questions.jsonl", selected), ("corpus.jsonl", list(documents.values()))):
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8", newline="\n")
    manifest["questions_sha256"] = file_sha256(out / "questions.jsonl")
    manifest["corpus_sha256"] = file_sha256(out / "corpus.jsonl")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_benchmark(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name in ("questions", "corpus"):
        if file_sha256(root / f"{name}.jsonl") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
    questions = read_records(root / "questions.jsonl")
    keyed(questions)
    docs = {row["doc_id"]: row for row in read_records(root / "corpus.jsonl")}
    split_docs = {}
    for q in questions:
        if q["split"] not in {"dev", "val", "test"}:
            raise ValueError("Unknown split")
        if q["legacy_document_exposure"] and q["split"] != "dev":
            raise ValueError("Legacy document exposed outside dev")
        for doc_id in q["expected_doc_ids"] + q["hard_negative_ids"]:
            if doc_id not in docs:
                raise ValueError(f"Missing document: {doc_id}")
        for doc_id in q["expected_doc_ids"]:
            if doc_id in split_docs and split_docs[doc_id] != q["split"]:
                raise ValueError("Gold document crosses splits")
            split_docs[doc_id] = q["split"]
    if len(questions) != manifest["questions"] or len(docs) != manifest["documents"]:
        raise ValueError("Manifest count mismatch")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/subset_documents/questions.jsonl"))
    parser.add_argument("--legacy", type=Path, default=Path("data/subset_documents/questions_100.jsonl"))
    parser.add_argument("--archive", type=Path, default=Path("data/all_documents.zip"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/final300-v2"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_benchmark(args.out) if args.verify else build_benchmark(args.source, args.legacy, args.archive, args.out)
        print(json.dumps({key: result[key] for key in ("questions", "documents", "multi_document_questions", "by_split")}, indent=2))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
