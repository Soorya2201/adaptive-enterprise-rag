"""Offline answer evaluation; semantic correctness requires explicit judgments.

Run with python -m adaptive_retrieval.answer_evaluation --help (PYTHONPATH=src).
No model or network access is required to score saved RAG traces.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean


def read_records(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{number}")
        records.append(record)
    if not records:
        raise ValueError(f"Empty input: {path}")
    return records


def keyed(records: list[dict]) -> dict[str, dict]:
    result = {}
    for record in records:
        key = record.get("question_id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Missing or duplicate question_id: {key!r}")
        result[key] = record
    return result


def score_answers(questions: list[dict], traces: list[dict], judgments: list[dict]) -> list[dict]:
    """Join by ID, or exact unique question text for existing CLI traces.

    Refusals on answerable questions are incorrect; unanswered correctness
    judgments remain null rather than silently counting as correct or wrong.
    """
    gold = keyed(questions)
    text_ids: dict[str, list[str]] = {}
    for key, question in gold.items():
        text_ids.setdefault(question["question"], []).append(key)
        if not isinstance(question.get("expected_doc_ids"), list):
            raise ValueError(f"expected_doc_ids must be a list: {key}")
    resolved = []
    for trace in traces:
        item = dict(trace)
        if "question_id" not in item:
            matches = text_ids.get(item.get("question"), [])
            if len(matches) != 1:
                raise ValueError("Trace question text must match exactly one question")
            item["question_id"] = matches[0]
        resolved.append(item)
    predictions = keyed(resolved)
    if predictions.keys() != gold.keys():
        raise ValueError("Traces must cover exactly the selected question IDs")
    labels = keyed(judgments) if judgments else {}
    if labels.keys() - gold.keys():
        raise ValueError("Judgments contain unknown question IDs")
    rows = []
    for key, question in gold.items():
        trace = predictions[key]
        if "question" in trace and trace["question"] != question["question"]:
            raise ValueError(f"Trace question differs from gold: {key}")
        status = trace.get("status")
        if status not in {"answer", "refusal"}:
            raise ValueError(f"Unknown trace status for {key}: {status}")
        answer = trace.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Trace must contain a nonempty answer: {key}")
        label = labels.get(key, {})
        if label:
            # Bind manual/model judgments to the precise output being scored.
            if label.get("answer") != answer or not label.get("judge"):
                raise ValueError(f"Judgment needs matching answer and judge: {key}")
            if "correct" in label and type(label["correct"]) is not bool:
                raise ValueError(f"correct must be boolean: {key}")
            if label.get("nli") not in {None, "entailment", "neutral", "contradiction"}:
                raise ValueError(f"Unknown NLI label: {key}")
        answerable = bool(question["expected_doc_ids"])
        refused = status == "refusal"
        correct = (not answerable) if refused else (False if not answerable else label.get("correct"))
        rows.append({
            "question_id": key, "split": question.get("split", "unspecified"),
            "question_type": question.get("question_type", "unspecified"),
            "answerable": answerable, "refused": refused,
            "correct": correct, "refusal_correct": refused == (not answerable),
            "nli": label.get("nli") if not refused else None,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    def average(values):
        return mean(values) if values else None

    judged = [row for row in rows if row["correct"] is not None]
    unanswerable = [row for row in rows if not row["answerable"]]
    answerable = [row for row in rows if row["answerable"]]
    nli = [row for row in rows if row["nli"] is not None]
    return {
        "questions": len(rows), "correctness_scored": len(judged),
        "correctness_coverage": len(judged) / len(rows) if rows else None,
        "answer_correctness": average([int(row["correct"]) for row in judged]),
        "refusal_accuracy": average([int(row["refusal_correct"]) for row in rows]),
        "unanswerable_refusal_recall": average([int(row["refused"]) for row in unanswerable]),
        "answerable_false_refusal_rate": average([int(row["refused"]) for row in answerable]),
        "nli_scored": len(nli),
        "nli_entailment_rate": average([int(row["nli"] == "entailment") for row in nli]),
    }


def report(rows: list[dict]) -> dict:
    result = {"overall": summarize(rows)}
    for field in ("split", "question_type"):
        groups = {}
        for row in rows:
            groups.setdefault(str(row[field]), []).append(row)
        result[f"by_{field}"] = {key: summarize(group) for key, group in groups.items()}
    return result


def paired_comparison(candidate: list[dict], baseline: list[dict], *, samples=2000, seed=13) -> dict:
    """Percentile paired bootstrap, candidate minus baseline, using identical IDs.

    Require full correctness coverage to avoid judging different subsets/systems.
    """
    left, right = keyed(candidate), keyed(baseline)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical nonempty question sets")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    result = {}
    for metric in ("correct", "refusal_correct"):
        if any(row[metric] is None for row in candidate + baseline):
            raise ValueError("Paired correctness comparison requires complete judgments")
        differences = [int(left[key][metric]) - int(right[key][metric]) for key in sorted(left)]
        rng = random.Random(seed)
        estimates = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
        result[metric] = {
            "delta": mean(differences),
            "ci95": [estimates[int((samples - 1) * .025)], estimates[int((samples - 1) * .975)]],
            "questions": len(differences), "samples": samples, "seed": seed,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--baseline-traces", type=Path)
    parser.add_argument("--baseline-judgments", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.baseline_judgments and not args.baseline_traces:
            raise ValueError("--baseline-judgments requires --baseline-traces")
        questions = read_records(args.questions)
        rows = score_answers(questions, read_records(args.traces), read_records(args.judgments) if args.judgments else [])
        result = report(rows)
        if args.baseline_traces:
            baseline = score_answers(questions, read_records(args.baseline_traces), read_records(args.baseline_judgments) if args.baseline_judgments else [])
            result["baseline"] = report(baseline)
            result["paired_comparison"] = paired_comparison(rows, baseline)
        result["per_question"] = rows
        result["run"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evaluator_version": 1,
            "inputs": {
                name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name in ("questions", "traces", "judgments", "baseline_traces", "baseline_judgments")
                if (path := getattr(args, name)) is not None
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Answer evaluation saved: {args.out}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
