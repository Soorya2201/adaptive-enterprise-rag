"""Export blinded correctness review tasks and evidence-bound label templates."""

import argparse
import json
from pathlib import Path

from .answer_evaluation import keyed, read_records, score_answers, trace_digest


def export_review(questions, traces, out):
    # Validate coverage and IDs before creating any output.
    score_answers(questions, traces, [])
    indexed = keyed(traces)
    tasks, templates = [], []
    for question in questions:
        trace = indexed[question["question_id"]]
        if trace["status"] != "answer" or not question["expected_doc_ids"]:
            continue
        tasks.append({"question_id": question["question_id"], "question": question["question"],
                      "answer": trace["answer"], "gold_answer": question["gold_answer"],
                      "answer_facts": question.get("answer_facts", []),
                      "expected_doc_ids": question["expected_doc_ids"],
                      "supplied_context": [p for p in trace.get("passages", []) if p.get("prompt_label") is not None],
                      "citations": trace.get("citations", []), "trace_sha256": trace_digest(trace)})
        templates.append({"question_id": question["question_id"], "answer": trace["answer"],
                          "trace_sha256": trace_digest(trace), "judge": "",
                          "judgment_kind": "manual", "review_notes": ""})
    out.mkdir(parents=True, exist_ok=False)
    for name, records in (("tasks.jsonl", tasks), ("judgments.jsonl", templates)):
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8", newline="\n")
    (out / "INSTRUCTIONS.md").write_text(
        "# Blinded answer review\n\n"
        "Inspect each task's question, gold answer, rubric, and actual supplied context. "
        "Add a boolean `correct` to each judgment, replace the empty `judge` with your "
        "reviewer identity, and record reasons in `review_notes`. Correct answers must "
        "satisfy essential rubric facts without material unsupported or contradictory claims. "
        "Optional facts are not mandatory; negative rubric constraints must be respected. "
        "Optionally add `nli` as entailment, neutral, or contradiction. "
        "Model suggestions and policy names are omitted to reduce reviewer bias. "
        "Do not alter answer or trace_sha256. Have a second reviewer adjudicate disagreements. "
        "Refusals and answers to explicitly unanswerable questions are scored automatically.\n",
        encoding="utf-8", newline="\n")
    return len(tasks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        count = export_review(read_records(args.questions), read_records(args.traces), args.out)
        print(f"Exported {count} blinded review tasks: {args.out}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
