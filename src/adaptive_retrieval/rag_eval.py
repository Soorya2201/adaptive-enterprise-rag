"""Run the full RAG pipeline over a question split and save comparable records.

Each record keeps the gold answer and facts so answer correctness and citation
entailment (week 8) can be scored later without re-running generation.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .evaluate import (
    TOP_K, document_ranking, git_state, load_questions, percentile, ranking_metrics, run_settings,
)
from .vector_index import file_sha256


def outcome(answerable: bool, answered: bool) -> str:
    if answerable:
        return "answered" if answered else "false_refusal"
    return "unsupported_answer" if answered else "correct_refusal"


def evaluate_answer(pipeline, question: dict, top_k: int) -> dict:
    result = pipeline.ask(question["question"], top_k)
    gold = set(question["expected_doc_ids"])
    answered = result["status"] == "answer"
    passages = result["passages"]
    context_docs = sorted({p["document_id"] for p in passages if p["prompt_label"]})
    cited_docs = sorted({c["document_id"] for c in result["citations"]})
    ranked = document_ranking([(SimpleChunk(p), p["score"]) for p in passages])
    return {
        "question_id": question["question_id"],
        "split": question.get("split"),
        "question_type": question.get("question_type"),
        "question": question["question"],
        "expected_doc_ids": question["expected_doc_ids"],
        "gold_answer": question.get("gold_answer"),
        "answer_facts": question.get("answer_facts"),
        "answerable": bool(gold),
        "outcome": outcome(bool(gold), answered),
        # Filled by the week-8 answer judge; never inferred from retrieval.
        "answer_correct": None,
        "context_has_gold": bool(gold & set(context_docs)) if gold else None,
        "cites_gold": bool(gold & set(cited_docs)) if gold and answered else None,
        "context_doc_ids": context_docs,
        "cited_doc_ids": cited_docs,
        **{f"passages_{key}": value for key, value in ranking_metrics(
            [item["doc_id"] for item in ranked], gold, TOP_K,
        ).items()},
        **{key: result[key] for key in (
            "status", "route", "answer", "reason", "citations", "passages", "retrieval",
            "metrics", "models", "settings", "raw_generation",
        )},
    }


class SimpleChunk:
    """Adapter so passage dictionaries can reuse document_ranking."""

    def __init__(self, passage: dict) -> None:
        self.document_id = passage["document_id"]
        self.id = passage["id"]


def _rate(values) -> float | None:
    values = [value for value in values if value is not None]
    return mean(float(value) for value in values) if values else None


def summarize(records: list[dict]) -> dict:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    answered = [r for r in records if r["status"] == "answer"]
    metric = lambda name: [r["metrics"][name] for r in records if r["metrics"].get(name) is not None]
    totals = metric("total_ms")
    return {
        "questions": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answer_rate": _rate(r["status"] == "answer" for r in records),
        "false_refusal_rate": _rate(r["outcome"] == "false_refusal" for r in answerable),
        "refusal_accuracy_unanswerable": _rate(r["outcome"] == "correct_refusal" for r in unanswerable),
        "context_gold_recall": _rate(r["context_has_gold"] for r in answerable),
        "cites_gold_rate": _rate(r["cites_gold"] for r in answered if r["answerable"]),
        f"passages_ndcg@{TOP_K}": _rate(r[f"passages_ndcg@{TOP_K}"] for r in answerable),
        "answer_correct_rate": _rate(r["answer_correct"] for r in records),
        "mean_total_ms": mean(totals) if totals else None,
        "p95_total_ms": percentile(totals, 0.95),
        "mean_retrieval_ms": _rate(metric("retrieval_ms")),
        "mean_generation_ms": _rate(metric("generation_ms")),
        "mean_llm_tokens": _rate(metric("total_llm_tokens")),
        "gate_rate": _rate(r["retrieval"].get("gate_fired", False) for r in records),
        "routes": dict(Counter(r["route"] for r in records)),
        # Kept apart: refusing an unanswerable question is correct, not a failure.
        "false_refusal_reasons": dict(Counter(r["reason"] for r in answerable if r["status"] != "answer")),
        "correct_refusal_reasons": dict(Counter(r["reason"] for r in unanswerable if r["status"] != "answer")),
    }


def run_rag_evaluation(
    pipeline, questions_path: Path, out_dir: Path, *, split: str | None, limit: int | None,
    top_k: int, progress=None,
) -> tuple[Path, dict]:
    questions = load_questions(questions_path, split, limit)
    pipeline.warmup()
    timestamp = datetime.now(timezone.utc)
    settings = run_settings(pipeline.session)
    run_dir = Path(out_dir) / f"rag_eval_{settings['retriever']}_{split or 'all'}_{timestamp:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    # Write each record as it finishes so an interrupted run keeps its progress.
    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as stream:
        for number, question in enumerate(questions, 1):
            record = evaluate_answer(pipeline, question, top_k)
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if progress:
                progress(number, len(questions), record)
    by_type: dict[str, list[dict]] = {}
    for record in records:
        by_type.setdefault(record["question_type"] or "unknown", []).append(record)
    summary = {
        "run": {
            "timestamp_utc": timestamp.isoformat(timespec="seconds"),
            **git_state(),
            "questions_path": str(questions_path),
            "questions_sha256": file_sha256(questions_path),
            "split": split or "all",
            "limit": limit,
            "top_k": top_k,
            "index": pipeline.session.metadata,
            "generator": pipeline.generator.model_name,
            "startup": pipeline.startup,
            **settings,
        },
        **summarize(records),
        "by_question_type": {name: summarize(rows) for name, rows in sorted(by_type.items())},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary


def format_rag_summary(summary: dict) -> str:
    run = summary["run"]
    fmt = lambda value, spec=".3f": "n/a" if value is None else format(value, spec)
    return "\n".join([
        f"Retriever: {run['retriever']} | split: {run['split']} | generator: {run['generator']}"
        + (f" | expander: {run['expander']}" if run.get("expander") else ""),
        f"Questions: {summary['questions']} ({summary['answerable']} answerable, "
        f"{summary['unanswerable']} unanswerable)",
        f"Answer rate (all questions): {fmt(summary['answer_rate'])} | false refusals: {fmt(summary['false_refusal_rate'])} | "
        f"correct refusals on unanswerable: {fmt(summary['refusal_accuracy_unanswerable'])}",
        f"False refusal reasons (answerable only): {summary['false_refusal_reasons']}",
        f"Gold doc in context: {fmt(summary['context_gold_recall'])} | answers citing gold doc: "
        f"{fmt(summary['cites_gold_rate'])}",
        f"Latency: mean {fmt(summary['mean_total_ms'], '.0f')} ms | p95 {fmt(summary['p95_total_ms'], '.0f')} ms | "
        f"retrieval {fmt(summary['mean_retrieval_ms'], '.0f')} ms | generation {fmt(summary['mean_generation_ms'], '.0f')} ms",
        f"Mean LLM tokens: {fmt(summary['mean_llm_tokens'], '.0f')} | gate fired: "
        f"{fmt(summary['gate_rate'], '.1%')} | routes: {summary['routes']}",
        "Answer correctness is not scored yet (answer_correct is null until the week-8 judge).",
    ])


def print_progress(number: int, total: int, record: dict) -> None:
    gate = " gate" if record["retrieval"].get("gate_fired") else ""
    print(
        f"[{number}/{total}] {record['question_id']} {record['outcome']}{gate} "
        f"{record['metrics']['total_ms'] / 1000:.1f}s",
        file=sys.stderr, flush=True,
    )
