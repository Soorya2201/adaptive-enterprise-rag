"""Document-level retrieval evaluation with quality, latency and gate statistics."""

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from .paths import PROJECT_ROOT
from .vector_index import file_sha256


TOP_K = 5
SPLITS = ("dev", "val", "test")


def load_questions(path: Path, split: str | None = None, limit: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Questions not found: {path}. Run 'prepare-subset' first.")
    questions = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not all(key in record for key in ("question_id", "question", "expected_doc_ids")):
                raise ValueError(f"Question record missing required fields at {path}:{line_number}")
            questions.append(record)
    if not questions:
        raise ValueError(f"Question file is empty: {path}")
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join(SPLITS)}")
        questions = [question for question in questions if question.get("split") == split]
        if not questions:
            raise ValueError(f"No questions in split {split!r}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    return questions


def document_ranking(results) -> list[dict]:
    """Collapse chunk results to documents, ranked by each document's best chunk."""
    ranked: list[dict] = []
    seen: set[str] = set()
    for chunk, score in results:
        if chunk.document_id in seen:
            continue
        seen.add(chunk.document_id)
        ranked.append({
            "rank": len(ranked) + 1,
            "doc_id": chunk.document_id,
            "score": score,
            "best_chunk_id": chunk.id,
        })
    return ranked


def rank_documents(session, question: str) -> list[dict]:
    return document_ranking(session.search(question, top_k=len(session.rows)))


def ranking_metrics(doc_ids: list[str], relevant: set[str], k: int = TOP_K) -> dict:
    """Binary-relevance metrics; None for unanswerable questions."""
    names = ("hit@1", f"hit@{k}", f"recall@{k}", f"ndcg@{k}", "reciprocal_rank")
    if not relevant:
        return dict.fromkeys(names)
    hits = [int(doc_id in relevant) for doc_id in doc_ids[:k]]
    gold_rank = next((rank for rank, doc_id in enumerate(doc_ids, 1) if doc_id in relevant), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        "hit@1": float(bool(doc_ids) and doc_ids[0] in relevant),
        f"hit@{k}": float(any(hits)),
        f"recall@{k}": sum(hits) / len(relevant),
        f"ndcg@{k}": dcg / ideal,
        "reciprocal_rank": 1 / gold_rank if gold_rank else 0.0,
    }


def evaluate_question(session, question: dict) -> dict:
    started = perf_counter()
    ranked = rank_documents(session, question["question"])
    elapsed_ms = (perf_counter() - started) * 1000
    gold = set(question["expected_doc_ids"])
    gold_rank = next((item["rank"] for item in ranked if item["doc_id"] in gold), None)
    trace = dict(getattr(session, "last_trace", {}) or {})
    return {
        "question_id": question["question_id"],
        "split": question.get("split"),
        "question_type": question.get("question_type"),
        "question": question["question"],
        "expected_doc_ids": question["expected_doc_ids"],
        "answerable": bool(gold),
        "gold_rank": gold_rank,
        **ranking_metrics([item["doc_id"] for item in ranked], gold),
        "retrieval_ms": elapsed_ms,
        "timings": dict(getattr(session, "last_timings", {}) or {}),
        "gate_fired": bool(trace.get("gate_fired", False)),
        "retrieval_rounds": trace.get("retrieval_rounds", 1),
        "expansion_tokens": trace.get("expansion_input_tokens", 0) + trace.get("expansion_output_tokens", 0),
        "retrieval": trace,
        "retrieved": ranked[:TOP_K],
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank ("higher") percentile, matching the optimization experiments."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def summarize_records(records: list[dict]) -> dict:
    scored = [record for record in records if record["answerable"]]
    latencies = [record["retrieval_ms"] for record in records]
    metrics = {}
    for name in ("hit@1", f"hit@{TOP_K}", f"recall@{TOP_K}", f"ndcg@{TOP_K}", "reciprocal_rank"):
        key = "mrr" if name == "reciprocal_rank" else name
        metrics[key] = mean(record[name] for record in scored) if scored else None
    return {
        "questions": len(records),
        "scored": len(scored),
        **metrics,
        "mean_ms": mean(latencies) if latencies else None,
        "p95_ms": percentile(latencies, 0.95),
        "gate_rate": mean(record["gate_fired"] for record in records) if records else None,
        "expansion_tokens": sum(record["expansion_tokens"] for record in records),
    }


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return {"git_commit": run("rev-parse", "HEAD"), "working_tree_dirty": bool(status) if status is not None else None}


def run_settings(session) -> dict:
    retriever = getattr(session, "retriever", "cosine")
    expander = getattr(session, "expander", None)
    return {
        "retriever": retriever,
        "score_type": getattr(session, "score_type", "cosine"),
        "candidate_depth": getattr(session, "candidate_depth", None) if retriever != "cosine" else None,
        "expander": getattr(expander, "model_name", None) if retriever in {"hybrid_lamer", "adaptive"} else None,
        "expansion_candidates": getattr(session, "expansion_candidates", None) if expander else None,
    }


def run_evaluation(
    session, questions_path: Path, out_dir: Path, *, split: str | None = None,
    limit: int | None = None,
) -> tuple[Path, dict]:
    """Evaluate the selected questions and save per-question records plus a summary."""
    questions = load_questions(questions_path, split, limit)
    # Load models before timing so the first question does not include startup.
    if hasattr(session.embedder, "warmup"):
        session.embedder.warmup()
    expander = getattr(session, "expander", None)
    if expander is not None and getattr(session, "retriever", "") in {"hybrid_lamer", "adaptive"}:
        expander.warmup()
    records = [evaluate_question(session, question) for question in questions]
    timestamp = datetime.now(timezone.utc)
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
            "index": session.metadata,
            **run_settings(session),
        },
        **summarize_records(records),
        "by_question_type": {name: summarize_records(rows) for name, rows in sorted(by_type.items())},
    }
    retriever = summary["run"]["retriever"]
    run_dir = Path(out_dir) / f"retrieval_eval_{retriever}_{split or 'all'}_{timestamp:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary


def format_summary(summary: dict) -> str:
    unscored = summary["questions"] - summary["scored"]
    run = summary.get("run", {})
    lines = [
        f"Retriever: {run.get('retriever', 'unknown')} | split: {run.get('split', 'all')}"
        + (f" | expander: {run['expander']}" if run.get("expander") else ""),
    ]
    if not summary["scored"]:
        lines.append(f"No questions have a correct document; nothing to score ({unscored} questions)")
        return "\n".join(lines)
    lines += [
        f"Scored {summary['scored']} of {summary['questions']} questions "
        f"({unscored} have no correct document)",
        f"nDCG@{TOP_K}: {summary[f'ndcg@{TOP_K}']:.3f} | Recall@{TOP_K}: {summary[f'recall@{TOP_K}']:.3f} | "
        f"Hit@1: {summary['hit@1']:.3f} | Hit@{TOP_K}: {summary[f'hit@{TOP_K}']:.3f} | MRR: {summary['mrr']:.3f}",
        f"Latency: mean {summary['mean_ms']:.1f} ms | p95 {summary['p95_ms']:.1f} ms | "
        f"gate fired: {summary['gate_rate']:.1%}",
    ]
    return "\n".join(lines)
