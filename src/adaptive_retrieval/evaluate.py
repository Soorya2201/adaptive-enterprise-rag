"""Document-level retrieval evaluation: Hit@5 and MRR over labeled questions."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .interactive import SearchSession
from .paths import PROJECT_ROOT
from .vector_index import file_sha256


TOP_K = 5


def load_questions(path: Path) -> list[dict]:
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
    return questions


def rank_documents(session: SearchSession, question: str) -> list[dict]:
    """Rank returned documents by their best-scoring chunk; labels are per document."""
    ranked: list[dict] = []
    seen: set[str] = set()
    for chunk, score in session.search(question, top_k=len(session.rows)):
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


def evaluate_question(session: SearchSession, question: dict) -> dict:
    ranked = rank_documents(session, question["question"])
    gold = set(question["expected_doc_ids"])
    gold_rank = next((item["rank"] for item in ranked if item["doc_id"] in gold), None)
    answerable = bool(gold)
    return {
        "question_id": question["question_id"],
        "split": question.get("split"),
        "question_type": question.get("question_type"),
        "question": question["question"],
        "expected_doc_ids": question["expected_doc_ids"],
        "answerable": answerable,
        "gold_rank": gold_rank,
        f"hit@{TOP_K}": (
            float(gold_rank is not None and gold_rank <= TOP_K) if answerable else None
        ),
        "reciprocal_rank": (1 / gold_rank if gold_rank else 0.0) if answerable else None,
        "retrieved": ranked[:TOP_K],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_evaluation(session: SearchSession, questions_path: Path, out_dir: Path) -> tuple[Path, dict]:
    """Evaluate every question and save per-question records plus a summary."""
    questions = load_questions(questions_path)
    records = [evaluate_question(session, question) for question in questions]
    scored = [record for record in records if record["answerable"]]
    timestamp = datetime.now(timezone.utc)
    summary = {
        "run": {
            "timestamp_utc": timestamp.isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "questions_path": str(questions_path),
            "questions_sha256": file_sha256(questions_path),
            "index": session.metadata,
            "retriever": getattr(session, "retriever", "cosine"),
            "score_type": getattr(session, "score_type", "cosine"),
            "candidate_depth": session.candidate_depth if getattr(session, "retriever", "cosine") == "hybrid" else None,
        },
        "questions": len(records),
        "scored": len(scored),
        f"hit@{TOP_K}": mean(record[f"hit@{TOP_K}"] for record in scored) if scored else None,
        "mrr": mean(record["reciprocal_rank"] for record in scored) if scored else None,
    }
    run_dir = Path(out_dir) / f"retrieval_eval_{timestamp:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary


def format_summary(summary: dict) -> str:
    unscored = summary["questions"] - summary["scored"]
    if not summary["scored"]:
        return f"No questions have a correct document; nothing to score ({unscored} questions)"
    return "\n".join([
        f"Scored {summary['scored']} of {summary['questions']} questions "
        f"({unscored} have no correct document)",
        f"Hit@{TOP_K}: {summary[f'hit@{TOP_K}']:.3f}   (correct document in the top {TOP_K})",
        f"MRR:   {summary['mrr']:.3f}   (average of 1 / rank of the correct document)",
    ])
