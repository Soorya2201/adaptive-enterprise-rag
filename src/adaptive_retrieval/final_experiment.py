"""Collect frozen fixed/adaptive answer traces without exposing gold labels to policy."""

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import platform
from time import perf_counter

from .answer_evaluation import read_records
from .final_benchmark import canonical_hash, verify_benchmark
from .generation import QwenGenerator
from .hybrid_search import OptimizedSearchSession
from .interactive import SearchSession
from .model_store import MODEL_REVISIONS
from .nli import NLI_MODEL, NLI_REVISION, NliJudge, claims
from .pipeline import RagPipeline
from .vector_index import file_sha256


def grounding_check(trace, judge):
    if trace["status"] != "answer":
        return {"passed": False, "reason": "pipeline_refusal", "claims": []}
    context = "\n\n".join(p["text"] for p in trace["passages"] if p.get("prompt_label") is not None)
    sentences = claims(trace["answer"])
    if not context.strip() or not sentences:
        return {"passed": False, "reason": "missing_context_or_claims", "claims": []}
    checks = [{"claim": text, **judge.assess(context, text)} for text in sentences]
    return {"passed": all(check["label"] == "entailment" for check in checks),
            "reason": "nli_checked", "claims": checks}


def ask_policy(pipeline, judge, question, *, mode, top_k=5, retry_top_k=10):
    """Both modes use the same NLI gate; adaptive allows one wider retry.

    No expected document IDs, answers, split labels, or other gold data enter
    the routing decision. Retrieval settings and thresholds are frozen upstream.
    """
    if mode not in {"fixed", "adaptive"} or not 1 <= top_k < retry_top_k <= 20:
        raise ValueError("Choose fixed/adaptive and 1 <= top_k < retry_top_k <= 20")
    started = perf_counter()
    attempts = []
    for depth in ([top_k] if mode == "fixed" else [top_k, retry_top_k]):
        trace = pipeline.ask(question, depth)
        check_started = perf_counter()
        checked = grounding_check(trace, judge)
        attempts.append({"trace": trace, "nli_check": checked,
                         "nli_ms": (perf_counter() - check_started) * 1000})
        if checked["passed"]:
            break
    result = dict(attempts[-1]["trace"])
    result["metrics"] = dict(result["metrics"])
    result["attempts"] = attempts
    result["experiment_mode"] = mode
    result["nli_judge"] = {"model": NLI_MODEL, "revision": NLI_REVISION, "threshold": judge.threshold}
    result["settings"] = {**result.get("settings", {}), "first_top_k": top_k, "retry_top_k": retry_top_k}
    if not attempts[-1]["nli_check"]["passed"] and result["status"] == "answer":
        result["unchecked_answer"] = result["answer"]
        result.update(status="refusal", answer="I couldn't verify an answer from the retrieved documents.",
                      reason="nli_not_entailed", citations=[])
    result["route"] = ("refusal" if result["status"] == "refusal" else
                       "second_pass" if len(attempts) == 2 else "single_pass")
    for key in ("prompt_tokens", "output_tokens", "generation_calls", "retrieval_rounds",
                "retrieval_ms", "generation_ms", "prompt_ms", "validation_ms"):
        result["metrics"][key] = sum(attempt["trace"]["metrics"].get(key, 0) for attempt in attempts)
    result["metrics"]["nli_ms"] = sum(attempt["nli_ms"] for attempt in attempts)
    result["metrics"]["total_ms"] = (perf_counter() - started) * 1000
    return result


def collect(pipeline, judge, questions, destination: Path, *, mode, protocol):
    """Exclusive creation prevents accidental repeat/append of a frozen run."""
    pipeline.warmup()
    with destination.open("x", encoding="utf-8") as stream:
        for question in questions:
            result = ask_policy(pipeline, judge, question["question"], mode=mode,
                                top_k=protocol["top_k"], retry_top_k=protocol["retry_top_k"])
            result["question_id"] = question["question_id"]
            result["protocol_sha256"] = protocol["protocol_sha256"]
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{mode} {question['question_id']}: {result['status']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/processed/final300-v2"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true", help="Freeze settings without running models")
    parser.add_argument("--split", choices=("dev", "val", "test"), default="val")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--retriever", choices=("cosine", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retry-top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=.8)
    args = parser.parse_args(argv)
    try:
        verify_benchmark(args.benchmark)
        if not 1 <= args.top_k < args.retry_top_k <= 20 or not .5 < args.threshold < 1:
            raise ValueError("Invalid retrieval depths or NLI threshold")
        protocol_path = args.out / "protocol.json"
        questions = [q for q in read_records(args.benchmark / "questions.jsonl") if q["split"] == args.split]
        if not questions:
            raise ValueError("Selected split is empty")
        with args.index.open(encoding="utf-8") as stream:
            index_metadata = json.loads(stream.readline())["_meta"]
        if index_metadata["corpus_sha256"] != file_sha256(args.benchmark / "corpus.jsonl"):
            raise ValueError("Index was not built from the frozen benchmark corpus")
        package_versions = {}
        for name in ("torch", "transformers", "mlx-lm", "sentence-transformers", "faiss-cpu", "bm25s"):
            try:
                package_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                package_versions[name] = None
        settings = {"benchmark_manifest_sha256": file_sha256(args.benchmark / "manifest.json"),
                    "index_sha256": file_sha256(args.index), "split": args.split,
                    "retriever": args.retriever, "top_k": args.top_k, "retry_top_k": args.retry_top_k,
                    "threshold": args.threshold, "max_new_tokens": 512, "max_prompt_tokens": 4096,
                    "selected_questions_canonical_sha256": canonical_hash(questions),
                    "package_versions": package_versions,
                    "implementation_sha256": {name: file_sha256(Path(__file__).with_name(name + ".py"))
                                              for name in ("final_experiment", "nli", "pipeline", "generation", "embedding", "interactive", "hybrid_search", "vector_index")},
                    "generation_revisions": MODEL_REVISIONS, "nli_revision": NLI_REVISION,
                    "policy": "nli-gated fixed vs one bounded wider-context retry v1"}
        if args.freeze:
            args.out.mkdir(parents=True, exist_ok=False)
            with protocol_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"frozen_utc": datetime.now(timezone.utc).isoformat(),
                                         "settings": settings}, indent=2) + "\n")
            (args.out / "questions.jsonl").write_text(
                "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8", newline="\n")
            print(f"Protocol frozen: {protocol_path}")
            return
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["settings"] != settings:
            raise ValueError("Run settings/input hashes differ from frozen protocol")
        if canonical_hash(read_records(args.out / "questions.jsonl")) != canonical_hash(questions):
            raise ValueError("Exported frozen questions were modified")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Frozen Qwen experiment requires the team's Apple Silicon/MLX runtime; no test run was started")
        destination = args.out / f"{args.mode}.jsonl"
        if destination.exists():
            raise ValueError("This mode already has a run; retain partial failures for audit")
        judge = NliJudge(threshold=args.threshold)
        judge.load()
        settings["protocol_sha256"] = file_sha256(protocol_path)
        corpus = args.benchmark / "corpus.jsonl"
        with OptimizedSearchSession(SearchSession(args.index, corpus), retriever=args.retriever) as session:
            pipeline = RagPipeline(session, QwenGenerator())
            collect(pipeline, judge, questions, destination, mode=args.mode, protocol=settings)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
