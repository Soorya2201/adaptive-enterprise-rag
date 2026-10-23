import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter
import numpy as np

from .backends import DenseIndex, LexicalIndex, fuse, document_ids, should_rerank
from .expansion import expand, expanded_query, request_json
from .prepare import ROOT, sha
from .model_worker import ModelWorker
from .metrics import quality, summarize, VARIANTS


class Suite:
    def __init__(self, chunks, vectors, config, encoder=None, reranker=None):
        import faiss

        self.chunks, self.config, self.encoder, self.reranker = (chunks, config, encoder, reranker)
        if len(chunks) != len(vectors) or len({c["id"] for c in chunks}) != len(chunks):
            raise ValueError("Need one vector per unique chunk")
        faiss.omp_set_num_threads(config["faiss_threads"])
        self.build = {}
        self.dense = {}
        for kind in ["python", "flat", "hnsw", "ivfpq"]:
            if kind != "flat" and kind not in config["variants"]:
                continue
            start = perf_counter()
            params = {k: config[k] for k in ["hnsw_m", "ef_search", "nlist", "nprobe", "pq_m", "pq_bits", "seed"]}
            self.dense[kind] = DenseIndex(vectors, kind, **params)
            self.build[kind] = {"build_ms": (perf_counter() - start) * 1000, "serialized_index_bytes": self.dense[kind].size_bytes()}
        self.lexical = {}
        for engine in ["reference", "bm25s"]:
            start = perf_counter()
            self.lexical[engine] = LexicalIndex([c["text"] for c in chunks], engine)
            self.build[engine] = {"build_ms": (perf_counter() - start) * 1000}

    def run(self, question, method):
        config = self.config
        start = perf_counter()
        trace = {
            "generation_calls": [],
            "embedding_ms": 0.0,
            "rerank_ms": 0.0,
            "expansion_ms": 0.0,
            "initial_search_ms": 0.0,
            "final_search_ms": 0.0,
            "fusion_ms": 0.0,
            "gate_fired": False,
        }
        depth = config["candidate_depth"]
        vector = None
        if method in {"hybrid_lamer_concise", "hybrid_lamer_gated"}:
            from .combinations import run_combination

            ranking = run_combination(self, question, method, trace)
        elif method in {"python", "flat", "hnsw", "ivfpq", "hybrid", "rerank", "gated_rerank"}:
            tick = perf_counter()
            vector = self.encoder(question)
            trace["embedding_ms"] = (perf_counter() - tick) * 1000
            tick = perf_counter()
            ranking = self.dense[method if method in self.dense else "flat"].search(vector, depth)
            trace["initial_search_ms"] = (perf_counter() - tick) * 1000
            if method == "hybrid":
                tick = perf_counter()
                lexical = self.lexical["bm25s"].search(question, depth)
                trace["final_search_ms"] = (perf_counter() - tick) * 1000
                tick = perf_counter()
                ranking = fuse(ranking, lexical)[:depth]
                trace["fusion_ms"] = (perf_counter() - tick) * 1000
            if method in {"rerank", "gated_rerank"}:
                trace["gate_fired"] = method == "rerank" or should_rerank(ranking, config["gate_margin"])
                if trace["gate_fired"]:
                    tick = perf_counter()
                    ranking = self.reranker.rerank(question, ranking, self.chunks)
                    trace["rerank_ms"] = (perf_counter() - tick) * 1000
        elif method in {"bm25_reference", "bm25s"}:
            tick = perf_counter()
            ranking = self.lexical["reference" if method == "bm25_reference" else "bm25s"].search(question, depth)
            trace["initial_search_ms"] = (perf_counter() - tick) * 1000
        elif method.startswith(("query2doc", "lamer")):
            base = "query2doc" if method.startswith("query2doc") else "lamer"
            initial = []
            if base == "lamer":
                tick = perf_counter()
                initial = self.lexical["bm25s"].search(question, depth)
                trace["initial_search_ms"] = (perf_counter() - tick) * 1000
            tick = perf_counter()
            texts, calls = expand(
                question,
                initial[: config["expansion_candidates"]],
                self.chunks,
                method=base,
                concise=not method.endswith("original"),
                model=config["generation_model"],
            )
            trace["expansion_ms"] = (perf_counter() - tick) * 1000
            trace["generation_calls"] = calls
            trace["expansion_texts"] = texts
            trace["candidate_chunk_ids"] = [self.chunks[i]["id"] for i, _ in initial[: config["expansion_candidates"]]]
            expanded = expanded_query(question, texts, base)
            tick = perf_counter()
            ranking = self.lexical["bm25s"].search(expanded, depth)
            trace["final_search_ms"] = (perf_counter() - tick) * 1000
            if method == "lamer_rrf":
                tick = perf_counter()
                ranking = fuse(initial, ranking)[:depth]
                trace["fusion_ms"] = (perf_counter() - tick) * 1000
        else:
            raise ValueError(f"Unknown method {method}")
        trace["total_ms"] = (perf_counter() - start) * 1000
        # Evaluation work is outside the timed request.
        if method in {"hnsw", "ivfpq"}:
            exact = {i for i, _ in self.dense["flat"].search(vector, depth)}
            trace["ann_overlap_at_depth"] = len(exact & {i for i, _ in ranking}) / len(exact)
        trace["input_tokens"] = sum(c["response"]["prompt_eval_count"] for c in trace["generation_calls"])
        trace["output_tokens"] = sum(c["response"]["eval_count"] for c in trace["generation_calls"])
        trace["truncated_generations"] = sum(c["response"].get("done_reason") == "length" for c in trace["generation_calls"])
        trace["compute_cost_usd"] = None
        return ranking, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/optimization-suite")
    p.add_argument("--config", type=Path, default=ROOT / "experiments/optimizations/suite.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, help="Readiness only; omit for full development run")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if (
        not config["variants"]
        or set(config["variants"]) - VARIANTS
        or len(set(config["variants"])) != len(config["variants"])
    ):
        p.error("Unknown/duplicate variants")
    for field in ["repeats", "budget_seconds", "top_k", "candidate_depth", "expansion_candidates", "faiss_threads"]:
        if config[field] < 1:
            p.error(f"{field} must be positive")
    if (
        config["top_k"] > config["candidate_depth"]
        or config["expansion_candidates"] > config["candidate_depth"]
    ):
        p.error("top_k and expansion_candidates must fit candidate_depth")
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text())
    for file, digest in manifest["hashes"].items():
        if sha(args.data / file) != digest:
            raise ValueError(f"Prepared input changed: {file}")
    chunks = [json.loads(l) for l in (args.data / "chunks.jsonl").read_text().splitlines()]
    questions = [json.loads(l) for l in (args.data / "subset_questions.jsonl").read_text().splitlines()]
    questions = [q for q in questions if q["split"] == config["split"]]
    random.Random(config["seed"]).shuffle(questions)
    if args.limit:
        questions = questions[: args.limit]
        config["repeats"] = 1  # Readiness uses one pass, not the full benchmark budget.
    if not questions or len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Need nonempty unique questions")
    doc_ids = {c["document_id"] for c in chunks}
    if any(not set(q["expected_doc_ids"]) <= doc_ids for q in questions):
        raise ValueError("Gold document missing from corpus")
    args.output.mkdir(parents=True, exist_ok=False)
    setup = perf_counter()
    worker = ModelWorker(manifest, config["device"], chunks, bool(set(config["variants"]) & {"rerank", "gated_rerank"}))
    import atexit

    atexit.register(worker.close)
    suite = Suite(chunks, np.load(args.data / "vectors.npy"), config, worker.encode, worker)
    runtime = None
    if any("query2doc" in m or "lamer" in m for m in config["variants"]):
        request_json(
            "/api/generate",
            {
                "model": config["generation_model"],
                "prompt": "Say ready.",
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_ctx": 4096, "num_predict": 4},
            },
        )
        runtime = {
            "version": request_json("/api/version"),
            "models": request_json("/api/ps"),
        }
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "data": manifest,
        "readiness_only": args.limit is not None,
        "selected_questions": [q["question_id"] for q in questions],
        "setup_ms": (perf_counter() - setup) * 1000,
        "index_build": suite.build,
        "generation_runtime": runtime,
        "code_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "code_hashes": {
            str(f.relative_to(ROOT)): sha(f)
            for f in (ROOT / "src/adaptive_retrieval").rglob("*.py")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ["faiss-cpu", "bm25s", "numpy", "torch", "sentence-transformers", "transformers"]
        },
        "model_runtime": "Separate spawned PyTorch worker to avoid conflicting FAISS OpenMP; IPC included in request timing",
        "timing": "Warm model; each dense request encodes its query; no app cache; native prompt caching possible; scoring excluded",
        "memory": "serialized index bytes only, not process RSS; vectors and multiple indexes coexist",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    records = []
    run_start = perf_counter()
    status = "complete"
    rng = random.Random(config["seed"])
    with (args.output / "per_query.jsonl").open("w") as out:
        for repeat in range(config["repeats"]):
            order = questions.copy()
            rng.shuffle(order)
            for q in order:
                # Soft budget: finish every variant in a paired question block.
                if perf_counter() - run_start >= config["budget_seconds"]:
                    status = "budget_exhausted"
                    break
                methods = config["variants"].copy()
                rng.shuffle(methods)
                for method in methods:
                    record = {
                        "question_id": q["question_id"],
                        "category": q["question_type"],
                        "repeat": repeat,
                        "method": method,
                    }
                    tick = perf_counter()
                    try:
                        ranking, trace = suite.run(q["question"], method)
                        ids = document_ids(ranking, chunks, config["top_k"])
                        record.update(trace)
                        record.update(quality(ids, set(q["expected_doc_ids"]), config["top_k"]))
                        record.update(
                            {
                                "document_ids": ids,
                                "ranked_chunks": [
                                    [chunks[i]["id"], s] for i, s in ranking
                                ],
                            }
                        )
                    except Exception as exc:
                        record.update(
                            {
                                "error": f"{type(exc).__name__}: {exc}",
                                "failed_ms": (perf_counter() - tick) * 1000,
                            }
                        )
                    records.append(record)
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                print(f"repeat={repeat+1} question={q['question_id']} elapsed={perf_counter()-run_start:.1f}s", flush=True)
            if status != "complete":
                break
    result = {
        "status": status,
        "elapsed_seconds": perf_counter() - run_start,
        "results": summarize(records, config),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    worker.close()
    atexit.unregister(worker.close)
    print(f"{status}: {len(records)} requests, {sum('error' in r for r in records)} failures; {args.output}")


if __name__ == "__main__":
    main()
