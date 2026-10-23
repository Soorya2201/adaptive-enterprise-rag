import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from adaptive_retrieval.experiments.metrics import PARENTS, summarize

METRICS = ["recall@5", "mrr@5", "ndcg@5"]
STAGES = ["embedding_ms", "initial_search_ms", "final_search_ms", "fusion_ms", "rerank_ms", "expansion_ms", "total_ms"]
EXTRA_COMPARISONS = [
    ("hybrid", "bm25s"),
    ("gated_rerank", "flat"),
    ("query2doc_concise", "bm25s"),
    ("lamer_concise", "bm25s"),
    ("lamer_rrf", "bm25s"),
    ("hybrid_lamer_concise", "lamer_concise"),
    ("hybrid_lamer_gated", "hybrid"),
]


def complete_passes(records, question_ids, methods):
    if any("error" in record for record in records):
        raise ValueError("Run contains failed requests; inspect per_query.jsonl before comparing methods")
    expected = {(question, method) for question in question_ids for method in methods}
    if not expected:
        raise ValueError("Need questions and methods")
    passes = defaultdict(list)
    seen = set()
    for record in records:
        pair = (record["question_id"], record["method"])
        key = (record["repeat"], *pair)
        if key in seen or pair not in expected:
            raise ValueError("Duplicate or unexpected request in run")
        seen.add(key)
        passes[record["repeat"]].append(record)
    complete = []
    for repeat, rows in sorted(passes.items()):
        if {(row["question_id"], row["method"]) for row in rows} == expected:
            complete.append(repeat)
    if not complete:
        raise ValueError("No complete pass; cannot make a balanced comparison")
    return [record for record in records if record["repeat"] in complete], complete


def query_means(records, metric):
    values = defaultdict(list)
    for record in records:
        if record[metric] is not None:
            values[record["question_id"]].append(record[metric])
    return { question: float(np.mean(scores)) for question, scores in sorted(values.items())}


def paired_comparison(left, right, metric, seed=42):
    right = {(row["question_id"], row["repeat"]): row for row in right}
    differences = []
    for row in left:
        other = right[(row["question_id"], row["repeat"])]
        if row[metric] is not None and other[metric] is not None:
            differences.append({"question_id": row["question_id"], metric: row[metric] - other[metric]})
    per_query = query_means(differences, metric)
    values = np.array(list(per_query.values()))
    interval = None
    if len(values):
        rng = np.random.default_rng(seed)
        bootstrap = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "paired_queries": len(values),
        "mean_delta": float(values.mean()) if len(values) else None,
        "bootstrap_95_interval": interval,
        "per_query_delta": per_query,
    }


def describe_method(records, summary):
    result = {
        "summary": summary,
        "stage_mean_ms": {stage: float(np.mean([row[stage] for row in records])) for stage in STAGES},
        "categories": {},
        "query_quality": {},
        "ranking_variability_queries": 0,
    }
    for category in sorted({row["category"] for row in records}):
        rows = [row for row in records if row["category"] == category]
        scores = {"questions": len({row["question_id"] for row in rows})}
        for metric in METRICS:
            values = list(query_means(rows, metric).values())
            scores[metric] = float(np.mean(values)) if values else None
        result["categories"][category] = scores
    for question in sorted({row["question_id"] for row in records}):
        rows = [row for row in records if row["question_id"] == question]
        result["query_quality"][question] = {metric: query_means(rows, metric).get(question) for metric in METRICS}
        if len({tuple(row["document_ids"]) for row in rows}) > 1:
            result["ranking_variability_queries"] += 1
    return result


def analyze(records, metadata, summary):
    methods = metadata["config"]["variants"]
    rows, repeats = complete_passes(records, metadata["selected_questions"], methods)
    scores = summarize(rows, metadata["config"])
    balanced = {"complete_repeats": repeats, "raw_requests": len(records), "analyzed_requests": len(rows), "results": scores}
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    result = {
        "raw_requests": len(records),
        "analysis_repeats": repeats,
        "excluded_partial_repeat_requests": len(records) - len(rows),
        "status": summary["status"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "requests": len(rows),
        "failures": [],
        "completed_blocks": len(rows) // len(methods),
        "methods": {},
        "comparisons": {},
    }
    for method in methods:
        result["methods"][method] = describe_method(by_method[method], scores[method])
    for method, parent in list(PARENTS.items()) + EXTRA_COMPARISONS:
        if method not in by_method or parent not in by_method:
            continue
        result["comparisons"][method + " vs " + parent] = {
            metric: paired_comparison(by_method[method], by_method[parent], metric, seed=metadata["config"]["seed"])
            for metric in METRICS + ["total_ms"]
        }
    return balanced, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, help="New directory; defaults to the run directory")
    args = parser.parse_args()
    output = args.output or args.run
    filenames = ["balanced_summary.json", "analysis.json"]
    if any((output / filename).exists() for filename in filenames):
        parser.error("Analysis already exists; use --output with a new directory")
    metadata = json.loads((args.run / "metadata.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    records = [json.loads(line) for line in (args.run / "per_query.jsonl").read_text().splitlines()]
    balanced, result = analyze(records, metadata, summary)
    output.mkdir(parents=True, exist_ok=True)
    for filename, contents in zip(filenames, [balanced, result]):
        (output / filename).write_text(json.dumps(contents, indent=2) + "\n")
    if output.resolve() != args.run.resolve():
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Analyzed {len(balanced['complete_repeats'])} complete passes: {len(records)} raw / {result['requests']} balanced requests")
    for method, scores in balanced["results"].items():
        print(method, {key: scores[key] for key in METRICS + ["mean_ms", "p95_ms"]})


if __name__ == "__main__":
    main()
