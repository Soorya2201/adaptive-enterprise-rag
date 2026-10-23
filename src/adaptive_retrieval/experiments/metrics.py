import math
from collections import defaultdict
import numpy as np

PARENTS = {
    "python": "flat",
    "hnsw": "flat",
    "ivfpq": "flat",
    "bm25s": "bm25_reference",
    "hybrid": "flat",
    "rerank": "flat",
    "gated_rerank": "rerank",
    "query2doc_original": "bm25s",
    "query2doc_concise": "query2doc_original",
    "lamer_original": "bm25s",
    "lamer_concise": "lamer_original",
    "lamer_rrf": "lamer_concise",
}
PARENTS.update(hybrid_lamer_concise="hybrid", hybrid_lamer_gated="hybrid_lamer_concise")
VARIANTS = {"flat", "bm25_reference", *PARENTS}


def quality(ids: list[str], relevant: set[str], k: int) -> dict:
    if k < 1 or len(ids) != len(set(ids)):
        raise ValueError("Positive k and unique ranked document IDs required")
    names = (f"recall@{k}", f"mrr@{k}", f"ndcg@{k}")
    if not relevant:
        return dict.fromkeys(names)

    # metrics
    hits = [int(doc_id in relevant) for doc_id in ids[:k]]
    recall = sum(hits) / len(relevant)
    rr = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))

    return dict(zip(names, (recall, rr, dcg / ideal)))


def summarize(records, config):
    summary = {}
    metric = f"recall@{config['top_k']}"
    for method in config["variants"]:
        rows = [r for r in records if r["method"] == method]
        good = [r for r in rows if "error" not in r]
        values = [r["total_ms"] for r in good]
        result = {
            "requests": len(rows),
            "failures": len(rows) - len(good),
            "mean_ms": float(np.mean(values)) if values else None,
            "p95_ms": (float(np.percentile(values, 95, method="higher")) if values else None),
            "input_tokens": sum(r["input_tokens"] for r in good),
            "output_tokens": sum(r["output_tokens"] for r in good),
            "truncated_generations": sum(r["truncated_generations"] for r in good),
            "gate_rate": (float(np.mean([r["gate_fired"] for r in good])) if good else None),
        }
        for m in [metric, f"mrr@{config['top_k']}", f"ndcg@{config['top_k']}"]:
            per_query = defaultdict(list)
            for r in good:
                if r[m] is not None:
                    per_query[r["question_id"]].append(r[m])
            vals = [np.mean(v) for v in per_query.values()]
            result[m] = float(np.mean(vals)) if vals else None
        overlap = [r["ann_overlap_at_depth"] for r in good if "ann_overlap_at_depth" in r]
        if overlap:
            result["ann_overlap_at_depth"] = float(np.mean(overlap))
        parent = PARENTS.get(method)
        # Paired repetitions, then avg per query before bootstrapping queries
        parent_rows = {(r["question_id"], r["repeat"]): r for r in records if r["method"] == parent and "error" not in r}
        differences = defaultdict(list)
        for r in good:
            other = parent_rows.get((r["question_id"], r["repeat"]))
            if other and r[metric] is not None and other[metric] is not None:
                differences[r["question_id"]].append(r[metric] - other[metric])
        if differences:
            delta = np.array([np.mean(v) for v in differences.values()])
            rng = np.random.default_rng(config["seed"])
            boots = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
            result["paired_recall"] = {
                "against": parent,
                "query_count": len(delta),
                "mean_delta": float(delta.mean()),
                "bootstrap_95_interval": np.quantile(boots, [0.025, 0.975]).tolist(),
                "improved": int((delta > 0).sum()),
                "worsened": int((delta < 0).sum()),
                "unchanged": int((delta == 0).sum()),
            }
        summary[method] = result
    return summary
