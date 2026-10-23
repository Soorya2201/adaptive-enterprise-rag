# for metrics to calculate combinations of optimizations

from time import perf_counter
from .backends import fuse
from .expansion import expand, expanded_query


def run_combination(suite, question, method, trace):
    config = suite.config
    depth = config["candidate_depth"]
    tick = perf_counter()
    vector = suite.encoder(question)
    trace["embedding_ms"] = (perf_counter() - tick) * 1000
    tick = perf_counter()
    dense = suite.dense["flat"].search(vector, depth)
    lexical = suite.lexical["bm25s"].search(question, depth)
    trace["initial_search_ms"] = (perf_counter() - tick) * 1000
    tick = perf_counter()
    baseline = fuse(dense, lexical)[:depth]
    trace["fusion_ms"] = (perf_counter() - tick) * 1000
    top_doc = lambda ranking: (suite.chunks[ranking[0][0]]["document_id"] if ranking else None)
    disagree = not dense or not lexical or top_doc(dense) != top_doc(lexical)
    trace["top_document_disagreement"] = bool(disagree)
    trace["gate_fired"] = method == "hybrid_lamer_concise" or bool(disagree)
    trace["gate_reason"] = ("always" if method == "hybrid_lamer_concise" else "top_document_disagreement")
    if not trace["gate_fired"]:
        return baseline
    candidates = lexical[: config["expansion_candidates"]]
    tick = perf_counter()
    texts, calls = expand(question, candidates, suite.chunks, method="lamer", concise=True, model=config["generation_model"])
    trace["expansion_ms"] = (perf_counter() - tick) * 1000
    trace["generation_calls"] = calls
    trace["expansion_texts"] = texts
    trace["candidate_chunk_ids"] = [suite.chunks[i]["id"] for i, _ in candidates]
    expanded = expanded_query(question, texts, "lamer")
    tick = perf_counter()
    expanded_ranking = suite.lexical["bm25s"].search(expanded, depth)
    trace["final_search_ms"] = (perf_counter() - tick) * 1000
    tick = perf_counter()
    ranking = fuse(dense, expanded_ranking)[:depth]
    trace["fusion_ms"] += (perf_counter() - tick) * 1000
    return ranking
