"""Build a 100-question evaluation subset with a minimal document corpus.

Questions are picked per type (round-robin across source types), split
70/15/15 into dev/val/test within each type, and the corpus is the gold docs
plus the top BM25 non-gold hits from the full corpus (hard negatives).

Run through ``python src/runner.py build-subset`` from any directory.
"""

import argparse
import json
import math
import random
import re
from array import array
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from .data import clean_document_text
from .paths import FULL_DOCUMENTS_DIR, PROCESSED_DIR

SEED = 13
QUOTAS = {
    "basic": 35,
    "semantic": 25,
    "intra_document_reasoning": 15,
    "constrained": 15,  # only single-gold questions are eligible
    "info_not_found": 10,
}
SPLIT = (0.70, 0.85)  # dev < 0.70 <= val < 0.85 <= test
HARD_NEG_PER_Q = 2
N_CACHE_SEEDS = 10
K1, B = 1.5, 0.75

STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "their this to was were what when where which who why will with does did do "
    "our we you your they them i can should would there been any about into than".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def doc_id_of(path):
    return path.name.split("__", 1)[0]


def source_of(path, root):
    return path.relative_to(root).parts[0]


def select_questions(questions, rng):
    selected = []
    for qtype, quota in QUOTAS.items():
        pool = [q for q in questions if q["question_type"] == qtype]
        if qtype == "constrained":
            pool = [q for q in pool if len(q["expected_doc_ids"]) == 1]
        by_source = defaultdict(list)
        for q in pool:
            by_source[q["source_types"][0] if q["source_types"] else "none"].append(q)
        for group in by_source.values():
            rng.shuffle(group)
        picked = []
        while len(picked) < quota and any(by_source.values()):
            for src in sorted(by_source):
                if by_source[src] and len(picked) < quota:
                    picked.append(by_source[src].pop())
        if len(picked) < quota:
            raise SystemExit(f"{qtype}: only {len(picked)} eligible, need {quota}")

        rng.shuffle(picked)
        n_dev, n_val = round(quota * SPLIT[0]), round(quota * (SPLIT[1] - SPLIT[0]))
        for i, q in enumerate(picked):
            split = "dev" if i < n_dev else "val" if i < n_dev + n_val else "test"
            selected.append({**q, "split": split})
    return selected


def _index_chunk(args):
    paths, vocab = args
    lengths = array("i")
    postings = defaultdict(lambda: (array("i"), array("i")))
    for local_idx, path in enumerate(paths):
        tokens = tokenize(path.read_text(encoding="utf-8", errors="ignore"))
        lengths.append(len(tokens))
        for term, tf in Counter(t for t in tokens if t in vocab).items():
            docs, tfs = postings[term]
            docs.append(local_idx)
            tfs.append(tf)
    return lengths, dict(postings)


def bm25_top_k(paths, queries, k):
    """Return, per query, the indices of the top-k paths by BM25 score."""
    query_terms = [set(tokenize(q)) for q in queries]
    vocab = set().union(*query_terms)
    chunk = 5000
    jobs = [(paths[i : i + chunk], vocab) for i in range(0, len(paths), chunk)]

    lengths = array("i")
    postings = defaultdict(list)  # term -> [(offset, docs, tfs), ...]
    with Pool() as pool:
        for i, (lens, posts) in enumerate(pool.imap(_index_chunk, jobs)):
            offset = i * chunk
            lengths.extend(lens)
            for term, (docs, tfs) in posts.items():
                postings[term].append((offset, docs, tfs))
            print(f"  indexed {len(lengths):,}/{len(paths):,}", end="\r", flush=True)
    print()

    n_docs = len(lengths)
    avgdl = sum(lengths) / n_docs
    results = []
    for terms in query_terms:
        scores = defaultdict(float)
        for term in terms:
            df = sum(len(docs) for _, docs, _ in postings.get(term, []))
            if not df:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for offset, docs, tfs in postings[term]:
                for d, tf in zip(docs, tfs):
                    gi = offset + d
                    norm = K1 * (1 - B + B * lengths[gi] / avgdl)
                    scores[gi] += idf * tf * (K1 + 1) / (tf + norm)
        results.append(sorted(scores, key=scores.get, reverse=True)[:k])
    return results


def print_dist(title, counter):
    total = sum(counter.values())
    print(f"\n{title} (n={total})")
    for key, n in counter.most_common():
        print(f"  {key:<26}{n:>4}  {n / total:6.1%}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=FULL_DOCUMENTS_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)
    rng = random.Random(SEED)

    questions = [json.loads(l) for l in open(args.root / "questions.jsonl")]
    selected = select_questions(questions, rng)

    paths = sorted(p for p in args.root.rglob("*.txt"))
    id_to_path = {doc_id_of(p): p for p in paths}
    print(f"corpus: {len(paths):,} documents")

    gold_ids = {d for q in selected for d in q["expected_doc_ids"]}
    missing = gold_ids - id_to_path.keys()
    if missing:
        raise SystemExit(f"gold doc ids not found in corpus: {sorted(missing)}")

    print("running BM25 over the full corpus for hard negatives...")
    top = bm25_top_k(paths, [q["question"] for q in selected], k=HARD_NEG_PER_Q + 10)

    neg_ids = {}  # doc_id -> question_ids it was mined for
    for q, hits in zip(selected, top):
        own_gold = set(q["expected_doc_ids"])
        negs = [doc_id_of(paths[i]) for i in hits if doc_id_of(paths[i]) not in own_gold]
        q["hard_negative_ids"] = negs[:HARD_NEG_PER_Q]
        for d in q["hard_negative_ids"]:
            neg_ids.setdefault(d, []).append(q["question_id"])
    neg_ids = {d: qs for d, qs in neg_ids.items() if d not in gold_ids}

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "subset_questions.jsonl", "w") as f:
        for q in selected:
            f.write(json.dumps(q) + "\n")

    with open(args.out / "subset_corpus.jsonl", "w") as f:
        for doc_id in sorted(gold_ids | neg_ids.keys()):
            path = id_to_path[doc_id]
            f.write(json.dumps({
                "doc_id": doc_id,
                "source": source_of(path, args.root),
                "path": str(path.relative_to(args.root)),
                "is_gold": doc_id in gold_ids,
                "mined_for": neg_ids.get(doc_id, []),
                "text": clean_document_text(path.read_text(encoding="utf-8", errors="ignore")),
            }) + "\n")

    dev_basic = [q for q in selected if q["question_type"] == "basic" and q["split"] == "dev"]
    with open(args.out / "cache_seeds.jsonl", "w") as f:
        for q in rng.sample(dev_basic, N_CACHE_SEEDS):
            f.write(json.dumps({"question_id": q["question_id"], "question": q["question"],
                                "paraphrases": [], "near_misses": []}) + "\n")

    print_dist("questions by type", Counter(q["question_type"] for q in selected))
    print_dist("questions by split", Counter(q["split"] for q in selected))
    print_dist("questions by source",
               Counter(q["source_types"][0] if q["source_types"] else "none" for q in selected))
    print_dist("gold docs by source", Counter(source_of(id_to_path[d], args.root) for d in gold_ids))
    print_dist("hard negatives by source",
               Counter(source_of(id_to_path[d], args.root) for d in neg_ids))
    print(f"\ncorpus: {len(gold_ids)} gold + {len(neg_ids)} hard negatives "
          f"= {len(gold_ids) + len(neg_ids)} documents -> {args.out}")


if __name__ == "__main__":
    main()
