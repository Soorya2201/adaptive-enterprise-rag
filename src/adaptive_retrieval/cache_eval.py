"""Cache experiment: warm with seeds, then shortlist and judge paraphrases and near-misses.

1. Warm-up: each seed runs through the full pipeline; checked answers are stored.
   ``warm_with="gold"`` stores the benchmark answers instead, to test the cache
   apart from retrieval, generation and the checker. It says nothing about the
   answer quality of the whole system.
2. Test: variants are only looked up, never generated or stored, so they cannot
   affect one another. Each judge rules once on every candidate that any threshold
   or key-term setting could select.
3. Scoring: decisions are replayed from the saved checks and verdicts.
4. Controls: every seed repeated word for word (should be YES) and paired with its
   most similar other seed (must be NO).

The threshold is fixed before the run; the sweep is descriptive. The judge is
selected on dev: fewest false reuses, then most correct matches, then fastest.
"""

import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from .cache import CHECKS, Candidate
from .judge import JUDGE_PROMPT
from .evaluate import git_state, percentile, run_settings
from .vector_index import file_sha256

THRESHOLDS = [round(0.70 + 0.01 * step, 2) for step in range(30)]
# name -> (checks switched off, whether the judge has the last word)
CONFIGS = {
    "similarity_only": (("key_terms", "sources"), False),
    "checks_without_judge": ((), False),
    "judge_without_key_terms": (("key_terms",), True),
    "full": ((), True),
}


def default_config(key_term_veto: bool) -> str:
    """The judged configuration that matches how the cache runs normally."""
    return "full" if key_term_veto else "judge_without_key_terms"

WARM_SOURCES = ("generated", "gold")
RIGHT = {"CORRECT", "MOSTLY CORRECT", "GOLD"}  # GOLD is right by construction
_SEED_HEADER = re.compile(
    r"^SEED (\d+) \| (qst_\d+) \| split: (\w+) \| type: (\w+)\s*$"
)
_FIELDS = ("QUESTION", "GOLD ANSWER", "CACHED ANSWER", "VERDICT", "TASK", "SOURCE",
           "EVIDENCE QUOTES", "PARAPHRASE", "NEAR_MISS", "NEAR_MISS_ANSWER")


def parse_seed_file(path: Path) -> list[dict]:
    """Parse cache_seed_questions_loaders(_complete).txt into one dict per seed."""
    seeds, current, field = [], None, None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        header = _SEED_HEADER.match(line)
        if header:
            if current:
                seeds.append(current)
            number, question_id, split, kind = header.groups()
            current = {"seed": int(number), "question_id": question_id, "split": split,
                       "question_type": kind}
            field = None
            continue
        if current is None:
            continue
        if (len(line) >= 3 and set(line) == {"="}) or line == "END":
            field = None
            continue
        key = next((name for name in _FIELDS if line.startswith(name + ":")), None)
        if key:
            field = key.lower().replace(" ", "_")
            current[field] = line[len(key) + 1:].strip()
            continue
        if field and line.strip():
            # Wrapped text continues the previous field.
            current[field] = (current[field] + " " + line.strip()).strip()
    if current:
        seeds.append(current)
    for seed in seeds:
        seed["task"] = seed.get("task", "").split()[0].lower() if seed.get("task") else None
        seed["verdict_label"] = (seed.get("verdict") or "").split(" - ")[0].strip()
    return seeds


def seed_variants(seeds: list[dict]) -> list[dict]:
    variants = []
    for seed in seeds:
        for kind in ("paraphrase", "near_miss"):
            text = (seed.get(kind) or "").strip()
            if text:
                variants.append({
                    "seed_question_id": seed["question_id"], "split": seed["split"],
                    "kind": kind, "required": seed.get("task") == kind, "question": text,
                    "near_miss_answer": seed.get("near_miss_answer") if kind == "near_miss" else None,
                })
    return variants


def missing_required(seeds: list[dict]) -> list[str]:
    return [s["question_id"] for s in seeds if s.get("task") and not (s.get(s["task"]) or "").strip()]


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def selectable(candidates: list[Candidate]) -> list[Candidate]:
    """Candidates that could reach the judge at some threshold or key-term setting."""
    chosen = {}
    for skip, judged in CONFIGS.values():
        if judged:
            candidate = next((c for c in candidates if c.passes(-2.0, skip)), None)
            if candidate:
                chosen[candidate.entry_id] = candidate
    return list(chosen.values())


def decide(
    record: dict, threshold: float, skip: tuple[str, ...] = (), judge: str | None = None,
) -> dict:
    """Replay one lookup. ``judge=None`` replays the deterministic checks alone."""
    candidates = [Candidate(**candidate) for candidate in record["candidates"]]
    selected = next((c for c in candidates if c.passes(threshold, skip)), None)
    judgment = record["judgments"][judge][selected.entry_id] if selected and judge else None
    hit = selected if selected and (judgment is None or judgment["verdict"] == "yes") else None
    seed = record["seed_question_id"]
    result = {"matched": hit.origin if hit else None, "judgment": judgment, "miss_reason": None}
    if record["kind"] == "near_miss":
        return {**result, "outcome": "wrong_reuse" if hit else "correct_miss"}
    if hit:
        return {**result, "outcome": "correct_hit" if hit.origin == seed else "wrong_entry"}
    if judgment:
        reason = f"judge_{judgment['verdict']}"
    else:
        own = next((c for c in candidates if c.origin == seed), None)
        if own is None:
            reason = "own_seed_not_in_top_candidates"
        elif "similar" not in skip and own.similarity < threshold:
            reason = "similar"
        else:
            reason = next(n for n in CHECKS if n not in skip and n != "similar" and not own.checks[n])
    return {**result, "outcome": "missed", "miss_reason": reason}


def score(
    records: list[dict], warm: dict, threshold: float,
    skip: tuple[str, ...] = (), judge: str | None = None,
) -> dict:
    decided = [(record, decide(record, threshold, skip, judge)) for record in records]
    outcomes = Counter(result["outcome"] for _, result in decided)
    hits = [r["seed_question_id"] for r, d in decided if d["outcome"] == "correct_hit"]
    quality = lambda question_id: warm[question_id]["answer_quality"]
    propagated = [q for q in hits if quality(q) in {"WRONG", "PARTLY CORRECT"}]
    paraphrases = sum(r["kind"] == "paraphrase" for r in records)
    near_misses = sum(r["kind"] == "near_miss" for r in records)
    calls = [d["judgment"] for _, d in decided if d["judgment"]]
    false_reuse = outcomes["wrong_reuse"] + outcomes["wrong_entry"]
    return {
        "threshold": threshold,
        "paraphrases": paraphrases,
        "near_misses": near_misses,
        "correct_matches": len(hits),
        "paraphrase_hit_rate": len(hits) / paraphrases if paraphrases else None,
        "wrong_entry": outcomes["wrong_entry"],
        "wrong_reuse": outcomes["wrong_reuse"],
        "near_miss_wrong_reuse_rate": outcomes["wrong_reuse"] / near_misses if near_misses else None,
        "wrong_reuse_seeds": [r["seed_question_id"] for r, d in decided if d["outcome"] == "wrong_reuse"],
        "served": len(hits) + false_reuse,
        "served_right": sum(quality(q) in RIGHT for q in hits),
        "propagated_errors": len(propagated),
        "propagated_error_seeds": propagated,
        "hits_on_unreviewed_answers": [q for q in hits if quality(q) == "UNREVIEWED"],
        "miss_reasons": dict(Counter(d["miss_reason"] for _, d in decided if d["outcome"] == "missed")),
        "judge_calls": len(calls),
        "mean_judge_ms": mean(c["judge_ms"] for c in calls) if calls else None,
        "judge_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls),
    }


def choose_judge(dev_scores: dict[str, dict]) -> str:
    """Fewest false reuses, then most correct matches, then fastest."""
    return min(dev_scores, key=lambda name: (
        dev_scores[name]["wrong_reuse"] + dev_scores[name]["wrong_entry"],
        -dev_scores[name]["correct_matches"],
        dev_scores[name]["mean_judge_ms"] or 0.0,
    ))


def _store_gold(pipeline, question: dict, top_k: int) -> dict:
    """Cache a benchmark answer; return the fields the warm-up log needs."""
    text, answer = question["question"], question["gold_answer"].strip()
    pipeline.cache.admit_reference(
        text, pipeline.session.embed_question(text), answer, pipeline.versions(top_k),
        origin=question["question_id"],
    )
    return {
        "status": "answer", "reason": "gold_answer", "answer": answer, "citations": [],
        "cache": {"admitted": True, "admission": "gold_answer"},
        "metrics": {"total_ms": None, "total_llm_tokens": None},
    }


def run_cache_evaluation(
    pipeline, judges: dict, seeds_path: Path, questions_path: Path, out_dir: Path, *,
    top_k: int = 5, warm_with: str = "generated", progress=None,
) -> tuple[Path, dict]:
    cache = pipeline.cache
    if cache is None or len(cache):
        raise ValueError("Cache evaluation needs an empty cache attached to the pipeline")
    seeds = parse_seed_file(seeds_path)
    if not seeds:
        raise ValueError(f"No seeds found in {seeds_path}")
    if any(seed["split"] == "test" for seed in seeds):
        raise ValueError("Seeds must not come from the test split")
    questions = {}
    for line in Path(questions_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            questions[row["question_id"]] = row
    if warm_with not in WARM_SOURCES:
        raise ValueError(f"warm_with must be one of {WARM_SOURCES}")
    gold = warm_with == "gold"
    if gold:
        if cache.require_exact_quotes:
            raise ValueError("Gold answers have no quotes, so exact quotes cannot be required")
        # From the benchmark: line wrapping in the seed file breaks hyphenated words.
        missing = [s["question_id"] for s in seeds
                   if not (questions[s["question_id"]].get("gold_answer") or "").strip()]
        if missing:
            raise ValueError(f"No gold answer for {missing}")
    variants = seed_variants(seeds)
    if not judges:
        raise ValueError("Cache evaluation needs at least one judge")
    pipeline.warmup()
    for judge in judges.values():
        judge.warmup()
    timestamp = datetime.now(timezone.utc)
    settings = run_settings(pipeline.session)
    label = f"{settings['retriever']}_gold" if gold else settings["retriever"]
    run_dir = Path(out_dir) / f"cache_eval_{label}_{timestamp:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)

    warm = {}
    with (run_dir / "warmup.jsonl").open("w", encoding="utf-8") as stream:
        for number, seed in enumerate(seeds, 1):
            question = questions[seed["question_id"]]
            if gold:
                result = _store_gold(pipeline, question, top_k)
                # Not comparable: the template holds generated answers.
                same, quality = None, "GOLD"
            else:
                result = pipeline.ask(question["question"], top_k, use_cache=False, admit=True,
                                      origin=seed["question_id"])
                same = _normalize(result["answer"]) == _normalize(seed.get("cached_answer"))
                # The template verdict only applies if the pipeline gave the same answer again.
                quality = seed["verdict_label"] if same else "UNREVIEWED"
            record = {
                "question_id": seed["question_id"], "split": seed["split"],
                "question": question["question"], "status": result["status"],
                "reason": result["reason"], "answer": result["answer"],
                "admitted": result["cache"]["admitted"], "admission": result["cache"]["admission"],
                "same_answer_as_template": same,
                "answer_quality": quality,
                "total_ms": result["metrics"]["total_ms"],
                "total_llm_tokens": result["metrics"]["total_llm_tokens"],
                "citations": result["citations"],
            }
            warm[seed["question_id"]] = record
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress:
                progress(f"warm {number}/{len(seeds)} {seed['question_id']} "
                         f"{'cached' if record['admitted'] else record['admission'] + '/' + record['reason']}")

    versions = pipeline.versions(top_k)
    records = []
    with (run_dir / "variants.jsonl").open("w", encoding="utf-8") as stream:
        for number, variant in enumerate(variants, 1):
            text = variant["question"]
            started = perf_counter()
            vector = pipeline.session.embed_question(text)
            embedded = perf_counter()
            candidates, _ = cache.shortlist(text, vector, versions, pipeline.chunks)
            checked = perf_counter()
            own_similarity = cache.similarity_to_origin(variant["seed_question_id"], vector)
            record = {
                **variant,
                "seed_cached": own_similarity is not None,
                "own_seed_similarity": own_similarity,
                "embedding_ms": (embedded - started) * 1000,
                "shortlist_ms": (checked - embedded) * 1000,
                "candidates": [asdict(candidate) for candidate in candidates],
                "judgments": {
                    name: {c.entry_id: asdict(judge.judge(text, c.question)) for c in selectable(candidates)}
                    for name, judge in judges.items()
                },
            }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress:
                progress(f"test {number}/{len(variants)} {variant['seed_question_id']} {variant['kind']}")

    controls = []
    with (run_dir / "controls.jsonl").open("w", encoding="utf-8") as stream:
        for entry in cache.entries:
            pairs = [("exact_repeat", entry)]
            others = [other for other in cache.entries if other.origin != entry.origin]
            if others:
                nearest = max(others, key=lambda o: cache.similarity_to_origin(o.origin, entry.vector))
                pairs.append(("other_seed", nearest))
            for kind, saved in pairs:
                record = {
                    "kind": kind, "seed_question_id": entry.origin, "saved_question_id": saved.origin,
                    "judgments": {n: asdict(j.judge(entry.question, saved.question)) for n, j in judges.items()},
                }
                controls.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    scored = [r for r in records if r["seed_cached"]]
    groups = {
        "dev": [r for r in scored if r["split"] == "dev"],
        "val": [r for r in scored if r["split"] == "val"],
        "all": scored,
    }

    def compare(threshold: float) -> dict:
        table = {}
        for config, (skip, judged) in CONFIGS.items():
            for name in judges if judged else (None,):
                label = f"{config}:{name}" if name else config
                table[label] = {
                    group: score(rows, warm, threshold, skip, name) for group, rows in groups.items()
                }
        return table

    results = compare(cache.threshold)
    default = default_config(cache.key_term_veto)
    chosen = choose_judge({name: results[f"{default}:{name}"]["dev"] for name in judges})
    cached_seeds = [w for w in warm.values() if w["admitted"]]
    judge_stats = {}
    for name, judge in judges.items():
        calls = [j for r in records for j in r["judgments"][name].values()]
        times = [call["judge_ms"] for call in calls]
        repeat = [c for c in controls if c["kind"] == "exact_repeat"]
        other = [c for c in controls if c["kind"] == "other_seed"]
        judge_stats[name] = {
            "backend": judge.describe(),
            "mean_ms": mean(times) if times else None,
            "p95_ms": percentile(times, 0.95),
            "mean_tokens": mean(c["input_tokens"] + c["output_tokens"] for c in calls) if calls else None,
            "verdicts": dict(Counter(c["verdict"] for c in calls)),
            "exact_repeat_yes": sum(c["judgments"][name]["verdict"] == "yes" for c in repeat),
            "exact_repeats": len(repeat),
            "other_seed_yes": sum(c["judgments"][name]["verdict"] == "yes" for c in other),
            "other_seeds": len(other),
        }
    summary = {
        "run": {
            "timestamp_utc": timestamp.isoformat(timespec="seconds"), **git_state(),
            "seeds_path": str(seeds_path), "seeds_sha256": file_sha256(seeds_path),
            "top_k": top_k, "threshold": cache.threshold,
            "max_candidates": cache.max_candidates,
            "require_exact_quotes": cache.require_exact_quotes,
            "key_term_veto": cache.key_term_veto, "default_config": default,
            "warm_with": warm_with,
            "judge_prompt": JUDGE_PROMPT,
            "versions": versions, "generator": pipeline.generator.model_name,
            **settings,
        },
        "seeds": len(seeds),
        "seeds_cached": len(cached_seeds),
        "seeds_not_cached": {w["question_id"]: w["reason"] for w in warm.values() if not w["admitted"]},
        "seed_answers_changed_since_template": [
            w["question_id"] for w in warm.values() if w["same_answer_as_template"] is False
        ],
        "variants": len(records),
        "variants_scored": len(scored),
        "variants_skipped_seed_not_cached": len(records) - len(scored),
        "judge_chosen_on_dev": chosen,
        "results": results,
        "judges": judge_stats,
        "latency": {
            "mean_embedding_ms": mean(r["embedding_ms"] for r in records) if records else None,
            "mean_shortlist_ms": mean(r["shortlist_ms"] for r in records) if records else None,
            # A gold warm-up never runs the pipeline, so it has no cost to compare with.
            "mean_full_pipeline_ms": (
                mean(w["total_ms"] for w in cached_seeds) if cached_seeds and not gold else None
            ),
            "mean_llm_tokens_full_pipeline": (
                mean(w["total_llm_tokens"] for w in cached_seeds) if cached_seeds and not gold else None
            ),
        },
        "sweeps": [{"threshold": t, "results": compare(t)} for t in THRESHOLDS],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary


def format_cache_summary(summary: dict) -> str:
    lines = [
        f"Seeds cached: {summary['seeds_cached']}/{summary['seeds']} | variants scored: "
        f"{summary['variants_scored']}/{summary['variants']} | threshold {summary['run']['threshold']:.2f}",
    ]
    if summary["run"].get("warm_with") == "gold":
        lines.insert(0, "CACHE-ONLY TEST: warmed with gold answers, so a matched answer is right by "
                        "construction. Not end-to-end accuracy.")
    default = summary["run"].get("default_config")
    for label, groups in summary["results"].items():
        lines.append(f"{label} (default)" if default and label.split(":")[0] == default else label)
        for group, row in groups.items():
            lines.append(
                f"  {group}: matched {row['correct_matches']}/{row['paraphrases']} "
                f"(right answer {row['served_right']}, propagated error {row['propagated_errors']}) | "
                f"wrong reuse {row['wrong_reuse']}/{row['near_misses']} {row['wrong_reuse_seeds']} | "
                f"wrong entry {row['wrong_entry']} | judge calls {row['judge_calls']} | misses {row['miss_reasons']}"
            )
    for name, row in summary["judges"].items():
        if row["mean_ms"] is not None:
            lines.append(
                f"Judge {name}: {row['mean_ms']:.0f} ms (p95 {row['p95_ms']:.0f}), "
                f"{row['mean_tokens']:.0f} tokens, verdicts {row['verdicts']} | exact repeats YES "
                f"{row['exact_repeat_yes']}/{row['exact_repeats']} | other seeds YES "
                f"{row['other_seed_yes']}/{row['other_seeds']}"
            )
    lines.append(f"Judge chosen on dev: {summary['judge_chosen_on_dev']}")
    if summary["seeds_not_cached"]:
        lines.append(f"Seeds not cached (their variants are skipped): {summary['seeds_not_cached']}")
    return "\n".join(lines)


def print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
