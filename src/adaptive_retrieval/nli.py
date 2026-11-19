"""Pinned local NLI inference and auditable automatic answer-quality proxies."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from .answer_evaluation import keyed, read_records, trace_digest
from .model_store import MODEL_CACHE

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


def claims(text):
    """Sentence-level claims; this is a segmentation heuristic, not a judge."""
    text = re.sub(r"\[S\d+\]", "", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


class NliJudge:
    def __init__(self, *, threshold=.8, batch_size=8):
        if not .5 < threshold < 1 or batch_size < 1:
            raise ValueError("NLI threshold must be between .5 and 1; batch size must be positive")
        self.threshold, self.batch_size = threshold, batch_size
        self.model = self.tokenizer = None

    def load(self, *, download=False):
        if self.model is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = snapshot_download(NLI_MODEL, revision=NLI_REVISION, cache_dir=str(MODEL_CACHE),
                                 local_files_only=not download,
                                 allow_patterns=["*.json", "model.safetensors", "spm.model"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
        labels = {str(value).lower() for value in self.model.config.id2label.values()}
        if labels != {"entailment", "neutral", "contradiction"}:
            raise ValueError("NLI model label mapping is not explicit")

    def assess(self, premise, hypothesis):
        self.load()
        import torch
        if not premise.strip() or not hypothesis.strip():
            raise ValueError("NLI premise and hypothesis must be nonempty")
        if len(self.tokenizer.encode(hypothesis, add_special_tokens=False)) > 384:
            raise ValueError("NLI hypothesis too long; split into smaller claims")
        # Every premise window is evaluated; evidence at the end is not dropped.
        tokens = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=512,
                                stride=64, return_overflowing_tokens=True,
                                padding=True, return_tensors="pt")
        tokens.pop("overflow_to_sample_mapping", None)
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(tokens["input_ids"]), self.batch_size):
                batch = {key: value[start:start + self.batch_size] for key, value in tokens.items()}
                probabilities.extend(self.model(**batch).logits.softmax(dim=-1).tolist())
        labels = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}
        scores = [{labels[i]: value for i, value in enumerate(row)} for row in probabilities]
        entailment = max(row["entailment"] for row in scores)
        contradiction = max(row["contradiction"] for row in scores)
        # Conflicting windows need review even if another window supports the claim.
        label = ("contradiction" if contradiction >= self.threshold else
                 "entailment" if entailment >= self.threshold else "neutral")
        return {"label": label, "max_entailment": entailment,
                "max_contradiction": contradiction, "windows": len(scores)}


def judge_trace(question, trace, judge):
    if trace["status"] != "answer":
        raise ValueError("Only answer traces require NLI judgments")
    supplied = [p for p in trace.get("passages", []) if p.get("prompt_label") is not None]
    if not supplied:
        raise ValueError("Cannot judge grounding without the actual prompt context")
    premise = "\n\n".join(f"[{p['prompt_label']}] {p['text']}" for p in supplied)
    answer_claims = claims(trace["answer"])
    if not answer_claims:
        raise ValueError("Answer contains no claims")
    grounding = [{"claim": claim, **judge.assess(premise, claim)} for claim in answer_claims]
    # Gold answers are declarative references; answer_facts often contain rubric
    # instructions (must/may/must not), so never feed those as asserted facts.
    reference_claims = claims(question.get("gold_answer", ""))
    coverage = [{"claim": claim, **judge.assess(trace["answer"], claim)} for claim in reference_claims]
    nli = ("contradiction" if any(item["label"] == "contradiction" for item in grounding) else
           "entailment" if all(item["label"] == "entailment" for item in grounding) else "neutral")
    proposed = bool(reference_claims) and nli == "entailment" and all(item["label"] == "entailment" for item in coverage)
    return {
        "question_id": question["question_id"], "answer": trace["answer"],
        "trace_sha256": trace_digest(trace),
        "judge": f"{NLI_MODEL}@{NLI_REVISION}", "judgment_kind": "automatic_nli_proxy",
        "correct": proposed, "nli": nli, "threshold": judge.threshold,
        "human_review_required": True, "grounding": grounding, "reference_coverage": coverage,
        "context_sha256": hashlib.sha256(premise.encode()).hexdigest(),
        "limitations": "Sentence-level NLI proxy; not human correctness or a calibrated factuality score.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        questions, traces = keyed(read_records(args.questions)), keyed(read_records(args.traces))
        if questions.keys() != traces.keys():
            raise ValueError("Questions and traces must have identical IDs")
        judge = NliJudge(threshold=args.threshold)
        judge.load(download=args.download)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            for key, question in questions.items():
                trace = traces[key]
                if trace.get("status") not in {"answer", "refusal"}:
                    raise ValueError(f"Invalid trace status: {key}")
                if trace.get("question") != question["question"]:
                    raise ValueError(f"Trace question mismatch: {key}")
                if trace["status"] == "answer":
                    record = judge_trace(question, trace, judge)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"Judged {key}: {record['nli']}", flush=True)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
