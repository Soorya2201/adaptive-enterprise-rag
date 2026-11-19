# Adith's final evaluation workflow

## Assignment status

| Assignment | Implemented | Outstanding |
| --- | --- | --- |
| Correctness/refusal accuracy with Naveen | Scorer, blinded review export, evidence-bound labels, split/category summaries | Actual Qwen traces and independently reviewed correctness labels |
| NLI with Naveen | Pinned local inference, per-claim checks, long-context windows, contradiction detection | Enterprise validation; threshold is not calibrated |
| Adaptive vs fixed RAG, paired CIs | Shared NLI gate, one bounded wider retrieval retry, grouped bootstrap | Collect both runs and calculate intervals from reviewed answers |
| About 300 questions with Shusei/Ty | 300 questions, 802 documents, 82 distinct multi-document cases, 210/45/45 splits | Team review of source rubrics and corpus policy |
| Frozen test run with Shusei/Ty | Frozen membership, protocol/input hashes, exclusive trace creation | Execute on the team's Apple Silicon/MLX runtime after validation |
| Final report/presentation with everyone | Methods and limitations below | Real experimental results and team conclusions |

Locally verified on Windows: 47 unit tests, compilation, the full benchmark
build/hash verification, and five actual pinned-model NLI sanity cases passed.
Synthetic NLI sanity cases are not benchmark accuracy. No held-out Qwen run,
human correctness judgments, or final answer-quality performance is claimed.

## 1. Reproduce the benchmark

Use the project virtual environment. Install evaluation dependencies with
`python -m pip install -e '.[evaluation,hybrid]'`. On Apple Silicon use the team's
existing demo environment for Qwen generation as well.

Download `all_documents.zip` from the [EnterpriseRAG-Bench v1.0.0 release](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/tag/v1.0.0)
to `data/all_documents.zip` (about 1.2 GB, ignored by Git). The builder reads only
needed archive members and does not extract paths.

```text
python -m adaptive_retrieval.final_benchmark
python -m adaptive_retrieval.final_benchmark --verify
```

Output is `data/processed/final300-v2/`. Compare generated question/corpus hashes
with `benchmarks/final300/manifest.json`. The committed questions and manifest
freeze membership; downloaded documents stay local. Existing output is never
replaced; use `--out` for an independent rebuild.

All original 100 evaluated IDs are excluded. Questions sharing gold documents
or identical text are grouped into one split. Groups touching legacy gold
documents are development only. No gold document crosses splits. The test set
has 45 new questions, including 12 multi-document cases and one explicit
unanswerable case; validation also has one unanswerable case. Refusal recall
with that denominator is highly uncertain. Report counts and inspect development
refusal examples; do not generalize from one held-out refusal question.

The bounded corpus contains selected gold documents plus legacy gold/distractor
documents. Distractors use capped term-frequency overlap within this corpus;
this is not full-archive hard-negative mining. Duplicate IDs with different
source text retain all sorted variants under the shared ID. The multi-document
count uses distinct IDs. `high_level` questions without expected documents are
excluded rather than mislabeled as explicit unanswerable cases. This benchmark
does not establish full-enterprise retrieval performance.

## 2. Prepare NLI and the new index

The [NLI model](https://huggingface.co/cross-encoder/nli-deberta-v3-small) is pinned
to `fa2804872c3b4bd748f38c0185cc85775361e735`. Normal inference uses cached
safetensors, with remote code disabled. Download and verify explicitly:

```text
python -m adaptive_retrieval.nli_smoke --download
```

This writes `data/outputs/nli_smoke.json` exclusively. Omit `--download` to run
offline. Checks cover entailment, contradiction, unrelated facts, a requested
deadline versus a promise, and evidence beyond the first context window.
The 0.8 threshold is fixed but not calibrated. Each claim is checked against all
512-token premise windows; hypotheses are not silently truncated. Conflicting
windows require review. Sentence-level NLI can miss multi-hop reasoning,
qualifications, dates, and conflicts. It is a diagnostic, not proof of truth.

On the supported Qwen runtime, build a new index for this exact corpus:

```text
python src/runner.py chunk --corpus data/processed/final300-v2/corpus.jsonl --out data/processed/final300-v2/chunks.jsonl
python src/runner.py embed --chunks data/processed/final300-v2/chunks.jsonl --out data/processed/final300-v2/qwen_index.jsonl
```

## 3. Freeze and collect both policies

Both modes use the same citation validator and NLI gate. Fixed makes one
attempt. Adaptive retries once at top-10 when top-5 refuses or fails grounding.
Final failed grounding becomes a refusal. Routing receives no gold answers or
expected document IDs. Attempts, cumulative tokens/calls/rounds, NLI time, and
total latency are saved. This is a bounded retry policy, not the separate
verified-answer cache assigned to other teammates.

Inspect development and settle settings on validation first:

```text
python -m adaptive_retrieval.final_experiment --index data/processed/final300-v2/qwen_index.jsonl --out data/outputs/final300-val --split val --freeze
python -m adaptive_retrieval.final_experiment --index data/processed/final300-v2/qwen_index.jsonl --out data/outputs/final300-val --split val --mode fixed
python -m adaptive_retrieval.final_experiment --index data/processed/final300-v2/qwen_index.jsonl --out data/outputs/final300-val --split val --mode adaptive
```

Once settings are settled, repeat these three commands with `--split test` and
a new directory `data/outputs/final300-test`. Do not tune using test outcomes.
The protocol pins benchmark/index hashes, models, threshold, retrieval depths,
and token budgets. Changed settings or inputs fail. Each mode creates traces
exclusively; partial failures remain for audit rather than being appended or
overwritten. Disclose any rerun under a new protocol directory. Windows stops
before creating a test trace because the current generator needs Apple Silicon.
Protocol creation alone is not an executed experiment.

## 4. Review correctness and compute paired results

Each run directory exports `questions.jsonl` for the exact split. Create blinded
review tasks for each policy:

```text
python -m adaptive_retrieval.answer_review --questions data/outputs/final300-test/questions.jsonl --traces data/outputs/final300-test/fixed.jsonl --out data/outputs/review-fixed
python -m adaptive_retrieval.answer_review --questions data/outputs/final300-test/questions.jsonl --traces data/outputs/final300-test/adaptive.jsonl --out data/outputs/review-adaptive
```

Adith/Naveen fill `judge` and boolean `correct`, with reasons, after reading the
question, gold answer, rubric, and supplied evidence. Optional facts are not
mandatory; negative constraints must be respected. Preserve answer/hash fields.
Policy names and model suggestions are hidden from tasks. A second reviewer
should adjudicate disagreements. Refusals and responses to explicitly
unanswerable questions are scored automatically.

Optional automatic NLI diagnostics:

```text
python -m adaptive_retrieval.nli --questions data/outputs/final300-test/questions.jsonl --traces data/outputs/final300-test/adaptive.jsonl --out data/outputs/adaptive-nli.jsonl
```

NLI checks grounding against context actually sent to the generator and reference
coverage against declarative `gold_answer` sentences. Imperative `answer_facts`
rubric items are not fed as asserted facts. Automatic correctness is explicitly
a proxy requiring human review; it is not interchangeable with human labels.
Using the same NLI model to route and evaluate is not independent evidence of
grounding quality, so report that diagnostic separately.

Primary comparison with reviewed labels:

```text
python -m adaptive_retrieval.answer_evaluation --questions data/outputs/final300-test/questions.jsonl --traces data/outputs/final300-test/adaptive.jsonl --judgments data/outputs/review-adaptive/judgments.jsonl --baseline-traces data/outputs/final300-test/fixed.jsonl --baseline-judgments data/outputs/review-fixed/judgments.jsonl --out data/outputs/final300-test/comparison.json
```

The scorer rejects incomplete coverage, duplicates, unknown IDs, changed text,
and stale evidence-bound labels. Unjudged correctness remains null with coverage
reported; paired comparison requires complete judgments in both runs. It reports
correctness, binary refusal accuracy, unanswerable refusal recall, false-refusal
rate, and split/type/document-scope summaries. Multi-document diagnostics include
final-passage gold recall and all-gold hit, separate from correctness.

The paired bootstrap resamples connected question groups, using 2,000 seeded
replicates, and reports candidate-minus-baseline deltas with 95% percentile CIs.
Older files without groups fall back to question-level resampling. The test set
is small, so intervals may be wide. Local token/latency traces do not measure
API dollar cost or energy. Final report conclusions must wait for real runs and
reviewed answers; the synthetic tests and NLI smoke checks are implementation
evidence only.
