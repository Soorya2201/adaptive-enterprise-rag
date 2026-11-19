# Answer evaluation (Adith)

See [the final experiment runbook](FINAL_EXPERIMENT.md) for the frozen
300-question benchmark, actual NLI model, batch collection, and current status.

The week 8 assignment is answer correctness, refusal accuracy, and NLI checking
with Naveen, comparing adaptive versus fixed single-pass RAG with paired CIs.
Week 9 also assigns Adith work with Shusei and Ty on roughly 300 questions,
including multi-document cases and a frozen test run.

Latest main already provides hybrid retrieval, local Qwen generation, citations,
and JSONL traces. Quote validation is not semantic entailment. The current
generator uses MLX on Apple Silicon; this offline evaluator also runs on Windows.

## Score saved traces

Select the intended question split into a JSONL file first. Supply exactly one
trace per selected question, using `question_id` or an exact unique `question`
text (compatible with existing `ask` traces). Duplicate, missing, extra, and
ambiguous predictions fail instead of silently changing the denominator.

Judgments are JSONL objects, for example:

```json
{"question_id":"qst_0046","answer":"The exact saved answer, including citations","judge":"human:Adith","correct":true,"nli":"entailment"}
```

Judge correctness against `gold_answer` / `answer_facts` and the source evidence.
Require all essential facts and reject material unsupported or contradictory
claims. NLI labels are `entailment`, `neutral`, or `contradiction`, judged against
the actual context supplied to the generator (`prompt_label` is non-null), not
outside knowledge. Record the judge identity or model/revision in `judge`.
NLI judgments can be imported or generated using the pinned local NLI model.
Automatic judgments are marked as proxies and require human review for primary
correctness results. Evidence-bound labels include a trace hash.
Correctness and entailment are separate judgments: a grounded answer can still
fail to answer the question. A quote match alone establishes neither.

```powershell
$env:PYTHONPATH = 'src'
python -m adaptive_retrieval.answer_evaluation --questions selected_questions.jsonl --traces candidate.jsonl --judgments candidate_judgments.jsonl --out results/answer_report.json
```

Reports include split, question-type, and document-scope groups. Correctness on unjudged answers
is null; coverage is always reported. A refusal on an answerable question and
an answer on an unanswerable question count as incorrect. Refusal accuracy is
the accuracy of the binary decision to answer/refuse across all questions;
unanswerable refusal recall and answerable false-refusal rate are also reported.
Answerability currently follows nonempty `expected_doc_ids`, as in retrieval
evaluation. Empty category denominators produce null, not zero.

For a paired comparison add `--baseline-traces fixed.jsonl
--baseline-judgments fixed_judgments.jsonl`. Both runs must cover identical IDs
and have complete correctness judgments. The seeded paired bootstrap resamples
gold-document-connected groups and reports candidate-minus-baseline deltas with
95% percentile intervals (2,000 replicates). Compare frozen settings on the
same test questions; do not tune using test results. Older question files without
evaluation groups use question-level resampling. Small test sets yield unstable intervals.

## Remaining work

- Execute batch trace collection on the supported Apple Silicon/MLX runtime.
- Validate NLI against independently reviewed enterprise examples; local
  inference and five synthetic model sanity checks are implemented and passed.
- Collect independently reviewed correctness labels with Naveen.
- Review the built 300-question subset with Shusei and Ty; membership and
  210/45/45 splits are frozen, including 82 distinct multi-document questions.
- Run the frozen fixed/adaptive comparison. The new experimental policy permits
  one wider-context retry; the default interactive pipeline is unchanged.

No held-out answer-quality results or human judgments have been produced yet.
