# Answer evaluation (Adith, first increment)

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
NLI judgments are imported in this increment; no NLI model is run automatically.
Correctness and entailment are separate judgments: a grounded answer can still
fail to answer the question. A quote match alone establishes neither.

```powershell
$env:PYTHONPATH = 'src'
python -m adaptive_retrieval.answer_evaluation --questions selected_questions.jsonl --traces candidate.jsonl --judgments candidate_judgments.jsonl --out results/answer_report.json
```

Reports include split and question-type groups. Correctness on unjudged answers
is null; coverage is always reported. A refusal on an answerable question and
an answer on an unanswerable question count as incorrect. Refusal accuracy is
the accuracy of the binary decision to answer/refuse across all questions;
unanswerable refusal recall and answerable false-refusal rate are also reported.
Answerability currently follows nonempty `expected_doc_ids`, as in retrieval
evaluation. Empty category denominators produce null, not zero.

For a paired comparison add `--baseline-traces fixed.jsonl
--baseline-judgments fixed_judgments.jsonl`. Both runs must cover identical IDs
and have complete correctness judgments. The seeded paired bootstrap resamples
question-level differences and reports candidate-minus-baseline deltas with
95% percentile intervals (2,000 replicates). Compare frozen settings on the
same test questions; do not tune using test results. Single-question resampling
assumes independent questions, so related paraphrases require a future grouped
bootstrap. Small test sets yield unstable intervals.

## Remaining work

- Add batch trace collection with question IDs on a supported model runtime.
- Integrate and validate an NLI judge, recording the model revision and context.
- Collect independently reviewed correctness labels with Naveen.
- Coordinate the 300-question/multi-document subset with Shusei and Ty; keep
  development, validation, and test membership fixed before final experiments.
- Run the frozen fixed/adaptive comparison once adaptive routing exists. Main
  currently performs one retrieval round, so no adaptive performance is claimed.

No real answer-quality or NLI scores have been produced by this increment.
