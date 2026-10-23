# Team progress review — October 18, 2026

## Branches and integration

- Pulled main from f5f36e3 to 0d30da1 (dataset archive ignore rule).
- origin/load_questions at 84d6bdb contains subset selection and documents.
- origin/feat/chunking-embedding at fc5d6fa includes load_questions plus the local retrieval pipeline, setup documentation, and evaluation.
- integration/team-progress combines current main with that feature branch. The only merge conflict was .gitignore; exclusions from both sides were retained.
- Main and the teammates' feature branches were left unchanged remotely.

## Discussion compared with checked-in work

| Area | Verified state |
| --- | --- |
| Dataset scope | Full EnterpriseRAG-Bench: ~500,000 documents and 500 questions. Initial proposal target: about 10,000 documents. Implemented subset: 290 documents (90 gold plus 200 other referenced documents), 100 questions. |
| Question mix | 35 basic, 25 semantic, 15 intra-document reasoning, 15 constrained, 10 info-not-found. |
| Splits | 69 development, 15 validation, 16 test. |
| Multi-document coverage | Zero questions have more than one expected document; current results do not validate multi-document retrieval. |
| Preparation | Reconstructs processed JSONL from checked-in documents; full-corpus subset builder also exists. |
| Chunking | Paragraph/fixed chunking, overlap, source offsets, and model-token fitting implemented. |
| Embeddings/search | MiniLM encoder and local JSONL vector index with cosine search; interactive source inspection implemented. This is not an external vector database. |
| Evaluation | Document-level Hit@5 and MRR, per-question outputs, metadata and saved results. |
| Generation and adaptive behavior | Interfaces/plans exist; generated answers, reranking, verified cache, adaptive second search, and evidence-based refusal remain unfinished. |

## Saved results

The checked-in results/retrieval_eval_20261016-191756 run reports 100 questions,
90 scored, Hit@5 = 0.844444 and MRR = 0.670343. Semantic questions have Hit@5 =
0.64. Ten info-not-found questions are logged but not scored for refusal.
These are existing teammate results, not a newly executed model evaluation.

MRR is the mean reciprocal rank; 1 / MRR is not the mean rank. Avoid describing
0.670 as an average rank of about 1.5. Metrics currently pool development,
validation, and test questions; report split-specific results before making
held-out performance claims or tuning against these outputs.

## Validation performed locally

- All 11 existing unit tests passed.
- Python compileall passed for src.
- Synthetic demo passed.
- prepare-subset produced 100 questions and 290 documents.
- check-subset confirmed all referenced documents and matching content.
- Default chunk command produced 3,098 chunks.
- MiniLM embedding/search/evaluation was not rerun: sentence-transformers is
  not installed in the local Python environment. The saved run uses max_words
  120 and 3,358 indexed chunks, so it must not be compared directly with the
  current default chunk output as if configurations were identical.

## Recommended next work

1. Review and merge this integration branch so everyone starts from the same code.
2. Reproduce the saved MiniLM baseline with recorded settings; add split/category
   summaries and direct tests of evaluation calculations.
3. Assign generation/integration work: grounded responses with source citations,
   evidence checks, and explicit refusal. Keep a fixed baseline for comparison.
4. Add reranking, then cache and bounded second retrieval as separate increments.
   Complete the discussed cache paraphrases and near-miss examples. The quick
   prepare-subset command does not generate cache_seeds.jsonl.
5. Add multi-document questions before claiming support for that objective.
6. Measure latency, usage/cost, answer correctness, faithfulness, and refusal
   accuracy; current retrieval scores do not measure those outcomes.

The Discord discussion mentions a presentation on the 20th or 22nd without
settling which; confirm the actual slot with the team.
