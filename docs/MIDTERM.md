# Midterm experiment protocol

Branch: `midterm-presentation`. Purpose: reproduce the existing retrieval
baseline and produce evidence for the two-slide midterm presentation. No
threshold or model tuning is performed against validation/test results.

## Completed run

The local reproduction matched all 100 gold-document ranks from the teammate
baseline. Start with the [results brief](../results/midterm/retrieval_eval_20261019-023503/MIDTERM_BRIEF.md)
and [manual evidence review](../results/midterm/retrieval_eval_20261019-023503/REVIEW.md).
The run directory also contains raw per-question results, summary metrics,
development examples, and the installed package versions. All 16 unit tests,
compilation, corpus validation, and dependency checks passed.

## Reproduce on Windows

From the repository root, using Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s src/tests -v
.\.venv\Scripts\python.exe src/runner.py prepare-subset
.\.venv\Scripts\python.exe src/runner.py check-subset
.\.venv\Scripts\python.exe src/runner.py chunk --strategy paragraph --max-words 120 --overlap-words 20
.\.venv\Scripts\python.exe src/runner.py embed --batch-size 32
.\.venv\Scripts\python.exe src/runner.py evaluate --out-dir results/midterm
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
After evaluation, pass the printed run directory to:

```powershell
.\.venv\Scripts\python.exe src/midterm_report.py results/midterm/retrieval_eval_TIMESTAMP
```

This writes `MIDTERM_BRIEF.md` and `examples.json` beside the raw results.
The examples include full gold source documents for manual inspection.

## Interpretation

- Hit@1/Hit@5: proportion of answerable questions whose first gold document is
  ranked in the first one/five documents. Chunks are collapsed by document.
- MRR: average reciprocal rank of the first gold document; not average rank.
- Questions without gold documents are excluded from retrieval metrics and
  counted separately. This is not a refusal benchmark yet.
- Report all-question metrics as exploratory. Split and type breakdowns expose
  the small denominators; earlier pooled scores were already inspected.
- Timing includes query embedding, exhaustive chunk scoring, and document
  ranking. The first query also includes lazy model loading. It excludes
  offline indexing and answer generation; do not present it as total RAG latency.
- Local inference has no paid API calls, but hardware/energy cost is unmeasured.
- No routing, cache, reranker, or generation claim is supported by this run.

Keep the dataset and model settings fixed when comparing this reproduction to
the saved October 16 baseline. Model/runtime versions and question/corpus hashes
are recorded with the evaluation. Model files and generated index remain local.

## Presentation readiness

Use the resulting brief for the takeaway, one success, and one failure example.
Before presenting, have a teammate inspect the cited evidence and rehearse the
limitations. Next implementation milestones remain grounded generation/refusal,
reranking, cache and bounded requery, then quality/latency/cost comparisons.
Slides: `docs/presentation/Team13-Midterm-v2.pptx` is the simplified two-slide
deck (speaker notes and Q&A prep included). Rebuild it with
`npm install --no-save pptxgenjs && node docs/presentation/build_v2.cjs`. The earlier
`Team13-Midterm.pptx` is kept for reference.
