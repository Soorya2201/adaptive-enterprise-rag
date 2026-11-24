# Cost-Aware Adaptive Retrieval for Enterprise Support

Team 13 studies whether adjusting search effort per question reduces latency and
compute without lowering answer quality. The pipeline is fully local: hybrid
BM25S + dense retrieval, an optional **disagreement gate** that runs LameR query
expansion only when keyword and dense search disagree on the top document, and
Qwen generation that answers with verified quotations or refuses.

Answer-quality evaluation: see [offline correctness, refusal, and paired-comparison instructions](docs/ANSWER_EVALUATION.md).

See [the architecture](docs/ARCHITECTURE.md) for how the files connect and
[the plan](docs/PLAN.md) for the remaining milestones.

## Setup

Answer generation targets **Apple Silicon macOS** (MLX). Run from the repository
root:

```bash
conda activate CS544-RAG

# One-time setup: install packages, then download the two pinned public models.
python -m pip install -r requirements.txt
python src/runner.py download-models

# Build the local index from the checked-in subset; no full dataset needed.
python src/runner.py prepare-subset
python src/runner.py check-subset
python src/runner.py chunk          # 240-word paragraph-aware chunks, 40-word overlap
python src/runner.py embed
```

`prepare-subset` decodes 32 benchmark documents with escaped line breaks (31
fully escaped, 1 partly). If your `data/processed/` predates this, re-run
`prepare-subset`, `chunk` and `embed`.

Models use about 3.5 GB of disk. Normal inference reads cached model files; it
does not download or call a hosted inference API.

**LameR expansion** (`--retriever adaptive` or `hybrid_lamer`) needs an expansion
model. The standard backend, used in the midterm, is Llama 3.2 3B (Q4_K_M) in
Ollama. No account or access token is needed: Ollama downloads the weights from
its own registry, unlike Meta's gated Hugging Face repository. Use is governed by
the Llama 3.2 Community License.

```bash
brew install ollama          # or install the app from https://ollama.com/download
ollama serve                 # skip if the Ollama app is already running
ollama pull llama3.2:3b
ollama list                  # ID must start with a80c4f17acd5
```

Warmup refuses to run if the installed `llama3.2:3b` digest differs from the one
pinned in `expansion.py`, because changed weights would change the results. Each
run records the Ollama version, model digest, quantization, sampling options,
embedding device and machine (chip and memory) in `summary.json`.

Pass `--expander mlx` to reuse the local Qwen generator instead. That avoids
Ollama but changes the expansion model, so its numbers are not comparable with
the midterm table.

## Retrieval modes

Every search, evaluation and RAG command accepts `--retriever`. Together they
form the ablation ladder for the final report:

| Mode | What it does | Hypothesis |
| --- | --- | --- |
| `cosine` | Dense-only exact cosine search, same FAISS worker as hybrid (baseline) | |
| `bm25` | Keyword-only BM25S | |
| `hybrid` (default) | RRF fusion (k = 60) of dense and BM25S, 50 candidates each | H1 |
| `hybrid_lamer` | Hybrid with LameR expansion on every question | H2 baseline |
| `adaptive` | Hybrid; expand only when dense and BM25S disagree on the top document | H2 |

When `adaptive` expands, the top five BM25S passages are given to the expansion
model, which writes two short possible answers. The question plus each answer
becomes a second BM25S query, and its ranking is fused with the original dense
ranking. Traces record whether the gate fired, the expansion text, tokens and time.
RRF scores are rank-based; compare ranks, not raw scores across modes.

## Evaluation

```bash
# Retrieval only (no generation): nDCG@5, Recall@5, Hit@1/5, MRR, latency, gate rate.
python src/runner.py evaluate --retriever hybrid --split dev
python src/runner.py evaluate --retriever adaptive --split dev

# Full RAG over a split: answer/refusal outcomes, citations, latency and tokens.
python src/runner.py rag-eval --retriever adaptive --split dev
python src/runner.py rag-eval --retriever hybrid --split dev --limit 5   # smoke test
```

**Midterm-comparable retrieval runs** use `--repeats 8`, which makes 8 passes,
each in a seeded random question order. Quality is averaged per question across
passes, then across questions; latency and gate rate use every request. Expansion
is seeded and usually repeats exactly (0 of 29 expansions differed between the
two saved runs compared), but not always, so use repeats for reported numbers.

```bash
python src/runner.py evaluate --retriever adaptive --split dev --repeats 8
```

The midterm's retrieval latencies came from an Apple M4 Max (64 GB); compare
latency only between runs on the same machine. The midterm also used 120-word
chunks; this repository now uses 240 words on a cleaned corpus, so its quality
numbers are close to, not identical with, the midterm table.

Both commands default to the **dev** split. Tune on `dev`/`val`; run `test` only
once per frozen configuration. Each run writes `per_question.jsonl` and
`summary.json` under `results/<kind>_<retriever>_<split>_<timestamp>/`, including
Git commit, index settings and expander model. Summaries include a breakdown by
question type.

`rag-eval` labels each question `answered`, `false_refusal`,
`correct_refusal` or `unsupported_answer`, and reports whether the gold document
reached the prompt and whether answers cite it. Records keep `gold_answer` and
`answer_facts` with `answer_correct: null`, so correctness and citation
entailment can be scored later without regenerating. Gold answers never enter
the index or the prompt.

## Verified-answer cache (H3)

The cache stores answers that passed the citation checker, never refusals, and
reuses one only for the same question:

1. **Frozen experiment:** documents and embeddings must not change while a cache
   is in use. If the corpus, index, embedding, generator, prompt or checker
   changes, startup throws out the whole cache (renamed to `*.stale-<id>`) and
   rebuilds it. Use a separate `--cache` file for each index you compare.
   Retriever, top-k, prompt budget and expander only separate entries: arms can
   share a file without sharing answers or discarding each other.
2. **Shortlist:** question cosine similarity of at least 0.80.
3. **Key-term veto (optional, off by default):** with `--key-term-veto`, numbers,
   `snake_case` identifiers and regions must match exactly. When it is off, the
   check is still recorded (`veto_would_block` in each trace's `cache` field).
4. **Sources assertion:** cited chunks and quotes are intact.
5. **Judge:** the most similar remaining entry goes to an LLM that sees only the
   two questions. Only an exact `YES` is a hit; NO, other text, errors and
   timeouts run the full pipeline.

```bash
python src/runner.py rag --retriever adaptive --cache data/processed/answer_cache.jsonl
python src/runner.py cache-eval --retriever adaptive      # compares both judges
python src/runner.py cache-eval --retriever adaptive --warm gold   # cache alone
```

`--cache-judge` selects `qwen` (default, the loaded MLX generator) or `llama`
(Ollama). `--cache-threshold` changes the shortlist threshold. `--key-term-veto`
switches the veto on for `rag`, `ask` and `cache-eval`; `cache-eval` reports
both settings either way and marks the one used as `(default)`. The cache is off
unless `--cache` is passed.

**Experimental.** Matching worked on the development variants, but a match does
not make the stored answer correct: 3 of 14 hits repeated a wrong answer
(correctness labeled by Claude; no human has verified it). The
experiment warms the cache with generated seed answers; a company deploying this
would build its own from valid generated answers. There is no access-control key,
so do not share one between users who may see different documents. See the [results](docs/ARCHITECTURE.md#second-experiment-same-question-judge).

`cache-eval` reads the completed
`data/cache_seeds/cache_seed_questions_loaders_complete.txt`, warms an empty
cache with the seeds, then judges each paraphrase and near-miss without
generating. It reports both judges, runs with checks switched off, controls and
a descriptive threshold sweep under `results/cache_eval_<retriever>_<timestamp>/`.

`--warm gold` isolates the cache from retrieval, generation and the checker: it
stores the benchmark's gold answer for every seed, in memory only, and writes to
`cache_eval_<retriever>_gold_<timestamp>/`. A matched answer is then right by
construction, so report it next to the generated run, never as end-to-end
accuracy. It records no pipeline time or tokens.

## Ask questions

```bash
python src/runner.py rag --top-k 5                      # persistent session
python src/runner.py rag --retriever adaptive           # with the gate
python src/runner.py ask "how to reset the password on gmail" --show-sources
python src/runner.py ask "how to reset the password on gmail" --json
```

In `rag`, `/sources` shows each retrieved passage, its rank, score, source path,
chunk ID and character offsets, and which passages were sent to the generator.
Output shows the route (`single_pass`, `second_pass` when the gate expanded, or
`refusal`), per-stage latency, and generation plus expansion tokens. Startup is
reported separately. Traces append to ignored `data/outputs/rag_*.jsonl`; use
`--trace PATH` to choose a file. Traces contain source text; keep them local.

Use `--max-prompt-tokens` (default 4096), `--max-new-tokens` (default 512) and
`--top-k` (default 5) to adjust the workload. Passages that do not fit the prompt
budget are omitted whole.

## Refusal and evidence checks

1. The generator sees the question and retrieved passages, and is instructed to
   answer only when they contain the required facts. It may return
   `insufficient_evidence`.
2. Code checks JSON structure, completion and source IDs. Each quote must be one
   contiguous span of whole tokens in the source it cites. It may differ only in
   whitespace, list/heading/block-quote markers, paired `*emphasis*`, backticks,
   typographic quotes and dashes, edge punctuation or ellipses, and the first
   capital. It may not cut a word or number, splice fragments with `...`, or crop
   off a negation in the three preceding words, a comparator, or half of a number
   range. Each citation records `quote_match` (`exact` or `formatting`).
   A verbatim quote can still mislead out of context (for example a dropped
   "only for admins" after it).
3. Inline markers (`[S1]`, `[S1][S2]` or `[S1, S2]`) must name only sources
   with a verified quote, and at least one marker is required.
4. Missing, malformed or invalid citations cause a refusal with an explicit reason.

**These checks do not prove that every claim follows from its quote.** A model
can cite a real but irrelevant sentence. Semantic checking (NLI) is planned work.

## Models and storage

| Role | Default | Runtime |
| --- | --- | --- |
| Embedding | `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions | Sentence Transformers / PyTorch MPS |
| Answer generation | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | MLX on Apple Silicon |
| Query expansion | `llama3.2:3b` (or the generator with `--expander mlx`) | Ollama / MLX |
| Search | BM25S + FAISS exact with reciprocal rank fusion | Local CPU worker |

`data/processed/qwen_index.jsonl` stores vectors with text and source metadata,
plus the model revision, query instruction, corpus hash and chunk settings.
Re-run `chunk` and `embed` when the corpus or chunk settings change.

## Other commands

```bash
python src/runner.py search "Medicore operational runbook turnaround timeline"
python src/runner.py interactive        # retrieval only: /doc N, /top N, /quit
python src/runner.py ablate DOC_ID      # copy corpus/index without one document
python -m unittest discover -s src/tests -v
```

`ablate` writes a separate corpus/index pair under `data/processed/ablation/`
without recomputing embeddings, for checking that answers refuse when their
source is removed. Pass it to other commands with `--corpus` and `--index`.

## Project files

| Path | Purpose |
| --- | --- |
| `src/runner.py` | CLI entry point |
| `src/adaptive_retrieval/model_store.py` | Pinned model IDs, downloads and offline loading |
| `src/adaptive_retrieval/chunking.py` | Paragraph-aware chunks with exact source offsets |
| `src/adaptive_retrieval/embedding.py` | Document and query embeddings |
| `src/adaptive_retrieval/vector_index.py` | Local vector storage and cosine search |
| `src/adaptive_retrieval/interactive.py` | Reusable search session and corpus checks |
| `src/adaptive_retrieval/hybrid_search.py` | BM25S, hybrid RRF and the disagreement gate |
| `src/adaptive_retrieval/expansion.py` | LameR prompt and Ollama/MLX expanders |
| `src/adaptive_retrieval/generation.py` | Context budget, Qwen generation, citation checks |
| `src/adaptive_retrieval/pipeline.py` | Connects retrieval, generation, checking and metrics |
| `src/adaptive_retrieval/evaluate.py` | Retrieval metrics by split and question type |
| `src/adaptive_retrieval/rag_eval.py` | Batch end-to-end RAG evaluation |
| `src/adaptive_retrieval/rag_cli.py` | Session setup, interactive loop and traces |
| `src/adaptive_retrieval/ablation.py` | Separate source-removal corpus/index pair |
| `data/subset_documents/` | Checked-in documents and question files |
| `data/processed/` | Ignored corpus, chunks, models and indexes |
| `results/` | Saved evaluation runs |
| `src/tests/` | Unit tests (models are faked) |

## Dataset

The checked-in subset has 290 documents and 100 single-document questions: 69
dev (62 answerable), 15 val and 16 test. To regenerate it from the full
[EnterpriseRAG-Bench v1.0.0 release](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/tag/v1.0.0),
extract `all_documents.zip` into `data/all_documents/`, put the separate
`questions.jsonl` there, and run `python src/runner.py build-subset`. Keep the
full dataset local; Git ignores it.
