# Cost-Aware Adaptive Retrieval for Enterprise Support

Team 13's midterm demo is a working **local RAG pipeline**: retrieve enterprise
passages, generate an answer with citations or refuse, and display measured
latency and token usage. The cost-aware contribution at this stage is
**instrumentation**. Adaptive routing and a verified-answer cache remain future
work; no cost savings or answer-accuracy benchmark is claimed.

See [demoplan.md](demoplan.md) for the implementation checklist and presentation
rehearsal, and [the architecture](docs/ARCHITECTURE.md) for how the files connect.

## Run the midterm demo

The full answer demo targets **Apple Silicon macOS** (MLX generation). It was
installed in `CS544-RAG` with Python 3.13.16 on an M4 Pro with 48 GB memory.
Run these commands from the repository root:

```bash
conda activate CS544-RAG

# One-time setup: install packages, then download the two pinned public models.
python -m pip install -r requirements.txt
python src/runner.py download-models

# Build the local index from the checked-in subset; no full dataset needed.
python src/runner.py prepare-subset
python src/runner.py check-subset
python src/runner.py chunk
python src/runner.py embed

# Start before presenting; startup loads and warms both models once.
python src/runner.py rag --top-k 5
```

The setup and index are already prepared on this machine. For a presentation,
only activate the environment and run `rag`. Models use about 3.5 GB of disk.
Normal inference reads cached model files; it does not download or call a hosted
inference API. The public models require no Hugging Face access token. If an
account is needed for a later download, run `hf auth login` in your own terminal
and use its hidden token prompt; do not put a token in code or chat.

At the prompt, try:

```text
Question> For a customer upgrading a private deployment in their own cloud environment, what restore time did support say is typical if the upgrade fails and a rollback is needed (assuming snapshots were taken)?
Question> /sources
Question> how to reset the password on gmail
Question> /sources
Question> /quit
```

`/sources` shows each full retrieved passage, rank, cosine score, source path,
chunk ID, and original character offsets. It also shows which passages were
actually sent to the generator. `--show-sources` prints them after every answer.
Cosine scores measure similarity; they are **not confidence probabilities**.

For one question or a machine-readable result:

```bash
python src/runner.py ask "how to reset the password on gmail" --show-sources
python src/runner.py ask "how to reset the password on gmail" --json

# Optional explicit offline rehearsal, after downloading and indexing:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/runner.py rag
```

Use `--max-prompt-tokens` (default 4096), `--max-new-tokens` (default 512), and
`--top-k` (default 5) to adjust the workload. Passages that do not fit the prompt
budget are omitted whole. If generation reaches its output limit, the response
is rejected as incomplete. The CLI's `--help` lists all commands.

## Models and storage

| Role | Midterm default | Runtime |
| --- | --- | --- |
| Embedding | `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions | Sentence Transformers / PyTorch MPS |
| Answer generation | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | MLX on Apple Silicon |
| Search | Exact cosine ranking of local vectors | Python |

Model revisions are pinned in `model_store.py`. Both documents and questions
use the same embedding model; only questions receive its retrieval instruction.
There is one default embedding model and one generator. Compatibility with old
MiniLM indexes remains for historical runs, but the demo does not compare models.
A larger vector dimension alone does not establish better retrieval quality.

`data/processed/qwen_index.jsonl` stores vectors together with text and source
metadata. A separate vector database is unnecessary for this small corpus.
The index records the model revision, query instruction and corpus hash to
prevent accidentally mixing incompatible artifacts. Re-run `chunk` and `embed`
when the corpus or embedding settings change.

## Refusal and evidence checks

1. The generator sees the question and retrieved passages, and is instructed
   to answer only when they contain the required facts. It may return
   `insufficient_evidence`.
2. Code checks JSON structure, completion, source IDs, matching inline citation
   markers and verbatim supporting quotes (ignoring whitespace differences).
3. Missing, malformed or invalid citations cause a refusal, with an explicit
   reason in the output and trace.

**These checks do not prove that every claim follows logically from its quote.**
A model can cite a real but irrelevant sentence. This is a preliminary refusal
mechanism, not a calibrated semantic verifier. Similarity thresholds, training,
and a verified-answer cache are not part of this increment. Manually review
answers before presenting them as correct.

## What the live metrics mean

Every query shows embedding, vector search, prompt construction, generation,
validation and total time; input/output tokens; time to first generated token;
decode throughput; retrieved/sent passage counts; and model call counts. A token
counter updates while generation runs in an interactive terminal. Startup is
reported separately and excluded from query totals. Returned answers are shown
after the citation checks finish.

Tokens include the prompt instructions and sources, plus generated JSON and
citations. First-token latency starts at generation, so it excludes retrieval.
Local execution makes zero paid inference API calls, but compute still consumes
resources. **Dollar cost and energy are not measured** (`cost_usd` is null).

Each run appends full structured traces to ignored `data/outputs/rag_*.jsonl`.
They contain the question, raw model output, answer/refusal reason, passages,
citations, timings, tokens, model revisions, corpus hash, settings and Git state.
Use `--trace PATH` to choose a file. Traces contain source text; keep them local.

## Other commands and checks

```bash
# Retrieval only: inspect passages without generating an answer.
python src/runner.py search "Medicore operational runbook turnaround timeline"
python src/runner.py interactive

# Existing synthetic keyword example (no downloaded models required).
python src/runner.py demo "reset password"

# Unit tests use fakes for the models; the demo rehearsal exercises real models.
python -m unittest discover -s src/tests -v
```

The retrieval-only `interactive` command supports `/doc N`, `/top N` and `/quit`;
`chat` is its alias. Use **`rag`** for generated answers.

The existing `evaluate` command scores document retrieval on the labeled
questions (Hit@5 and MRR), saves results under `results/`, and does not measure
answer correctness. Its historical MiniLM results do not describe this new Qwen
pipeline. No new baseline evaluation is required for the midterm demo.

The checked-in subset contains 100 questions and 290 documents. Its question
splits are intended for development, validation and held-out testing. Runtime
retrieval and generation do not read gold answers or split labels. Rehearsal
uses development questions; it is not a held-out evaluation.

## Project files

| Path | Purpose |
| --- | --- |
| `src/runner.py` | CLI entry point |
| `src/adaptive_retrieval/model_store.py` | Pinned model IDs, downloads and offline loading |
| `src/adaptive_retrieval/embedding.py` | Document and query embeddings |
| `src/adaptive_retrieval/vector_index.py` | Local vector storage and cosine search |
| `src/adaptive_retrieval/interactive.py` | Reusable search session and corpus checks |
| `src/adaptive_retrieval/generation.py` | Context budget, Qwen generation, citation checks |
| `src/adaptive_retrieval/pipeline.py` | Connects retrieval, generation, checking and metrics |
| `src/adaptive_retrieval/rag_cli.py` | Presentation output, interactive loop and traces |
| `data/subset_documents/` | Checked-in documents and question files |
| `data/processed/` | Ignored corpus, chunks, models and indexes |
| `data/outputs/` | Ignored demo traces |
| `src/tests/` | Unit tests |
| `docs/PLAN.md` | Broader research plan; includes future work |

## Optional full dataset

The demo uses the checked-in subset. To regenerate it from the full
[EnterpriseRAG-Bench v1.0.0 release](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/tag/v1.0.0),
extract `all_documents.zip` into `data/all_documents/` and put the separate
`questions.jsonl` there. The directory should directly contain `questions.jsonl`
and source folders such as `confluence/`, `gmail/` and `slack/`. Then run
`python src/runner.py build-subset`. Keep the full dataset local; Git ignores it.
