# Midterm RAG demo

## Goal

Run one local RAG pipeline in the `CS544-RAG` Conda environment: retrieve
enterprise passages, generate a cited answer or refusal, and display the time
and tokens used. No model training or baseline comparison is required.

## Models

- Embeddings: `Qwen/Qwen3-Embedding-0.6B`, via Sentence Transformers.
  Apply a retrieval instruction to questions, not to source passages.
- Generation: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, via MLX-LM on
  Apple Silicon. Load models once per interactive session.
- Download models and build the index before presenting. Keep model files,
  indexes, and transcripts under ignored `data/processed/` or `data/outputs/`.

## Implementation checklist

- [x] Install and verify dependencies in `conda activate CS544-RAG`.
- [x] Build a Qwen embedding index with source paths and exact chunk offsets.
- [x] Add generation, context budgeting, and citation validation.
- [x] Connect retrieval and generation in `pipeline.py`.
- [x] Add `ask` and persistent `rag` CLI commands with live metrics.
- [x] Test valid answers, refusals, invalid citations, and model/index mismatch.
- [x] Rehearse a supported question and the Gmail password question offline.
- [x] Document exact launch commands and observed demo results.

## Pipeline

```text
Question
  -> Qwen query embedding
  -> top-k passages from the local index
  -> numbered source passages within the generator's context budget
  -> Qwen answer with citations OR insufficient-evidence response
  -> citation/output validation
  -> answer/refusal, source passages, and measured metrics
```

`embedding.py` handles vectors; `interactive.py` owns the search session;
`generation.py` owns the local generator; `pipeline.py` connects the steps;
`rag_cli.py` presents results; `runner.py` exposes commands.

## Refusal scope

The generator must use the supplied passages and abstain when they do not
answer the question. Code rejects malformed responses, missing citations,
unknown citation IDs, and unsupported citation quotes. Citation/quote validation
does not prove that every claim logically follows from the source. Manually
review the demonstration answers. This is a preliminary refusal mechanism,
not a calibrated verifier or a verified-answer cache.

## Metrics and presentation

Show query-embedding time, vector-search time, prompt construction time,
generation time, validation time, total query time, prompt/output tokens,
time to first token, and generation throughput. Report startup separately.
Save structured per-question traces including model/index identity.

These are cost-profiling measurements. Local execution makes no paid model API
calls, but time and tokens are not dollar costs or energy measurements.
Automatic routing, a verified-answer cache, and baseline comparisons are outside
this midterm increment.

## Rehearsal

Use a question from the development questions whose supporting passage is
present. Then ask `how to reset the password on gmail`; show that unrelated
retrieved passages should lead to refusal. Use `/sources` to inspect the actual
evidence and `/quit` to exit. Keep saved transcripts as a backup.

## Setup and launch

Commands run from the repository root. Setup has been performed in `CS544-RAG`.

```bash
conda activate CS544-RAG
python -m pip install -r requirements.txt
python src/runner.py download-models
python src/runner.py prepare-subset
python src/runner.py check-subset
python src/runner.py chunk
python src/runner.py embed
```

For the presentation, only these commands are needed:

```bash
conda activate CS544-RAG
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/runner.py rag --top-k 5
```

Wait for `Ready`, then ask questions. Use `/sources` to reveal ranked evidence.
Startup loads model weights once; every question still performs retrieval and
generation. This is model reuse, not an answer-cache hit.

## Implementation decisions from rehearsal

- The initial 120-word chunks sometimes retrieved the right document while
  excluding the needed sentence. The CLI now uses 240 words with 40 words of
  overlap, retaining exact source offsets. This is a development adjustment,
  not a demonstrated improvement in held-out retrieval quality.
- The initial generator prompt produced malformed JSON on two rehearsal
  questions. Output checks rejected it. Prompt version `midterm-v3` uses a
  minimal `{"status":"insufficient_evidence"}` refusal, a synthetic answer-format
  example, and a synthetic example distinguishing requests from commitments.
  Neither example contains a benchmark answer.
- An intermediate rehearsal misattributed a customer's requested deadline to
  a supplier despite a valid quote. The prompt was strengthened, but source
  quote checks still do not establish semantic correctness.
- Chat-template `.jinja` files must be downloaded along with weights and
  tokenizer files. Downloads now include them.
- GPU inference was run outside the Codex sandbox because the sandbox cannot
  access Metal. A normal local terminal can use the Mac GPU.
- Public downloads succeeded without a Hugging Face token.

## Checks

```bash
conda activate CS544-RAG
python -m unittest discover -s src/tests -v
python -m pip check
```

20 unit tests pass; the installed dependencies have no reported conflicts.
All 1,516 rebuilt chunks were checked against their original source character
offsets. Tests cover model reuse, context budgets, citation failure paths,
corpus changes, embedding-model mismatch, and query-instruction mismatch.
The real-model rehearsal below is separate from those model-free unit tests.

## Observed offline rehearsal

Hardware: Apple M4 Pro, 48 GB RAM. Environment: `CS544-RAG`, Python 3.13.16,
PyTorch MPS embeddings and MLX generation. Both offline environment flags were
enabled. The persistent session loaded once in **4.43 seconds**, then handled
all three questions below. `/sources` and `/quit` were also exercised.

Defaults: top-k 5, prompt budget 4096, output limit 512, greedy generation.
Index: 1,516 chunks from 290 documents, 1024 dimensions. Rebuilding it took
about 192 seconds including embedding-model startup.

| Question | Observed outcome | Query total | Generation | Input / output tokens |
| --- | --- | ---: | ---: | ---: |
| Private deployment rollback (`qst_0026`, dev) | Answer: typically tens of minutes, with snapshot and database-size caveats; two valid citations | 6.21 s | 6.05 s | 2957 / 129 |
| `how to reset the password on gmail` | Model explicitly reported insufficient evidence | 3.96 s | 3.80 s | 2681 / 9 |
| Medicore runbook turnaround (`qst_0046`, dev) | Model explicitly reported insufficient evidence in the retrieved passages | 3.64 s | 3.49 s | 2447 / 9 |

These are individual observations, not averages or a quality benchmark.
The first answer was manually checked against the retrieved text and the
development reference answer. No gold answer is passed to the runtime pipeline.
All three requests used one retrieval round and one generator call.

The Medicore reference answer is **5 business days**, but its top five passages
omit that statement. The final prompt refuses instead of repeating the
customer's requested deadline as the supplier's proposal. This illustrates a
remaining retrieval limitation: an answer can exist in the corpus and still
be absent from the context the generator sees.

Full local backup:
[`data/outputs/midterm_rehearsal_v3.jsonl`](data/outputs/midterm_rehearsal_v3.jsonl).
It contains exact passages, raw responses, citations, model revisions, settings
and measurements. Earlier diagnostic traces are retained separately.

### Presentation script

1. Start `rag` and wait for `Ready`. Explain that startup is reported separately.
2. Paste this supported question:

   ```text
   For a customer upgrading a private deployment in their own cloud environment, what restore time did support say is typical if the upgrade fails and a rollback is needed (assuming snapshots were taken)?
   ```

3. Show the answer, cited quotes and `/sources`. Explain that the source text
   provides the answer and that similarity scores are not confidence.
4. Show the stage times and tokens. In this run generation was the largest
   measured stage. These measurements identify where future routing or caching
   could help; they do not establish savings.
5. Ask `how to reset the password on gmail`, then show `/sources`. The passages
   do not contain a Gmail password-reset procedure, so the model refuses.
6. Explain the current limit: exact quotes can be checked mechanically, but
   semantic correctness and refusal reliability still need broader evaluation.
   Finish with `/quit`.

No generator-without-RAG baseline, model training, paid inference API,
verified-answer cache or adaptive cost controller was added.
