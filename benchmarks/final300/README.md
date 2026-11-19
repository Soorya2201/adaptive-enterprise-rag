# Frozen final benchmark v2

`questions.jsonl` freezes 300 questions and split/group membership.
`manifest.json` records source hashes, document hashes, corpus policy, and the
expected question/corpus file hashes. The archive and generated corpus stay
local in ignored data paths.

Reproduce with `python -m adaptive_retrieval.final_benchmark`, then run the same
command with `--verify`. See [the experiment runbook](../../docs/FINAL_EXPERIMENT.md).
This snapshot contains frozen inputs, not final test results.
