# Evaluations

The judging pipeline's code: how generations were judged, and how the raw
judge output was merged into one canonical score file. This is the code
only — the data it reads and writes (constraints, generation records, judge
batches/scores, human-validation records) is published on the paired
[Hugging Face dataset](https://huggingface.co/datasets/solusops/incremental-instruction-creative-writing),
not tracked here. Fetch what a given script needs from there before running
it locally.

## Contents

- `judge_config.json` — the canonical rubric: adherence scale (0/0.5/1) with
  definitions, the five anchored 1–5 creative-quality dimensions, and the
  blinding contract (a judge call never sees model identity, condition, or
  other provenance).
- `prepare_native_judge_batches.mjs` — builds the blinded input batches sent
  to the judge (`{record_id, story_id, full_instruction, constraints[],
  text}`, joined from the `generations` and `docs/constraints.jsonl`
  configs on the HF dataset by `story_id`).
- `native_judge_worker_instructions.md`, `native_judge_long_worker_instructions.md`,
  `native_judge_repair_worker_instructions.md` — the exact instructions
  given to each judge run (standard batches, long-output batches, and
  one-record repair batches respectively). This is the judge prompt itself.
- `merge_scores.py` — combines the four raw judging passes (legacy
  automated, native, native long-output, native repairs) into one
  canonical, validated score file, one row per generation record. See the
  script's own docstring/comments for the exact merge and validation
  contract.
- `human_validation/evaluate_evidence_first.mjs` — runs the independent
  evidence-first pairwise evaluator (audits every constraint
  satisfied/partial/violated before a final preference) over the fixed
  30-case sample.
- `human_validation/validate_pairwise_validation.mjs` — validates a
  pairwise-evaluator run's structural integrity (SHA-256 freeze receipts,
  primary/reversed orientation consistency).

See [`analysis/`](../analysis/) for the human-vs-model agreement statistics
computed from this pipeline's output.
