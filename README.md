# The Effects of Incremental Instruction Delivery on Language-Model Creative Writing

Does splitting a story's instructions across a conversation, instead of
giving them all at once, change what a language model writes?

## Research question

When a language model receives the same final story specification, does its
output differ when the instructions are presented incrementally across a
conversation rather than all at once in a single prompt?

The study compares two semantically matched conditions:

- **Full instruction:** the complete story specification is supplied before
  generation.
- **Incremental instruction:** the same specification is revealed across
  multiple conversational turns.

The analysis focuses on constraint retention, contradiction, coherence, and
the natural integration of later instructions. Model family, parameter
count, and quantization are recorded as inference configurations, not
treated as causal variables.

## What the scripts do

This legacy SISTER repository contains runnable generation, evaluation,
validation, and analysis tooling:

- `runs/run_experiment.py` runs the FULL and incremental-instruction
  generation protocol against a local LM Studio or Ollama server.
- `runs/merge_results.py` validates and merges separately produced result
  directories.
- `evaluations/prepare_native_judge_batches.mjs` and the accompanying worker
  instructions define blinded automated-evaluation inputs.
- `evaluations/merge_scores.py` validates and combines score exports.
- `evaluations/human_validation/` contains the two pairwise-evaluator
  protocols, and `analysis/analyze_human_model_agreement.py` analyzes their
  agreement with a supplied human-judgment export.

## Generation

Generation runs against a local model server (LM Studio or Ollama, via
their OpenAI-compatible endpoints) using only the Python standard library
— no dependencies to install for this step.

```bash
# list installed models on your local backend
python3 runs/run_experiment.py --backend lmstudio --list-models
python3 runs/run_experiment.py --backend ollama --list-models

# run one explicitly selected model
python3 runs/run_experiment.py --backend lmstudio --models publisher/model-id
```

## Evaluation methodology

The evaluation tooling defines a per-constraint 0 / 0.5 / 1 adherence rubric,
five anchored creative-quality dimensions, and a blinding contract. Its key
components are:

- `evaluations/judge_config.json` — the rubric and blinding contract.
- `evaluations/prepare_native_judge_batches.mjs` — builds blinded evaluation
  inputs.
- `evaluations/native_judge_worker_instructions.md` (+ `_long_`/`_repair_`
  variants) — the exact instructions given to the judge for standard,
  long-output, and repair batches respectively.
- `evaluations/merge_scores.py` — merges and validates the raw judging
  passes into the canonical `model_evaluations` scores.

## Pairwise-validation scripts

`evaluations/human_validation/validate_pairwise_validation.mjs` supports a
standard pairwise preference evaluation. Its companion,
`evaluate_evidence_first.mjs`, supports an evidence-first comparison that
audits constraints before recording a preference. Both support original and
reversed presentation order for position-bias checks.

## Agreement analysis

`analysis/analyze_human_model_agreement.py` computes exact and directional
agreement plus weighted Cohen's kappa between supplied human and evaluator
judgments.

## Acknowledgements

This work was undertaken in the Artificial Intelligence/Machine Learning
track of Synthica's SISTER (Summer Institute of Science, Technology, and
Engineering Research) program, held from June 29 to August 2, 2026. It placed
among the top six projects in that track. Anshuman Singh led the research and
manuscript development. Special thanks to Abrar Eyasir for PI-like early
project guidance, including coordination of the related-work effort and group
meetings. Haseeb Yaqoob and John Manavalan were members of the research group;
their participation in the related-work effort and research meetings is
gratefully acknowledged.

## Credits

- **Research and paper lead:** Anshuman Singh
- **Human annotation:** Abrar Eyasir, Haseeb Yaqoob, and John Manavalan

## Rights and access

This repository is not offered under an open-source or Creative Commons
license. The retained scripts and documentation are subject to the
[SISTER Research Software Notice](LICENSE).
