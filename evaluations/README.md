# Evaluations

This folder holds the study's evaluation materials and derived scores,
separate from the immutable generation evidence in `../runs/results/`.

Keep all model generations in `../runs/results/`. Any evaluation record here
must reference the relevant generation `record_id` and must never overwrite or
edit a raw generation file.

## Contents

- `constraints.jsonl` — atomic creative-writing constraints extracted from
  every one of the 160 tasks in `../runs/benchmark_data.json`. One record per
  `story_id` (`domain_ItemIdZeroPadded3`, e.g. `fantasy_001` — needed because
  `item_id` alone is only unique within a domain). See `constraint_schema.json`
  for the field definitions, taxonomy, and extraction provenance.
- `constraint_schema.json` — schema and taxonomy for `constraints.jsonl`,
  including how it was generated and its current validation status.

- `human_validation/` â€” matched 30-pair evaluator validation: sanitized
  primary and reversed case files, frozen model judgments and SHA-256
  receipts, ordinal analysis-only encoding, position-consistency results, and
  the reproducible validation script. The source human export is external to
  the repository; its 11 completed cases are represented by the committed
  agreement summary and disagreement table, while the remaining 19 cases are
  not imputed or silently dropped from the reported denominator.

## Not yet built

Pointwise-score aggregation and the full statistics/analysis layer. The
matched pairwise human-vs-model analysis is currently limited to the 11 cases
completed in the supplied export.
