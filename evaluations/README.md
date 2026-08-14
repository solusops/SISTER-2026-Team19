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

## Not yet built

Judge harness (constraint adherence + quality scoring), pairwise blind
evaluation, and the statistics/analysis layer. See project discussion for the
proposed design.
