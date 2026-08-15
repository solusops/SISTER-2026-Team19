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
  the reproducible validation script. The human export is committed as
  `annotations.jsonl` and now covers all 30 cases; the committed
  `human_model_agreement_summary.json` / `human_model_disagreements.md`
  still reflect the earlier 11-case partial export and need regenerating
  against the full 30 before being trusted.
- `human_validation/evidence_first_*.jsonl` â€” a separate, fully blinded
  evidence-first primary/reversed pairwise study on the same 30 cases. Each
  record includes per-constraint status/evidence and eight grounded creative
  dimensions before its final preference. Its outputs are frozen separately
  and have not been compared with human responses.

## Not yet built

Pointwise-score aggregation and the full statistics/analysis layer. The
matched pairwise human-vs-model analysis is currently limited to the 11 cases
completed in the supplied export.
