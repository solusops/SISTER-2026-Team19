# Isolated Repair Judge Worker

Read the project router only as required by project instructions, then only
the batch file named in your dispatch. Do not read other results, scores,
provenance, model details, conditions, or batches.

Return one adherence result for **every** supplied constraint, using its exact
`constraint_id` once and only once. Assign `0`, `0.5`, or `1` and provide a
short evidence-based reason. Before writing, compare your returned IDs against
the input list in order and confirm that none is missing.

Write `quality` with exactly `craft`, `structure_coherence`, `originality`,
`genre_effectiveness`, `characterization`, and `characterization_na`. The
first four must be integers 1–5. Characterization is an integer 1–5 with
`characterization_na: false`, or null with `characterization_na: true` only
when the task has no character requirement and the text has no meaningful
identifiable characters.

Write one JSONL object with `record_id`, `story_id`, `adherence`, `quality`,
and `source: "codex_native_sol_high_repair"` to the assigned output file.
Use `apply_patch`; do not delegate; return only status and count.
