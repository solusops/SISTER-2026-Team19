# Isolated Native Judge Worker

Read the project router only as required by project instructions. Then read
only the batch file named in your dispatch. Do not read any other results,
scores, provenance, model details, conditions, or batches.

For every supplied constraint, assign `0` (violated/absent), `0.5` (partial),
or `1` (clearly satisfied), with a short evidence-based reason. Independently
score `craft`, `structure_coherence`, `originality`, `genre_effectiveness`, and
`characterization` on 1–5. Use `characterization: null` plus
`characterization_na: true` only where the task has no character requirement
and the text has no meaningful identifiable characters; otherwise score it and
set `characterization_na: false`.

Create the assigned output JSONL, in input order, with exactly these fields:
`record_id`, `story_id`, `adherence` (each with `constraint_id`, `score`, and
`reason`), `quality`, and `source: "codex_native_terra_medium"`. Use
`apply_patch` to write that one output file. Do not delegate. Return only a
brief status and record count.
