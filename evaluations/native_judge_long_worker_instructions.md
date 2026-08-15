# Isolated Long-Output Judge Worker

Read the project router only as required by project instructions, then only
the one batch file named in your dispatch. Do not read other results, scores,
provenance, model details, conditions, or batches.

Assess every supplied constraint independently: use `0` for violated/absent,
`0.5` for partial satisfaction, and `1` for clear satisfaction, with a short,
specific reason grounded in the text. Treat an explicit length or sentence
constraint as adherence: score it down if the text violates it. Do **not**
apply a separate global penalty merely for long prose; assess craft,
structure/coherence, originality, genre effectiveness, and characterization
independently on the 1–5 rubric. Closely examine whether apparent length adds
narrative value or reflects repetition, digression, restart, or incoherence.

Write one JSONL object with `record_id`, `story_id`, `adherence`, `quality`,
and `source: "codex_native_sol_high_long_output"` to the assigned file.
`quality` must contain exactly `craft`, `structure_coherence`, `originality`,
`genre_effectiveness`, `characterization`, and `characterization_na`; the
first four are integers 1–5, and characterization is either an integer 1–5
with `characterization_na: false`, or `null` with
`characterization_na: true`. Use `apply_patch`; do not delegate; return only
status and count.
