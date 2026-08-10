# The Effects of Incremental Instruction Delivery on Language-Model Creative Writing

This repository contains the study materials, evaluation software, generation
records, and manuscript source for an empirical study of instruction-delivery
robustness in language-model creative writing.

> **Project status:** Manuscript in preparation (2026). This repository is not
> an arXiv preprint or a publication record yet.

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
the natural integration of later instructions. It does not treat model family,
parameter count, or quantization as causal variables; these are recorded as
inference configurations for reproducibility.

## Repository guide

| Path | Contents |
| --- | --- |
| [`runs/`](runs/) | Evaluation runner, fixed task data, documented procedures, merger, dashboard, and active results. |
| [`runs/results/`](runs/results/) | The active, flat generation dataset: one JSONL file per model, `all_results.jsonl`, and `index.json` provenance. |
| [`runs/test/older_outputs/`](runs/test/older_outputs/) | Isolated partial, diagnostic, and superseded outputs; not part of the active study dataset. |
| [`evaluations/`](evaluations/) | Reserved for future derived evaluation data and protocol documentation; separate from raw model generations. |
| [`paper/`](paper/) | Editable manuscript source, bibliography, figures, tables, and macros retained as a research draft; no PDF build workflow is maintained here. |

The root is deliberately reserved for project-level material: this README,
citation metadata, licensing, and the three project areas above. Operational
details belong in [`runs/README.md`](runs/README.md), rather than here.

## Reproducing a model run

The evaluator supports LM Studio and Ollama through their local
OpenAI-compatible interfaces. Start with the exact installed model identifier:

```bash
python3 runs/run_experiment.py --backend lmstudio --list-models
python3 runs/run_experiment.py --backend ollama --list-models
```

Then run one explicitly selected model:

```bash
python3 runs/run_experiment.py --backend lmstudio --models publisher/model-id
```

Use the full [evaluation guide](runs/README.md) for smoke tests,
continuations, result validation, and merging independently completed model
runs. The active result root is intentionally flat so that each model’s output
and the combined dataset can be inspected without a run-directory hierarchy.

## Active data

The current active dataset contains completed generation records for six local
model configurations. Each active model has 320 final conditions and 1,156 raw
records. `runs/results/index.json` records the exact model identity, quant,
generation settings, context segments, progress, and integrity metadata;
`all_results.jsonl` is the combined independent dataset.

Generation records are research artifacts, not interpreted findings. Analysis
tables and paper claims should be added only after the evaluation protocol and
scoring procedure have been finalized.

Future derived scores and annotations belong in [`evaluations/`](evaluations/).
They must reference generation `record_id` values and must not modify the raw
files under `runs/results/`.

## Citation

If this repository or its materials inform your work, cite the current
manuscript as follows:

```bibtex
@misc{incremental_instruction_creative_writing_2026,
  title  = {The Effects of Incremental Instruction Delivery on Language-Model Creative Writing},
  author = {Anshuman Singh and Abrar Eyasir and Haseeb Yaqoob and John Manavalan},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

The citation key is an internal BibTeX label; it is not part of the paper
title or the author list. Citation metadata is also available in
[`CITATION.cff`](CITATION.cff).

## Contact

Anshuman Singh — [anshumanr434@gmail.com](mailto:anshumanr434@gmail.com)

## Manuscript source

[`paper/`](paper/) retains the editable manuscript source as part of the study
record. GitHub Actions compiles `paper/main.tex` and exposes `paper/main.pdf`
as a downloadable workflow artifact; generated PDFs and local build scripts
are not committed to the repository.
