# The Effects of Incremental Instruction Delivery on Language-Model Creative Writing

This repository contains the evaluation software, generation
records, and manuscript source for an empirical study of instruction-delivery
robustness in language-model creative writing.

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

## Active data

The current active dataset contains completed generation records for six local
model configurations. Each active model has 320 final conditions and 1,156 raw
records. `runs/results/index.json` records the exact model identity, quant,
generation settings, context segments, progress, and integrity metadata;
`all_results.jsonl` is the combined independent dataset.

## Datasets on Hugging Face

This repository holds the working code and files; two of those files are
also published on the Hub as standalone, independently versioned dataset
repos so they can be cited and reused apart from this codebase:

- **[`sister-benchmark`](https://huggingface.co/datasets/SolusOps/sister-benchmark)**:
  the 160-task benchmark (`runs/benchmark_data.json`). The reusable
  artifact: load this to evaluate your own model against the same tasks.
- **[`sister-benchmark-generations`](https://huggingface.co/datasets/SolusOps/sister-benchmark-generations)**:
  our baseline models' raw outputs on that benchmark (`runs/results/`),
  one split per model. Evidence for this paper, not part of the benchmark
  definition itself.

They're separate repos, not branches or configs of one repo, because they
version independently: the benchmark grows over time (new task versions,
tagged releases), while the generations repo grows with new experiment
runs. See `runs/BENCHMARK.md`, `runs/results/README.md`, and
`runs/hf_upload/README.md` for the full rationale and the push scripts.



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
