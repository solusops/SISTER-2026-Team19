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

## Setup

```bash
pip install -r requirements.txt   # datasets, huggingface_hub
```

Generation runs against a local model server (LM Studio or Ollama, via
their OpenAI-compatible endpoints). Reproducing the judging/analysis steps
needs no local model server — only the recorded outputs and scores below.

## Data

The benchmark, generations, judge scores, and evaluator/human-validation
outputs are published on Hugging Face as one dataset repo, six configs:

- **[`incremental-instruction-creative-writing`](https://huggingface.co/datasets/solusops/incremental-instruction-creative-writing)**
  — `benchmark`, `generations`, `pointwise_scores`, `pairwise_validation`,
  `evidence_first_validation`, `human_eval`. Load any config with
  `datasets.load_dataset(repo_id, config_name=...)`.

An earlier, narrower two-repo split
([`sister-benchmark`](https://huggingface.co/datasets/SolusOps/sister-benchmark),
[`sister-benchmark-generations`](https://huggingface.co/datasets/SolusOps/sister-benchmark-generations))
remains published for existing citations to it.

Locally, `runs/benchmark_data.json` is the benchmark and `runs/results/`
holds the generation records (`runs/results/index.json` carries run
provenance: model/quant identity, seeds, and context-length handling).
`evaluations/constraints.jsonl` holds the atomic constraints extracted from
every task, used by the judge.

## Generation

```bash
# list installed models on your local backend
python3 runs/run_experiment.py --backend lmstudio --list-models
python3 runs/run_experiment.py --backend ollama --list-models

# run one explicitly selected model against the benchmark
python3 runs/run_experiment.py --backend lmstudio --models publisher/model-id
```

## Evaluation methodology

Each final generation is judged against its task's atomic constraints on a
0 / 0.5 / 1 adherence scale (with a short reason per constraint), plus five
anchored 1–5 creative-quality dimensions (craft, structure and coherence,
originality, genre effectiveness, characterization). The full rubric and
blinding contract are in `evaluations/judge_config.json`.

`evaluations/scores.jsonl` is the canonical, validated merge of every
judging pass, one row per final generation record; build it from the raw
per-pass files with:

```bash
python3 evaluations/merge_scores.py
```

## Evaluator and human validation

Two independent pairwise evaluators — a standard preference judge and an
evidence-first evaluator that audits every constraint before choosing a
preference — were each run in original and reversed response order over a
fixed 30-case blind sample, to check for position bias.
`evaluations/human_validation/` holds that sample, both evaluators' outputs,
methodology reports, and the human-validation annotations used in the
current study over the same sample.

## Result analysis

Paired statistical analysis over `evaluations/scores.jsonl` (constraint-loss
vs. creative-quality effects across conditions) is in progress; this section
will be filled in once that lands.

## Citation

```bibtex
@misc{incremental_instruction_creative_writing_2026,
  title  = {The Effects of Incremental Instruction Delivery on Language-Model Creative Writing},
  author = {Anshuman Singh and Abrar Eyasir and Haseeb Yaqoob and John Manavalan},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```


