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

## Data

The benchmark, generations, and every evaluation artifact are published on
Hugging Face as one dataset repo:

**[`incremental-instruction-creative-writing`](https://huggingface.co/datasets/solusops/incremental-instruction-creative-writing)**

| Config | What it is |
|---|---|
| `benchmark` | The 160 writing tasks. |
| `generations` | Model-generated responses, one split per baseline model. |
| `model_evaluations` | 1,920 per-response LLM judge scores (constraint adherence + creative-quality dimensions). |
| `judge_comparisons` | 30-pair A/B judge evaluations, primary and reversed response order. |
| `judge_audit` | An independent evidence-first robustness evaluation over the same 30 pairs, auditing every constraint before a final preference. |
| `human_eval_cases` | The 30 response pairs shown to human annotators. |
| `human_eval` | The corresponding 30 human judgments. |

Load any config with `datasets.load_dataset(repo_id, config_name=...)`.

## Generation

Generation runs against a local model server (LM Studio or Ollama, via
their OpenAI-compatible endpoints) using only the Python standard library
— no dependencies to install for this step.

```bash
# fetch the benchmark tasks this script expects at runs/benchmark_data.json
python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('solusops/incremental-instruction-creative-writing', 'benchmark')['tasks']
json.dump(list(ds), open('runs/benchmark_data.json', 'w'))
"  # needs: pip install datasets

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
originality, genre effectiveness, characterization). The exact rubric,
blinding contract, and judge instructions are in `evaluations/`:

- `evaluations/judge_config.json` — the rubric and blinding contract.
- `evaluations/prepare_native_judge_batches.mjs` — builds the blinded input
  batches from the `generations` and benchmark configs.
- `evaluations/native_judge_worker_instructions.md` (+ `_long_`/`_repair_`
  variants) — the exact instructions given to the judge for standard,
  long-output, and repair batches respectively.
- `evaluations/merge_scores.py` — merges and validates the raw judging
  passes into the canonical `model_evaluations` scores.

## Evaluator and human validation

Two independent pairwise evaluators — a standard preference judge
(`judge_comparisons`, run by `evaluations/human_validation/validate_pairwise_validation.mjs`)
and an evidence-first evaluator that audits every constraint before
choosing a preference (`judge_audit`, run by
`evaluations/human_validation/evaluate_evidence_first.mjs`) — were each run
in original and reversed response order over the same fixed 30-case sample
also shown to human annotators (`human_eval_cases` / `human_eval`), to
check for position bias and evaluator-human agreement.

## Result analysis

`analysis/analyze_human_model_agreement.py` computes exact/directional
agreement and weighted Cohen's kappa between the human judgments and the
model evaluator's primary-order judgments over the full 30-case sample.
Paired statistical analysis over the judge scores (constraint-loss vs.
creative-quality effects across conditions) is in progress; this section
will be filled in further once that lands.

## Acknowledgements

This work began in Synthica's SISTER program, where it placed among the top
six projects in the AI/ML track. Special thanks to Abrar Eyasir for early
project guidance in a PI-like role. Anshuman Singh led the research and paper
development.

## Credits

- **Research and paper lead:** Anshuman Singh
- **Human annotation:** Abrar Eyasir, Haseeb Yaqoob, and John Manavalan


