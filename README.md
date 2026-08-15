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

An earlier, narrower two-repo split
([`sister-benchmark`](https://huggingface.co/datasets/SolusOps/sister-benchmark),
[`sister-benchmark-generations`](https://huggingface.co/datasets/SolusOps/sister-benchmark-generations))
remains published for existing citations to it.

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
originality, genre effectiveness, characterization). The full rubric and
blinding contract are published alongside the scores in the dataset repo
(`methodology/` files, and `model_evaluations` for the scores themselves).

## Evaluator and human validation

Two independent pairwise evaluators — a standard preference judge
(`judge_comparisons`) and an evidence-first evaluator that audits every
constraint before choosing a preference (`judge_audit`) — were each run in
original and reversed response order over the same fixed 30-case sample
also shown to human annotators (`human_eval_cases` / `human_eval`), to
check for position bias and evaluator-human agreement.

## Result analysis

Paired statistical analysis over the judge scores (constraint-loss vs.
creative-quality effects across conditions) is in progress; this section
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


