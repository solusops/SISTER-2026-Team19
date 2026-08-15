"""Compute human-vs-model pairwise agreement statistics over the completed
human annotation set, and the disagreement table.

Reproduces (and, with more annotations available, extends) an earlier ad
hoc analysis computed for a partial N=11 export -- see
evaluations/human_validation/human_model_agreement_summary.json /
human_model_disagreements.md for the current output.

Ordinal encoding (5-level, matches the file's own documented "weighting"):
    A clearly better  -> -2
    A slightly better -> -1
    Tie                ->  0
    B slightly better ->  1
    B clearly better   ->  2

Only PRIMARY-orientation model judgments are compared against human
judgments (not reversed, not an average of the two) -- this matches the
original file's methodology exactly (verified by reproducing its N=11
figures before trusting this script for N=30).

Reads its input from evaluations/human_validation/ (not tracked in this
repo -- fetch judge_comparisons/human_eval_cases/human_eval from the paired
HF dataset first; see the main README).

Usage (run from the repo root):
    python analysis/analyze_human_model_agreement.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = REPO_ROOT / "evaluations" / "human_validation"

ORDINAL_LABELS = {
    -2: "A clearly better",
    -1: "A slightly better",
    0: "Tie",
    1: "B slightly better",
    2: "B clearly better",
}
LABEL_TO_ORDINAL = {v: k for k, v in ORDINAL_LABELS.items()}

ANNOTATION_CODE_TO_ORDINAL = {
    "a_clear": -2,
    "a_slight": -1,
    "tie": 0,
    "b_slight": 1,
    "b_clear": 2,
}

CATEGORIES = [-2, -1, 0, 1, 2]  # index i in 0..4 maps to CATEGORIES[i]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collapse(ordinal: int) -> int:
    """5-level -> 3-level: favor A (-1), tie (0), favor B (1)."""
    if ordinal < 0:
        return -1
    if ordinal > 0:
        return 1
    return 0


def cohens_kappa(pairs: list[tuple[int, int]], categories: list[int], weight_fn) -> float:
    """Generic weighted Cohen's kappa. weight_fn(cat_i, cat_j) -> agreement
    weight in [0, 1], 1 on the diagonal. Pass `lambda a, b: 1.0 if a == b
    else 0.0` for the unweighted case."""
    n = len(pairs)
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    observed = [[0] * k for _ in range(k)]
    for human, model in pairs:
        observed[idx[human]][idx[model]] += 1
    row_marginal = [sum(observed[i]) for i in range(k)]
    col_marginal = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    p_o = 0.0
    p_e = 0.0
    for i in range(k):
        for j in range(k):
            w = weight_fn(categories[i], categories[j])
            p_o += w * observed[i][j] / n
            p_e += w * (row_marginal[i] / n) * (col_marginal[j] / n)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def quadratic_weight(a: int, b: int) -> float:
    return 1 - ((a - b) / 4) ** 2


def unweighted_weight(a: int, b: int) -> float:
    return 1.0 if a == b else 0.0


def compute_metrics(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    exact = sum(1 for h, m in pairs if h == m) / n
    directional = sum(1 for h, m in pairs if collapse(h) == collapse(m)) / n
    weighted_kappa = cohens_kappa(pairs, CATEGORIES, quadratic_weight)
    collapsed_pairs = [(collapse(h), collapse(m)) for h, m in pairs]
    collapsed_kappa = cohens_kappa(collapsed_pairs, [-1, 0, 1], unweighted_weight)
    mean_abs_disagreement = sum(abs(h - m) for h, m in pairs) / n
    # "Severe" = a two-step-or-more gap on the 5-level ordinal scale
    # (e.g. human "clearly A" vs. model "tie" or worse). NOTE: every other
    # metric in this function was validated to reproduce the original N=11
    # human_model_agreement_summary.json bit-for-bit; this one reproduces
    # the constraint_following figure (2 cases) exactly but not the
    # creative_writing_quality figure (2 vs. the stored 3) -- several
    # alternative rules were tried (opposite-sign-only, collapsed-bucket,
    # cross-checking the reversed-order model judgment) and none reproduced
    # both dimensions at once, so the original selection rule for that one
    # field isn't recoverable from the stored numbers alone. Using this
    # explicit, simple definition going forward rather than a rule tuned to
    # match a single stale figure.
    severe = [1 for h, m in pairs if abs(h - m) >= 3]
    return {
        "n": n,
        "exact_five_level_agreement": exact,
        "directional_agreement": directional,
        "weighted_cohens_kappa_quadratic": weighted_kappa,
        "collapsed_cohens_kappa": collapsed_kappa,
        "mean_absolute_disagreement": mean_abs_disagreement,
        "severe_disagreement_rate": len(severe) / n,
        "severe_disagreement_cases": len(severe),
    }


def load_human_ordinals() -> dict:
    """case_id -> (constraint_ordinal, creative_ordinal, raw_row)"""
    rows = read_jsonl(BASE / "annotations.jsonl")
    out = {}
    for row in rows:
        out[row["case_id"]] = (
            ANNOTATION_CODE_TO_ORDINAL[row["constraint_following"]],
            ANNOTATION_CODE_TO_ORDINAL[row["creative_quality"]],
            row,
        )
    return out


def load_model_ordinals() -> dict:
    """case_id -> (primary_constraint_ordinal, primary_creative_ordinal)"""
    rows = read_jsonl(BASE / "ordinal_model_judgments.jsonl")
    return {r["case_id"]: (r["primary_constraint_ordinal"], r["primary_creative_ordinal"]) for r in rows}


def load_model_primary_reasons() -> dict:
    """case_id -> {constraint_reason, creative_reason}"""
    rows = read_jsonl(BASE / "model_pairwise_primary.jsonl")
    return {r["case_id"]: r for r in rows}


def main() -> None:
    human = load_human_ordinals()
    model = load_model_ordinals()
    model_reasons = load_model_primary_reasons()

    case_ids = sorted(human.keys(), key=lambda c: int(c.split("-")[1]))
    missing = [c for c in case_ids if c not in model]
    if missing:
        raise SystemExit(f"No model judgment for human-annotated cases: {missing}")

    constraint_pairs = [(human[c][0], model[c][0]) for c in case_ids]
    creative_pairs = [(human[c][1], model[c][1]) for c in case_ids]

    summary = {
        "human_cases_completed": len(case_ids),
        "human_case_ids": case_ids,
        "ordinal_encoding": {
            "A clearly better": -2,
            "A slightly better": -1,
            "Tie": 0,
            "B slightly better": 1,
            "B clearly better": 2,
        },
        "weighting": "quadratic ordinal weights: 1 - ((i - j) / 4)^2 on the ordered five-level scale",
        "constraint_following": compute_metrics(constraint_pairs),
        "creative_writing_quality": compute_metrics(creative_pairs),
    }

    summary_path = BASE / "human_model_agreement_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {summary_path}")

    # Disagreement table: only cases where human and model differ on either dimension
    lines = [
        "# Human–Model Pairwise Disagreements",
        "",
        "Only completed human cases are included. Categories are deliberately not inferred automatically; use the notes as evidence during diagnosis.",
        "",
        "| case_id | human constraint | model constraint | human creative | model creative | human notes | model constraint reason | model creative reason | diagnostic category |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    disagreement_count = 0
    for c in case_ids:
        h_constraint_ord, h_creative_ord, raw = human[c]
        m_constraint_ord, m_creative_ord = model[c]
        if h_constraint_ord == m_constraint_ord and h_creative_ord == m_creative_ord:
            continue
        disagreement_count += 1
        reasons = model_reasons.get(c, {})
        notes = raw.get("notes", "").replace("\n", " ")
        lines.append(
            f"| {c} | {ORDINAL_LABELS[h_constraint_ord]} | {ORDINAL_LABELS[m_constraint_ord]} | "
            f"{ORDINAL_LABELS[h_creative_ord]} | {ORDINAL_LABELS[m_creative_ord]} | {notes} | "
            f"{reasons.get('constraint_reason', '')} | {reasons.get('creative_reason', '')} | "
            "Not categorized automatically; interpret only from the supplied notes. |"
        )

    disagreements_path = BASE / "human_model_disagreements.md"
    with disagreements_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {disagreements_path} ({disagreement_count}/{len(case_ids)} cases disagree on at least one dimension)")


if __name__ == "__main__":
    main()
