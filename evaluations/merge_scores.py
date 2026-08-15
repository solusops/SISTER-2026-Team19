"""Merge the four judged-score sources into one canonical evaluations/scores.jsonl.

Sources:
    1. evaluations/scores_auto_nonoverlap.jsonl   -- legacy ad hoc automated judging
    2. evaluations/native_judge_scores/*.jsonl    -- native Codex judging (gpt-5.6-terra)
    3. evaluations/native_judge_long_scores/*.jsonl -- long-output judging (gpt-5.6-sol)
    4. evaluations/native_judge_repair_scores/*.jsonl -- one-record repairs for defective
       native/long rows (8 native + 4 long), which supersede those record_ids

Each row already carries its own "source" tag from the run that produced it -- this
script does not invent new ones. It only: excludes repaired record_ids from the
native/long pools, derives story_id where a source omits it (the legacy auto rows),
validates every row's constraint-ID coverage against evaluations/constraints.jsonl,
and writes one merged, deterministically-ordered evaluations/scores.jsonl.

Usage:
    python evaluations/merge_scores.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "evaluations"

CANONICAL_QUALITY_KEYS = {
    "craft",
    "structure_coherence",
    "originality",
    "genre_effectiveness",
    "characterization",
    "characterization_na",
}
ADHERENCE_SCALE = {0, 0.5, 1}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_glob(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(Path(p) for p in glob.glob(pattern)):
        rows.extend(read_jsonl(path))
    return rows


def story_id_from_constraint_id(constraint_id: str) -> str:
    # constraint_id = story_id + "_c" + N ; story_id itself may contain underscores
    # (domain_ItemIdZeroPadded3), so strip only the trailing "_c<digits>".
    match = re.match(r"^(.*)_c\d+$", constraint_id)
    if not match:
        raise ValueError(f"Cannot derive story_id from constraint_id {constraint_id!r}")
    return match.group(1)


def load_constraints(path: Path) -> dict[str, set[str]]:
    by_story = {}
    for row in read_jsonl(path):
        by_story[row["story_id"]] = {c["constraint_id"] for c in row["constraints"]}
    return by_story


def load_final_record_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_final"):
                ids.add(row["record_id"])
    return ids


def load_sources(eval_dir: Path = EVAL_DIR) -> list[dict]:
    repair_rows = read_jsonl_glob(str(eval_dir / "native_judge_repair_scores" / "*.jsonl"))
    repair_ids = {r["record_id"] for r in repair_rows}

    auto_rows = read_jsonl(eval_dir / "scores_auto_nonoverlap.jsonl")
    for row in auto_rows:
        if "story_id" not in row:
            row["story_id"] = story_id_from_constraint_id(row["adherence"][0]["constraint_id"])

    native_rows = [
        r for r in read_jsonl_glob(str(eval_dir / "native_judge_scores" / "*.jsonl"))
        if r["record_id"] not in repair_ids
    ]
    long_rows = [
        r for r in read_jsonl_glob(str(eval_dir / "native_judge_long_scores" / "*.jsonl"))
        if r["record_id"] not in repair_ids
    ]

    all_rows = auto_rows + native_rows + long_rows + repair_rows

    counts = {
        "auto_nonoverlap": len(auto_rows),
        "native": len(native_rows),
        "long": len(long_rows),
        "repair": len(repair_rows),
    }
    return all_rows, counts


def validate(rows: list[dict], constraints_by_story: dict[str, set[str]], final_ids: set[str]) -> None:
    errors = []
    seen_record_ids: set[str] = set()

    for row in rows:
        record_id = row.get("record_id")
        story_id = row.get("story_id")

        if record_id in seen_record_ids:
            errors.append(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        if record_id not in final_ids:
            errors.append(f"{record_id}: not a final generation record in all_results.jsonl")

        expected = constraints_by_story.get(story_id)
        if expected is None:
            errors.append(f"{record_id}: unknown story_id {story_id!r}")
        else:
            got = {a["constraint_id"] for a in row["adherence"]}
            if got != expected:
                missing = expected - got
                extra = got - expected
                errors.append(
                    f"{record_id}: constraint-ID mismatch for {story_id} "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )

        for a in row["adherence"]:
            if a["score"] not in ADHERENCE_SCALE:
                errors.append(f"{record_id}: adherence score {a['score']!r} out of scale for {a['constraint_id']}")

        quality = row.get("quality", {})
        if set(quality.keys()) != CANONICAL_QUALITY_KEYS:
            errors.append(f"{record_id}: quality keys {sorted(quality.keys())} != canonical set")
        else:
            na = quality["characterization_na"]
            char = quality["characterization"]
            if na and char is not None:
                errors.append(f"{record_id}: characterization_na=True but characterization={char!r}")
            if not na and char is None:
                errors.append(f"{record_id}: characterization_na=False but characterization is null")
            for key in ("craft", "structure_coherence", "originality", "genre_effectiveness"):
                if not isinstance(quality[key], int) or not (1 <= quality[key] <= 5):
                    errors.append(f"{record_id}: quality.{key} = {quality[key]!r} not an int in 1..5")

        if not row.get("source"):
            errors.append(f"{record_id}: missing source tag")

    if errors:
        raise SystemExit("merge_scores validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors[:50])
                          + (f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""))


def main() -> None:
    constraints_by_story = load_constraints(EVAL_DIR / "constraints.jsonl")
    final_ids = load_final_record_ids(REPO_ROOT / "runs" / "results" / "all_results.jsonl")

    rows, counts = load_sources()
    print("Loaded:", counts, "total:", len(rows))

    validate(rows, constraints_by_story, final_ids)

    rows.sort(key=lambda r: r["record_id"])

    out_path = EVAL_DIR / "scores.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")

    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    print("By source:", by_source)


if __name__ == "__main__":
    main()
