"""Validate and persist blinded creative-writing judge scores.

This module consumes blinded input batches and structured judge responses under
``evaluations/``. It writes only derived evaluation artifacts and never edits
the immutable generation evidence under ``runs/results/``.
"""

import hashlib
from numbers import Real


ADHERENCE_SCORES = {0, 0.5, 1}
QUALITY_DIMENSIONS = (
    "craft",
    "structure_coherence",
    "originality",
    "genre_effectiveness",
)


def evaluation_id(record_id, judge_prompt_version):
    """Return the reproducible identifier for one record and prompt version."""
    value = f"{record_id}{judge_prompt_version}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def validate_response(batch, response, judge_prompt_version):
    """Validate a judge response against one blinded batch and derive score rows."""
    expected_by_record_id = _validate_batch(batch)
    evaluations = _required_list(response, "evaluations", "response")
    returned_by_record_id = _index_unique(evaluations, "record_id", "evaluation")
    _require_exact_ids(
        expected_by_record_id,
        returned_by_record_id,
        "response record",
    )

    rows = []
    for source_record in batch:
        record_id = source_record["record_id"]
        evaluation = returned_by_record_id[record_id]
        adherence = _validate_adherence(source_record, evaluation)
        quality = _validate_quality(evaluation.get("quality"))
        rows.append(
            {
                "record_id": record_id,
                "story_id": source_record["story_id"],
                "evaluation_id": evaluation_id(record_id, judge_prompt_version),
                "adherence": adherence,
                "I_i": sum(item["score"] for item in adherence) / len(adherence),
                "quality": quality,
            }
        )
    return rows


def _validate_batch(batch):
    if not isinstance(batch, list) or not batch:
        raise ValueError("batch must be a non-empty list")
    records = _index_unique(batch, "record_id", "batch record")
    for record in batch:
        if not isinstance(record.get("story_id"), str) or not record["story_id"]:
            raise ValueError("batch record must include a non-empty story_id")
        constraints = _required_list(record, "constraints", "batch record")
        if not constraints:
            raise ValueError("batch record must include at least one constraint")
        _index_unique(constraints, "constraint_id", "batch constraint")
    return records


def _validate_adherence(source_record, evaluation):
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    returned = _required_list(evaluation, "adherence", "evaluation")
    returned_by_id = _index_unique(returned, "constraint_id", "adherence constraint")
    expected_constraints = source_record["constraints"]
    expected_by_id = _index_unique(expected_constraints, "constraint_id", "batch constraint")
    _require_exact_ids(expected_by_id, returned_by_id, "constraint")

    normalized = []
    for constraint in expected_constraints:
        result = returned_by_id[constraint["constraint_id"]]
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, Real) or score not in ADHERENCE_SCORES:
            raise ValueError(f"constraint {constraint['constraint_id']} has an invalid score")
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"constraint {constraint['constraint_id']} requires a non-empty reason")
        normalized.append(
            {
                "constraint_id": constraint["constraint_id"],
                "score": score,
                "reason": reason.strip(),
            }
        )
    return normalized


def _validate_quality(quality):
    if not isinstance(quality, dict):
        raise ValueError("quality must be an object")
    normalized = {}
    for dimension in QUALITY_DIMENSIONS:
        value = quality.get(dimension)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"quality {dimension} must be an integer from 1 to 5")
        normalized[dimension] = value

    characterization_na = quality.get("characterization_na")
    characterization = quality.get("characterization")
    if not isinstance(characterization_na, bool):
        raise ValueError("characterization_na must be a boolean")
    if characterization_na:
        if characterization is not None:
            raise ValueError("characterization must be null when characterization_na is true")
    elif isinstance(characterization, bool) or not isinstance(characterization, int) or not 1 <= characterization <= 5:
        raise ValueError("characterization must be an integer from 1 to 5 when characterization_na is false")
    normalized["characterization"] = characterization
    normalized["characterization_na"] = characterization_na
    return normalized


def _required_list(document, field, label):
    if not isinstance(document, dict) or not isinstance(document.get(field), list):
        raise ValueError(f"{label} must include a {field} list")
    return document[field]


def _index_unique(records, key, label):
    indexed = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"{label} must include a non-empty {key}")
        if record[key] in indexed:
            raise ValueError(f"duplicate {label} {record[key]}")
        indexed[record[key]] = record
    return indexed


def _require_exact_ids(expected, returned, label):
    missing = sorted(set(expected) - set(returned))
    extra = sorted(set(returned) - set(expected))
    if missing or extra:
        raise ValueError(f"{label} IDs do not match batch (missing={missing}, extra={extra})")
