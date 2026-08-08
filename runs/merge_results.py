"""Merge completed, distinct-model evaluation directories into one result set."""

import argparse
import json
from pathlib import Path

from run_experiment import atomic_write_json, canonical_json, read_jsonl, sha256_file, sha256_value


def load_index(source):
    index_path = source / "index.json"
    try:
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing index: {index_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid index JSON: {index_path}") from exc
    if index.get("schema_version") != 1 or not isinstance(index.get("runs"), list):
        raise RuntimeError(f"Unsupported index schema: {index_path}")
    return index


def validate_source(source):
    index = load_index(source)
    all_records = read_jsonl(source / "all_results.jsonl")
    all_by_id = {record["record_id"]: record for record in all_records}
    if len(all_by_id) != len(all_records):
        raise RuntimeError(f"Duplicate record IDs in {source / 'all_results.jsonl'}")

    selected = []
    expected_by_id = {}
    for run in index["runs"]:
        model = (run.get("model") or {}).get("id")
        if run.get("status") != "completed":
            raise RuntimeError(f"{source.name}: {model or 'unknown model'} is not completed")
        output_name = Path((run.get("proof") or {}).get("run_output_path") or "").name
        if not model or not output_name.startswith("results_"):
            raise RuntimeError(f"{source.name}: completed run lacks a valid model output reference")
        records = read_jsonl(source / output_name)
        for record in records:
            if record.get("model_id") != model:
                raise RuntimeError(f"{source.name}: {output_name} contains another model's record")
            record_id = record["record_id"]
            if record_id in expected_by_id:
                raise RuntimeError(f"{source.name}: duplicate record ID {record_id}")
            expected_by_id[record_id] = record
        selected.append((model, output_name, run, records))

    if set(expected_by_id) != set(all_by_id):
        raise RuntimeError(f"{source.name}: per-model files do not match all_results.jsonl")
    for record_id, record in expected_by_id.items():
        if canonical_json(record) != canonical_json(all_by_id[record_id]):
            raise RuntimeError(f"{source.name}: record {record_id} differs from all_results.jsonl")
    return selected


def prepare_output(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def merge(sources, output):
    prepare_output(output)
    merged = []
    seen_models = set()
    seen_records = set()
    source_summaries = []

    for source in sources:
        source = Path(source)
        selected = validate_source(source)
        source_summaries.append({"name": source.name, "run_count": len(selected)})
        for model, output_name, run, records in selected:
            if model in seen_models:
                raise RuntimeError(f"Model appears in more than one source: {model}")
            record_ids = {record["record_id"] for record in records}
            if seen_records & record_ids:
                raise RuntimeError(f"Duplicate record IDs across sources for {model}")
            seen_models.add(model)
            seen_records.update(record_ids)
            merged.append((model, output_name, run, records))

    all_records = []
    merged_runs = []
    for model, output_name, run, records in merged:
        output_path = output / output_name
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record))
                handle.write("\n")
        copied_run = json.loads(json.dumps(run))
        copied_run["proof"] = {
            "run_output_path": output_name,
            "all_output_path": "all_results.jsonl",
            "record_count": len(records),
            "record_ids_sha256": sha256_value([record["record_id"] for record in records]),
            "run_output_sha256": sha256_file(output_path),
            "all_output_sha256_at_update": None,
        }
        merged_runs.append(copied_run)
        all_records.extend(records)

    all_path = output / "all_results.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(canonical_json(record))
            handle.write("\n")
    all_hash = sha256_file(all_path)
    for run in merged_runs:
        run["proof"]["all_output_sha256_at_update"] = all_hash

    atomic_write_json(
        output / "index.json",
        {
            "schema_version": 1,
            "batch_models": [model for model, _, _, _ in merged],
            "merged_from": source_summaries,
            "runs": merged_runs,
        },
    )
    return len(merged_runs), len(all_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New, empty merged result directory")
    parser.add_argument("sources", nargs="+", help="Completed collaborator result directories")
    args = parser.parse_args()
    try:
        run_count, record_count = merge(args.sources, Path(args.output))
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"Merged {run_count} models and {record_count} records into {args.output}")


if __name__ == "__main__":
    main()
