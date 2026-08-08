import json
import tempfile
import unittest
from pathlib import Path

from merge_results import merge
from run_experiment import canonical_json, read_jsonl


def write_source(root, model, record_id):
    root.mkdir()
    record = {
        "schema_version": 1,
        "record_id": record_id,
        "model_id": model,
        "model_name": model,
        "parameters": "7B",
        "quant_file": "Q4",
        "sequence": 1,
        "item": {"domain": "test", "item_id": 1, "title": "Test"},
        "condition": "full",
        "turn": 1,
        "is_final": True,
        "attempt_id": "attempt",
        "text": "response",
        "metrics": {"finish_reason": "stop"},
    }
    output_name = f"results_{model}.jsonl"
    for name in (output_name, "all_results.jsonl"):
        (root / name).write_text(canonical_json(record) + "\n", encoding="utf-8")
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": f"run-{model}",
                        "status": "completed",
                        "model": {"id": model},
                        "proof": {"run_output_path": output_name},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class MergeResultsTests(unittest.TestCase):
    def test_merges_distinct_models_and_rebuilds_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, output = root / "first", root / "second", root / "merged"
            write_source(first, "model-a", "record-a")
            write_source(second, "model-b", "record-b")

            self.assertEqual(merge([first, second], output), (2, 2))
            self.assertEqual(
                [record["record_id"] for record in read_jsonl(output / "all_results.jsonl")],
                ["record-a", "record-b"],
            )
            with (output / "index.json").open(encoding="utf-8") as handle:
                index = json.load(handle)
            self.assertEqual(index["batch_models"], ["model-a", "model-b"])
            self.assertEqual(index["merged_from"], [{"name": "first", "run_count": 1}, {"name": "second", "run_count": 1}])
            self.assertTrue(all(run["proof"]["all_output_sha256_at_update"] for run in index["runs"]))

    def test_rejects_duplicate_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, output = root / "first", root / "second", root / "merged"
            write_source(first, "model-a", "record-a")
            write_source(second, "model-a", "record-b")

            with self.assertRaisesRegex(RuntimeError, "Model appears"):
                merge([first, second], output)


if __name__ == "__main__":
    unittest.main()
