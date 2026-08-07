import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_experiment as runner


ITEM = {"domain": "test", "item_id": 1, "title": "One", "full_instruction": "Write one.", "shards": ["Write one."]}
ENTRY = {"id": "model-a", "context_length": 4096, "reasoning_effort": None}
ARGS = SimpleNamespace(
    temperature=0.8,
    seed=123,
    base_url="http://localhost:1234/v1",
    management_url="http://localhost:1234/api/v1",
)


class RawResultStoreTests(unittest.TestCase):
    def make_run(self):
        return runner.make_run(ENTRY, None, "dataset-hash", "runner-hash", ARGS)

    def record(self, run, record_id, sequence, condition="full"):
        return {
            "schema_version": 1,
            "record_id": record_id,
            "sequence": sequence,
            "created_at": "2026-08-07T00:00:00+00:00",
            "item": {"domain": "test", "item_id": 1, "title": "One"},
            "condition": condition,
            "turn": 1,
            "is_final": True,
            "attempt_id": "attempt",
            "text": "raw response",
            "metrics": {"elapsed_seconds": 1.0},
        }

    def test_dual_write_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run()
            store = runner.RunStore(temporary, run["run_id"])
            store.create(run)
            first = self.record(run, "first", 1)
            store.append(first)
            self.assertEqual(runner.read_output_document(store.output_path)["outputs"], [first])
            self.assertEqual(store._aggregate_group()["outputs"], [first])

            second = self.record(run, "second", 2, "sharded")
            local = runner.read_output_document(store.output_path)
            local["outputs"].append(second)
            runner.atomic_write_json(store.output_path, local)
            repaired = runner.RunStore(temporary, run["run_id"])
            repaired.reconcile()
            aggregate = repaired._aggregate_group()["outputs"]
            self.assertEqual([record["record_id"] for record in aggregate], ["first", "second"])
            self.assertEqual(repaired.done_keys(), {("test", 1, "full"), ("test", 1, "sharded")})

    def test_resume_selects_only_the_exact_completed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = runner.RunIndex(Path(temporary) / "index.json")
            draft = self.make_run()
            run, store, done, completed = runner.select_run(index, temporary, draft, [ITEM], False)
            self.assertFalse(completed)
            self.assertEqual(done, set())
            store.append(self.record(run, "full", 1, "full"))
            store.append(self.record(run, "sharded", 2, "sharded"))
            done = store.done_keys()
            runner.sync_proof(index, store, run["run_id"], [ITEM], done, "completed")

            resumed, resumed_store, resumed_done, completed = runner.select_run(
                index, temporary, self.make_run(), [ITEM], True
            )
            self.assertTrue(completed)
            self.assertEqual(resumed["run_id"], run["run_id"])
            self.assertEqual(resumed_store.proof()["record_count"], 2)
            self.assertEqual(resumed_done, done)


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
        self.loaded = []
        self.last_turn_metadata = None

    def get_model_record(self, model):
        return {"key": model, "quantization": {}, "max_context_length": 4096}

    def list_loaded_instances(self):
        return list(self.loaded)

    def load_model(self, model, context_length):
        self.loaded = [{"model": model, "instance_id": "instance-1"}]
        return "instance-1"

    def unload_model(self, instance_id):
        self.loaded = []

    def chat(self, model, messages, options):
        self.calls += 1
        if self.fail:
            raise RuntimeError("planned chat failure")
        self.last_turn_metadata = {"usage": {"total_tokens": 5}, "finish_reason": "stop"}
        return f"response-{self.calls}"


class RunnerIntegrationTests(unittest.TestCase):
    def invoke(self, client, output, *arguments):
        argv = ["run_experiment.py", "--models", "model-a", "--limit", "1", "--output", str(output), *arguments]
        with patch.object(runner, "LMStudioClient", return_value=client), patch("sys.argv", argv):
            runner.main()

    def test_runner_dual_writes_every_response_and_skips_exact_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            client = FakeClient()
            self.invoke(client, output)
            index = json_load(output / "index.json")
            run = index["runs"][0]
            run_records = runner.read_output_document(output / run["run_id"] / "outputs.json")["outputs"]
            with (output / "all.json").open(encoding="utf-8") as source:
                aggregate_run = json.load(source)["runs"][0]
            all_records = aggregate_run["outputs"]
            self.assertEqual(run["status"], "completed")
            self.assertEqual(len(run_records), 1 + len(runner.load_items(1)[0]["shards"]))
            self.assertEqual(run_records, all_records)
            self.assertEqual(aggregate_run["model"]["id"], "model-a")
            self.assertNotIn("model", run_records[0])
            calls = client.calls
            self.invoke(client, output, "--resume")
            self.assertEqual(client.calls, calls)
            self.assertEqual(len(json_load(output / "index.json")["runs"]), 1)

    def test_runner_returns_nonzero_after_a_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with self.assertRaises(SystemExit) as exit_code:
                self.invoke(FakeClient(fail=True), output)
            self.assertEqual(exit_code.exception.code, 1)
            self.assertEqual(json_load(output / "index.json")["runs"][0]["status"], "failed")


def json_load(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


if __name__ == "__main__":
    unittest.main()
