"""Run the creative-writing benchmark against an LM Studio model sequence.

Each generated response is written immediately to two raw datasets:

* ``results/<run-id>/outputs.json`` is the self-contained dataset for one
  model/configuration run.
* ``results/all.json`` is the independently usable dataset across all runs.

``results/index.json`` records the evidence needed to reproduce and audit a
run: model and inference settings, protocol hashes, progress, record IDs, and
file hashes.  Run ``--sequence`` to use the tracked ``models.json`` plan.
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


DATA_PATH = "benchmark_data.json"
PLAN_PATH = "models.json"
OUT_PATH = "results"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_TIMEOUT = 600
DEFAULT_CONTEXT_LENGTH = 65536
DEFAULT_CONTEXT_LENGTH_OVERRIDES = {
    "google/gemma-4-26b-a4b-qat": 32768,
    "mistral-7b-instruct-v0.2": 32768,
}
SYSTEM_PROMPT = (
    "You are a skilled creative writer. Follow the user's instructions "
    "for the story precisely, including any exact phrases, constraints, "
    "or stylistic requirements."
)
CONDITIONS = ("full", "sharded")
REASONING_EFFORTS = {"off", "on", "low", "medium", "high"}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path, document):
    """Atomically replace one small JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".experiment-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def read_output_document(path):
    """Read one compact raw-output document without accepting partial schemas."""
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "outputs": []}
    with path.open("r", encoding="utf-8") as source:
        try:
            document = json.load(source)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid raw output JSON in {path}") from exc
    if document.get("schema_version") != 1 or not isinstance(document.get("outputs"), list):
        raise RuntimeError(f"Invalid raw output schema in {path}")
    for record in document["outputs"]:
        if not isinstance(record, dict) or not record.get("record_id"):
            raise RuntimeError(f"Invalid raw record in {path}")
    return document


def load_items(limit=None):
    dataset_path = Path(__file__).resolve().parent / DATA_PATH
    with dataset_path.open("r", encoding="utf-8") as source:
        items = json.load(source)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    return items[:limit] if limit is not None else items


def summarize_model_record(record):
    if not record:
        return None
    quantization = record.get("quantization") or {}
    return {
        "publisher": record.get("publisher"),
        "model_id": record.get("key"),
        "display_name": record.get("display_name"),
        "architecture": record.get("architecture"),
        "parameters": record.get("params_string"),
        "quantization": quantization.get("name"),
        "quantization_bits": quantization.get("bits_per_weight"),
        "max_context_length": record.get("max_context_length"),
        "selected_variant": record.get("selected_variant"),
    }


def parse_context_lengths(specs):
    overrides = {}
    for spec in specs or []:
        try:
            model, raw_length = spec.rsplit("=", 1)
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --context-length-for value {spec!r}; expected MODEL=INTEGER"
            ) from exc
        if not model or length <= 0:
            raise ValueError(
                f"Invalid --context-length-for value {spec!r}; model and positive length are required"
            )
        overrides[model] = length
    return overrides


def parse_reasoning_efforts(specs):
    overrides = {}
    for spec in specs or []:
        try:
            model, effort = spec.rsplit("=", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --reasoning-effort-for value {spec!r}; expected MODEL=EFFORT"
            ) from exc
        if not model or effort not in REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning effort {spec!r}; use one of {sorted(REASONING_EFFORTS)}"
            )
        overrides[model] = effort
    return overrides


def load_model_plan(path):
    """Read the tracked sequence; every entry owns its context configuration."""
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            document = json.load(source)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Model plan not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model plan is not valid JSON: {path}") from exc
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise RuntimeError(f"Model plan {path} must contain schema_version 1 and a models list")
    plan = []
    seen = set()
    for entry in document["models"]:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Model plan {path} contains a non-object entry")
        model = entry.get("id")
        context_length = entry.get("context_length")
        reasoning_effort = entry.get("reasoning_effort")
        if not isinstance(model, str) or not model:
            raise RuntimeError(f"Model plan {path} contains an entry without a model id")
        if model in seen:
            raise RuntimeError(f"Model plan {path} contains {model!r} more than once")
        if not isinstance(context_length, int) or context_length <= 0:
            raise RuntimeError(f"Model plan entry {model!r} needs a positive context_length")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise RuntimeError(
                f"Model plan entry {model!r} has an invalid reasoning_effort"
            )
        seen.add(model)
        plan.append(
            {
                "id": model,
                "context_length": context_length,
                "reasoning_effort": reasoning_effort,
            }
        )
    if not plan:
        raise RuntimeError(f"Model plan {path} contains no models")
    return plan


def resolve_model_plan(args, context_overrides, effort_overrides):
    if args.sequence and args.models:
        raise ValueError("--sequence cannot be combined with --models")
    if args.sequence:
        plan_path = Path(args.plan)
        if not plan_path.is_absolute():
            plan_path = Path(__file__).resolve().parent / plan_path
        plan = load_model_plan(plan_path)
        plan_provenance = {"path": str(plan_path), "sha256": sha256_file(plan_path)}
    elif args.models:
        plan = [
            {
                "id": model,
                "context_length": DEFAULT_CONTEXT_LENGTH_OVERRIDES.get(
                    model, DEFAULT_CONTEXT_LENGTH
                ),
                "reasoning_effort": None,
            }
            for model in args.models
        ]
        plan_provenance = None
    else:
        raise ValueError("--models or --sequence is required unless --list-models is used by itself")
    resolved = []
    for entry in plan:
        model = entry["id"]
        context_length = context_overrides.get(
            model, args.context_length if args.context_length is not None else entry["context_length"]
        )
        if context_length <= 0:
            raise ValueError("--context-length must be positive")
        resolved.append(
            {
                "id": model,
                "context_length": context_length,
                "reasoning_effort": effort_overrides.get(
                    model,
                    args.reasoning_effort
                    if args.reasoning_effort is not None
                    else entry["reasoning_effort"],
                ),
            }
        )
    return resolved, plan_provenance


class RunIndex:
    """Atomic index of every run and its proof/progress metadata."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as source:
                self.document = json.load(source)
            if (
                self.document.get("schema_version") != 1
                or not isinstance(self.document.get("runs"), list)
            ):
                raise RuntimeError(f"Index {self.path} does not use schema version 1")
        else:
            self.document = {"schema_version": 1, "runs": []}

    def write(self):
        atomic_write_json(self.path, self.document)

    def find(self, fingerprint):
        return [
            run
            for run in self.document["runs"]
            if run.get("config_fingerprint") == fingerprint
        ]

    def add(self, run):
        self.document["runs"].append(run)
        self.write()

    def update(self, run_id, **changes):
        run = next(run for run in self.document["runs"] if run["run_id"] == run_id)
        run.update(changes)
        self.write()
        return run


class RunStore:
    """One model run's raw data plus the immediately mirrored aggregate data."""

    def __init__(self, output_root, run_id):
        self.root = Path(output_root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.output_path = self.run_dir / "outputs.json"
        self.all_path = self.root / "all.json"
        self.records = []
        self.output_document = None
        self.all_document = None

    @staticmethod
    def _aggregate_run(manifest):
        return {
            "run_id": manifest["run_id"],
            "config_fingerprint": manifest["config_fingerprint"],
            "dataset": manifest["dataset"],
            "protocol": {"sha256": manifest["protocol"]["sha256"]},
            "generation": manifest["generation"],
            "model": manifest["model"],
            "outputs": [],
        }

    def _load_all_document(self):
        if self.all_path.exists():
            with self.all_path.open("r", encoding="utf-8") as source:
                document = json.load(source)
            if document.get("schema_version") != 1 or not isinstance(document.get("runs"), list):
                raise RuntimeError(f"Invalid aggregate output schema in {self.all_path}")
            return document
        return {"schema_version": 1, "runs": []}

    def _aggregate_group(self):
        group = next(
            (group for group in self.all_document["runs"] if group.get("run_id") == self.run_id),
            None,
        )
        if group is None or not isinstance(group.get("outputs"), list):
            raise RuntimeError(f"Aggregate output lacks run {self.run_id}")
        return group

    def create(self, manifest):
        if self.run_dir.exists():
            raise RuntimeError(f"Run directory already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        atomic_write_json(self.manifest_path, manifest)
        self.output_document = {"schema_version": 1, "outputs": []}
        atomic_write_json(self.output_path, self.output_document)
        self.all_document = self._load_all_document()
        self.all_document["runs"].append(self._aggregate_run(manifest))
        atomic_write_json(self.all_path, self.all_document)
        self.records = []

    def load(self):
        if not self.manifest_path.exists():
            raise RuntimeError(f"Run manifest is missing: {self.manifest_path}")
        self.output_document = read_output_document(self.output_path)
        self.all_document = self._load_all_document()
        self._aggregate_group()
        self.records = self.output_document["outputs"]
        return self.records

    def reconcile(self):
        """Repair an interrupted dual append without regenerating model output."""
        self.load()
        local = {record["record_id"]: record for record in self.records}
        if len(local) != len(self.records):
            raise RuntimeError(f"Duplicate record IDs in {self.output_path}")
        global_records = self._aggregate_group()["outputs"]
        aggregate = {record["record_id"]: record for record in global_records}
        if len(aggregate) != len(global_records):
            raise RuntimeError(f"Duplicate record IDs for run {self.run_id} in {self.all_path}")
        for record_id in local.keys() & aggregate.keys():
            if canonical_json(local[record_id]) != canonical_json(aggregate[record_id]):
                raise RuntimeError(
                    f"Record {record_id} differs between {self.output_path} and {self.all_path}"
                )
        for record in self.records:
            if record["record_id"] not in aggregate:
                global_records.append(record)
        for record in global_records:
            if record["record_id"] not in local:
                self.records.append(record)
        self.records.sort(key=lambda record: record["sequence"])
        global_records.sort(key=lambda record: record["sequence"])
        self.output_document["outputs"] = self.records
        atomic_write_json(self.output_path, self.output_document)
        atomic_write_json(self.all_path, self.all_document)
        return self.records

    def append(self, record):
        if record["record_id"] in {existing["record_id"] for existing in self.records}:
            raise RuntimeError(f"Duplicate record id {record['record_id']}")
        self.records.append(record)
        self.output_document["outputs"] = self.records
        atomic_write_json(self.output_path, self.output_document)
        self._aggregate_group()["outputs"].append(record)
        atomic_write_json(self.all_path, self.all_document)

    def update_manifest(self, **changes):
        with self.manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
        manifest.update(changes)
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def done_keys(self):
        return {
            (record["item"]["domain"], record["item"]["item_id"], record["condition"])
            for record in self.records
            if record.get("is_final")
        }

    def proof(self):
        record_ids = [record["record_id"] for record in self.records]
        return {
            "run_output_path": str(self.output_path),
            "all_output_path": str(self.all_path),
            "record_count": len(record_ids),
            "record_ids_sha256": sha256_value(record_ids),
            "run_output_sha256": sha256_file(self.output_path)
            if self.output_path.exists()
            else None,
            "all_run_records_sha256": sha256_value(self._aggregate_group()["outputs"]),
            "all_output_sha256_at_update": sha256_file(self.all_path)
            if self.all_path.exists()
            else None,
        }


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible and native APIs."""

    def __init__(self, base_url, api_key, timeout, management_url=None):
        self.base_url = base_url.rstrip("/")
        self.management_url = (
            management_url or self._derive_management_url(self.base_url)
        ).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.last_turn_metadata = None

    @staticmethod
    def _derive_management_url(base_url):
        return f"{base_url[:-3]}/api/v1" if base_url.endswith("/v1") else f"{base_url}/api/v1"

    def _request_json(self, method, path, payload=None, base_url=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_object = request.Request(
            f"{base_url or self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with request.urlopen(request_object, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LM Studio returned HTTP {exc.code} for {base_url or self.base_url}{path}: {details}"
            ) from exc
        except (TimeoutError, error.URLError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"Could not reach LM Studio at {base_url or self.base_url}: {reason}") from exc

    def list_models(self):
        response = self._request_json("GET", "/models", base_url=self.management_url)
        return [record["key"] for record in response.get("models", [])]

    def get_model_record(self, model):
        response = self._request_json("GET", "/models", base_url=self.management_url)
        return next((record for record in response.get("models", []) if record.get("key") == model), None)

    def list_loaded_instances(self):
        response = self._request_json("GET", "/models", base_url=self.management_url)
        return [
            {"model": model.get("key"), "instance_id": instance.get("id")}
            for model in response.get("models", [])
            for instance in model.get("loaded_instances", [])
        ]

    def load_model(self, model, context_length):
        response = self._request_json(
            "POST",
            "/models/load",
            {"model": model, "context_length": context_length},
            base_url=self.management_url,
        )
        try:
            return response["instance_id"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"LM Studio load response did not contain an instance ID: {response}") from exc

    def unload_model(self, instance_id):
        self._request_json(
            "POST", "/models/unload", {"instance_id": instance_id}, base_url=self.management_url
        )
        deadline = time.monotonic() + min(self.timeout, 60)
        while time.monotonic() < deadline:
            if not any(instance["instance_id"] == instance_id for instance in self.list_loaded_instances()):
                return
            time.sleep(1)
        raise RuntimeError(f"LM Studio still reports instance {instance_id} as loaded after unload")

    def chat(self, model, messages, options):
        metadata = {"usage": None, "finish_reason": None}
        self.last_turn_metadata = metadata
        try:
            response = self._request_json(
                "POST", "/chat/completions", {"model": model, "messages": messages, **options}
            )
        except Exception as exc:
            metadata["error"] = repr(exc)
            raise
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LM Studio response did not contain assistant content: {response}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"LM Studio returned non-text assistant content: {content!r}")
        metadata.update({"usage": response.get("usage"), "finish_reason": choice.get("finish_reason")})
        return content


def output_metrics(metadata, elapsed_seconds):
    metrics = {"elapsed_seconds": round(elapsed_seconds, 3)}
    if metadata.get("usage") is not None:
        metrics["usage"] = metadata["usage"]
    if metadata.get("finish_reason") is not None:
        metrics["finish_reason"] = metadata["finish_reason"]
    return metrics


def make_raw_record(run, item, condition, turn, is_final, attempt_id, text, metrics, sequence):
    return {
        "schema_version": 1,
        "record_id": uuid.uuid4().hex,
        "sequence": sequence,
        "created_at": now(),
        "item": {key: item[key] for key in ("domain", "item_id", "title")},
        "condition": condition,
        "turn": turn,
        "is_final": is_final,
        "attempt_id": attempt_id,
        "text": text,
        "metrics": metrics,
    }


def progress(items, done, store):
    completed = sum(
        (item["domain"], item["item_id"], condition) in done
        for item in items
        for condition in CONDITIONS
    )
    return {
        "completed_conditions": completed,
        "total_conditions": len(items) * len(CONDITIONS),
        "raw_records": len(store.records),
    }


def sync_proof(index, store, run_id, items, done, status):
    state = {"status": status, "progress": progress(items, done, store), "proof": store.proof()}
    store.update_manifest(**state)
    index.update(run_id, **state)


def write_turn(store, index, run, items, done, item, condition, turn, is_final, attempt_id, text, metadata):
    record = make_raw_record(
        run,
        item,
        condition,
        turn,
        is_final,
        attempt_id,
        text,
        metadata,
        len(store.records) + 1,
    )
    store.append(record)
    if is_final:
        done.add((item["domain"], item["item_id"], condition))
    sync_proof(index, store, run["run_id"], items, done, "running")


def run_full(client, store, index, run, items, done, item, options):
    started = time.monotonic()
    reply = client.chat(
        run["model"]["id"],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item["full_instruction"]}],
        options,
    )
    write_turn(
        store,
        index,
        run,
        items,
        done,
        item,
        "full",
        1,
        True,
        uuid.uuid4().hex,
        reply,
        output_metrics(client.last_turn_metadata, time.monotonic() - started),
    )


def run_sharded(client, store, index, run, items, done, item, options):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    attempt_id = uuid.uuid4().hex
    for turn, shard in enumerate(item["shards"], start=1):
        messages.append({"role": "user", "content": shard})
        started = time.monotonic()
        reply = client.chat(run["model"]["id"], messages, options)
        write_turn(
            store,
            index,
            run,
            items,
            done,
            item,
            "sharded",
            turn,
            turn == len(item["shards"]),
            attempt_id,
            reply,
            output_metrics(client.last_turn_metadata, time.monotonic() - started),
        )
        messages.append({"role": "assistant", "content": reply})


def make_run(entry, identity, dataset_sha256, runner_sha256, args, plan_provenance=None):
    generation = {"temperature": args.temperature, "seed": args.seed}
    protocol = {
        "system_prompt": SYSTEM_PROMPT,
        "conditions": list(CONDITIONS),
        "sha256": sha256_value({"system_prompt": SYSTEM_PROMPT, "conditions": CONDITIONS}),
    }
    configuration = {
        "dataset_sha256": dataset_sha256,
        "runner_sha256": runner_sha256,
        "model": entry,
        "generation": generation,
        "protocol": protocol,
        "model_plan": plan_provenance,
    }
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "config_fingerprint": sha256_value(configuration),
        "started_at": now(),
        "status": "running",
        "dataset": {"path": DATA_PATH, "sha256": dataset_sha256},
        "runner": {"name": Path(__file__).name, "sha256": runner_sha256},
        "provider": {
            "name": "lm-studio",
            "base_url": args.base_url,
            "management_url": args.management_url,
        },
        "model_plan": plan_provenance,
        "protocol": protocol,
        "generation": generation,
        "model": {"id": entry["id"], **{key: value for key, value in entry.items() if key != "id"}, "identity": identity},
    }


def expected_complete(items, done):
    return len(done) == len(items) * len(CONDITIONS)


def select_run(index, store_root, draft, items, resume):
    """Reuse only the exact same configuration; different contexts make a new run."""
    matches = index.find(draft["config_fingerprint"])
    if resume and matches:
        completed = [run for run in matches if run.get("status") == "completed"]
        selected = completed[-1] if completed else matches[-1]
        store = RunStore(store_root, selected["run_id"])
        store.reconcile()
        done = store.done_keys()
        if selected.get("status") == "completed" and not expected_complete(items, done):
            raise RuntimeError(
                f"Run {selected['run_id']} is marked completed but its raw output is incomplete"
            )
        return selected, store, done, selected.get("status") == "completed"
    store = RunStore(store_root, draft["run_id"])
    manifest = {
        **draft,
        "progress": {
            "completed_conditions": 0,
            "total_conditions": len(items) * len(CONDITIONS),
            "raw_records": 0,
        },
        "proof": None,
    }
    store.create(manifest)
    manifest["proof"] = store.proof()
    store.update_manifest(proof=manifest["proof"])
    index.add({**manifest, "run_directory": str(store.run_dir)})
    return draft, store, set(), False


def execute_run(client, index, output_root, entry, items, dataset_sha256, runner_sha256, args, plan_provenance):
    identity = None
    metadata_error = None
    if args.manage_models:
        try:
            identity = summarize_model_record(client.get_model_record(entry["id"]))
        except Exception as exc:
            metadata_error = repr(exc)
    draft = make_run(entry, identity, dataset_sha256, runner_sha256, args, plan_provenance)
    run, store, done, already_complete = select_run(index, output_root, draft, items, args.resume)
    if already_complete:
        sync_proof(index, store, run["run_id"], items, done, "completed")
        index.update(run["run_id"], last_verified_at=now())
        store.update_manifest(last_verified_at=now())
        print(f"Already complete: {entry['id']} ({run['run_id']})")
        return False

    if run["run_id"] != draft["run_id"]:
        print(f"Resuming {entry['id']} ({run['run_id']})")
        index.update(run["run_id"], status="running", resumed_at=now())
    else:
        index.update(run["run_id"], model_metadata_error=metadata_error)
    print(f"Run: {run['run_id']} | output: {store.run_dir}")
    options = run["generation"].copy()
    if entry["reasoning_effort"] is not None:
        options["reasoning_effort"] = entry["reasoning_effort"]
    status = "completed"
    error_message = None
    instance_id = None
    model_failed = False
    try:
        if args.manage_models:
            existing = client.list_loaded_instances()
            if existing:
                raise RuntimeError(f"Refusing to load another model while instances are loaded: {existing}")
            print(f"Loading {entry['id']} at context {entry['context_length']}...")
            instance_id = client.load_model(entry["id"], entry["context_length"])
            loaded = client.list_loaded_instances()
            if loaded != [{"model": entry["id"], "instance_id": instance_id}]:
                raise RuntimeError(f"LM Studio did not report exactly the requested model: {loaded}")
            index.update(run["run_id"], model_instance_id=instance_id)
            store.update_manifest(model_instance_id=instance_id)
        for item in items:
            full_key = (item["domain"], item["item_id"], "full")
            sharded_key = (item["domain"], item["item_id"], "sharded")
            if full_key not in done:
                try:
                    run_full(client, store, index, run, items, done, item, options)
                except Exception:
                    model_failed = True
                    status = "failed"
                    error_message = traceback.format_exc()
            if model_failed:
                break
            if sharded_key not in done:
                try:
                    run_sharded(client, store, index, run, items, done, item, options)
                except Exception:
                    model_failed = True
                    status = "failed"
                    error_message = traceback.format_exc()
            if model_failed:
                break
    except KeyboardInterrupt:
        status = "stopped_by_user"
        error_message = "KeyboardInterrupt"
        raise
    except Exception:
        status = "failed"
        error_message = traceback.format_exc()
        model_failed = True
    finally:
        if instance_id is not None:
            try:
                client.unload_model(instance_id)
            except Exception:
                status = "unload_failed"
                error_message = traceback.format_exc()
                model_failed = True
        if status == "completed" and not expected_complete(items, done):
            status = "failed"
            error_message = "Run ended before all required final outputs were written"
            model_failed = True
        final = {"finished_at": now(), "error": error_message}
        sync_proof(index, store, run["run_id"], items, done, status)
        index.update(run["run_id"], **final)
        store.update_manifest(**final)
    if model_failed:
        print(f"FAILED {entry['id']}: {error_message.splitlines()[-1] if error_message else status}")
    else:
        print(f"Completed {entry['id']}")
    return model_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Exact LM Studio model IDs")
    parser.add_argument("--sequence", action="store_true", help="Run the tracked models.json plan")
    parser.add_argument("--plan", default=PLAN_PATH, help=f"Sequence plan path (default: {PLAN_PATH})")
    parser.add_argument("--list-models", action="store_true", help="List installed LM Studio model IDs")
    parser.add_argument("--base-url", default=os.environ.get("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--management-url", default=os.environ.get("LM_STUDIO_MANAGEMENT_URL"))
    parser.add_argument("--api-key", default=os.environ.get("LM_STUDIO_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, help="Only run the first N benchmark items")
    parser.add_argument("--resume", action="store_true", help="Resume or skip only an exact matching run")
    parser.add_argument("--output", default=OUT_PATH, help=f"Result directory (default: {OUT_PATH})")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--context-length", type=int, help="Override every selected model's context length")
    parser.add_argument("--context-length-for", action="append", default=[], metavar="MODEL=TOKENS")
    parser.add_argument("--reasoning-effort", choices=sorted(REASONING_EFFORTS))
    parser.add_argument("--reasoning-effort-for", action="append", default=[], metavar="MODEL=EFFORT")
    parser.add_argument("--manage-models", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        context_overrides = parse_context_lengths(args.context_length_for)
        effort_overrides = parse_reasoning_efforts(args.reasoning_effort_for)
        plan, plan_provenance = (
            resolve_model_plan(args, context_overrides, effort_overrides)
            if (args.models or args.sequence)
            else (None, None)
        )
        items = load_items(args.limit) if plan else None
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    client = LMStudioClient(args.base_url, args.api_key, args.timeout, args.management_url)
    if args.list_models:
        for model in client.list_models():
            print(model)
        if plan is None:
            return
    if plan is None:
        parser.error("--models or --sequence is required unless --list-models is used by itself")
    dataset_path = Path(__file__).resolve().parent / DATA_PATH
    dataset_sha256 = sha256_file(dataset_path)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    index = RunIndex(output_root / "index.json")
    failures = False
    for entry in plan:
        failures |= execute_run(
            client,
            index,
            output_root,
            entry,
            items,
            dataset_sha256,
            runner_sha256,
            args,
            plan_provenance,
        )
    if failures:
        raise SystemExit(1)
    print(f"Done. Raw outputs are in {output_root}")


if __name__ == "__main__":
    main()
