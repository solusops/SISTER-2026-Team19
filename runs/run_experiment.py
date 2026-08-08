"""Run the creative-writing benchmark against explicit LM Studio or Ollama models.

Each generated response is appended immediately to two JSONL datasets:

* ``results/results_<model>.jsonl`` contains one model's self-identifying rows.
* ``results/all_results.jsonl`` is the independent combined dataset.

``results/index.json`` is the single run-status and provenance index.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


DATA_PATH = "benchmark_data.json"
OUT_PATH = "results"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_TIMEOUT = 600
CONTEXT_LENGTH = 16384
LMSTUDIO_LOAD_SETTINGS = {
    "eval_batch_size": 2048,
    "physical_batch_size": 512,
    "parallel": 1,
    "flash_attention": True,
    "offload_kv_cache_to_gpu": True,
}
SYSTEM_PROMPT = (
    "You are a skilled creative writer. Follow the user's instructions "
    "for the story precisely, including any exact phrases, constraints, "
    "or stylistic requirements."
)
CONDITIONS = ("full", "sharded")


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


def read_jsonl(path):
    """Read a raw result file whose lines are independently usable records."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL record in {path} line {line_number}") from exc
            if not isinstance(record, dict) or not record.get("record_id"):
                raise RuntimeError(f"Invalid raw record in {path} line {line_number}")
            records.append(record)
    return records


def append_jsonl(path, record):
    """Durably append one independently usable result record."""
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(canonical_json(record))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def load_items(limit=None):
    dataset_path = Path(__file__).resolve().parent / DATA_PATH
    with dataset_path.open("r", encoding="utf-8") as source:
        items = json.load(source)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    return items[:limit] if limit is not None else items


def summarize_model_record(record, backend):
    if not record:
        return None
    if backend == "ollama":
        details = record.get("details") or {}
        return {
            "publisher": "ollama",
            "model_id": record.get("name") or record.get("model"),
            "display_name": record.get("name") or record.get("model"),
            "architecture": details.get("family"),
            "parameters": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "quantization_bits": None,
            "max_context_length": None,
            "selected_variant": record.get("digest"),
        }
    quantization = record.get("quantization") or {}
    reasoning = (record.get("capabilities") or {}).get("reasoning") or {}
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
        "reasoning_options": reasoning.get("allowed_options") or [],
    }


def resolve_models(models):
    if not models:
        raise ValueError("--models is required unless --list-models is used by itself")
    if len(set(models)) != len(models):
        raise ValueError("--models cannot contain duplicate model IDs")
    return [{"id": model, "context_length": CONTEXT_LENGTH} for model in models]


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

    def set_batch(self, models):
        self.document["batch_models"] = list(models)
        self.write()

    def update(self, run_id, **changes):
        run = next(run for run in self.document["runs"] if run["run_id"] == run_id)
        run.update(changes)
        self.write()
        return run


class RunStore:
    """Flat per-model JSONL output plus one independent combined JSONL file."""

    def __init__(self, output_root, model_id):
        self.root = Path(output_root)
        self.model_id = model_id
        self.model_slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
        self.output_path = self.root / f"results_{self.model_slug}.jsonl"
        self.all_path = self.root / "all_results.jsonl"
        self.records = []

    def create(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise RuntimeError(f"Result file already exists: {self.output_path}")
        self.output_path.touch()
        self.records = []

    def load(self):
        self.records = read_jsonl(self.output_path)
        return self.records

    def reconcile(self):
        """Repair an interrupted append to the combined dataset."""
        self.load()
        local = {record["record_id"]: record for record in self.records}
        if len(local) != len(self.records):
            raise RuntimeError(f"Duplicate record IDs in {self.output_path}")
        aggregate = {record["record_id"]: record for record in read_jsonl(self.all_path)}
        for record_id in local.keys() & aggregate.keys():
            if canonical_json(local[record_id]) != canonical_json(aggregate[record_id]):
                raise RuntimeError(f"Record {record_id} differs between {self.output_path} and {self.all_path}")
        for record_id, record in local.items():
            if record_id not in aggregate:
                append_jsonl(self.all_path, record)
        return self.records

    def append(self, record):
        if record["record_id"] in {existing["record_id"] for existing in self.records}:
            raise RuntimeError(f"Duplicate record id {record['record_id']}")
        append_jsonl(self.output_path, record)
        append_jsonl(self.all_path, record)
        self.records.append(record)

    def done_keys(self):
        return {
            (record["item"]["domain"], record["item"]["item_id"], record["condition"])
            for record in self.records
            if record.get("is_final")
        }

    def proof(self):
        record_ids = [record["record_id"] for record in self.records]
        return {
            "run_output_path": self.output_path.name,
            "all_output_path": self.all_path.name,
            "record_count": len(record_ids),
            "record_ids_sha256": sha256_value(record_ids),
            "run_output_sha256": sha256_file(self.output_path)
            if self.output_path.exists()
            else None,
            "all_output_sha256_at_update": sha256_file(self.all_path)
            if self.all_path.exists()
            else None,
        }


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible and native APIs."""

    backend = "lmstudio"
    manages_models = True

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
            {"model": model, "context_length": context_length, **LMSTUDIO_LOAD_SETTINGS},
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

    def chat(self, model, messages, options, context_length):
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


class OllamaClient:
    """Client for Ollama's local model, metadata, and chat APIs."""

    backend = "ollama"
    manages_models = False

    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_turn_metadata = None

    def _request_json(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request_object = request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with request.urlopen(request_object, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code} for {path}: {details}") from exc
        except (TimeoutError, error.URLError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {reason}") from exc

    def list_models(self):
        return [record["name"] for record in self._request_json("GET", "/api/tags").get("models", [])]

    def get_model_record(self, model):
        for record in self._request_json("GET", "/api/tags").get("models", []):
            if model in (record.get("name"), record.get("model")):
                return record
        return None

    def chat(self, model, messages, options, context_length):
        metadata = {"usage": None, "finish_reason": None}
        self.last_turn_metadata = metadata
        response = self._request_json(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {**options, "num_ctx": context_length},
            },
        )
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Ollama response did not contain assistant content: {response}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Ollama returned non-text assistant content: {content!r}")
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        usage = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        metadata.update({"usage": usage, "finish_reason": response.get("done_reason")})
        return content


def output_metrics(metadata, elapsed_seconds):
    metrics = {"elapsed_seconds": round(elapsed_seconds, 3)}
    if metadata.get("usage") is not None:
        metrics["usage"] = metadata["usage"]
    if metadata.get("finish_reason") is not None:
        metrics["finish_reason"] = metadata["finish_reason"]
    return metrics


def make_raw_record(run, item, condition, turn, is_final, attempt_id, text, metrics, sequence):
    identity = run["model"].get("identity") or {}
    return {
        "schema_version": 1,
        "record_id": uuid.uuid4().hex,
        "model_id": run["model"]["id"],
        "model_name": identity.get("display_name") or run["model"]["id"],
        "parameters": identity.get("parameters"),
        "quant_file": identity.get("selected_variant"),
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
        run["model"]["context_length"],
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
        reply = client.chat(run["model"]["id"], messages, options, run["model"]["context_length"])
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


def make_run(entry, identity, dataset_sha256, runner_sha256, args):
    generation = {"temperature": args.temperature, "seed": args.seed}
    if args.backend == "lmstudio" and "off" in (identity or {}).get("reasoning_options", []):
        generation["reasoning"] = "off"
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
        "provider": args.backend,
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
            "name": args.backend,
            "base_url": args.base_url,
            **({"management_url": args.management_url} if args.backend == "lmstudio" else {}),
        },
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
        store = RunStore(store_root, selected["model"]["id"])
        store.reconcile()
        done = store.done_keys()
        if selected.get("status") == "completed" and not expected_complete(items, done):
            raise RuntimeError(
                f"Run {selected['run_id']} is marked completed but its raw output is incomplete"
            )
        return selected, store, done, selected.get("status") == "completed"
    store = RunStore(store_root, draft["model"]["id"])
    run = {
        **draft,
        "progress": {
            "completed_conditions": 0,
            "total_conditions": len(items) * len(CONDITIONS),
            "raw_records": 0,
        },
        "proof": None,
    }
    store.create()
    run["proof"] = store.proof()
    index.add(run)
    return draft, store, set(), False


def execute_run(client, index, output_root, entry, items, dataset_sha256, runner_sha256, args):
    identity = None
    metadata_error = None
    try:
        identity = summarize_model_record(client.get_model_record(entry["id"]), args.backend)
    except Exception as exc:
        metadata_error = repr(exc)
    draft = make_run(entry, identity, dataset_sha256, runner_sha256, args)
    run, store, done, already_complete = select_run(index, output_root, draft, items, args.resume)
    if already_complete:
        sync_proof(index, store, run["run_id"], items, done, "completed")
        index.update(run["run_id"], last_verified_at=now())
        print(f"Already complete: {entry['id']} ({run['run_id']})")
        return False

    if run["run_id"] != draft["run_id"]:
        print(f"Resuming {entry['id']} ({run['run_id']})")
        index.update(run["run_id"], status="running", resumed_at=now())
    else:
        index.update(run["run_id"], model_metadata_error=metadata_error)
    print(f"Run: {run['model']['id']} | output: {store.output_path}")
    options = run["generation"].copy()
    status = "completed"
    error_message = None
    instance_id = None
    model_failed = False
    try:
        if client.manages_models:
            existing = client.list_loaded_instances()
            if existing:
                raise RuntimeError(f"Refusing to load another model while instances are loaded: {existing}")
            print(f"Loading {entry['id']} at context {entry['context_length']}...")
            instance_id = client.load_model(entry["id"], entry["context_length"])
            loaded = client.list_loaded_instances()
            if loaded != [{"model": entry["id"], "instance_id": instance_id}]:
                raise RuntimeError(f"LM Studio did not report exactly the requested model: {loaded}")
            index.update(run["run_id"], model_instance_id=instance_id)
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
    if model_failed:
        print(f"FAILED {entry['id']}: {error_message.splitlines()[-1] if error_message else status}")
    else:
        print(f"Completed {entry['id']}")
    return model_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lmstudio", "ollama"), default="lmstudio")
    parser.add_argument("--models", nargs="+", help="Exact model IDs or Ollama tags to run")
    parser.add_argument("--list-models", action="store_true", help="List locally available models for the selected backend")
    parser.add_argument("--base-url")
    parser.add_argument("--management-url", default=os.environ.get("LM_STUDIO_MANAGEMENT_URL"))
    parser.add_argument("--api-key", default=os.environ.get("LM_STUDIO_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, help="Only run the first N benchmark items")
    parser.add_argument("--resume", action="store_true", help="Resume or skip only an exact matching run")
    parser.add_argument("--output", default=OUT_PATH, help=f"Result directory (default: {OUT_PATH})")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.base_url is None:
        args.base_url = os.environ.get(
            "LM_STUDIO_BASE_URL" if args.backend == "lmstudio" else "OLLAMA_BASE_URL",
            LMSTUDIO_BASE_URL if args.backend == "lmstudio" else OLLAMA_BASE_URL,
        )
    try:
        models = resolve_models(args.models) if args.models else None
        items = load_items(args.limit) if models else None
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    client = (
        LMStudioClient(args.base_url, args.api_key, args.timeout, args.management_url)
        if args.backend == "lmstudio"
        else OllamaClient(args.base_url, args.timeout)
    )
    if args.list_models:
        for model in client.list_models():
            print(model)
        if models is None:
            return
    if models is None:
        parser.error("--models is required unless --list-models is used by itself")
    dataset_path = Path(__file__).resolve().parent / DATA_PATH
    dataset_sha256 = sha256_file(dataset_path)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    index = RunIndex(output_root / "index.json")
    index.set_batch(entry["id"] for entry in models)
    failures = False
    for entry in models:
        failures |= execute_run(
            client,
            index,
            output_root,
            entry,
            items,
            dataset_sha256,
            runner_sha256,
            args,
        )
    if failures:
        raise SystemExit(1)
    print(f"Done. Raw outputs are in {output_root}")


if __name__ == "__main__":
    main()
