"""
Runs the creative-writing "lost in conversation" benchmark against models
served by an OpenAI-compatible LM Studio endpoint.

For each benchmark item and each model, generates TWO outputs:
  - "full":    single prompt containing the complete fully-specified instruction
  - "sharded": a multi-turn conversation where each shard is sent as a
               separate user turn (mirrors the "LLMs Get Lost in Multi-Turn
               Conversation" methodology), and the FINAL assistant turn
               (after the last shard) is treated as the story to evaluate.

Usage:
    python3 run_experiment.py --list-models
    python3 run_experiment.py --models <model-id-1> <model-id-2> --limit 3
    python3 run_experiment.py --models <model-id> --resume

Recommended model sequence:
    openai/gpt-oss-20b
    qwen/qwen3.5-9b
    google/gemma-4-26b-a4b-qat
    mistral-7b-instruct-v0.2
    prism-ml/bonsai-27b
    liquid/lfm2-24b-a2b
    thedrummer_cydonia-24b-v4.2.0

LM Studio must be running its local server (the default endpoint is
`http://localhost:1234/v1`). The runner explicitly loads one model through
LM Studio's native `/api/v1/models/load` endpoint, evaluates it, and unloads
that instance before moving to the next model. Use the exact model IDs
returned by `--list-models`; the names shown in the LM Studio model picker are
not always the IDs accepted by the API.

Example:
    python3 run_experiment.py \
        --models "<model-id-1-from-list-models>" \
                 "<model-id-2-from-list-models>" --resume
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
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


DATA_PATH = "benchmark_data.json"
OUT_PATH = "results.jsonl"
INDEX_PATH = "run_index.json"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_TIMEOUT = 600
DEFAULT_CONTEXT_LENGTH = 65536
DEFAULT_CONTEXT_LENGTH_OVERRIDES = {
    "google/gemma-4-26b-a4b-qat": 32768,
    "mistral-7b-instruct-v0.2": 32768,
}
DEFAULT_MODEL_SEQUENCE = (
    "openai/gpt-oss-20b",
    "qwen/qwen3.5-9b",
    "google/gemma-4-26b-a4b-qat",
    "mistral-7b-instruct-v0.2",
    "prism-ml/bonsai-27b",
    "liquid/lfm2-24b-a2b",
    "thedrummer_cydonia-24b-v4.2.0",
)

SYSTEM_PROMPT = (
    "You are a skilled creative writer. Follow the user's instructions "
    "for the story precisely, including any exact phrases, constraints, "
    "or stylistic requirements."
)

FULL_RESULT_FIELDS = (
    "domain", "item_id", "title", "model", "condition", "story", "seconds"
)
SHARDED_RESULT_FIELDS = (
    "domain", "item_id", "title", "model", "condition", "story", "transcript", "seconds"
)


def load_items(limit=None):
    dataset_path = Path(__file__).resolve().parent / DATA_PATH
    with dataset_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return items[:limit] if limit else items


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_model_record(record):
    """Keep stable LM Studio identity fields without storing lifecycle state."""
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
        "size_bytes": record.get("size_bytes"),
        "max_context_length": record.get("max_context_length"),
        "format": record.get("format"),
        "selected_variant": record.get("selected_variant"),
        "variants": record.get("variants"),
        "capabilities": record.get("capabilities"),
    }


def load_done_keys(output_path):
    """Return result keys already written to ``output_path`` for --resume."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add((row["domain"], row["item_id"], row["model"], row["condition"]))
                except Exception:
                    continue
    return done


def model_output_path(base_output, model):
    """Append a filesystem-safe model name to the configured output path."""
    path = Path(base_output)
    suffix = path.suffix or ".jsonl"
    stem = path.stem if path.suffix else path.name
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._")
    return str(path.with_name(f"{stem}_{safe_model}{suffix}"))


def append_telemetry(telemetry_path, record):
    """Append verbose progress and per-turn diagnostics to telemetry."""
    append_jsonl(telemetry_path, record)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


class IndexWriter:
    """Persist compact hierarchical batch/run summaries as one JSON document."""

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as source:
                    self.document = json.load(source)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Index {path} is not structured JSON; use a new --index path"
                ) from exc
            if not isinstance(self.document, dict) or not isinstance(
                self.document.get("batches"), list
            ):
                raise RuntimeError(f"Index {path} has an invalid structured format")
        else:
            self.document = {"schema_version": 1, "batches": []}

    def _write(self):
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, temporary_path = tempfile.mkstemp(
            prefix=".run-index-", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(self.document, output, ensure_ascii=False, indent=2)
                output.write("\n")
            os.replace(temporary_path, self.path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def start_batch(self, batch):
        batch_id = uuid.uuid4().hex
        self.document["batches"].append({"batch_id": batch_id, **batch, "runs": []})
        self._write()
        return batch_id

    def start_run(self, batch_id, run):
        batch = next(batch for batch in self.document["batches"] if batch["batch_id"] == batch_id)
        batch["runs"].append(run)
        self._write()

    def finish_run(self, batch_id, run_id, result):
        batch = next(batch for batch in self.document["batches"] if batch["batch_id"] == batch_id)
        run = next(run for run in batch["runs"] if run["run_id"] == run_id)
        run.update(result)
        self._write()


def seed_model_output_from_legacy(model_output, legacy_output, model):
    """Copy this model's old combined rows into its new per-model file once."""
    if os.path.exists(model_output) or not legacy_output or not os.path.exists(legacy_output):
        return 0

    rows = []
    with open(legacy_output, "r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("model") == model:
                rows.append(line)

    if not rows:
        return 0
    with open(model_output, "w", encoding="utf-8") as destination:
        destination.writelines(rows)
    return len(rows)


def clean_result_file(path, allowed_models):
    """Normalize one JSONL artifact and remove disallowed/duplicate rows."""
    if not path or not os.path.exists(path):
        return {"path": path, "kept": 0, "removed": 0, "duplicates": 0}

    kept_rows = []
    seen = set()
    removed = 0
    duplicates = 0
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
                key = (row["domain"], row["item_id"], row["model"], row["condition"])
            except Exception:
                removed += 1
                continue
            if row["model"] not in allowed_models:
                removed += 1
                continue
            if key in seen:
                duplicates += 1
                removed += 1
                continue
            fields = SHARDED_RESULT_FIELDS if row["condition"] == "sharded" else FULL_RESULT_FIELDS
            if row["condition"] not in {"full", "sharded"}:
                removed += 1
                continue
            kept_rows.append({field: row[field] for field in fields if field in row})
            seen.add(key)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary_path = tempfile.mkstemp(prefix=".results-clean-", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            for row in kept_rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return {"path": path, "kept": len(kept_rows), "removed": removed, "duplicates": duplicates}


def count_completed(items, done, model):
    """Count completed full/sharded rows for this model and item selection."""
    return sum(
        (items_item["domain"], items_item["item_id"], model, condition) in done
        for items_item in items
        for condition in ("full", "sharded")
    )


def progress_metrics(total_rows, completed_rows, initial_completed, started_monotonic):
    elapsed_seconds = time.monotonic() - started_monotonic
    observed_rows = completed_rows - initial_completed
    remaining_rows = max(total_rows - completed_rows, 0)
    average_seconds = elapsed_seconds / observed_rows if observed_rows else None
    if remaining_rows == 0:
        eta_seconds = 0
    else:
        eta_seconds = remaining_rows * average_seconds if average_seconds is not None else None
    return {
        "completed_rows": completed_rows,
        "total_rows": total_rows,
        "remaining_rows": remaining_rows,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 1),
    }


def format_eta(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_context_lengths(specs):
    """Parse repeated MODEL=TOKEN_COUNT overrides for per-model loading."""
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
    """Parse repeated MODEL=EFFORT overrides for LM Studio reasoning."""
    allowed = {"off", "on", "low", "medium", "high"}
    overrides = {}
    for spec in specs or []:
        try:
            model, effort = spec.rsplit("=", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --reasoning-effort-for value {spec!r}; expected MODEL=EFFORT"
            ) from exc
        if not model or effort not in allowed:
            raise ValueError(
                f"Invalid reasoning effort {spec!r}; use one of {sorted(allowed)}"
            )
        overrides[model] = effort
    return overrides


class ResultWriter:
    """Write each row to its model-specific JSONL file."""

    def __init__(self, base_output):
        self.base_output = base_output
        self.files = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for output in self.files.values():
            output.close()

    def write(self, line):
        row = json.loads(line)
        model = row["model"]
        if model not in self.files:
            output_path = model_output_path(self.base_output, model)
            self.files[model] = open(output_path, "a", encoding="utf-8")
        self.files[model].write(line)

    def flush(self):
        for output in self.files.values():
            output.flush()


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
        if base_url.endswith("/v1"):
            return f"{base_url[:-3]}/api/v1"
        return f"{base_url}/api/v1"

    def _request_json(self, method, path, payload=None, base_url=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            f"{(base_url or self.base_url)}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LM Studio returned HTTP {exc.code} for {(base_url or self.base_url)}{path}: {details}"
            ) from exc
        except (TimeoutError, error.URLError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(
                f"Could not reach LM Studio at {base_url or self.base_url}: {reason}"
            ) from exc

    def list_models(self):
        response = self._request_json("GET", "/models")
        return [model["id"] for model in response.get("data", [])]

    def get_model_record(self, model):
        """Return native LM Studio metadata for one model key, if available."""
        response = self._request_json("GET", "/models", base_url=self.management_url)
        return next(
            (record for record in response.get("models", []) if record.get("key") == model),
            None,
        )

    def list_loaded_instances(self):
        """Return native-API model instances currently loaded in LM Studio."""
        response = self._request_json(
            "GET", "/models", base_url=self.management_url
        )
        instances = []
        for model in response.get("models", []):
            for instance in model.get("loaded_instances", []):
                instances.append(
                    {"model": model.get("key"), "instance_id": instance.get("id")}
                )
        return instances

    def load_model(self, model, context_length=None):
        payload = {"model": model}
        if context_length is not None:
            payload["context_length"] = context_length
        response = self._request_json(
            "POST", "/models/load", payload, base_url=self.management_url
        )
        try:
            return response["instance_id"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"LM Studio load response did not contain an instance ID: {response}"
            ) from exc

    def unload_model(self, instance_id):
        self._request_json(
            "POST",
            "/models/unload",
            {"instance_id": instance_id},
            base_url=self.management_url,
        )
        # Do not start another model until LM Studio confirms this instance is
        # gone. A successful unload response alone is not enough to protect
        # against overlapping engine cleanup on a heavily loaded machine.
        deadline = time.monotonic() + min(self.timeout, 60)
        while time.monotonic() < deadline:
            if not any(
                instance["instance_id"] == instance_id
                for instance in self.list_loaded_instances()
            ):
                return
            time.sleep(1)
        raise RuntimeError(
            f"LM Studio still reports instance {instance_id} as loaded after unload"
        )

    def chat(self, model, messages, options, context_length=None):
        prompt_characters = sum(
            len(message.get("content") or "") for message in messages
        )
        metadata = {
            "model": model,
            "context_length": context_length,
            "prompt_characters": prompt_characters,
            "estimated_prompt_tokens": math.ceil(prompt_characters / 4),
            "usage": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
            "response_characters": None,
            "reasoning_characters": None,
            "context_limit_reached": False,
            "context_limit_signal": None,
        }
        self.last_turn_metadata = metadata
        try:
            response = self._request_json(
                "POST",
                "/chat/completions",
                {"model": model, "messages": messages, **options},
            )
        except Exception as exc:
            message = str(exc).lower()
            if "context" in message or "token" in message or "length" in message:
                metadata["context_limit_reached"] = True
                metadata["context_limit_signal"] = "error"
            metadata["error"] = repr(exc)
            raise
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"LM Studio response did not contain assistant content: {response}"
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError(
                f"LM Studio returned non-text assistant content: {content!r}"
            )
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(reasoning_content, str):
            reasoning_content = json.dumps(reasoning_content, ensure_ascii=False)
        metadata.update(
            {
                "usage": usage or None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens")
                or completion_details.get("reasoning_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "response_characters": len(content),
                "reasoning_characters": len(reasoning_content),
            }
        )
        estimated_total_tokens = math.ceil(
            (prompt_characters + len(content) + len(reasoning_content)) / 4
        )
        metadata["estimated_total_tokens"] = estimated_total_tokens
        if context_length:
            if metadata["total_tokens"] is not None:
                if metadata["total_tokens"] >= context_length:
                    metadata["context_limit_reached"] = True
                    metadata["context_limit_signal"] = "usage"
                elif metadata["total_tokens"] >= context_length * 0.9:
                    metadata["context_limit_signal"] = "usage_near_limit"
            elif estimated_total_tokens >= context_length:
                metadata["context_limit_reached"] = True
                metadata["context_limit_signal"] = "estimate"
            elif estimated_total_tokens >= context_length * 0.9:
                metadata["context_limit_signal"] = "estimate_near_limit"
            if metadata["finish_reason"] == "length":
                metadata["context_limit_reached"] = True
                metadata["context_limit_signal"] = "finish_reason_length"
        return content


def run_full(client, model, item, options, context_length, turn_callback=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["full_instruction"]},
    ]
    try:
        reply = client.chat(model, messages, options, context_length)
    except Exception:
        if turn_callback is not None:
            turn_callback(1, client.last_turn_metadata)
        raise
    if turn_callback is not None:
        turn_callback(1, client.last_turn_metadata)
    return reply


def run_sharded(client, model, item, options, context_length, turn_callback=None):
    """Send each shard as its own user turn; the model's reply to each turn
    is added to the conversation history. Only the FINAL reply (after the
    last shard) is scored as the finished story, matching how a real user
    would gradually refine a request in chat."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_reply = None
    turn_metadata = []
    for turn_number, shard in enumerate(item["shards"], start=1):
        messages.append({"role": "user", "content": shard})
        try:
            reply = client.chat(model, messages, options, context_length)
        except Exception:
            if turn_callback is not None:
                turn_callback(turn_number, client.last_turn_metadata)
            raise
        metadata = {"turn_number": turn_number, **client.last_turn_metadata}
        turn_metadata.append(metadata)
        if turn_callback is not None:
            turn_callback(turn_number, client.last_turn_metadata)
        messages.append({"role": "assistant", "content": reply})
        last_reply = reply
    return last_reply, messages, turn_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Exact LM Studio model IDs; pass several IDs to evaluate several models")
    ap.add_argument("--sequence", action="store_true",
                    help="Run the configured model sequence in the documented order")
    ap.add_argument("--list-models", action="store_true",
                    help="List model IDs exposed by LM Studio and exit unless --models is also supplied")
    ap.add_argument("--base-url", default=os.environ.get("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL),
                    help=f"LM Studio API base URL (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--management-url", default=os.environ.get("LM_STUDIO_MANAGEMENT_URL"),
                    help="Native LM Studio management API base URL; derived from --base-url by default")
    ap.add_argument("--api-key", default=os.environ.get("LM_STUDIO_API_KEY", DEFAULT_API_KEY),
                    help="Bearer token sent to the endpoint (default: LM_STUDIO_API_KEY or lm-studio)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N benchmark items (smoke test)")
    ap.add_argument("--resume", action="store_true", help="Skip rows already in each model's output JSONL file")
    ap.add_argument("--output", default=OUT_PATH,
                    help=f"Base output path; model names are appended per file (default: {OUT_PATH})")
    ap.add_argument("--index", default=INDEX_PATH,
                    help=f"Run metadata JSONL index (default: {INDEX_PATH})")
    ap.add_argument("--telemetry", default="run_telemetry.jsonl",
                    help="Verbose progress/turn telemetry path (default: run_telemetry.jsonl)")
    ap.add_argument("--legacy-output", default=None,
                    help="Optional old combined JSONL file to seed model-specific outputs once")
    ap.add_argument("--clean-results", action="store_true",
                    help="Normalize result JSONL files, remove disallowed models, and deduplicate keys")
    ap.add_argument("--clean-only", action="store_true",
                    help="Perform --clean-results and exit without contacting LM Studio")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--context-length", type=int, default=None,
                    help=(
                        "Context length for models without a default override "
                        f"(default: {DEFAULT_CONTEXT_LENGTH}; Gemma/Mistral: 32768)"
                    ))
    ap.add_argument("--context-length-for", action="append", default=[], metavar="MODEL=TOKENS",
                    help="Override context length for one model; repeat for per-model settings")
    ap.add_argument("--reasoning-effort", choices=("off", "on", "low", "medium", "high"),
                    default=None, help="Default LM Studio reasoning effort for models that support it")
    ap.add_argument("--reasoning-effort-for", action="append", default=[], metavar="MODEL=EFFORT",
                    help="Override reasoning effort for one model; repeat for per-model settings")
    ap.add_argument("--manage-models", action=argparse.BooleanOptionalAction, default=True,
                    help="Load and unload each model through LM Studio's native API (default: enabled)")
    args = ap.parse_args()

    try:
        context_length_overrides = parse_context_lengths(args.context_length_for)
        reasoning_effort_overrides = parse_reasoning_efforts(args.reasoning_effort_for)
    except ValueError as exc:
        ap.error(str(exc))

    client = LMStudioClient(args.base_url, args.api_key, args.timeout, args.management_url)
    if args.sequence and args.models:
        ap.error("--sequence cannot be combined with --models")
    if args.sequence:
        args.models = list(DEFAULT_MODEL_SEQUENCE)
    if args.list_models:
        for model in client.list_models():
            print(model)
        if args.models is None:
            return
    if not args.models:
        ap.error("--models is required unless --list-models is used by itself")

    if args.clean_only and not args.clean_results:
        ap.error("--clean-only requires --clean-results")

    if args.clean_results:
        paths = [model_output_path(args.output, model) for model in args.models]
        if args.legacy_output:
            paths.append(args.legacy_output)
        cleaned_paths = set()
        for path in paths:
            if path in cleaned_paths:
                continue
            summary = clean_result_file(path, set(args.models))
            cleaned_paths.add(path)
            if summary["path"] and os.path.exists(summary["path"]):
                print(
                    f"Cleaned {path}: kept {summary['kept']}, removed {summary['removed']} "
                    f"({summary['duplicates']} duplicate keys)"
                )
        if args.clean_only:
            return

    items = load_items(args.limit)
    dataset_path = Path(__file__).resolve().parent / DATA_PATH
    dataset_sha256 = sha256_file(dataset_path)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    model_metadata = {}
    model_metadata_errors = {}
    if args.manage_models:
        for model in args.models:
            try:
                record = client.get_model_record(model)
                model_metadata[model] = summarize_model_record(record)
                if record is None:
                    model_metadata_errors[model] = "LM Studio did not return a matching model record"
            except Exception as exc:
                model_metadata[model] = None
                model_metadata_errors[model] = repr(exc)
    else:
        for model in args.models:
            model_metadata[model] = None
            model_metadata_errors[model] = "unavailable because --no-manage-models was selected"
    index_writer = IndexWriter(args.index)
    batch_id = index_writer.start_batch(
        {
            "provider": "lm-studio",
            "dataset": {
                "path": DATA_PATH,
                "sha256": dataset_sha256,
                "items": len(items),
            },
            "runner": {
                "name": Path(__file__).name,
                "sha256": runner_sha256,
            },
            "protocol": {
                "conditions": ["full", "sharded"],
                "temperature": args.temperature,
                "result_rows_per_model": len(items) * 2,
            },
            "output_base": args.output,
        }
    )
    with ResultWriter(args.output) as out:
        for model in args.models:
            if model in context_length_overrides:
                context_length = context_length_overrides[model]
            elif args.context_length is not None:
                context_length = args.context_length
            else:
                context_length = DEFAULT_CONTEXT_LENGTH_OVERRIDES.get(
                    model, DEFAULT_CONTEXT_LENGTH
                )
            reasoning_effort = reasoning_effort_overrides.get(model, args.reasoning_effort)
            options = {"temperature": args.temperature}
            if reasoning_effort is not None:
                options["reasoning_effort"] = reasoning_effort
            output_path = model_output_path(args.output, model)
            legacy_rows_reused = 0
            if args.resume:
                legacy_rows_reused = seed_model_output_from_legacy(
                    output_path, args.legacy_output, model
                )
            done = load_done_keys(output_path) if args.resume else set()
            print(f"Output: {output_path}")
            total_rows = len(items) * 2
            initial_completed = count_completed(items, done, model)
            completed_rows = initial_completed
            started_monotonic = time.monotonic()
            run_id = uuid.uuid4().hex
            started_at = datetime.now(timezone.utc).isoformat()
            identity = model_metadata.get(model)
            summary_base = {
                "run_id": run_id,
                "model": model,
                "provider": "lm-studio",
                "parameters": identity.get("parameters") if identity else None,
                "quantization": identity.get("quantization") if identity else None,
                "quantization_bits": identity.get("quantization_bits") if identity else None,
                "architecture": identity.get("architecture") if identity else None,
                "selected_variant": identity.get("selected_variant") if identity else None,
                "max_context_length": identity.get("max_context_length") if identity else None,
                "model_metadata_error": model_metadata_errors.get(model),
                "dataset": DATA_PATH,
                "dataset_sha256": dataset_sha256,
                "runner": Path(__file__).name,
                "runner_sha256": runner_sha256,
                "items_requested": len(items),
                "conditions": ["full", "sharded"],
                "temperature": args.temperature,
                "reasoning_effort": reasoning_effort,
                "context_length": context_length,
                "output_path": output_path,
                "started_at": started_at,
            }
            telemetry_base = {
                **summary_base,
                "base_url": args.base_url,
                "management_url": client.management_url if args.manage_models else None,
                "model_metadata": identity,
                "timeout": args.timeout,
                "limit": args.limit,
                "manage_models": args.manage_models,
                "resume": args.resume,
                "legacy_output": args.legacy_output,
                "legacy_rows_reused": legacy_rows_reused,
                "telemetry_path": args.telemetry,
            }
            index_writer.start_run(
                batch_id,
                {
                    "run_id": run_id,
                    "model": {
                        "id": model,
                        "parameters": identity.get("parameters") if identity else None,
                        "quantization": identity.get("quantization") if identity else None,
                        "quantization_bits": identity.get("quantization_bits") if identity else None,
                        "architecture": identity.get("architecture") if identity else None,
                        "selected_variant": identity.get("selected_variant") if identity else None,
                        "max_context_length": identity.get("max_context_length") if identity else None,
                    },
                    "settings": {
                        "reasoning_effort": reasoning_effort,
                        "context_length": context_length,
                    },
                    "output_path": output_path,
                    "started_at": started_at,
                    "status": "running",
                },
            )
            append_telemetry(args.telemetry, {**telemetry_base, "event": "started", "status": "started"})

            def write_progress(status="running"):
                metrics = progress_metrics(
                    total_rows, completed_rows, initial_completed, started_monotonic
                )
                append_telemetry(
                    args.telemetry,
                    {**telemetry_base, **metrics, "event": "progress", "status": status},
                )
                print(
                    f"Progress {model}: {metrics['completed_rows']}/{metrics['total_rows']} "
                    f"complete | {metrics['remaining_rows']} remaining | "
                    f"ETA {format_eta(metrics['eta_seconds'])}"
                )

            def record_turn(condition, item, turn_number, metadata):
                if not metadata:
                    return
                turn_record = {
                    **telemetry_base,
                    "event": "turn",
                    "status": "completed" if not metadata.get("error") else "failed",
                    "condition": condition,
                    "domain": item["domain"],
                    "item_id": item["item_id"],
                    "turn_number": turn_number,
                    **metadata,
                }
                append_telemetry(args.telemetry, turn_record)
                if metadata.get("context_limit_reached"):
                    append_telemetry(
                        args.telemetry,
                        {**turn_record, "event": "context_limit", "status": "context_limit"},
                    )
                    print(
                        f"  CONTEXT LIMIT SIGNAL: {model} | {item['domain']} "
                        f"#{item['item_id']} | {condition} turn {turn_number} | "
                        f"{metadata.get('context_limit_signal')}"
                    )
                elif metadata.get("context_limit_signal"):
                    append_telemetry(
                        args.telemetry,
                        {**turn_record, "event": "context_warning", "status": "near_context_limit"},
                    )

            write_progress()
            if initial_completed >= total_rows:
                write_progress("already_complete")
                index_writer.finish_run(
                    batch_id,
                    run_id,
                    {
                        **progress_metrics(
                            total_rows, completed_rows, initial_completed, started_monotonic
                        ),
                        "status": "already_complete",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "rows_written": 0,
                        "error": None,
                    },
                )
                print(f"Already complete: {model}; skipping model load")
                continue
            instance_id = None
            rows_written = 0
            status = "completed"
            error_message = None
            try:
                if args.manage_models:
                    existing = client.list_loaded_instances()
                    if existing:
                        raise RuntimeError(
                            "Refusing to load another model while LM Studio has "
                            f"loaded instances: {existing}"
                        )
                    print(f"Loading {model}...")
                    instance_id = client.load_model(model, context_length)
                    loaded = client.list_loaded_instances()
                    if loaded != [{"model": model, "instance_id": instance_id}]:
                        raise RuntimeError(
                            "LM Studio did not report exactly the requested model "
                            f"as loaded: {loaded}"
                        )
                    print(f"Loaded {model} as instance {instance_id}")

                model_failed = False
                for item in items:
                    key_full = (item["domain"], item["item_id"], model, "full")
                    key_shard = (item["domain"], item["item_id"], model, "sharded")

                    if key_full not in done:
                        print(f"{model} | {item['domain']} #{item['item_id']} | full")
                        try:
                            t0 = time.time()
                            story = run_full(
                                client,
                                model,
                                item,
                                options,
                                context_length,
                                lambda turn_number, metadata: record_turn(
                                    "full", item, turn_number, metadata
                                ),
                            )
                            row = {
                                "domain": item["domain"], "item_id": item["item_id"], "title": item["title"],
                                "model": model, "condition": "full", "story": story,
                                "seconds": round(time.time() - t0, 1),
                            }
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out.flush()
                            rows_written += 1
                            completed_rows += 1
                            write_progress()
                        except Exception:
                            print(f"  FAILED: {traceback.format_exc()}")
                            model_failed = True
                            status = "failed"
                            error_message = traceback.format_exc()

                    if model_failed:
                        print(f"  SKIPPING remaining items for {model} after the first failure")
                        break

                    if key_shard not in done:
                        print(f"{model} | {item['domain']} #{item['item_id']} | sharded")
                        try:
                            t0 = time.time()
                            story, transcript, _ = run_sharded(
                                client,
                                model,
                                item,
                                options,
                                context_length,
                                lambda turn_number, metadata: record_turn(
                                    "sharded", item, turn_number, metadata
                                ),
                            )
                            row = {
                                "domain": item["domain"], "item_id": item["item_id"], "title": item["title"],
                                "model": model, "condition": "sharded", "story": story,
                                "transcript": transcript, "seconds": round(time.time() - t0, 1),
                            }
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out.flush()
                            rows_written += 1
                            completed_rows += 1
                            write_progress()
                        except Exception:
                            print(f"  FAILED: {traceback.format_exc()}")
                            model_failed = True
                            status = "failed"
                            error_message = traceback.format_exc()

                    if model_failed:
                        print(f"  SKIPPING remaining items for {model} after the first failure")
                        break
            except KeyboardInterrupt:
                status = "stopped_by_user"
                error_message = "KeyboardInterrupt"
                print(f"  STOPPED BY USER: {model}")
                raise
            except Exception as exc:
                status = "failed"
                error_message = repr(exc)
                print(f"  MODEL FAILED: {traceback.format_exc()}")
                if args.manage_models:
                    raise
            finally:
                unload_error = None
                if instance_id is not None:
                    print(f"Unloading {model} instance {instance_id}...")
                    try:
                        client.unload_model(instance_id)
                    except Exception as exc:
                        unload_error = exc
                        status = "unload_failed"
                        error_message = repr(exc)
                        print(f"  UNLOAD FAILED: {traceback.format_exc()}")
                    else:
                        print(f"Unloaded {model} instance {instance_id}")
                index_writer.finish_run(
                    batch_id,
                    run_id,
                    {
                        **progress_metrics(
                            total_rows, completed_rows, initial_completed, started_monotonic
                        ),
                        "status": status,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "rows_written": rows_written,
                        "error": error_message,
                    },
                )
                if unload_error is not None:
                    raise RuntimeError(
                        f"Refusing to continue because LM Studio instance {instance_id} could not be unloaded"
                    ) from unload_error

    print(f"\nDone. Results written to model-specific files using base {args.output}")


if __name__ == "__main__":
    main()
