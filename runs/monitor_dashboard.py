"""Serve a local, read-only dashboard for the benchmark results directory.

Run with ``python3 monitor_dashboard.py`` and open http://127.0.0.1:8765.
The server uses only the Python standard library and never changes experiment
artifacts.
"""

import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
INDEX = RESULTS / "index.json"
HOST = "127.0.0.1"
PORT = 8765


def read_json(path, fallback):
    try:
        with Path(path).open(encoding="utf-8") as source:
            return json.load(source)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def read_jsonl(path):
    records = []
    try:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    records.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return records


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds_since(value):
    started = parse_time(value)
    if started is None:
        return None
    return max(0, (datetime.now(timezone.utc) - started).total_seconds())


def duration_label(seconds):
    if seconds is None:
        return "--"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def output_name(run):
    name = run.get("proof", {}).get("run_output_path")
    return Path(name).name if name else ""


def current_status():
    index = read_json(INDEX, {"runs": []})
    runs = index.get("runs", [])
    batch_models = index.get("batch_models") or []
    all_records = read_jsonl(RESULTS / "all_results.jsonl")
    all_ids = {record.get("record_id") for record in all_records}
    anomalies = []
    summaries = []
    active = None

    for run in runs:
        progress = run.get("progress") or {}
        model = run.get("model") or {}
        records = read_jsonl(RESULTS / output_name(run))
        record_ids = {record.get("record_id") for record in records}
        context = model.get("context_length")
        prompt_tokens = [
            record.get("metrics", {}).get("usage", {}).get("prompt_tokens")
            for record in records
        ]
        prompt_tokens = [value for value in prompt_tokens if isinstance(value, int)]
        max_prompt = max(prompt_tokens, default=None)
        incomplete_rows = sum(
            not all(record.get(field) for field in ("model_name", "parameters", "quant_file"))
            for record in records
        )
        non_stop = sum(
            record.get("metrics", {}).get("finish_reason") not in (None, "stop")
            for record in records
        )
        missing_combined = len(record_ids - all_ids)
        if run.get("error"):
            anomalies.append({"level": "error", "message": f"{model.get('id')}: {run['error'].splitlines()[-1]}"})
        if incomplete_rows:
            anomalies.append({"level": "warn", "message": f"{model.get('id')}: {incomplete_rows} rows lack model metadata."})
        if non_stop:
            anomalies.append({"level": "warn", "message": f"{model.get('id')}: {non_stop} rows ended without a normal stop."})
        if missing_combined:
            anomalies.append({"level": "error", "message": f"{model.get('id')}: {missing_combined} rows are absent from all_results.jsonl."})
        if context and max_prompt and max_prompt / context >= 0.85:
            anomalies.append({"level": "warn", "message": f"{model.get('id')}: prompt uses {max_prompt / context:.0%} of context."})

        summary = {
            "model": model.get("id", "unknown"),
            "status": run.get("status", "unknown"),
            "conditions_done": progress.get("completed_conditions", 0),
            "conditions_total": progress.get("total_conditions", 0),
            "records": len(records),
            "context": context,
            "max_prompt": max_prompt,
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "output": output_name(run),
        }
        summaries.append(summary)
        if run.get("status") == "running":
            active = (run, summary)

    active_payload = None
    batch_eta = None
    if active:
        run, summary = active
        elapsed = seconds_since(run.get("started_at"))
        done = summary["conditions_done"]
        total = summary["conditions_total"]
        seconds_per_condition = elapsed / done if elapsed and done else None
        remaining_current = max(0, total - done)
        active_payload = {
            **summary,
            "elapsed_seconds": elapsed,
            "rate_seconds_per_condition": seconds_per_condition,
            "eta_seconds": seconds_per_condition * remaining_current if seconds_per_condition else None,
            "headroom": summary["context"] - summary["max_prompt"] if summary["context"] and summary["max_prompt"] else None,
        }
        completed_models = {item["model"] for item in summaries if item["status"] == "completed"}
        pending_models = [model for model in batch_models if model not in completed_models and model != summary["model"]]
        if seconds_per_condition:
            batch_eta = seconds_per_condition * (remaining_current + len(pending_models) * total)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": active_payload,
        "runs": summaries,
        "batch_models": batch_models,
        "all_records": len(all_records),
        "batch_eta_seconds": batch_eta,
        "anomalies": anomalies,
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SISTER run monitor</title>
<style>
:root{color-scheme:dark;--canvas:#111211;--surface:#1a1b19;--surface-2:#22231f;--line:rgba(241,237,224,.10);--ink:#efece2;--muted:#a9a69b;--faint:#77766f;--ok:#b5d19b;--warn:#e1b76a;--bad:#e17f72;--accent:#d7cf93;--r-sm:5px;--r-md:9px}
*{box-sizing:border-box} body{margin:0;background:var(--canvas);color:var(--ink);font-family:"JetBrains Mono","DejaVu Sans Mono",ui-monospace,monospace;font-size:14px;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:var(--r-sm);min-height:40px;padding:0 12px;cursor:pointer;transition-property:transform,background-color,border-color;transition-duration:140ms}button:active{transform:scale(.96)}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}@media(hover:hover){button:hover{background:var(--surface-2);border-color:rgba(241,237,224,.24)}}
main{max-width:1440px;margin:0 auto;padding:28px clamp(18px,4vw,52px) 48px}.topline{display:flex;align-items:center;justify-content:space-between;gap:18px;border-bottom:1px solid var(--line);padding-bottom:18px}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px}h1{font-size:clamp(24px,3.5vw,40px);letter-spacing:-.04em;line-height:1;margin:0;font-weight:700}.live{display:flex;align-items:center;gap:8px;color:var(--muted);white-space:nowrap}.dot{width:8px;height:8px;border-radius:50%;background:var(--faint)}.dot.live{background:var(--ok);box-shadow:0 0 0 3px rgba(181,209,155,.12)}.dot.failed{background:var(--bad)}
.summary{display:grid;grid-template-columns:2fr repeat(4,minmax(118px,1fr));border-bottom:1px solid var(--line)}.metric{padding:25px 18px 21px;border-left:1px solid var(--line)}.metric:first-child{border-left:0;padding-left:0}.metric .label{color:var(--muted);font-size:11px;letter-spacing:.09em;text-transform:uppercase;margin-bottom:9px}.metric .value{font-size:clamp(17px,2vw,25px);letter-spacing:-.04em;white-space:nowrap}.metric .sub{margin-top:8px;color:var(--faint);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:clip}
.grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(270px,.9fr);gap:36px;margin-top:29px}.section-title{color:var(--muted);font-size:11px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 13px}.run-list{border-top:1px solid var(--line)}.run{display:grid;grid-template-columns:minmax(150px,1fr) 96px 130px 78px;gap:18px;align-items:center;border-bottom:1px solid var(--line);padding:15px 0}.model{min-width:0}.model-name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.file{font-size:11px;color:var(--faint);margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.status{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}.status.running{color:var(--ok)}.status.failed,.status.unload_failed{color:var(--bad)}.num{text-align:right}.num b{display:block;font-size:15px;font-weight:500}.num span{display:block;font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.07em;margin-top:3px}.meter{height:6px;background:var(--surface-2);border-radius:99px;overflow:hidden;margin-top:12px}.meter>i{display:block;height:100%;background:var(--accent);border-radius:inherit}.guard{background:var(--surface);padding:20px;border-radius:var(--r-md);box-shadow:0 8px 25px rgba(0,0,0,.14)}.guard-value{font-size:32px;letter-spacing:-.05em;margin:6px 0}.guard small{color:var(--muted);line-height:1.55;display:block}.headroom{margin:18px 0 0;height:10px;background:#2d2e29;border-radius:99px;overflow:hidden}.headroom i{display:block;height:100%;background:var(--ok);border-radius:inherit}.anomalies{margin-top:28px;border-top:1px solid var(--line)}.notice{padding:13px 0;border-bottom:1px solid var(--line);line-height:1.45;color:var(--muted)}.notice.warn{color:var(--warn)}.notice.error{color:var(--bad)}.fine{color:var(--muted);line-height:1.55;padding:13px 0;border-bottom:1px solid var(--line)}.footer{margin-top:26px;color:var(--faint);font-size:11px;display:flex;justify-content:space-between;gap:16px}.hidden{display:none}
@media(max-width:850px){main{padding-top:20px}.topline{align-items:flex-start}.summary{grid-template-columns:repeat(2,1fr)}.metric:first-child{grid-column:span 2;padding-left:0}.metric:nth-child(3){border-left:0}.grid{grid-template-columns:1fr;gap:26px}.run{grid-template-columns:minmax(130px,1fr) 72px 90px}.run .num:last-child{display:none}.footer{display:block}.footer span{display:block;margin-bottom:7px}}@media(max-width:420px){.topline{gap:10px}.live{font-size:11px}.summary .metric{padding:19px 10px 17px}.metric:first-child{padding-left:0}.run{gap:9px}.model-name{font-size:12px}}
</style>
</head>
<body>
<main>
  <header class="topline"><div><div class="eyebrow">SISTER, experiment console</div><h1>Run monitor</h1></div><div class="live"><span id="dot" class="dot"></span><span id="live-label">Reading results</span><button id="refresh" aria-label="Refresh benchmark status">Refresh</button></div></header>
  <section class="summary" aria-label="Run summary">
    <div class="metric"><div class="label">Current model</div><div class="value" id="model">--</div><div class="sub" id="model-file">Waiting for index.json</div></div>
    <div class="metric"><div class="label">Conditions</div><div class="value" id="conditions">--</div><div class="sub" id="condition-rate">--</div></div>
    <div class="metric"><div class="label">Current ETA</div><div class="value" id="current-eta">--</div><div class="sub" id="elapsed">--</div></div>
    <div class="metric"><div class="label">Batch ETA</div><div class="value" id="batch-eta">--</div><div class="sub" id="all-records">Selected models only</div></div>
    <div class="metric"><div class="label">Context headroom</div><div class="value" id="headroom">--</div><div class="sub" id="context-detail">--</div></div>
  </section>
  <section class="grid"><div><h2 class="section-title">Model progression</h2><div class="run-list" id="runs"></div></div><aside><h2 class="section-title">Context guardrail</h2><div class="guard"><div class="label" id="guard-model">No active model</div><div class="guard-value" id="guard-value">--</div><small id="guard-copy">The monitor will flag prompts at 85% of their configured context limit.</small><div class="headroom"><i id="guard-bar" style="width:0%"></i></div></div><h2 class="section-title" style="margin-top:28px">Anomaly log</h2><div class="anomalies" id="anomalies"></div></aside></section>
  <footer class="footer"><span id="updated">Not yet refreshed</span><span>Local only, read-only, auto-refreshes every 10 seconds</span></footer>
</main>
<script>
const fmt=(s)=>{if(s==null)return'--';s=Math.max(0,Math.round(s));let d=Math.floor(s/86400);s%=86400;let h=Math.floor(s/3600);s%=3600;let m=Math.floor(s/60);return d?`${d}d ${h}h`:h?`${h}h ${m}m`:m?`${m}m`:`${s}s`};
const esc=(v)=>String(v??'--').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function set(id,value){document.getElementById(id).textContent=value}
function render(data){const a=data.active;const dot=document.getElementById('dot');dot.className='dot '+(a?'live':'');set('live-label',a?'Run active':'No active run');set('model',a?a.model:'--');set('model-file',a?a.output:'No active result file');set('conditions',a?`${a.conditions_done} / ${a.conditions_total}`:'--');set('condition-rate',a?.rate_seconds_per_condition?`${a.rate_seconds_per_condition.toFixed(1)} sec / condition`:'Rate will appear after first condition');set('current-eta',fmt(a?.eta_seconds));set('elapsed',a?`elapsed ${fmt(a.elapsed_seconds)}`:'');set('batch-eta',fmt(data.batch_eta_seconds));set('all-records',`${data.batch_models.length} selected, ${data.all_records.toLocaleString()} rows`);set('headroom',a?.headroom!=null?`${a.headroom.toLocaleString()} tok`:'--');set('context-detail',a?.context?`${a.max_prompt?.toLocaleString()||0} / ${a.context.toLocaleString()} used`:'');set('guard-model',a?`${a.model} context use`:'No active model');set('guard-value',a?.context?`${((a.max_prompt||0)/a.context*100).toFixed(1)}%`:'--');set('guard-copy',a?.context?`${(a.max_prompt||0).toLocaleString()} prompt tokens, ${(a.context-(a.max_prompt||0)).toLocaleString()} remain before the configured limit.`:'The monitor flags prompts at 85% of the configured context limit.');document.getElementById('guard-bar').style.width=a?.context?`${Math.min(100,(a.max_prompt||0)/a.context*100)}%`:'0%';
document.getElementById('runs').innerHTML=data.runs.length?data.runs.map(r=>{let pct=r.conditions_total?100*r.conditions_done/r.conditions_total:0;return `<div class="run"><div class="model"><div class="model-name" title="${esc(r.model)}">${esc(r.model)}</div><div class="file">${esc(r.output)}</div><div class="meter"><i style="width:${pct}%"></i></div></div><div class="status ${esc(r.status)}">${esc(r.status)}</div><div class="num"><b>${r.conditions_done} / ${r.conditions_total}</b><span>conditions</span></div><div class="num"><b>${r.records}</b><span>rows</span></div></div>`}).join(''):'<div class="fine">No runs are recorded yet. Start the evaluator, then refresh.</div>';
document.getElementById('anomalies').innerHTML=data.anomalies.length?data.anomalies.map(n=>`<div class="notice ${esc(n.level)}">${esc(n.message)}</div>`).join(''):'<div class="fine">No errors, context-pressure warnings, metadata gaps, or combined-output mismatches.</div>';set('updated',`Updated ${new Date(data.generated_at).toLocaleTimeString()}`)}
async function refresh(){try{const res=await fetch('/api/status',{cache:'no-store'});if(!res.ok)throw Error('status unavailable');render(await res.json())}catch(error){set('live-label','Cannot read results');document.getElementById('dot').className='dot failed';document.getElementById('anomalies').innerHTML='<div class="notice error">The dashboard could not read the local result files. Check the runner and refresh.</div>'}}
document.getElementById('refresh').addEventListener('click',refresh);refresh();setInterval(refresh,10000);
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self.respond_json(current_status())
        elif path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def respond_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"SISTER monitor: http://{HOST}:{PORT}")
    server.serve_forever()
