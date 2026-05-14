import json
from pathlib import PurePosixPath

import plotly.graph_objects as go

from apps.runs.models import Run
from apps.vm.services import default_vm_target, run_shell


def _bundle_root_for_run(run: Run) -> PurePosixPath | None:
    if run.remote_status_path and run.remote_status_path.startswith("/"):
        return PurePosixPath(run.remote_status_path).parent
    if run.bundle_config_path and run.bundle_config_path.startswith("/"):
        config_root = PurePosixPath(run.bundle_config_path).parent.parent.parent
        return config_root / "automation" / "tcga_slide_triads" / run.run_id
    return None


def build_live_monitor_snapshot(run: Run) -> dict:
    bundle_root = _bundle_root_for_run(run)
    if bundle_root is None:
        return {}

    payload = {
        "bundle_root": str(bundle_root),
        "bundle_config_path": run.bundle_config_path,
        "launch_log_path": run.remote_launch_log_path,
        "run_id": run.run_id,
    }
    remote_script = f"""
import json
import re
import subprocess
from pathlib import Path

payload = json.loads({json.dumps(json.dumps(payload))})
bundle_root = Path(payload["bundle_root"])
bundle_config_path = Path(payload["bundle_config_path"]) if payload.get("bundle_config_path") else None
launch_log_path = Path(payload["launch_log_path"]) if payload.get("launch_log_path") else None
run_id = payload["run_id"]
root_mil = bundle_root / "slideflow_project" / "mil"
approaches_root = bundle_root / "approaches"

epoch_progress_re = re.compile(r"Epoch\\s+(\\d+)/(\\d+)")
epoch_summary_re = re.compile(r"^\\s*(\\d+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+(\\d{{2}}:\\d{{2}})\\s*$")
repeat_fold_re = re.compile(r"repeat_(\\d+)_fold_(\\d+)")

def parse_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        raw = (result.stdout or "").strip()
        if not raw:
            return {{}}
        util, mem_used, mem_total, temp = [item.strip() for item in raw.split(",")]
        return {{
            "utilization": int(util),
            "memory_used": int(mem_used),
            "memory_total": int(mem_total),
            "temperature": int(temp),
        }}
    except Exception:
        return {{}}

def parse_ram():
    try:
        result = subprocess.run(["free", "-m"], text=True, capture_output=True, check=False)
        for line in (result.stdout or "").splitlines():
            if line.startswith("Mem:"):
                parts = [part for part in line.split() if part]
                total = int(parts[1])
                used = int(parts[2])
                available = int(parts[6]) if len(parts) > 6 else int(parts[-1])
                return {{
                    "total_gb": round(total / 1024, 1),
                    "used_gb": round(used / 1024, 1),
                    "available_gb": round(available / 1024, 1),
                }}
    except Exception:
        return {{}}
    return {{}}

def parse_workers():
    workers = []
    try:
        result = subprocess.run(["ps", "-eo", "pid,pcpu,pmem,etimes,cmd"], text=True, capture_output=True, check=False)
        for line in (result.stdout or "").splitlines():
            if "--stage train-approach" not in line or run_id not in line:
                continue
            parts = line.strip().split(None, 4)
            if len(parts) < 5:
                continue
            pid, cpu, mem, etimes, cmd = parts
            label = ""
            if "--approach-label" in cmd:
                label = cmd.split("--approach-label", 1)[1].strip()
            workers.append(
                {{
                    "pid": int(pid),
                    "cpu": float(cpu),
                    "mem": float(mem),
                    "elapsed_seconds": int(etimes),
                    "label": label,
                }}
            )
    except Exception:
        return []
    return workers

def parse_latest_events(log_path: Path, source: str, limit: int = 5):
    if not log_path.exists():
        return []
    interesting = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("[") or "INFO" in text or "Better model found" in text or "slide-level AUC" in text:
            interesting.append({{"source": source, "message": text[-240:]}})
    return interesting[-limit:]

snapshot = {{
    "gpu": parse_gpu(),
    "ram": parse_ram(),
    "workers": parse_workers(),
    "approaches": [],
    "events": [],
}}

specs = []
if bundle_config_path and bundle_config_path.exists():
    try:
        specs = json.loads(bundle_config_path.read_text()).get("specs", [])
    except Exception:
        specs = []

for spec in specs:
    label = spec.get("approach_label") or ""
    experiment_id = str(spec.get("experiment_id") or "").lower()
    tag = f"{{experiment_id}}_{{label.lower()}}"
    dirs = sorted([p.name for p in root_mil.iterdir() if p.is_dir() and tag in p.name]) if root_mil.exists() else []
    latest_dir = dirs[-1] if dirs else ""
    latest_repeat = None
    latest_fold = None
    match = repeat_fold_re.search(latest_dir)
    if match:
        latest_repeat = int(match.group(1))
        latest_fold = int(match.group(2))

    status_payload = {{}}
    status_path = approaches_root / label / "status.json"
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text())
        except Exception:
            status_payload = {{}}

    epoch_rows = []
    current_epoch = None
    total_epochs = None
    runner_log = approaches_root / label / "runner.log"
    if runner_log.exists():
        runner_lines = runner_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in runner_lines:
            progress_match = epoch_progress_re.search(line)
            if progress_match:
                current_epoch = int(progress_match.group(1))
                total_epochs = int(progress_match.group(2))
            summary_match = epoch_summary_re.match(line.strip())
            if summary_match:
                epoch_rows.append(
                    {{
                        "epoch": int(summary_match.group(1)) + 1,
                        "train_loss": float(summary_match.group(2)),
                        "valid_loss": float(summary_match.group(3)),
                        "auroc": float(summary_match.group(4)),
                        "time": summary_match.group(5),
                    }}
                )
        snapshot["events"].extend(parse_latest_events(runner_log, label, limit=4))

    snapshot["approaches"].append(
        {{
            "label": label,
            "state": status_payload.get("state") or "",
            "mil_model": spec.get("mil_model") or "",
            "feature_extractor": spec.get("feature_extractor") or "",
            "n_folds": int(spec.get("n_folds") or 0),
            "n_repeats": int(spec.get("n_repeats") or 0),
            "total_expected": int(spec.get("n_folds") or 0) * int(spec.get("n_repeats") or 0),
            "completed_count": len(dirs),
            "latest_dir": latest_dir,
            "latest_repeat": latest_repeat,
            "latest_fold": latest_fold,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "epoch_rows": epoch_rows[-30:],
        }}
    )

if launch_log_path:
    snapshot["events"].extend(parse_latest_events(launch_log_path, "launch", limit=6))
recovery_log = launch_log_path.parent / f"{{run_id}}-recovery.log" if launch_log_path else None
if recovery_log:
    snapshot["events"].extend(parse_latest_events(recovery_log, "recovery", limit=6))
snapshot["events"] = snapshot["events"][-16:]
print(json.dumps(snapshot))
"""
    command = f"/usr/bin/python3 - <<'PY'\n{remote_script}\nPY"
    result = run_shell(default_vm_target(), command, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        return {"error": (result.stderr or result.stdout or "Unable to load live snapshot").strip()}
    return json.loads(result.stdout)


def build_epoch_chart_json(rows: list[dict], title: str) -> str:
    if not rows:
        return ""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["epoch"] for row in rows],
            y=[row["valid_loss"] for row in rows],
            mode="lines+markers",
            name="Valid loss",
            line={"color": "#0e7be6", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[row["epoch"] for row in rows],
            y=[row["auroc"] for row in rows],
            mode="lines+markers",
            name="AUROC",
            yaxis="y2",
            line={"color": "#1f8a55", "width": 3},
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        margin={"l": 28, "r": 28, "t": 44, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#162334", "size": 11},
        height=280,
        xaxis={"title": "Epoch", "dtick": 1, "gridcolor": "rgba(22,35,52,0.08)"},
        yaxis={"title": "Valid loss", "gridcolor": "rgba(22,35,52,0.08)"},
        yaxis2={"title": "AUROC", "overlaying": "y", "side": "right", "range": [0.0, 1.0]},
        legend={"orientation": "h"},
    )
    return fig.to_json()
