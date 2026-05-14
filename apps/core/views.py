import ctypes
from collections import Counter
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db.models import Q
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
import plotly.graph_objects as go
import torch

from apps.runs.models import Run
from apps.runs.services import create_run_from_payload
from apps.runs.vm_runtime import sync_run_status
from apps.vm.services import default_vm_target, run_shell
from .inference import get_inference_metadata, predict_upload
from .library_data import compact_prediction_history_rows, compact_storage_manifest, delete_prediction_history_entry, find_storage_sample, load_prediction_history, load_storage_manifest
from .predict_jobs import create_prediction_job, create_prediction_job_from_path, get_prediction_job


STATE_LABELS = {
    "matching_annotations": "Matching annotations",
    "downloading_slides": "Downloading slides",
    "extracting_tiles": "Extracting tiles",
    "retrying_tiles": "Retrying tiles",
    "generating_features": "Generating features",
    "prepared": "Prepared",
    "training_parallel": "Training in parallel",
    "completed": "Completed",
    "failed": "Failed",
    "pending": "Pending",
    "spawned": "Queued",
    "training": "Training",
}

STATE_PROGRESS = {
    "matching_annotations": 8,
    "downloading_slides": 18,
    "extracting_tiles": 36,
    "retrying_tiles": 42,
    "generating_features": 56,
    "prepared": 64,
    "training_parallel": 72,
    "training": 72,
    "completed": 100,
    "failed": 100,
}

LIVE_STATES = {
    "matching_annotations",
    "downloading_slides",
    "extracting_tiles",
    "retrying_tiles",
    "generating_features",
    "prepared",
    "training_parallel",
    "training",
    "spawned",
    "pending",
}

METRIC_FIELDS = [
    ("AUROC", "mean_auroc"),
    ("F1 Macro", "mean_f1_macro"),
    ("AUPRC", "mean_auprc"),
    ("Bal Acc", "mean_balanced_accuracy"),
    ("MSI-H Recall", "mean_recall_msi_h"),
    ("Specificity", "mean_specificity"),
]


def _next_app_url(path: str = "/") -> str:
    base = str(getattr(settings, "NEXT_APP_URL", "http://127.0.0.1:3000") or "http://127.0.0.1:3000").rstrip("/")
    clean_path = path if path.startswith("/") else f"/{path}"
    return f"{base}{clean_path}"


def frontend_redirect(request: HttpRequest) -> HttpResponse:
    return redirect(_next_app_url("/"))


def retired_frontend_partial(request: HttpRequest) -> HttpResponse:
    return HttpResponse(status=204)


def _clock_delta_seconds(start: str, end: str) -> int | None:
    def _to_seconds(value: str) -> int | None:
        try:
            hours, minutes, seconds = [int(part) for part in value.split(":", 2)]
        except (TypeError, ValueError):
            return None
        return hours * 3600 + minutes * 60 + seconds

    start_seconds = _to_seconds(start)
    end_seconds = _to_seconds(end)
    if start_seconds is None or end_seconds is None:
        return None
    if end_seconds < start_seconds:
        end_seconds += 24 * 3600
    return end_seconds - start_seconds


def _feature_runtime_snapshot(run: Run) -> dict:
    snapshot = getattr(run, "live_runtime", {}) or {}
    if run.state != "generating_features":
        return {}
    current_extractor = str(snapshot.get("current_extractor") or "").strip()
    bag_counts = snapshot.get("bag_counts") or {}
    selected = int(snapshot.get("selected_slide_count") or run.selected_slide_display or 0)
    if not current_extractor or not selected:
        return snapshot
    bag_key = f"{current_extractor}_{run.tile_px}px_{run.tile_um}um"
    current_bags = int((bag_counts.get(bag_key) or {}).get("pt") or 0)
    sequence = snapshot.get("extractor_sequence") or []
    total_extractors = int(snapshot.get("spec_count") or len(sequence) or len(run.feature_extractor_candidates or []) or 0)
    current_index = 0
    for idx, item in enumerate(sequence, start=1):
        if item.get("extractor") == current_extractor:
            current_index = idx
    completed_extractors = max(0, current_index - 1)
    average_seconds = None
    if len(sequence) >= 2:
        deltas = []
        for previous, current in zip(sequence, sequence[1:]):
            delta = _clock_delta_seconds(previous.get("time", ""), current.get("time", ""))
            if delta and delta > 0:
                deltas.append(delta)
        if deltas:
            average_seconds = int(sum(deltas) / len(deltas))
    fraction_done = min(1.0, current_bags / max(1, selected))
    eta_to_training = None
    if average_seconds and total_extractors:
        remaining_extractors = max(0.0, total_extractors - completed_extractors - fraction_done)
        eta_to_training = timedelta(seconds=int(average_seconds * remaining_extractors))
    snapshot["current_bags"] = current_bags
    snapshot["total_extractors"] = total_extractors
    snapshot["current_index"] = current_index
    snapshot["completed_extractors"] = completed_extractors
    snapshot["eta_to_training"] = eta_to_training
    return snapshot


def format_duration(delta: timedelta | None) -> str:
    if not delta:
        return "n/a"
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def metric_display(value, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _safe_float(value):
    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_to_seconds(value: str) -> int | None:
    text = (value or "").strip()
    if not text or ":" not in text:
        return None
    try:
        minutes, seconds = [int(part) for part in text.split(":", 1)]
    except ValueError:
        return None
    return minutes * 60 + seconds


def _tail_text_lines(path: Path, limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [line for line in lines[-limit:] if str(line).strip()]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def _load_json_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _runtime_batch_status() -> dict:
    runtime_root = Path(settings.BASE_DIR) / "runtime" / "storage_batches"
    if not runtime_root.exists():
        return {
            "has_active_batch": False,
            "batch_name": "",
            "phase": "idle",
            "phase_label": "Idle",
            "current_index": 0,
            "total": 0,
            "completed": 0,
            "percent": 0,
            "current_file": "",
            "queued_batches": [],
            "updated_at": timezone.now().isoformat(),
        }

    try:
        process_result = subprocess.run(
            ["pgrep", "-af", "storage_batch_vm_runner.py"],
            text=True,
            capture_output=True,
            check=False,
        )
        process_lines = [line.strip() for line in process_result.stdout.splitlines() if line.strip()]
    except Exception:
        process_lines = []

    active_batch_name = ""
    for line in process_lines:
        match = re.search(r"--batch-name\s+([A-Za-z0-9._-]+)", line)
        if match:
            active_batch_name = match.group(1).strip()
            break

    batch_dirs = [path for path in runtime_root.iterdir() if path.is_dir()]
    batch_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    batch_map = {path.name: path for path in batch_dirs}

    active_dir = batch_map.get(active_batch_name)
    if active_dir is None:
        for candidate in batch_dirs:
            summary_path = candidate / "summary.json"
            selected_count = _load_json_count(candidate / "selected_rows.json")
            summary_count = 0
            if summary_path.exists():
                try:
                    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary_count = int(summary_payload.get("count") or 0) if isinstance(summary_payload, dict) else 0
                except Exception:
                    summary_count = 0
            if selected_count and summary_count < selected_count:
                active_dir = candidate
                active_batch_name = candidate.name
                break

    queued_batches: list[str] = []
    for candidate in batch_dirs:
        if candidate.name == active_batch_name:
            continue
        selected_count = _load_json_count(candidate / "selected_rows.json")
        result_count = _count_jsonl_rows(candidate / "prediction_results.jsonl")
        if selected_count and result_count < selected_count:
            queued_batches.append(candidate.name)
            continue
        has_autostart = any(candidate.glob("autostart_after_*.sh"))
        has_runner = (candidate / "runner.log").exists()
        if has_autostart and not has_runner:
            queued_batches.append(candidate.name)

    if active_dir is None:
        return {
            "has_active_batch": False,
            "batch_name": "",
            "phase": "idle",
            "phase_label": "Idle",
            "current_index": 0,
            "total": 0,
            "completed": 0,
            "percent": 0,
            "current_file": "",
            "queued_batches": queued_batches,
            "updated_at": timezone.now().isoformat(),
        }

    selected_count = _load_json_count(active_dir / "selected_rows.json")
    completed_count = _count_jsonl_rows(active_dir / "prediction_results.jsonl")
    phase = "preparing"
    phase_label = "Preparing"
    current_index = min(completed_count, selected_count)
    current_file = ""
    log_lines = _tail_text_lines(active_dir / "runner.log")
    for line in reversed(log_lines):
        download_match = re.search(r"\[download\s+(\d+)/(\d+)\]\s+(.+)$", line)
        if download_match:
            phase = "downloading"
            phase_label = "Downloading slides"
            current_index = int(download_match.group(1))
            selected_count = max(selected_count, int(download_match.group(2)))
            current_file = download_match.group(3).strip()
            break
        predict_match = re.search(r"\[(?:predict|skip|timeout|fallback)\s+(\d+)/(\d+)\]\s+(.+?)(?:\s+mode=.*)?$", line)
        if predict_match:
            phase = "predicting"
            phase_label = "Running predictions"
            current_index = int(predict_match.group(1))
            selected_count = max(selected_count, int(predict_match.group(2)))
            current_file = predict_match.group(3).strip()
            break
        done_match = re.search(r"\[done\]\s+batch=([A-Za-z0-9._-]+)\s+slides=(\d+)", line)
        if done_match:
            phase = "completed"
            phase_label = "Completed"
            current_index = int(done_match.group(2))
            selected_count = max(selected_count, current_index)
            current_file = ""
            break

    if phase == "predicting":
        percent = int(round((completed_count / max(1, selected_count)) * 100))
    elif phase == "downloading":
        percent = int(round((current_index / max(1, selected_count)) * 100))
    elif phase == "completed":
        percent = 100
    else:
        percent = int(round((completed_count / max(1, selected_count)) * 100)) if selected_count else 0

    return {
        "has_active_batch": True,
        "batch_name": active_batch_name,
        "phase": phase,
        "phase_label": phase_label,
        "current_index": current_index,
        "total": selected_count,
        "completed": completed_count,
        "percent": max(0, min(100, percent)),
        "current_file": current_file,
        "queued_batches": queued_batches,
        "updated_at": timezone.now().isoformat(),
    }


def _slide_type_for_name(name: str) -> str:
    upper = str(name or "").strip().upper()
    if not upper:
        return ""
    stem = upper[:-4] if upper.endswith(".SVS") else upper
    for part in stem.split("-"):
        if part.startswith(("DX", "TS", "BS", "MS")):
            return part.split(".")[0]
    return ""


def _batch_progress_rows() -> list[dict[str, Any]]:
    runtime_root = Path(settings.BASE_DIR) / "runtime" / "storage_batches"
    if not runtime_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for batch_dir in sorted((path for path in runtime_root.iterdir() if path.is_dir()), key=lambda item: item.name):
        selected_count = _load_json_count(batch_dir / "selected_rows.json")
        completed_count = _count_jsonl_rows(batch_dir / "prediction_results.jsonl")
        watcher = next(batch_dir.glob("autostart_after_*.sh"), None)
        rows.append(
            {
                "batch_name": batch_dir.name,
                "selected_total": selected_count,
                "completed_total": completed_count,
                "status": "completed" if selected_count and completed_count >= selected_count else "queued" if watcher and not (batch_dir / "runner.log").exists() else "running" if (batch_dir / "runner.log").exists() else "idle",
                "has_autostart": bool(watcher),
            }
        )
    return rows


def _history_analysis_payload() -> dict[str, Any]:
    rows = load_prediction_history(limit=None)
    scored_rows = [
        item
        for item in rows
        if str(item.get("label") or "") in {"MSS", "MSI-H"}
        and str(item.get("expected_label") or "") in {"MSS", "MSI-H"}
    ]
    total = len(scored_rows)
    correct = sum(1 for item in scored_rows if str(item.get("label") or "") == str(item.get("expected_label") or ""))
    wrong = total - correct
    false_positive = sum(1 for item in scored_rows if str(item.get("expected_label") or "") == "MSS" and str(item.get("label") or "") == "MSI-H")
    false_negative = sum(1 for item in scored_rows if str(item.get("expected_label") or "") == "MSI-H" and str(item.get("label") or "") == "MSS")
    type_counter = Counter(_slide_type_for_name(str(item.get("uploaded_name") or "")) or "Unknown" for item in scored_rows)
    confidence_counter = Counter(str(item.get("confidence_level") or "Unknown") for item in scored_rows)
    source_groups = sorted({str(item.get("source_group") or "Unknown") for item in scored_rows})
    by_source_group: list[dict[str, Any]] = []
    for group in source_groups:
        group_rows = [item for item in scored_rows if str(item.get("source_group") or "Unknown") == group]
        group_total = len(group_rows)
        group_correct = sum(1 for item in group_rows if str(item.get("label") or "") == str(item.get("expected_label") or ""))
        by_source_group.append(
            {
                "name": group,
                "total": group_total,
                "correct": group_correct,
                "wrong": group_total - group_correct,
                "accuracy": round((group_correct / group_total) if group_total else 0.0, 4),
            }
        )
    by_slide_type: list[dict[str, Any]] = []
    for slide_type in sorted(type_counter):
        type_rows = [item for item in scored_rows if (_slide_type_for_name(str(item.get("uploaded_name") or "")) or "Unknown") == slide_type]
        type_total = len(type_rows)
        type_correct = sum(1 for item in type_rows if str(item.get("label") or "") == str(item.get("expected_label") or ""))
        by_slide_type.append(
            {
                "name": slide_type,
                "total": type_total,
                "correct": type_correct,
                "wrong": type_total - type_correct,
                "accuracy": round((type_correct / type_total) if type_total else 0.0, 4),
            }
        )
    by_pipeline_mode: list[dict[str, Any]] = []
    for mode in sorted({str(item.get("pipeline_mode") or "unknown") for item in scored_rows}):
        mode_rows = [item for item in scored_rows if str(item.get("pipeline_mode") or "unknown") == mode]
        mode_total = len(mode_rows)
        mode_correct = sum(1 for item in mode_rows if str(item.get("label") or "") == str(item.get("expected_label") or ""))
        by_pipeline_mode.append(
            {
                "name": mode,
                "total": mode_total,
                "correct": mode_correct,
                "wrong": mode_total - mode_correct,
                "accuracy": round((mode_correct / mode_total) if mode_total else 0.0, 4),
            }
        )
    recent_wrong = []
    for item in scored_rows:
        if str(item.get("label") or "") == str(item.get("expected_label") or ""):
            continue
        recent_wrong.append(
            {
                "uploaded_name": item.get("uploaded_name"),
                "patient": item.get("patient"),
                "expected_label": item.get("expected_label"),
                "label": item.get("label"),
                "probability": item.get("probability"),
                "confidence_level": item.get("confidence_level"),
                "source_group": item.get("source_group"),
                "saved_at": item.get("saved_at"),
                "slide_type": _slide_type_for_name(str(item.get("uploaded_name") or "")) or "Unknown",
            }
        )
    recent_wrong.sort(key=lambda item: str(item.get("saved_at") or ""), reverse=True)
    inference_meta = get_inference_metadata()
    return {
        "overview": {
            "total_scored": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": round((correct / total) if total else 0.0, 4),
            "false_positive": false_positive,
            "false_negative": false_negative,
            "type_i_error": false_positive,
            "type_ii_error": false_negative,
        },
        "confusion": {
            "MSS_to_MSS": sum(1 for item in scored_rows if str(item.get("expected_label") or "") == "MSS" and str(item.get("label") or "") == "MSS"),
            "MSS_to_MSI_H": false_positive,
            "MSI_H_to_MSI_H": sum(1 for item in scored_rows if str(item.get("expected_label") or "") == "MSI-H" and str(item.get("label") or "") == "MSI-H"),
            "MSI_H_to_MSS": false_negative,
        },
        "by_source_group": by_source_group,
        "by_slide_type": by_slide_type,
        "by_pipeline_mode": by_pipeline_mode,
        "confidence_distribution": [
            {"name": level, "count": confidence_counter[level]}
            for level in ["High", "Medium", "Low", "Unknown"]
            if confidence_counter[level]
        ],
        "batch_progress": _batch_progress_rows(),
        "current_model": {
            "pipeline_mode": inference_meta.get("pipeline_mode"),
            "pipeline_style": inference_meta.get("pipeline_style"),
            "approach_label": inference_meta.get("approach_label"),
            "mil_model": inference_meta.get("mil_model"),
            "encoder_label": (inference_meta.get("encoder") or {}).get("encoder_label"),
            "feature_dim": inference_meta.get("feature_dim"),
            "selected_checkpoint_count": inference_meta.get("selected_checkpoint_count"),
            "available_checkpoints": inference_meta.get("available_checkpoints"),
            "mean_threshold": inference_meta.get("mean_threshold"),
        },
        "recent_wrong_cases": recent_wrong[:12],
        "updated_at": timezone.now().isoformat(),
    }


def _slugish(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def _find_local_approach_dir(run: Run, link) -> Path | None:
    bundle_roots = [
        Path(settings.BASE_DIR) / "ten" / run.run_id / "approaches",
        Path(settings.BASE_DIR) / "iresuts-history" / "hybrid-03" / "approaches",
    ]
    label_no_space = link.approach_template.label.replace(" ", "")
    key_slug = _slugish(link.approach_template.label)
    for root in bundle_roots:
        if not root.exists():
            continue
        direct = root / label_no_space
        if direct.exists():
            return direct
        for candidate in root.rglob("metrics.json"):
            parent = candidate.parent
            normalized_parent = _slugish(str(parent.relative_to(root)))
            if label_no_space in str(parent) or key_slug in normalized_parent:
                return parent
    return None


def _build_run_metric_map(links) -> str:
    chart_links = [link for link in links if any(getattr(link, field, None) is not None for _, field in METRIC_FIELDS)]
    if not chart_links:
        return ""
    z = []
    labels = []
    for link in chart_links:
        labels.append(link.approach_template.label)
        z.append([_safe_float(getattr(link, field, None)) for _, field in METRIC_FIELDS])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=[label for label, _ in METRIC_FIELDS],
                y=labels,
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                text=[[metric_display(value) for value in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="Approach=%{y}<br>Metric=%{x}<br>Value=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 12, "r": 12, "t": 20, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=max(260, 54 * len(labels)),
    )
    return fig.to_json()


def _build_run_metric_map_3d(links) -> str:
    chart_links = [link for link in links if any(getattr(link, field, None) is not None for _, field in METRIC_FIELDS)]
    if not chart_links:
        return ""
    z = []
    labels = []
    for link in chart_links:
        labels.append(link.approach_template.label)
        z.append([_safe_float(getattr(link, field, None)) for _, field in METRIC_FIELDS])
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=list(range(len(METRIC_FIELDS))),
                y=list(range(len(labels))),
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
                customdata=[[metric_display(value) for value in row] for row in z],
                hovertemplate="Approach=%{y}<br>Metric=%{x}<br>Value=%{customdata}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 12, "r": 12, "t": 20, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=max(320, 70 * len(labels)),
        scene={
            "xaxis": {"title": "Metric", "tickvals": list(range(len(METRIC_FIELDS))), "ticktext": [label for label, _ in METRIC_FIELDS]},
            "yaxis": {"title": "Approach", "tickvals": list(range(len(labels))), "ticktext": labels},
            "zaxis": {"title": "Score", "range": [0.0, 1.0]},
            "camera": {"eye": {"x": 1.6, "y": 1.4, "z": 0.8}},
        },
    )
    return fig.to_json()


def _build_tradeoff_map(links) -> str:
    rows = []
    for link in links:
        recall = _safe_float(getattr(link, "mean_recall_msi_h", None))
        specificity = _safe_float(getattr(link, "mean_specificity", None))
        auroc = _safe_float(getattr(link, "mean_auroc", None))
        auprc = _safe_float(getattr(link, "mean_auprc", None))
        if recall is None or specificity is None or auroc is None:
            continue
        rows.append((link.approach_template.label, recall, specificity, auroc, auprc or 0.0))
    if not rows:
        return ""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[item[2] for item in rows],
                y=[item[1] for item in rows],
                mode="markers+text",
                text=[item[0] for item in rows],
                textposition="top center",
                marker={
                    "size": [20 + item[3] * 18 for item in rows],
                    "color": [item[4] for item in rows],
                    "colorscale": "Turbo",
                    "showscale": True,
                    "colorbar": {"title": "AUPRC"},
                    "line": {"color": "rgba(255,255,255,0.55)", "width": 1},
                },
                hovertemplate="Approach=%{text}<br>Specificity=%{x:.3f}<br>Recall=%{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 24, "r": 12, "t": 20, "b": 32},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=320,
        xaxis={"title": "Specificity", "range": [0.7, 1.0], "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"title": "MSI-H Recall", "range": [0.7, 1.0], "gridcolor": "rgba(255,255,255,0.08)"},
    )
    return fig.to_json()


def _build_tradeoff_map_3d(links) -> str:
    rows = []
    for link in links:
        recall = _safe_float(getattr(link, "mean_recall_msi_h", None))
        specificity = _safe_float(getattr(link, "mean_specificity", None))
        auroc = _safe_float(getattr(link, "mean_auroc", None))
        auprc = _safe_float(getattr(link, "mean_auprc", None))
        if recall is None or specificity is None or auroc is None:
            continue
        rows.append((link.approach_template.label, recall, specificity, auroc, auprc or 0.0))
    if not rows:
        return ""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[item[2] for item in rows],
                y=[item[1] for item in rows],
                z=[item[3] for item in rows],
                mode="markers+text",
                text=[item[0] for item in rows],
                textposition="top center",
                marker={
                    "size": [8 + item[4] * 18 for item in rows],
                    "color": [item[4] for item in rows],
                    "colorscale": "Turbo",
                    "showscale": True,
                    "colorbar": {"title": "AUPRC"},
                    "line": {"color": "rgba(255,255,255,0.55)", "width": 1},
                    "opacity": 0.9,
                },
                hovertemplate="Approach=%{text}<br>Specificity=%{x:.3f}<br>Recall=%{y:.3f}<br>AUROC=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 24, "r": 12, "t": 20, "b": 32},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=320,
        scene={
            "xaxis": {"title": "Specificity", "range": [0.7, 1.0]},
            "yaxis": {"title": "MSI-H Recall", "range": [0.7, 1.0]},
            "zaxis": {"title": "AUROC", "range": [0.7, 1.0]},
            "camera": {"eye": {"x": 1.5, "y": 1.3, "z": 0.95}},
        },
    )
    return fig.to_json()


def _build_fold_metric_chart(link) -> str:
    rows = getattr(link, "fold_metrics", None) or []
    if not rows:
        return ""
    folds = list(range(1, len(rows) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=folds, y=[_safe_float(row.get("auroc")) for row in rows], mode="lines+markers", name="AUROC"))
    fig.add_trace(go.Scatter(x=folds, y=[_safe_float(row.get("f1_macro")) for row in rows], mode="lines+markers", name="F1"))
    fig.add_trace(go.Scatter(x=folds, y=[_safe_float(row.get("balanced_accuracy")) for row in rows], mode="lines+markers", name="Bal Acc"))
    fig.add_trace(go.Scatter(x=folds, y=[_safe_float(row.get("recall_msi_h")) for row in rows], mode="lines+markers", name="Recall"))
    fig.update_layout(
        margin={"l": 28, "r": 12, "t": 20, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=280,
        xaxis={"title": "Fold", "dtick": 1, "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"title": "Score", "range": [0.0, 1.0], "gridcolor": "rgba(255,255,255,0.08)"},
        legend={"orientation": "h"},
    )
    return fig.to_json()


def _build_fold_metric_chart_3d(link) -> str:
    rows = getattr(link, "fold_metrics", None) or []
    if not rows:
        return ""
    folds = list(range(1, len(rows) + 1))
    series = [
        ("AUROC", "auroc"),
        ("F1", "f1_macro"),
        ("Bal Acc", "balanced_accuracy"),
        ("Recall", "recall_msi_h"),
    ]
    z = [[_safe_float(row.get(field)) for row in rows] for _, field in series]
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=folds,
                y=list(range(len(series))),
                colorscale="Plasma",
                cmin=0.0,
                cmax=1.0,
                hovertemplate="Fold=%{x}<br>Metric=%{y}<br>Score=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 28, "r": 12, "t": 20, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=280,
        scene={
            "xaxis": {"title": "Fold", "tickvals": folds},
            "yaxis": {"title": "Metric", "tickvals": list(range(len(series))), "ticktext": [label for label, _ in series]},
            "zaxis": {"title": "Score", "range": [0.0, 1.0]},
            "camera": {"eye": {"x": 1.4, "y": 1.2, "z": 0.85}},
        },
    )
    return fig.to_json()


def _build_epoch_chart(run: Run, link) -> tuple[str, str]:
    approach_dir = _find_local_approach_dir(run, link)
    if not approach_dir:
        return "", ""
    runner_log = approach_dir / "runner.log"
    if not runner_log.exists():
        return "", ""
    pattern = re.compile(
        r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d{2}:\d{2})\s*$"
    )
    epochs = []
    cumulative_seconds = 0
    for line in runner_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        epoch_idx = int(match.group(1)) + 1
        train_loss = _safe_float(match.group(2))
        valid_loss = _safe_float(match.group(3))
        auroc = _safe_float(match.group(4))
        epoch_seconds = _duration_to_seconds(match.group(5)) or 0
        cumulative_seconds += epoch_seconds
        epochs.append(
            {
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "auroc": auroc,
                "epoch_seconds": epoch_seconds,
                "elapsed_minutes": cumulative_seconds / 60.0,
            }
        )
    if not epochs:
        return "", ""

    epoch_numbers = [row["epoch"] for row in epochs]
    fig_epoch = go.Figure()
    fig_epoch.add_trace(go.Scatter(x=epoch_numbers, y=[row["train_loss"] for row in epochs], mode="lines+markers", name="Train loss"))
    fig_epoch.add_trace(go.Scatter(x=epoch_numbers, y=[row["valid_loss"] for row in epochs], mode="lines+markers", name="Valid loss"))
    fig_epoch.add_trace(go.Scatter(x=epoch_numbers, y=[row["auroc"] for row in epochs], mode="lines+markers", name="Val AUROC", yaxis="y2"))
    fig_epoch.update_layout(
        margin={"l": 32, "r": 32, "t": 20, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        xaxis={"title": "Epoch", "dtick": 1, "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"title": "Loss", "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis2={"title": "AUROC", "overlaying": "y", "side": "right", "range": [0.0, 1.0]},
        legend={"orientation": "h"},
        height=300,
    )

    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(x=epoch_numbers, y=[row["epoch_seconds"] for row in epochs], name="Epoch seconds"))
    fig_time.add_trace(go.Scatter(x=epoch_numbers, y=[row["elapsed_minutes"] for row in epochs], mode="lines+markers", name="Cumulative minutes", yaxis="y2"))
    fig_time.update_layout(
        margin={"l": 32, "r": 32, "t": 20, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        xaxis={"title": "Epoch", "dtick": 1, "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"title": "Seconds", "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis2={"title": "Elapsed minutes", "overlaying": "y", "side": "right"},
        legend={"orientation": "h"},
        height=260,
    )
    return fig_epoch.to_json(), fig_time.to_json()


def _build_epoch_chart_3d(run: Run, link) -> tuple[str, str]:
    approach_dir = _find_local_approach_dir(run, link)
    if not approach_dir:
        return "", ""
    runner_log = approach_dir / "runner.log"
    if not runner_log.exists():
        return "", ""
    pattern = re.compile(
        r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d{2}:\d{2})\s*$"
    )
    epochs = []
    cumulative_seconds = 0
    for line in runner_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        epoch_idx = int(match.group(1)) + 1
        train_loss = _safe_float(match.group(2))
        valid_loss = _safe_float(match.group(3))
        auroc = _safe_float(match.group(4))
        epoch_seconds = _duration_to_seconds(match.group(5)) or 0
        cumulative_seconds += epoch_seconds
        epochs.append(
            {
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "auroc": auroc,
                "epoch_seconds": epoch_seconds,
                "elapsed_minutes": cumulative_seconds / 60.0,
            }
        )
    if not epochs:
        return "", ""

    epoch_numbers = [row["epoch"] for row in epochs]
    fig_epoch = go.Figure()
    metric_lanes = [("Train loss", 0, "train_loss"), ("Valid loss", 1, "valid_loss"), ("Val AUROC", 2, "auroc")]
    for label, lane, field in metric_lanes:
        fig_epoch.add_trace(
            go.Scatter3d(
                x=epoch_numbers,
                y=[lane] * len(epochs),
                z=[row[field] for row in epochs],
                mode="lines+markers",
                name=label,
                hovertemplate=f"{label}<br>Epoch=%{{x}}<br>Value=%{{z:.3f}}<extra></extra>",
            )
        )
    fig_epoch.update_layout(
        margin={"l": 32, "r": 32, "t": 20, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=300,
        scene={
            "xaxis": {"title": "Epoch", "tickvals": epoch_numbers},
            "yaxis": {"title": "Metric", "tickvals": [0, 1, 2], "ticktext": [item[0] for item in metric_lanes]},
            "zaxis": {"title": "Value"},
            "camera": {"eye": {"x": 1.35, "y": 1.15, "z": 0.9}},
        },
    )

    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter3d(
            x=epoch_numbers,
            y=[row["elapsed_minutes"] for row in epochs],
            z=[row["epoch_seconds"] for row in epochs],
            mode="lines+markers",
            name="Epoch time",
            hovertemplate="Epoch=%{x}<br>Elapsed min=%{y:.2f}<br>Epoch sec=%{z:.0f}<extra></extra>",
        )
    )
    fig_time.add_trace(
        go.Scatter3d(
            x=epoch_numbers,
            y=[row["elapsed_minutes"] for row in epochs],
            z=[row["auroc"] for row in epochs],
            mode="lines+markers",
            name="AUROC vs time",
            hovertemplate="Epoch=%{x}<br>Elapsed min=%{y:.2f}<br>AUROC=%{z:.3f}<extra></extra>",
        )
    )
    fig_time.update_layout(
        margin={"l": 32, "r": 32, "t": 20, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff", "size": 11},
        height=260,
        scene={
            "xaxis": {"title": "Epoch", "tickvals": epoch_numbers},
            "yaxis": {"title": "Elapsed minutes"},
            "zaxis": {"title": "Epoch seconds / AUROC"},
            "camera": {"eye": {"x": 1.35, "y": 1.25, "z": 0.85}},
        },
    )
    return fig_epoch.to_json(), fig_time.to_json()


def infer_run_progress(run: Run) -> tuple[int, str]:
    if run.state in {"training_parallel", "training"} and run.total_link_count:
        progress = 72 + int(round((run.completed_link_count / max(1, run.total_link_count)) * 24))
        detail = f"{run.completed_link_count}/{run.total_link_count} approaches synced"
        return min(96, progress), detail
    feature_runtime = _feature_runtime_snapshot(run)
    if feature_runtime:
        current_index = int(feature_runtime.get("current_index") or 0)
        total_extractors = int(feature_runtime.get("total_extractors") or 0)
        current_bags = int(feature_runtime.get("current_bags") or 0)
        selected = int(feature_runtime.get("selected_slide_count") or run.selected_slide_display or 0)
        if current_index and total_extractors and selected:
            base = 56
            span = 16
            progress = base + int(round(((current_index - 1) + (current_bags / max(1, selected))) / total_extractors * span))
            detail = f"Extractor {current_index}/{total_extractors}: {run.extractor_display} {current_bags}/{selected} bags"
            return min(71, progress), detail
    base = STATE_PROGRESS.get(run.state, 0)
    if run.state == "completed":
        return 100, "Bundle finished"
    if run.state == "failed":
        return 100, "Runner needs recovery"
    return base, run.state_display


def infer_eta_copy(run: Run) -> str:
    if run.state == "completed":
        return "Completed"
    if run.state == "failed":
        return "Needs recovery"

    elapsed = timezone.now() - run.created_at
    feature_runtime = _feature_runtime_snapshot(run)
    if feature_runtime.get("eta_to_training"):
        return f"ETA {format_duration(feature_runtime['eta_to_training'])} to training"
    if run.state in {"training_parallel", "training"} and run.completed_link_count > 0 and run.total_link_count > run.completed_link_count:
        completed = max(1, run.completed_link_count)
        per_approach_seconds = elapsed.total_seconds() / completed
        remaining = run.total_link_count - run.completed_link_count
        eta = timedelta(seconds=per_approach_seconds * remaining)
        return f"ETA {format_duration(eta)}"

    if run.state in {"extracting_tiles", "generating_features"}:
        return "ETA stabilizes after the first synced branch"
    if run.state in {"matching_annotations", "downloading_slides", "prepared"}:
        return "ETA pending first runtime sync"
    return "ETA pending"


def build_scientific_metrics(link) -> list[dict[str, str]]:
    return [
        {"label": "AUROC", "value": metric_display(link.mean_auroc)},
        {"label": "F1 Macro", "value": metric_display(getattr(link, "mean_f1_macro", None))},
        {"label": "AUPRC", "value": metric_display(getattr(link, "mean_auprc", None))},
        {"label": "Bal Acc", "value": metric_display(getattr(link, "mean_balanced_accuracy", None))},
        {"label": "MSI-H Recall", "value": metric_display(getattr(link, "mean_recall_msi_h", None))},
        {"label": "Specificity", "value": metric_display(getattr(link, "mean_specificity", None))},
        {"label": "Precision", "value": metric_display(getattr(link, "mean_precision", None))},
        {"label": "Brier", "value": metric_display(getattr(link, "mean_brier_score", None))},
        {"label": "Threshold", "value": metric_display(getattr(link, "mean_best_threshold", None))},
    ]


def build_stage_copy(run: Run) -> tuple[str, str]:
    if run.sync_error:
        return (
            f"Last known stage is {run.state_display}.",
            run.sync_detail or "Live VM status could not be refreshed, so this headline is showing the latest saved state.",
        )
    if run.state == "matching_annotations":
        return (
            "Matching annotations against the TCGA cohort.",
            f"Preparing the cohort map before downloads start for {run.requested_slide_limit} requested slides.",
        )
    if run.state == "downloading_slides":
        return (
            f"Downloading {run.selected_slide_display} slides from the bucket.",
            f"Current class balance is {run.label_counts_msi_h} MSI-H and {run.label_counts_mss} MSS.",
        )
    if run.state == "extracting_tiles":
        return (
            f"Tiling is running for {run.selected_slide_display} selected slides.",
            f"Slides are downloaded. The runner is now cutting pathology tiles for {run.extractor_display}.",
        )
    if run.state == "retrying_tiles":
        return (
            "Tile extraction is retrying failed slides.",
            "The VM is reprocessing the incomplete slides before feature generation continues.",
        )
    if run.state == "generating_features":
        feature_runtime = _feature_runtime_snapshot(run)
        if feature_runtime:
            current_bags = int(feature_runtime.get("current_bags") or 0)
            selected = int(feature_runtime.get("selected_slide_count") or run.selected_slide_display or 0)
            total_extractors = int(feature_runtime.get("total_extractors") or 0)
            current_index = int(feature_runtime.get("current_index") or 0)
            missing_count = int(feature_runtime.get("missing_bag_count") or 0)
            missing_preview = ", ".join(feature_runtime.get("missing_bag_slides") or [])
            detail = f"Extractor {current_index}/{total_extractors} has emitted {current_bags}/{selected} bags."
            if missing_count:
                detail = f"{detail} {missing_count} slide is currently missing a bag."
                if missing_preview:
                    detail = f"{detail} Latest missing example: {missing_preview}."
            return (
                f"Feature bags are building for {run.extractor_display}.",
                detail,
            )
        return (
            f"Feature bags are building for {run.extractor_display}.",
            f"Tiles are ready. Embeddings are being generated across {run.selected_slide_display} slides.",
        )
    if run.state in {"training_parallel", "training"}:
        if run.active_link is not None:
            return (
                f"{run.completed_link_count}/{run.total_link_count} approaches completed. {run.active_link.approach_template.label} is active.",
                f"{run.running_link_count} approach branches are still running and {run.pending_link_count} are queued.",
            )
        return (
            "Approach training has started.",
            f"{run.running_link_count} approach branches are running across the current bundle.",
        )
    if run.state == "completed" and run.best_link:
        return (
            f"Run completed. {run.best_link.approach_template.label} is leading.",
            f"Best synced metrics are AUROC {run.best_link.mean_auroc_display} and F1 {run.best_link.mean_f1_display}.",
        )
    if run.state == "failed":
        return (
            "Run entered a failure state.",
            "The bundle needs recovery, relaunch, or a deeper VM-side log check.",
        )
    return (run.state_display, f"Latest known extractor is {run.extractor_display}.")


def build_milestone_items(live_runs: list[Run]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for run in live_runs:
        title, detail = build_stage_copy(run)
        meta = f"Run {run.run_id} - Updated {run.last_sync_display}"
        if run.sync_error:
            meta = f"{meta} - VM sync warning"
        items.append(
            {
                "run_id": run.run_id,
                "stage": run.state_display,
                "title": title,
                "detail": detail,
                "meta": meta,
                "variant": "warning" if run.sync_error else "live",
            }
        )

    if not items:
        items.append(
            {
                "run_id": "No live run",
                "stage": "Idle",
                "title": "No active milestone yet.",
                "detail": "Launch a run and this flash strip will start rotating live pipeline updates.",
                "meta": "Waiting for the next bundle",
                "variant": "idle",
            }
        )
    return items


def infer_stage_from_launch_log(run: Run) -> tuple[str | None, str]:
    if not run.remote_launch_log_path.startswith("/"):
        return None, ""

    command = f"tail -n 80 {shlex.quote(run.remote_launch_log_path)}"
    result = run_shell(default_vm_target(), command, timeout=20)
    stdout = result.stdout or ""
    if result.returncode != 0 or not stdout.strip():
        return None, ""

    log_text = stdout.lower()
    if "traceback" in log_text or "runtimeerror" in log_text or "exception" in log_text:
        return "failed", "The launch log shows a failure signature after the last VM update."
    if "using feature extractor:" in log_text or "generating..." in log_text:
        return "generating_features", "The launch log reached feature generation before the live status path disappeared."
    if "finished tile extraction" in log_text or "extracting (" in log_text:
        return "extracting_tiles", "The launch log reached tile extraction before the live status path disappeared."
    if "downloading" in log_text:
        return "downloading_slides", "The launch log still shows slide downloads as the latest visible stage."
    return None, ""


def hydrate_run(run: Run, *, allow_sync_failure: bool, sync_remote: bool) -> Run | None:
    run.sync_error = ""
    run.sync_detail = ""
    if sync_remote and (run.remote_status_path.startswith("/") or run.bundle_config_path.startswith("/")):
        try:
            sync_payload = sync_run_status(run)
            run.live_runtime = sync_payload.get("live_runtime", {})
            run.refresh_from_db()
        except Exception as exc:
            run.sync_error = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            inferred_state, inferred_detail = infer_stage_from_launch_log(run)
            if inferred_state:
                run.state = inferred_state
            run.sync_detail = inferred_detail
            if not allow_sync_failure:
                return None

    run.label_counts_msi_h = (run.label_counts or {}).get("MSI-H", 0)
    run.label_counts_mss = (run.label_counts or {}).get("MSS", 0)
    run.state_display = STATE_LABELS.get(run.state, run.state.replace("_", " ").title())
    run.is_complete = run.state == "completed"
    run.selected_slide_display = run.selected_slide_count or run.requested_slide_limit
    run.extractor_display = (run.feature_extractor_used or "pending").replace(",", ", ")
    run.best_link = None
    run.active_link = None
    run.last_sync_display = timezone.localtime(run.updated_at).strftime("%d %b %I:%M:%S %p")
    run.elapsed_display = format_duration(timezone.now() - run.created_at)
    links = list(run.approach_links.select_related("approach_template").all())
    run.display_links = links
    for link in links:
        link.state_display = STATE_LABELS.get(link.state, link.state.replace("_", " ").title())
        link.mean_auroc_display = f"{link.mean_auroc:.3f}" if link.mean_auroc is not None else ""
        link.mean_f1_display = f"{link.mean_f1_macro:.3f}" if link.mean_f1_macro is not None else ""
        link.mean_auprc_display = metric_display(getattr(link, "mean_auprc", None))
        link.mean_balanced_accuracy_display = metric_display(getattr(link, "mean_balanced_accuracy", None))
        link.mean_recall_display = metric_display(getattr(link, "mean_recall_msi_h", None))
        link.mean_specificity_display = metric_display(getattr(link, "mean_specificity", None))
        link.mean_precision_display = metric_display(getattr(link, "mean_precision", None))
        link.mean_brier_display = metric_display(getattr(link, "mean_brier_score", None))
        link.mean_threshold_display = metric_display(getattr(link, "mean_best_threshold", None))
        link.available_bag_slide_count_display = getattr(link, "available_bag_slide_count", None) or "-"
        link.missing_bag_slide_count = len(getattr(link, "missing_bag_slides", []) or [])
        link.chart_score_pct = max(2, min(100, int(round((link.mean_auroc or 0.0) * 100)))) if link.mean_auroc is not None else 6
        link.fold_chart_json = _build_fold_metric_chart(link)
        link.fold_chart_3d_json = _build_fold_metric_chart_3d(link)
        link.epoch_chart_json = ""
        link.epoch_chart_3d_json = ""
        link.time_chart_json = ""
        link.time_chart_3d_json = ""
    completed_links = [link for link in links if link.mean_auroc is not None]
    if completed_links:
        run.best_link = max(completed_links, key=lambda item: item.mean_auroc or 0.0)
        run.best_link.scientific_metrics = build_scientific_metrics(run.best_link)
        run.best_link.epoch_chart_json, run.best_link.time_chart_json = _build_epoch_chart(run, run.best_link)
        run.best_link.epoch_chart_3d_json, run.best_link.time_chart_3d_json = _build_epoch_chart_3d(run, run.best_link)
    else:
        run.best_link = None
    run.active_link = next((link for link in links if link.state in {"training", "spawned"}), None)
    run.total_link_count = len(links)
    run.completed_link_count = sum(1 for link in links if link.state == "completed")
    run.failed_link_count = sum(1 for link in links if link.state == "failed")
    run.running_link_count = sum(1 for link in links if link.state in {"training", "spawned"})
    run.pending_link_count = max(0, run.total_link_count - run.completed_link_count - run.failed_link_count - run.running_link_count)
    run.progress_percent, run.progress_detail = infer_run_progress(run)
    run.eta_display = infer_eta_copy(run)
    run.external_cohort_count = len(getattr(run, "external_cohorts", []) or [])
    run.stage_title, run.stage_detail = build_stage_copy(run)
    run.metric_map_chart_json = _build_run_metric_map(links)
    run.metric_map_chart_3d_json = _build_run_metric_map_3d(links)
    run.tradeoff_map_chart_json = _build_tradeoff_map(links)
    run.tradeoff_map_chart_3d_json = _build_tradeoff_map_3d(links)
    return run


def dashboard(request: HttpRequest) -> HttpResponse:
    return frontend_redirect(request)


@require_GET
def dashboard_metrics_partial(request: HttpRequest) -> HttpResponse:
    return retired_frontend_partial(request)


@require_GET
def live_runs_partial(request: HttpRequest) -> HttpResponse:
    return retired_frontend_partial(request)


@require_GET
def milestone_ticker_partial(request: HttpRequest) -> HttpResponse:
    return retired_frontend_partial(request)


@require_GET
def history_partial(request: HttpRequest) -> HttpResponse:
    return retired_frontend_partial(request)


@require_GET
def history_page(request: HttpRequest) -> HttpResponse:
    return frontend_redirect(request)


@require_GET
def results_beta_page(request: HttpRequest) -> HttpResponse:
    return redirect(_next_app_url("/"))


def _predict_uploaded_file(uploaded) -> dict:
    if uploaded is None:
        raise ValueError("Choose a slide, image, or trusted feature bag first.")
    temp_path = None
    try:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            for chunk in uploaded.chunks():
                handle.write(chunk)
            temp_path = Path(handle.name)
        result = predict_upload(temp_path)
        result["uploaded_name"] = uploaded.name
        return result
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _bytes_to_gib(value: int) -> float:
    return round(value / float(1024**3), 2)


def _ram_snapshot() -> dict:
    try:
        if sys.platform.startswith("linux"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total = int(page_size * total_pages)
            available = int(page_size * avail_pages)
            return {
                "total_gib": _bytes_to_gib(total),
                "available_gib": _bytes_to_gib(available),
                "used_gib": _bytes_to_gib(max(total - available, 0)),
            }
        if sys.platform.startswith("win"):
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                total = int(state.ullTotalPhys)
                available = int(state.ullAvailPhys)
                return {
                    "total_gib": _bytes_to_gib(total),
                    "available_gib": _bytes_to_gib(available),
                    "used_gib": _bytes_to_gib(max(total - available, 0)),
                }
    except Exception:
        pass
    return {"total_gib": None, "available_gib": None, "used_gib": None}


def _gpu_snapshot() -> dict:
    payload = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "name": "",
        "memory_total_gib": None,
        "memory_reserved_gib": None,
        "memory_allocated_gib": None,
        "driver_line": "",
    }
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            payload["name"] = props.name
            payload["memory_total_gib"] = _bytes_to_gib(int(props.total_memory))
            payload["memory_reserved_gib"] = _bytes_to_gib(int(torch.cuda.memory_reserved(device)))
            payload["memory_allocated_gib"] = _bytes_to_gib(int(torch.cuda.memory_allocated(device)))
        except Exception:
            pass
    try:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            line = (result.stdout or "").strip().splitlines()
            if line:
                payload["driver_line"] = line[0].strip()
    except Exception:
        pass
    return payload


def _system_profile() -> dict:
    from .inference import _pipeline_mode

    disk = shutil.disk_usage(settings.BASE_DIR)
    ram = _ram_snapshot()
    gpu = _gpu_snapshot()
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "platform_summary": platform.platform(),
        "base_dir": str(settings.BASE_DIR),
        "disk_total_gib": _bytes_to_gib(disk.total),
        "disk_free_gib": _bytes_to_gib(disk.free),
        "disk_used_gib": _bytes_to_gib(disk.used),
        "ram": ram,
        "gpu": gpu,
        "prefer_device": getattr(settings, "MSI_PREFER_DEVICE", "auto"),
        "pipeline_mode": _pipeline_mode(),
        "max_inference_tiles": int(getattr(settings, "MSI_MAX_INFERENCE_TILES", 24) or 24),
    }


def _requested_predict_mode(request: HttpRequest) -> str:
    raw = (
        request.GET.get("mode")
        or request.POST.get("mode")
        or request.headers.get("X-Predict-Mode")
        or ""
    ).strip().lower()
    if raw == "fast":
        return "manager2"
    return "manager1"


def _normalize_expected_label(value: str) -> str:
    raw = (value or "").strip().upper()
    if raw in {"MSI", "MSI-H", "MSIH"}:
        return "MSI-H"
    if raw == "MSS":
        return "MSS"
    return ""


def _apply_prediction_cors(request: HttpRequest, response: HttpResponse) -> HttpResponse:
    origin = str(request.headers.get("Origin") or "").rstrip("/")
    allowed_origins = {
        str(getattr(settings, "NEXT_APP_URL", "http://127.0.0.1:3000") or "http://127.0.0.1:3000").rstrip("/"),
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://34.126.112.227:3000",
    }
    if origin and origin in allowed_origins:
        response["Access-Control-Allow-Origin"] = origin
        response["Vary"] = "Origin"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, X-Predict-Mode"
    return response


@require_GET
def predict_metadata_api(request: HttpRequest) -> JsonResponse:
    from .inference import temporary_pipeline_mode

    with temporary_pipeline_mode(_requested_predict_mode(request)):
        return _apply_prediction_cors(request, JsonResponse(
            {
                "inference": get_inference_metadata(),
                "system": _system_profile(),
            }
        ))


@csrf_exempt
@require_POST
def predict_job_create_api(request: HttpRequest) -> JsonResponse:
    uploaded = request.FILES.get("prediction_input")
    if uploaded is None:
        return _apply_prediction_cors(request, JsonResponse({"error": "Choose a slide, image, or trusted feature bag first."}, status=400))
    expected_label = _normalize_expected_label(str(request.POST.get("expected_label") or ""))
    try:
        payload = create_prediction_job(
            uploaded,
            _requested_predict_mode(request),
            history_context={"expected_label": expected_label} if expected_label else None,
        )
    except Exception as exc:
        return _apply_prediction_cors(request, JsonResponse({"error": str(exc).strip() or "Could not create prediction job."}, status=400))
    return _apply_prediction_cors(request, JsonResponse(payload, status=202))


@require_GET
def predict_job_status_api(request: HttpRequest, job_id: str) -> JsonResponse:
    payload = get_prediction_job(job_id)
    if payload is None:
        return _apply_prediction_cors(request, JsonResponse({"error": "Prediction job not found."}, status=404))
    if payload.get("status") == "completed":
        from .inference import temporary_pipeline_mode

        with temporary_pipeline_mode(str(payload.get("pipeline_mode") or "manager1")):
            payload["inference"] = get_inference_metadata()
            payload["system"] = _system_profile()
    return _apply_prediction_cors(request, JsonResponse(payload))


@require_GET
def storage_samples_api(request: HttpRequest) -> JsonResponse:
    payload = load_storage_manifest()
    bucket_name = str(request.GET.get("bucket_name") or "").strip()
    compact = str(request.GET.get("compact") or "").strip() == "1"
    if bucket_name:
        item = next((entry for entry in (payload.get("files", []) or []) if str(entry.get("bucket_name") or "") == bucket_name), None)
        if not item:
            return _apply_prediction_cors(request, JsonResponse({"error": "Stored sample not found."}, status=404))
        return _apply_prediction_cors(request, JsonResponse(item))
    if compact:
        payload = compact_storage_manifest(payload)
    return _apply_prediction_cors(request, JsonResponse(payload))


@require_GET
def batch_status_api(request: HttpRequest) -> JsonResponse:
    return _apply_prediction_cors(request, JsonResponse(_runtime_batch_status()))


@require_GET
def analysis_summary_api(request: HttpRequest) -> JsonResponse:
    return _apply_prediction_cors(request, JsonResponse(_history_analysis_payload()))


@csrf_exempt
def prediction_history_api(request: HttpRequest) -> JsonResponse:
    if request.method not in {"GET", "DELETE"}:
        return _apply_prediction_cors(request, JsonResponse({"error": "Method not allowed."}, status=405))
    if request.method == "DELETE":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        job_id = str(payload.get("job_id") or "").strip()
        saved_at = str(payload.get("saved_at") or "").strip()
        uploaded_name = str(payload.get("uploaded_name") or "").strip()
        deleted = delete_prediction_history_entry(job_id=job_id, saved_at=saved_at, uploaded_name=uploaded_name)
        status = 200 if deleted else 404
        return _apply_prediction_cors(request, JsonResponse({"deleted": deleted}, status=status))
    raw_limit = str(request.GET.get("limit") or "").strip()
    if raw_limit:
        try:
            limit = max(1, int(raw_limit))
        except ValueError:
            limit = None
    else:
        limit = None
    rows = load_prediction_history(limit=limit)
    job_id = str(request.GET.get("job_id") or "").strip()
    saved_at = str(request.GET.get("saved_at") or "").strip()
    uploaded_name = str(request.GET.get("uploaded_name") or "").strip()
    compact = str(request.GET.get("compact") or "").strip() == "1"
    if job_id or (saved_at and uploaded_name):
        for item in rows:
            if job_id and str(item.get("job_id") or "") == job_id:
                return _apply_prediction_cors(request, JsonResponse(item))
            if saved_at and uploaded_name and str(item.get("saved_at") or "") == saved_at and str(item.get("uploaded_name") or "") == uploaded_name:
                return _apply_prediction_cors(request, JsonResponse(item))
        return _apply_prediction_cors(request, JsonResponse({"error": "Saved result not found."}, status=404))
    if compact:
        rows = compact_prediction_history_rows(rows)
    return _apply_prediction_cors(
        request,
        JsonResponse(
            {
                "count": len(rows),
                "items": rows,
            }
        ),
    )


@csrf_exempt
@require_POST
def storage_sample_test_api(request: HttpRequest) -> JsonResponse:
    bucket_name = str(request.POST.get("bucket_name") or "").strip()
    if not bucket_name:
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        bucket_name = str(payload.get("bucket_name") or "").strip()
    if not bucket_name:
        return _apply_prediction_cors(request, JsonResponse({"error": "Choose a stored sample first."}, status=400))
    sample = find_storage_sample(bucket_name)
    if not sample:
        return _apply_prediction_cors(request, JsonResponse({"error": "Stored sample not found."}, status=404))
    sample_path = Path(str(sample.get("local_vm_path") or "")).expanduser()
    try:
        job = create_prediction_job_from_path(
            sample_path,
            bucket_name,
            _requested_predict_mode(request),
            history_context={
                "patient": sample.get("patient"),
                "expected_label": sample.get("msi_status"),
                "source_group": sample.get("source_group"),
                "sample_bucket_name": sample.get("bucket_name"),
            },
        )
    except Exception as exc:
        return _apply_prediction_cors(request, JsonResponse({"error": str(exc).strip() or "Could not queue stored sample."}, status=400))
    return _apply_prediction_cors(request, JsonResponse(job, status=202))


def _monitor_eta(snapshot: dict) -> str:
    approaches = snapshot.get("approaches") or []
    workers = snapshot.get("workers") or []
    if not approaches or not workers:
        return "ETA pending"
    total_done = sum(int(item.get("completed_count") or 0) for item in approaches)
    total_expected = sum(int(item.get("total_expected") or 0) for item in approaches)
    remaining = max(0, total_expected - total_done)
    elapsed_seconds = max(int(item.get("elapsed_seconds") or 0) for item in workers) if workers else 0
    if total_done <= 0 or elapsed_seconds <= 0:
        return "ETA pending"
    pace_per_minute = total_done / max(elapsed_seconds / 60.0, 1e-6)
    eta_minutes = remaining / max(pace_per_minute, 1e-6)
    if eta_minutes >= 60:
        hours = int(eta_minutes // 60)
        minutes = int(round(eta_minutes % 60))
        return f"{hours}h {minutes}m remaining"
    return f"{int(round(eta_minutes))}m remaining"


def _monitor_cards(snapshot: dict) -> dict:
    gpu = snapshot.get("gpu") or {}
    ram = snapshot.get("ram") or {}
    workers = snapshot.get("workers") or []
    cpu_total = sum(float(item.get("cpu") or 0.0) for item in workers)
    cpu_avg = cpu_total / len(workers) if workers else 0.0
    return {
        "gpu_util": gpu.get("utilization"),
        "gpu_memory": f"{gpu.get('memory_used', 0)} / {gpu.get('memory_total', 0)} MiB" if gpu else "-",
        "gpu_temp": gpu.get("temperature"),
        "cpu_total": round(cpu_total, 1) if workers else None,
        "cpu_avg": round(cpu_avg, 1) if workers else None,
        "ram_used": f"{ram.get('used_gb', 0)} / {ram.get('total_gb', 0)} GB" if ram else "-",
        "ram_available": ram.get("available_gb"),
    }


@require_GET
def live_monitor_page(request: HttpRequest) -> HttpResponse:
    return redirect(_next_app_url("/"))


def results_beta_infer(request: HttpRequest) -> HttpResponse:
    return redirect(_next_app_url("/"))


@csrf_exempt
@require_POST
def predict_upload_api(request: HttpRequest) -> JsonResponse:
    uploaded = request.FILES.get("prediction_input")
    if uploaded is None:
        return _apply_prediction_cors(request, JsonResponse({"error": "Choose a slide, image, or trusted feature bag first."}, status=400))
    from .inference import temporary_pipeline_mode

    try:
        with temporary_pipeline_mode(_requested_predict_mode(request)):
            result = _predict_uploaded_file(uploaded)
            result["inference"] = get_inference_metadata()
    except Exception as exc:
        return _apply_prediction_cors(request, JsonResponse({"error": str(exc).strip() or "Prediction failed."}, status=400))
    result["system"] = _system_profile()
    return _apply_prediction_cors(request, JsonResponse(result))


@require_POST
def launch_run(request: HttpRequest) -> HttpResponse:
    external_cohorts_text = request.POST.get("external_cohorts_json", "").strip()
    try:
        external_cohorts = json.loads(external_cohorts_text) if external_cohorts_text else []
    except json.JSONDecodeError:
        external_cohorts = []
    payload = {
        "experiment_name": request.POST.get("experiment_name") or "tcga3-semi-final-200x10f-uni2h-virchow2-gigapath-conch15-hoptimus-midnight-dinov2large-dinov3vitb16-chief-retccl-256tiles",
        "source_uri": request.POST.get("source_uri") or "gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD",
        "annotations_csv": request.POST.get("annotations_csv") or "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/tcga3_vm_annotations.csv",
        "requested_slide_limit": int(request.POST.get("requested_slide_limit") or 200),
        "n_folds": int(request.POST.get("n_folds") or 10),
        "n_repeats": int(request.POST.get("n_repeats") or 1),
        "max_tiles_per_slide": int(request.POST.get("max_tiles_per_slide") or 256),
        "feature_extractor_candidates": [
            item.strip()
            for item in (
                request.POST.get("feature_extractors")
                or "uni2-h,virchow2,prov-gigapath,conchv1_5,h-optimus-0,midnight,dinov2-large,dinov3-vitb16,chief,retccl"
            ).split(",")
            if item.strip()
        ],
        "external_cohorts": external_cohorts,
        "approaches": [slot["key"] for slot in settings.MSI_DEFAULT_APPROACHES],
        "n8n_webhook_url": request.POST.get("n8n_webhook_url") or "",
    }
    create_run_from_payload(payload)
    if request.headers.get("HX-Request") == "true":
        return dashboard_metrics_partial(request)
    return redirect("dashboard")


def hydrate_live_runs(sync_remote: bool = True):
    recent_cutoff = timezone.now() - timedelta(hours=8)
    runs = list(
        Run.objects.filter(state__in=LIVE_STATES, updated_at__gte=recent_cutoff)
        .order_by("-updated_at")[:6]
    )
    hydrated_runs = []
    for run in runs:
        hydrated = hydrate_run(run, allow_sync_failure=True, sync_remote=sync_remote)
        if hydrated and (hydrated.state or hydrated.selected_slide_display or hydrated.best_link):
            hydrated_runs.append(hydrated)
    return hydrated_runs


def hydrate_recent_runs(sync_remote: bool = False, limit: int = 8):
    runs = list(Run.objects.order_by("-updated_at")[:limit])
    hydrated_runs = []
    for run in runs:
        hydrated = hydrate_run(run, allow_sync_failure=True, sync_remote=sync_remote)
        if hydrated is not None:
            hydrated_runs.append(hydrated)
    return hydrated_runs


def hydrate_history_runs(limit: int = 16):
    terminal = Q(state__in=["completed", "failed"])
    runs = list(
        Run.objects.filter(terminal)
        .order_by("-updated_at")[:limit]
    )
    hydrated_runs = []
    for run in runs:
        hydrated = hydrate_run(run, allow_sync_failure=True, sync_remote=False)
        if hydrated is not None:
            hydrated_runs.append(hydrated)
    return hydrated_runs
