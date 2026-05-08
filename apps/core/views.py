import json
import re
import shlex
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST
import plotly.graph_objects as go

from apps.approaches.registry import build_approach_slots
from apps.archives.models import BatchArchive
from apps.runs.models import Run
from apps.runs.services import create_run_from_payload, dashboard_summary
from apps.runs.vm_runtime import sync_run_status
from apps.vm.services import default_vm_target, run_shell
from .results_beta import build_results_beta_context
from .services import integration_summary


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
    summary = dashboard_summary()
    live_runs = hydrate_live_runs(sync_remote=False)
    recent_runs = hydrate_recent_runs(sync_remote=False, limit=8)
    milestone_items = build_milestone_items(live_runs)
    initial_tab = "history" if request.GET.get("tab") == "history" else "live"
    history_runs = hydrate_history_runs() if initial_tab == "history" else []
    archive_records = BatchArchive.objects.order_by("-updated_at")[:8] if initial_tab == "history" else []
    context = {
        "summary": summary,
        "approach_slots": build_approach_slots(),
        "chart_json": json.dumps(summary["chart"]),
        "recent_runs": recent_runs,
        "live_runs": live_runs,
        "history_runs": history_runs,
        "archive_records": archive_records,
        "integrations": integration_summary(),
        "initial_tab": initial_tab,
        "milestone_items": milestone_items,
        "milestone_items_json": json.dumps(milestone_items),
    }
    return render(request, "core/dashboard.html", context)


@require_GET
def dashboard_metrics_partial(request: HttpRequest) -> HttpResponse:
    summary = dashboard_summary()
    return render(
        request,
        "core/partials/metrics_panel.html",
        {
            "summary": summary,
            "chart_json": json.dumps(summary["chart"]),
        },
    )


@require_GET
def live_runs_partial(request: HttpRequest) -> HttpResponse:
    live_runs = hydrate_live_runs(sync_remote=True)
    return render(
        request,
        "core/partials/live_runs_panel.html",
        {
            "live_runs": live_runs,
        },
    )


@require_GET
def milestone_ticker_partial(request: HttpRequest) -> HttpResponse:
    live_runs = hydrate_live_runs(sync_remote=False)
    milestone_items = build_milestone_items(live_runs)
    return render(
        request,
        "core/partials/milestone_ticker.html",
        {
            "milestone_items": milestone_items,
            "milestone_items_json": json.dumps(milestone_items),
        },
    )


@require_GET
def history_partial(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "core/partials/history_panel.html",
        {
            "history_runs": hydrate_history_runs(),
            "archive_records": BatchArchive.objects.order_by("-updated_at")[:8],
        },
    )


@require_GET
def history_page(request: HttpRequest) -> HttpResponse:
    summary = dashboard_summary()
    live_runs = hydrate_live_runs(sync_remote=False)
    recent_runs = hydrate_recent_runs(sync_remote=False, limit=12)
    milestone_items = build_milestone_items(live_runs)
    context = {
        "summary": summary,
        "approach_slots": build_approach_slots(),
        "chart_json": json.dumps(summary["chart"]),
        "recent_runs": recent_runs,
        "live_runs": live_runs,
        "history_runs": hydrate_history_runs(limit=30),
        "archive_records": BatchArchive.objects.order_by("-updated_at")[:20],
        "integrations": integration_summary(),
        "initial_tab": "history",
        "milestone_items": milestone_items,
        "milestone_items_json": json.dumps(milestone_items),
    }
    return render(request, "core/dashboard.html", context)


@require_GET
def results_beta_page(request: HttpRequest) -> HttpResponse:
    context = {
        "results_beta": build_results_beta_context(),
    }
    return render(request, "core/results_beta.html", context)


@require_POST
def results_beta_infer(request: HttpRequest) -> HttpResponse:
    return redirect("results-beta-page")


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
