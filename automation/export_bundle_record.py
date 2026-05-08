from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import plotly.graph_objects as go


METRIC_COLUMNS = [
    ("mean_auroc", "AUROC"),
    ("mean_f1_macro", "F1 Macro"),
    ("mean_auprc", "AUPRC"),
    ("mean_balanced_accuracy", "Balanced Accuracy"),
    ("mean_recall_msi_h", "MSI-H Recall"),
    ("mean_specificity", "Specificity"),
    ("mean_best_threshold", "Best Threshold"),
]

EPOCH_PATTERN = re.compile(
    r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d{2}:\d{2})\s*$"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value):
    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_dual_figure(fig: go.Figure, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(html_path.with_suffix(".png"), scale=2, width=1400, height=900)
    except Exception:
        pass


def duration_to_seconds(value: str) -> int:
    minutes, seconds = [int(part) for part in value.split(":", 1)]
    return minutes * 60 + seconds


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def approach_sort_key(label: str) -> tuple[int, str]:
    match = re.search(r"Approach(\d+)", label)
    if match:
        return int(match.group(1)), label
    return 999, label


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_epoch_rows(runner_log: Path, approach_label: str) -> list[dict]:
    if not runner_log.exists():
        return []
    rows: list[dict] = []
    cumulative_seconds = 0
    for line in runner_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_PATTERN.match(line.strip())
        if not match:
            continue
        epoch = int(match.group(1)) + 1
        train_loss = safe_float(match.group(2))
        valid_loss = safe_float(match.group(3))
        auroc = safe_float(match.group(4))
        epoch_seconds = duration_to_seconds(match.group(5))
        cumulative_seconds += epoch_seconds
        rows.append(
            {
                "approach_label": approach_label,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "val_auroc": auroc,
                "epoch_seconds": epoch_seconds,
                "elapsed_minutes": round(cumulative_seconds / 60.0, 4),
            }
        )
    return rows


def make_leaderboard_chart(rows: list[dict], out_path: Path) -> None:
    labels = [row["approach_label"] for row in rows]
    aurocs = [row["mean_auroc"] for row in rows]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=aurocs,
                marker={
                    "color": [row["mean_f1_macro"] for row in rows],
                    "colorscale": "Turbo",
                    "colorbar": {"title": "F1 Macro"},
                },
                text=[f"{value:.4f}" for value in aurocs],
                textposition="outside",
                hovertemplate="Approach=%{x}<br>AUROC=%{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Approach Leaderboard",
        xaxis_title="Approach",
        yaxis_title="Mean AUROC",
        yaxis={"range": [0.0, 1.05]},
        template="plotly_white",
    )
    write_dual_figure(fig, out_path)


def make_metric_heatmap(rows: list[dict], out_path: Path) -> None:
    z = [[safe_float(row[key]) for key, _ in METRIC_COLUMNS[:-1]] for row in rows]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=[label for _, label in METRIC_COLUMNS[:-1]],
                y=[row["approach_label"] for row in rows],
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                text=[[f"{value:.3f}" if value is not None else "-" for value in line] for line in z],
                texttemplate="%{text}",
            )
        ]
    )
    fig.update_layout(title="Metric Heatmap", template="plotly_white", height=max(400, len(rows) * 42))
    write_dual_figure(fig, out_path)


def make_metric_surface(rows: list[dict], out_path: Path) -> None:
    z = [[safe_float(row[key]) for key, _ in METRIC_COLUMNS[:-1]] for row in rows]
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=list(range(len(METRIC_COLUMNS) - 1)),
                y=list(range(len(rows))),
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
            )
        ]
    )
    fig.update_layout(
        title="Metric Surface 3D",
        template="plotly_white",
        scene={
            "xaxis": {"title": "Metric", "tickvals": list(range(len(METRIC_COLUMNS) - 1)), "ticktext": [label for _, label in METRIC_COLUMNS[:-1]]},
            "yaxis": {"title": "Approach", "tickvals": list(range(len(rows))), "ticktext": [row["approach_label"] for row in rows]},
            "zaxis": {"title": "Score", "range": [0.0, 1.0]},
        },
    )
    write_dual_figure(fig, out_path)


def make_tradeoff_charts(rows: list[dict], out_2d: Path, out_3d: Path) -> None:
    labels = [row["approach_label"] for row in rows]
    recall = [row["mean_recall_msi_h"] for row in rows]
    specificity = [row["mean_specificity"] for row in rows]
    auroc = [row["mean_auroc"] for row in rows]
    auprc = [row["mean_auprc"] for row in rows]

    fig_2d = go.Figure(
        data=[
            go.Scatter(
                x=specificity,
                y=recall,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"size": [20 + value * 18 for value in auroc], "color": auprc, "colorscale": "Turbo", "showscale": True},
            )
        ]
    )
    fig_2d.update_layout(
        title="Recall vs Specificity",
        xaxis_title="Specificity",
        yaxis_title="MSI-H Recall",
        template="plotly_white",
    )
    write_dual_figure(fig_2d, out_2d)

    fig_3d = go.Figure(
        data=[
            go.Scatter3d(
                x=specificity,
                y=recall,
                z=auroc,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"size": [8 + value * 18 for value in auprc], "color": auprc, "colorscale": "Turbo", "showscale": True},
            )
        ]
    )
    fig_3d.update_layout(
        title="Tradeoff Space 3D",
        template="plotly_white",
        scene={
            "xaxis": {"title": "Specificity"},
            "yaxis": {"title": "MSI-H Recall"},
            "zaxis": {"title": "AUROC"},
        },
    )
    write_dual_figure(fig_3d, out_3d)


def make_fold_chart(fold_rows: list[dict], out_path: Path) -> None:
    by_approach: dict[str, list[dict]] = {}
    for row in fold_rows:
        by_approach.setdefault(row["approach_label"], []).append(row)

    fig = go.Figure()
    for approach_label, rows in sorted(by_approach.items(), key=lambda item: approach_sort_key(item[0])):
        rows = sorted(rows, key=lambda item: int(item["fold"]))
        fig.add_trace(
            go.Scatter(
                x=[row["fold"] for row in rows],
                y=[row["auroc"] for row in rows],
                mode="lines+markers",
                name=approach_label,
            )
        )
    fig.update_layout(
        title="Fold AUROC Profiles",
        xaxis_title="Fold",
        yaxis_title="AUROC",
        yaxis={"range": [0.0, 1.05]},
        template="plotly_white",
    )
    write_dual_figure(fig, out_path)


def make_epoch_charts(epoch_rows: list[dict], out_dir: Path) -> None:
    by_approach: dict[str, list[dict]] = {}
    for row in epoch_rows:
        by_approach.setdefault(row["approach_label"], []).append(row)

    for approach_label, rows in sorted(by_approach.items(), key=lambda item: approach_sort_key(item[0])):
        rows = sorted(rows, key=lambda item: int(item["epoch"]))
        epochs = [row["epoch"] for row in rows]
        file_stub = safe_name(approach_label)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=[row["train_loss"] for row in rows], mode="lines+markers", name="Train loss"))
        fig.add_trace(go.Scatter(x=epochs, y=[row["valid_loss"] for row in rows], mode="lines+markers", name="Valid loss"))
        fig.add_trace(go.Scatter(x=epochs, y=[row["val_auroc"] for row in rows], mode="lines+markers", name="Val AUROC", yaxis="y2"))
        fig.update_layout(
            title=f"{approach_label} Epoch Curves",
            xaxis_title="Epoch",
            yaxis={"title": "Loss"},
            yaxis2={"title": "AUROC", "overlaying": "y", "side": "right", "range": [0.0, 1.0]},
            template="plotly_white",
        )
        write_dual_figure(fig, out_dir / f"{file_stub}_epoch_curves.html")

        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(x=epochs, y=[row["epoch_seconds"] for row in rows], name="Epoch seconds"))
        fig_time.add_trace(go.Scatter(x=epochs, y=[row["elapsed_minutes"] for row in rows], mode="lines+markers", name="Elapsed minutes", yaxis="y2"))
        fig_time.update_layout(
            title=f"{approach_label} Time Curves",
            xaxis_title="Epoch",
            yaxis={"title": "Seconds"},
            yaxis2={"title": "Elapsed minutes", "overlaying": "y", "side": "right"},
            template="plotly_white",
        )
        write_dual_figure(fig_time, out_dir / f"{file_stub}_time_curves.html")


def export_bundle(bundle_root: Path, output_dir: Path) -> None:
    summary = load_json(bundle_root / "final_summary.json")
    approaches = summary.get("approaches", {})
    sorted_labels = sorted(approaches, key=approach_sort_key)

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    epoch_rows: list[dict] = []

    for label in sorted_labels:
        payload = approaches[label]
        summary_row = {
            "approach_label": label,
            "feature_extractor_used": payload.get("feature_extractor_used", ""),
            "mil_model": payload.get("mil_model", ""),
            "folds": payload.get("folds", ""),
            "epochs": payload.get("epochs", ""),
            "available_bag_slide_count": payload.get("available_bag_slide_count", ""),
            "external_metrics_present": bool(payload.get("external_metrics")),
        }
        for key, _label in METRIC_COLUMNS:
            summary_row[key] = payload.get(key, "")
        summary_rows.append(summary_row)

        for idx, fold_metric in enumerate(payload.get("fold_metrics", []), start=1):
            row = {"approach_label": label, "fold": idx}
            row.update(fold_metric)
            fold_rows.append(row)

        runner_log = bundle_root / "approaches" / label / "runner.log"
        epoch_rows.extend(parse_epoch_rows(runner_log, label))

    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    epoch_graphs_dir = graphs_dir / "epochs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    epoch_graphs_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "approach_summary.csv",
        summary_rows,
        [
            "approach_label",
            "feature_extractor_used",
            "mil_model",
            "folds",
            "epochs",
            "available_bag_slide_count",
            "external_metrics_present",
            *[key for key, _ in METRIC_COLUMNS],
        ],
    )

    if fold_rows:
        fold_fields = sorted({key for row in fold_rows for key in row.keys()}, key=lambda item: (item not in {"approach_label", "fold"}, item))
        write_csv(output_dir / "fold_metrics_long.csv", fold_rows, fold_fields)

    if epoch_rows:
        write_csv(
            output_dir / "epoch_history_long.csv",
            epoch_rows,
            ["approach_label", "epoch", "train_loss", "valid_loss", "val_auroc", "epoch_seconds", "elapsed_minutes"],
        )

    make_leaderboard_chart(summary_rows, graphs_dir / "leaderboard.html")
    make_metric_heatmap(summary_rows, graphs_dir / "metric_heatmap.html")
    make_metric_surface(summary_rows, graphs_dir / "metric_surface_3d.html")
    make_tradeoff_charts(summary_rows, graphs_dir / "tradeoff_2d.html", graphs_dir / "tradeoff_3d.html")
    if fold_rows:
        make_fold_chart(fold_rows, graphs_dir / "fold_auroc_profiles.html")
    if epoch_rows:
        make_epoch_charts(epoch_rows, epoch_graphs_dir)

    manifest = {
        "bundle_root": str(bundle_root),
        "output_dir": str(output_dir),
        "selected_slide_count": summary.get("selected_slide_count"),
        "label_counts": summary.get("label_counts"),
        "best_approach": summary.get("best_approach"),
        "approach_count": len(summary_rows),
        "graphs": sorted(str(path.relative_to(output_dir)) for path in graphs_dir.rglob("*") if path.suffix.lower() in {".html", ".png"}),
        "files": {
            "approach_summary_csv": "approach_summary.csv",
            "fold_metrics_long_csv": "fold_metrics_long.csv" if fold_rows else "",
            "epoch_history_long_csv": "epoch_history_long.csv" if epoch_rows else "",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete local record and graph set for a saved result bundle.")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    if not bundle_root.exists():
        raise FileNotFoundError(bundle_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else bundle_root / "complete_record"
    export_bundle(bundle_root, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
