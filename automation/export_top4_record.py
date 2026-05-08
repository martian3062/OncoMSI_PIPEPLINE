from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import plotly.graph_objects as go

TOP4_ORDER = [
    "Approach2-Virchow2",
    "Approach6-Midnight-12k",
    "Approach1-UNI2-h",
    "Approach5-H-Optimus-0",
]

METRIC_COLUMNS = [
    ("mean_auroc", "AUROC"),
    ("mean_f1_macro", "F1 Macro"),
    ("mean_auprc", "AUPRC"),
    ("mean_balanced_accuracy", "Balanced Accuracy"),
    ("mean_recall_msi_h", "MSI-H Recall"),
    ("mean_specificity", "Specificity"),
]

EPOCH_PATTERN = re.compile(
    r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d{2}:\d{2})\s*$"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def safe_float(value):
    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def duration_to_seconds(value: str) -> int:
    minutes, seconds = [int(part) for part in value.split(":", 1)]
    return minutes * 60 + seconds


def write_dual(fig: go.Figure, html_path: Path, *, width: int = 1400, height: int = 900) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    fig.write_image(html_path.with_suffix(".png"), scale=2, width=width, height=height)


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
        cumulative_seconds += duration_to_seconds(match.group(5))
        rows.append(
            {
                "approach_label": approach_label,
                "epoch": epoch,
                "train_loss": float(match.group(2)),
                "valid_loss": float(match.group(3)),
                "val_auroc": float(match.group(4)),
                "epoch_seconds": duration_to_seconds(match.group(5)),
                "elapsed_minutes": round(cumulative_seconds / 60.0, 4),
            }
        )
    return rows


def make_leaderboard(rows: list[dict], out: Path) -> None:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[row["approach_label"] for row in rows],
                y=[row["mean_auroc"] for row in rows],
                text=[f'{row["mean_auroc"]:.4f}' for row in rows],
                textposition="outside",
                marker={"color": [row["mean_auprc"] for row in rows], "colorscale": "Turbo", "colorbar": {"title": "AUPRC"}},
            )
        ]
    )
    fig.update_layout(title="Top 4 Leaderboard by AUROC", template="plotly_white", yaxis={"range": [0.0, 1.05]})
    write_dual(fig, out, width=1500, height=900)


def make_grouped_metrics(rows: list[dict], out: Path) -> None:
    fig = go.Figure()
    for key, label in METRIC_COLUMNS[:4]:
        fig.add_trace(go.Bar(name=label, x=[row["approach_label"] for row in rows], y=[row[key] for row in rows]))
    fig.update_layout(title="Top 4 Core Metrics", barmode="group", template="plotly_white", yaxis={"range": [0.0, 1.05]})
    write_dual(fig, out, width=1700, height=900)


def make_metric_heatmap(rows: list[dict], out: Path) -> None:
    z = [[row[key] for key, _ in METRIC_COLUMNS] for row in rows]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=[label for _, label in METRIC_COLUMNS],
                y=[row["approach_label"] for row in rows],
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                text=[[f"{value:.3f}" for value in line] for line in z],
                texttemplate="%{text}",
            )
        ]
    )
    fig.update_layout(title="Top 4 Metric Heatmap", template="plotly_white", height=800)
    write_dual(fig, out, width=1500, height=1000)


def make_metric_surface(rows: list[dict], out: Path) -> None:
    z = [[row[key] for key, _ in METRIC_COLUMNS] for row in rows]
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=list(range(len(METRIC_COLUMNS))),
                y=list(range(len(rows))),
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
            )
        ]
    )
    fig.update_layout(
        title="Top 4 Metric Surface 3D",
        template="plotly_white",
        scene={
            "xaxis": {"title": "Metric", "tickvals": list(range(len(METRIC_COLUMNS))), "ticktext": [label for _, label in METRIC_COLUMNS]},
            "yaxis": {"title": "Approach", "tickvals": list(range(len(rows))), "ticktext": [row["approach_label"] for row in rows]},
            "zaxis": {"title": "Score", "range": [0.0, 1.0]},
        },
    )
    write_dual(fig, out, width=1600, height=1100)


def make_tradeoff(rows: list[dict], out2d: Path, out3d: Path) -> None:
    labels = [row["approach_label"] for row in rows]
    spec = [row["mean_specificity"] for row in rows]
    recall = [row["mean_recall_msi_h"] for row in rows]
    auroc = [row["mean_auroc"] for row in rows]
    auprc = [row["mean_auprc"] for row in rows]

    fig2d = go.Figure(
        data=[
            go.Scatter(
                x=spec,
                y=recall,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"size": [24 + value * 16 for value in auroc], "color": auprc, "colorscale": "Turbo", "showscale": True},
            )
        ]
    )
    fig2d.update_layout(title="Top 4 Recall vs Specificity", template="plotly_white", xaxis_title="Specificity", yaxis_title="MSI-H Recall")
    write_dual(fig2d, out2d, width=1400, height=900)

    fig3d = go.Figure(
        data=[
            go.Scatter3d(
                x=spec,
                y=recall,
                z=auroc,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"size": [10 + value * 14 for value in auprc], "color": auprc, "colorscale": "Turbo", "showscale": True},
            )
        ]
    )
    fig3d.update_layout(
        title="Top 4 Tradeoff Space 3D",
        template="plotly_white",
        scene={"xaxis": {"title": "Specificity"}, "yaxis": {"title": "MSI-H Recall"}, "zaxis": {"title": "AUROC"}},
    )
    write_dual(fig3d, out3d, width=1600, height=1100)


def make_thresholds_and_bags(rows: list[dict], out: Path) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Best Threshold", x=[row["approach_label"] for row in rows], y=[row["mean_best_threshold"] for row in rows]))
    fig.add_trace(go.Bar(name="Bag Slides / 200", x=[row["approach_label"] for row in rows], y=[row["available_bag_slide_count"] / 200.0 for row in rows]))
    fig.update_layout(title="Top 4 Thresholds and Bag Coverage", barmode="group", template="plotly_white", yaxis={"range": [0.0, 1.05]})
    write_dual(fig, out, width=1600, height=900)


def make_stability_chart(rows: list[dict], out: Path) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AUROC Std", x=[row["approach_label"] for row in rows], y=[row["auroc_std"] for row in rows]))
    fig.add_trace(go.Scatter(name="AUROC CI Low", x=[row["approach_label"] for row in rows], y=[row["auroc_ci_low"] for row in rows], mode="lines+markers"))
    fig.add_trace(go.Scatter(name="AUROC CI High", x=[row["approach_label"] for row in rows], y=[row["auroc_ci_high"] for row in rows], mode="lines+markers"))
    fig.update_layout(title="Top 4 Stability Snapshot", template="plotly_white")
    write_dual(fig, out, width=1600, height=900)


def make_confusion_chart(rows: list[dict], out: Path) -> None:
    labels = [row["approach_label"] for row in rows]
    fig = go.Figure()
    for key in ("tp", "tn", "fp", "fn"):
        fig.add_trace(go.Bar(name=key.upper(), x=labels, y=[row[key] for row in rows]))
    fig.update_layout(title="Top 4 Aggregate Confusion Counts", barmode="stack", template="plotly_white")
    write_dual(fig, out, width=1600, height=900)


def make_fold_profiles(fold_rows: list[dict], metric_key: str, title: str, out: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in fold_rows:
        grouped.setdefault(row["approach_label"], []).append(row)
    fig = go.Figure()
    for label in TOP4_ORDER:
        rows = sorted(grouped.get(label, []), key=lambda item: int(item["fold"]))
        if not rows:
            continue
        fig.add_trace(go.Scatter(x=[row["fold"] for row in rows], y=[safe_float(row[metric_key]) for row in rows], mode="lines+markers", name=label))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Fold", yaxis_title=metric_key)
    write_dual(fig, out, width=1500, height=900)


def make_epoch_charts(epoch_rows: list[dict], out_dir: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in epoch_rows:
        grouped.setdefault(row["approach_label"], []).append(row)
    for label in TOP4_ORDER:
        rows = sorted(grouped.get(label, []), key=lambda item: int(item["epoch"]))
        if not rows:
            continue
        stub = safe_name(label)
        epochs = [row["epoch"] for row in rows]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=[row["train_loss"] for row in rows], mode="lines+markers", name="Train loss"))
        fig.add_trace(go.Scatter(x=epochs, y=[row["valid_loss"] for row in rows], mode="lines+markers", name="Valid loss"))
        fig.add_trace(go.Scatter(x=epochs, y=[row["val_auroc"] for row in rows], mode="lines+markers", name="Val AUROC", yaxis="y2"))
        fig.update_layout(
            title=f"{label} Epoch Curves",
            template="plotly_white",
            xaxis_title="Epoch",
            yaxis={"title": "Loss"},
            yaxis2={"title": "AUROC", "overlaying": "y", "side": "right", "range": [0.0, 1.0]},
        )
        write_dual(fig, out_dir / f"{stub}_epoch_curves.html", width=1500, height=900)

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=epochs, y=[row["epoch_seconds"] for row in rows], name="Epoch seconds"))
        fig_t.add_trace(go.Scatter(x=epochs, y=[row["elapsed_minutes"] for row in rows], mode="lines+markers", name="Elapsed minutes", yaxis="y2"))
        fig_t.update_layout(
            title=f"{label} Time Curves",
            template="plotly_white",
            xaxis_title="Epoch",
            yaxis={"title": "Seconds"},
            yaxis2={"title": "Elapsed minutes", "overlaying": "y", "side": "right"},
        )
        write_dual(fig_t, out_dir / f"{stub}_time_curves.html", width=1500, height=900)


def export_top4(bundle_root: Path, output_dir: Path) -> None:
    summary = load_json(bundle_root / "final_summary.json")
    approaches = summary["approaches"]
    selected = [label for label in TOP4_ORDER if label in approaches]
    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    epoch_rows: list[dict] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    epochs_dir = graphs_dir / "epochs"
    artifacts_dir = output_dir / "approach_artifacts"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for label in selected:
        payload = approaches[label]
        confusion = payload.get("aggregate_confusion_matrix", {})
        row = {
            "approach_label": label,
            "feature_extractor_used": payload.get("feature_extractor_used", ""),
            "mil_model": payload.get("mil_model", ""),
            "folds": payload.get("folds", ""),
            "epochs": payload.get("epochs", ""),
            "available_bag_slide_count": payload.get("available_bag_slide_count", 0),
            "auroc_std": payload.get("auroc_std", 0.0),
            "auroc_ci_low": payload.get("auroc_ci_low", 0.0),
            "auroc_ci_high": payload.get("auroc_ci_high", 0.0),
            "tp": confusion.get("tp", 0),
            "tn": confusion.get("tn", 0),
            "fp": confusion.get("fp", 0),
            "fn": confusion.get("fn", 0),
            "external_metrics_present": bool(payload.get("external_metrics")),
        }
        row.update({key: payload.get(key, "") for key, _ in METRIC_COLUMNS})
        row["mean_best_threshold"] = payload.get("mean_best_threshold", "")
        summary_rows.append(row)

        for idx, metric in enumerate(payload.get("fold_metrics", []), start=1):
            fold_row = {"approach_label": label, "fold": idx}
            fold_row.update(metric)
            fold_rows.append(fold_row)

        epoch_rows.extend(parse_epoch_rows(bundle_root / "approaches" / label / "runner.log", label))

        src_dir = bundle_root / "approaches" / label
        if src_dir.exists():
            shutil.copytree(src_dir, artifacts_dir / safe_name(label), dirs_exist_ok=True)

    write_csv(
        output_dir / "top4_summary.csv",
        summary_rows,
        [
            "approach_label",
            "feature_extractor_used",
            "mil_model",
            "folds",
            "epochs",
            "available_bag_slide_count",
            "auroc_std",
            "auroc_ci_low",
            "auroc_ci_high",
            "tp",
            "tn",
            "fp",
            "fn",
            "external_metrics_present",
            *[key for key, _ in METRIC_COLUMNS],
            "mean_best_threshold",
        ],
    )
    write_csv(output_dir / "top4_fold_metrics_long.csv", fold_rows, sorted({key for row in fold_rows for key in row.keys()}))
    write_csv(output_dir / "top4_epoch_history_long.csv", epoch_rows, ["approach_label", "epoch", "train_loss", "valid_loss", "val_auroc", "epoch_seconds", "elapsed_minutes"])

    make_leaderboard(summary_rows, graphs_dir / "leaderboard_top4.html")
    make_grouped_metrics(summary_rows, graphs_dir / "core_metrics_top4.html")
    make_metric_heatmap(summary_rows, graphs_dir / "metric_heatmap_top4.html")
    make_metric_surface(summary_rows, graphs_dir / "metric_surface_top4_3d.html")
    make_tradeoff(summary_rows, graphs_dir / "tradeoff_top4_2d.html", graphs_dir / "tradeoff_top4_3d.html")
    make_thresholds_and_bags(summary_rows, graphs_dir / "thresholds_and_bags_top4.html")
    make_stability_chart(summary_rows, graphs_dir / "stability_top4.html")
    make_confusion_chart(summary_rows, graphs_dir / "confusion_top4.html")
    make_fold_profiles(fold_rows, "auroc", "Top 4 Fold AUROC Profiles", graphs_dir / "fold_auroc_top4.html")
    make_fold_profiles(fold_rows, "f1_macro", "Top 4 Fold F1 Macro Profiles", graphs_dir / "fold_f1_top4.html")
    make_fold_profiles(fold_rows, "balanced_accuracy", "Top 4 Fold Balanced Accuracy Profiles", graphs_dir / "fold_balanced_accuracy_top4.html")
    make_epoch_charts(epoch_rows, epochs_dir)

    manifest = {
        "bundle_root": str(bundle_root),
        "selected_top4": selected,
        "best_overall_10_model": summary.get("best_approach"),
        "output_dir": str(output_dir),
        "files": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_top4(Path(args.bundle_root).resolve(), Path(args.output_dir).resolve())
    print(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
