from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import plotly.graph_objects as go


METRICS = [
    ("mean_auroc", "AUROC"),
    ("mean_f1_macro", "F1 Macro"),
    ("mean_auprc", "AUPRC"),
    ("mean_balanced_accuracy", "Balanced Accuracy"),
    ("mean_recall_msi_h", "MSI-H Recall"),
    ("mean_specificity", "Specificity"),
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dual(fig: go.Figure, html_path: Path, *, width: int = 1500, height: int = 900) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(html_path.with_suffix(".png"), scale=2, width=width, height=height)
    except Exception:
        pass


def safe_float(value: str | float | int | None) -> float:
    if value in (None, "", "NA"):
        return 0.0
    return float(value)


def load_rows(summary_csv: Path) -> list[dict]:
    rows = read_csv(summary_csv)
    for row in rows:
        for key in (
            "folds",
            "epochs",
            "available_bag_slide_count",
            "mean_auroc",
            "mean_f1_macro",
            "mean_auprc",
            "mean_balanced_accuracy",
            "mean_recall_msi_h",
            "mean_specificity",
            "mean_best_threshold",
        ):
            row[key] = safe_float(row.get(key))
    return rows


def add_confusion_estimates(rows: list[dict], *, pos_count: int, neg_count: int) -> list[dict]:
    enriched: list[dict] = []
    prevalence = pos_count / (pos_count + neg_count)
    for row in rows:
        tp = round(row["mean_recall_msi_h"] * pos_count)
        fn = pos_count - tp
        tn = round(row["mean_specificity"] * neg_count)
        fp = neg_count - tn
        new_row = dict(row)
        new_row["class_prevalence"] = prevalence
        new_row["positive_count"] = pos_count
        new_row["negative_count"] = neg_count
        new_row["approx_tp"] = tp
        new_row["approx_fp"] = fp
        new_row["approx_tn"] = tn
        new_row["approx_fn"] = fn
        enriched.append(new_row)
    return enriched


def add_deltas_and_ranks(rows: list[dict]) -> list[dict]:
    best_values = {metric: max(row[metric] for row in rows) for metric, _ in METRICS}
    rank_maps: dict[str, dict[str, int]] = {}
    for metric, _ in METRICS:
        ranked = sorted(rows, key=lambda row: (-row[metric], row["approach_label"]))
        rank_maps[metric] = {row["approach_label"]: index + 1 for index, row in enumerate(ranked)}

    enriched: list[dict] = []
    for row in rows:
        new_row = dict(row)
        for metric, label in METRICS:
            slug = label.lower().replace(" ", "_").replace("-", "_")
            new_row[f"delta_{slug}"] = round(best_values[metric] - row[metric], 6)
            new_row[f"rank_{slug}"] = rank_maps[metric][row["approach_label"]]
        enriched.append(new_row)
    return enriched


def build_clinical_labels(rows: list[dict]) -> tuple[list[dict], dict]:
    screening = max(rows, key=lambda row: (row["mean_recall_msi_h"], row["mean_balanced_accuracy"], row["mean_auroc"]))
    confirmation = max(rows, key=lambda row: (row["mean_specificity"], row["mean_auroc"], row["mean_auprc"]))
    overall = max(rows, key=lambda row: (row["mean_auroc"], row["mean_auprc"], row["mean_f1_macro"]))

    labels = []
    for row in rows:
        role = "generalist"
        if row["approach_label"] == screening["approach_label"]:
            role = "screening"
        elif row["approach_label"] == confirmation["approach_label"]:
            role = "confirmation"
        labels.append(
            {
                "approach_label": row["approach_label"],
                "clinical_preference": role,
                "why": (
                    "highest MSI-H recall with strong balance"
                    if role == "screening"
                    else "highest specificity with strongest confirmation profile"
                    if role == "confirmation"
                    else "high-performing support model"
                ),
            }
        )
    return labels, {
        "screening_model": screening["approach_label"],
        "confirmation_model": confirmation["approach_label"],
        "overall_best_model": overall["approach_label"],
    }


def make_prevalence_chart(out_dir: Path, *, pos_count: int, neg_count: int) -> None:
    total = pos_count + neg_count
    fig = go.Figure(
        data=[
            go.Bar(
                x=["MSI-H", "MSS"],
                y=[pos_count, neg_count],
                text=[f"{pos_count} ({pos_count/total:.1%})", f"{neg_count} ({neg_count/total:.1%})"],
                textposition="outside",
                marker={"color": ["#d64550", "#315fdb"]},
            )
        ]
    )
    fig.update_layout(title="Class Prevalence in the 200-Slide Cohort", template="plotly_white", yaxis_title="Slides")
    write_dual(fig, out_dir / "class_prevalence.html")


def make_confusion_chart(out_dir: Path, rows: list[dict]) -> None:
    labels = [row["approach_label"] for row in rows]
    fig = go.Figure()
    for key, label, color in (
        ("approx_tp", "Approx TP", "#1f9d55"),
        ("approx_fp", "Approx FP", "#f08c00"),
        ("approx_tn", "Approx TN", "#1c7ed6"),
        ("approx_fn", "Approx FN", "#c92a2a"),
    ):
        fig.add_trace(go.Bar(name=label, x=labels, y=[row[key] for row in rows], marker={"color": color}))
    fig.update_layout(title="Approximate Aggregate Confusion Counts", template="plotly_white", barmode="stack", yaxis_title="Slides")
    write_dual(fig, out_dir / "approx_confusion_counts.html", width=1700, height=950)


def make_delta_heatmap(out_dir: Path, rows: list[dict]) -> None:
    delta_keys = [
        ("delta_auroc", "AUROC"),
        ("delta_f1_macro", "F1 Macro"),
        ("delta_auprc", "AUPRC"),
        ("delta_balanced_accuracy", "Balanced Accuracy"),
        ("delta_msi_h_recall", "MSI-H Recall"),
        ("delta_specificity", "Specificity"),
    ]
    z = [[row[key] for key, _ in delta_keys] for row in rows]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=[label for _, label in delta_keys],
                y=[row["approach_label"] for row in rows],
                colorscale="YlOrRd",
                zmin=0.0,
                zmax=max(max(line) for line in z) if z else 0.1,
                text=[[f"{value:.4f}" for value in line] for line in z],
                texttemplate="%{text}",
            )
        ]
    )
    fig.update_layout(title="Delta from Metric Leader", template="plotly_white", height=max(500, len(rows) * 45))
    write_dual(fig, out_dir / "delta_from_best_heatmap.html", width=1600, height=1000)


def make_rank_heatmap(out_dir: Path, rows: list[dict]) -> None:
    rank_keys = [
        ("rank_auroc", "AUROC"),
        ("rank_f1_macro", "F1 Macro"),
        ("rank_auprc", "AUPRC"),
        ("rank_balanced_accuracy", "Balanced Accuracy"),
        ("rank_msi_h_recall", "MSI-H Recall"),
        ("rank_specificity", "Specificity"),
    ]
    z = [[row[key] for key, _ in rank_keys] for row in rows]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=[label for _, label in rank_keys],
                y=[row["approach_label"] for row in rows],
                colorscale="Blues_r",
                zmin=1,
                zmax=len(rows),
                text=[[str(int(value)) for value in line] for line in z],
                texttemplate="%{text}",
            )
        ]
    )
    fig.update_layout(title="Metric Ranking by Approach", template="plotly_white", height=max(500, len(rows) * 45))
    write_dual(fig, out_dir / "metric_rank_heatmap.html", width=1600, height=1000)


def make_clinical_chart(out_dir: Path, rows: list[dict], preference_labels: dict[str, str]) -> None:
    labels = [row["approach_label"] for row in rows]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[row["mean_specificity"] for row in rows],
                y=[row["mean_recall_msi_h"] for row in rows],
                mode="markers+text",
                text=[f"{label}<br>{preference_labels.get(label, 'generalist')}" for label in labels],
                textposition="top center",
                marker={
                    "size": [28 + row["mean_auroc"] * 10 for row in rows],
                    "color": [row["mean_balanced_accuracy"] for row in rows],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Balanced Accuracy"},
                },
            )
        ]
    )
    fig.add_vline(x=max(row["mean_specificity"] for row in rows), line_dash="dot", line_color="#666")
    fig.add_hline(y=max(row["mean_recall_msi_h"] for row in rows), line_dash="dot", line_color="#666")
    fig.update_layout(
        title="Clinical Positioning: Screening vs Confirmation",
        template="plotly_white",
        xaxis_title="Specificity",
        yaxis_title="MSI-H Recall",
    )
    write_dual(fig, out_dir / "clinical_positioning.html", width=1500, height=950)


def write_markdown(out_dir: Path, summary: dict, clinical_rows: list[dict], *, prevalence: float, pos_count: int, neg_count: int) -> None:
    lines = [
        "# Additional Result Analysis",
        "",
        f"- Cohort prevalence: `{pos_count} MSI-H / {neg_count} MSS` (`{prevalence:.2%}` MSI-H)",
        f"- Best overall discriminator: `{summary['overall_best_model']}`",
        f"- Best clinical screening candidate: `{summary['screening_model']}`",
        f"- Best clinical confirmation candidate: `{summary['confirmation_model']}`",
        "",
        "## Clinical Preference Labels",
        "",
        "| Approach | Suggested role | Why |",
        "| --- | --- | --- |",
    ]
    for row in clinical_rows:
        lines.append(f"| {row['approach_label']} | `{row['clinical_preference']}` | {row['why']} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--positive-count", type=int, default=74)
    parser.add_argument("--negative-count", type=int, default=126)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    graph_dir = out_dir / "graphs"

    rows = load_rows(summary_csv)
    rows = add_confusion_estimates(rows, pos_count=args.positive_count, neg_count=args.negative_count)
    rows = add_deltas_and_ranks(rows)
    clinical_rows, summary = build_clinical_labels(rows)
    prevalence = args.positive_count / (args.positive_count + args.negative_count)

    prevalence_rows = [
        {"class_label": "MSI-H", "count": args.positive_count, "rate": round(prevalence, 6)},
        {"class_label": "MSS", "count": args.negative_count, "rate": round(1.0 - prevalence, 6)},
    ]
    write_csv(out_dir / "class_prevalence.csv", prevalence_rows, ["class_label", "count", "rate"])

    confusion_rows = [
        {
            "approach_label": row["approach_label"],
            "approx_tp": row["approx_tp"],
            "approx_fp": row["approx_fp"],
            "approx_tn": row["approx_tn"],
            "approx_fn": row["approx_fn"],
        }
        for row in rows
    ]
    write_csv(out_dir / "approx_confusion_counts.csv", confusion_rows, ["approach_label", "approx_tp", "approx_fp", "approx_tn", "approx_fn"])

    delta_fields = ["approach_label"] + [f"delta_{label.lower().replace(' ', '_').replace('-', '_')}" for _, label in METRICS]
    write_csv(out_dir / "delta_from_best.csv", [{field: row.get(field) for field in delta_fields} for row in rows], delta_fields)

    rank_fields = ["approach_label"] + [f"rank_{label.lower().replace(' ', '_').replace('-', '_')}" for _, label in METRICS]
    write_csv(out_dir / "model_metric_ranks.csv", [{field: row.get(field) for field in rank_fields} for row in rows], rank_fields)

    write_csv(out_dir / "clinical_preference_models.csv", clinical_rows, ["approach_label", "clinical_preference", "why"])

    make_prevalence_chart(graph_dir, pos_count=args.positive_count, neg_count=args.negative_count)
    make_confusion_chart(graph_dir, rows)
    make_delta_heatmap(graph_dir, rows)
    make_rank_heatmap(graph_dir, rows)
    make_clinical_chart(graph_dir, rows, {row["approach_label"]: row["clinical_preference"] for row in clinical_rows})

    write_markdown(out_dir, summary, clinical_rows, prevalence=prevalence, pos_count=args.positive_count, neg_count=args.negative_count)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "positive_count": args.positive_count,
                "negative_count": args.negative_count,
                "outputs": {
                    "class_prevalence_csv": "class_prevalence.csv",
                    "approx_confusion_counts_csv": "approx_confusion_counts.csv",
                    "delta_from_best_csv": "delta_from_best.csv",
                    "model_metric_ranks_csv": "model_metric_ranks.csv",
                    "clinical_preference_models_csv": "clinical_preference_models.csv",
                    "graphs_dir": "graphs",
                },
                "clinical_summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
