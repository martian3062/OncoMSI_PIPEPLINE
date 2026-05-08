from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev

from top4_analysis_common import monte_carlo_rank, read_json

ROOT = Path(__file__).resolve().parents[1]
APPROACH_ROOT = ROOT / "ten" / "run-8635c038adcc" / "approaches"
TOP4 = [
    "Approach1-UNI2-h",
    "Approach2-Virchow2",
    "Approach5-H-Optimus-0",
    "Approach6-Midnight-12k",
]


def read_fold_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_column(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        values.append(float(raw))
    return values


def weakest_fold(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    if not rows:
        return None
    valid = [row for row in rows if row.get(key) not in (None, "")]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row[key]))


def summarize() -> dict[str, dict]:
    report: dict[str, dict] = {}
    for name in TOP4:
        metrics_path = APPROACH_ROOT / name / "metrics.json"
        fold_path = APPROACH_ROOT / name / "fold_metrics.csv"
        metrics = read_json(metrics_path)
        rows = read_fold_rows(fold_path)
        monte_path = APPROACH_ROOT / name / "monte_carlo_summary.json"
        monte = read_json(monte_path) if monte_path.exists() else {}
        auroc_summary = monte.get("metrics", {}).get("AUROC", {})

        auroc_values = float_column(rows, "auroc")
        f1_values = float_column(rows, "f1_macro")
        recall_values = float_column(rows, "recall_msi_h")
        rank_metrics = {
            "mean_auroc": metrics.get("mean_auroc"),
            "mean_auprc": metrics.get("mean_auprc"),
            "mean_balanced_accuracy": metrics.get("mean_balanced_accuracy"),
            "mean_recall_msi_h": metrics.get("mean_recall_msi_h"),
            "mean_brier_score": metrics.get("mean_brier_score"),
            "auroc_stability_ratio": auroc_summary.get("stability_ratio"),
        }
        rank_score = monte_carlo_rank(rank_metrics)

        report[name] = {
            "feature_extractor": metrics.get("feature_extractor_used"),
            "mean_auroc": metrics.get("mean_auroc"),
            "mean_auprc": metrics.get("mean_auprc"),
            "mean_balanced_accuracy": metrics.get("mean_balanced_accuracy"),
            "mean_recall_msi_h": metrics.get("mean_recall_msi_h"),
            "mean_specificity": metrics.get("mean_specificity"),
            "mean_brier_score": metrics.get("mean_brier_score"),
            "ranking_score": rank_score,
            "ranking_std_source": "auroc_stability_ratio",
            "fold_auroc_mean": mean(auroc_values) if auroc_values else None,
            "fold_auroc_std": pstdev(auroc_values) if len(auroc_values) > 1 else 0.0,
            "fold_f1_mean": mean(f1_values) if f1_values else None,
            "fold_f1_std": pstdev(f1_values) if len(f1_values) > 1 else 0.0,
            "fold_recall_mean": mean(recall_values) if recall_values else None,
            "fold_recall_std": pstdev(recall_values) if len(recall_values) > 1 else 0.0,
            "auroc_seed_var": auroc_summary.get("seed_var"),
            "auroc_fold_var": auroc_summary.get("fold_var"),
            "auroc_stability_ratio": auroc_summary.get("stability_ratio"),
            "weakest_auroc_fold": weakest_fold(rows, "auroc"),
            "weakest_f1_fold": weakest_fold(rows, "f1_macro"),
            "weakest_recall_fold": weakest_fold(rows, "recall_msi_h"),
            "prediction_files_present_locally": False,
            "external_metrics_present": bool(metrics.get("external_metrics")),
        }
    return report


def main() -> None:
    report = summarize()
    ranked = sorted(report.items(), key=lambda item: item[1]["ranking_score"], reverse=True)
    output = {
        "top4": report,
        "ranking_order": [name for name, _ in ranked],
    }
    print(output)


if __name__ == "__main__":
    main()
