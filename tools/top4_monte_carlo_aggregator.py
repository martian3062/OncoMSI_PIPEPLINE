from __future__ import annotations

import argparse
from pathlib import Path

from top4_analysis_common import (
    DEFAULT_TOP4,
    load_fold_metrics,
    monte_carlo_rank,
    read_json,
    summarize_monte_carlo_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--approaches", default=",".join(DEFAULT_TOP4))
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    approaches = [item.strip() for item in args.approaches.split(",") if item.strip()]
    output_json = Path(args.output_json) if args.output_json else run_root / "top4_montecarlo_aggregate.json"

    rows: list[dict] = []
    for approach in approaches:
        metrics_path = run_root / "approaches" / approach / "metrics.json"
        metrics = read_json(metrics_path)
        fold_df = load_fold_metrics(run_root, approach)
        monte = summarize_monte_carlo_frame(fold_df)
        auroc = monte.get("metrics", {}).get("AUROC", {})
        row = {
            "approach_label": approach,
            "mil_model": metrics.get("mil_model"),
            "feature_extractor_used": metrics.get("feature_extractor_used"),
            "mean_auroc": metrics.get("mean_auroc"),
            "mean_auprc": metrics.get("mean_auprc"),
            "mean_balanced_accuracy": metrics.get("mean_balanced_accuracy"),
            "mean_recall_msi_h": metrics.get("mean_recall_msi_h"),
            "mean_specificity": metrics.get("mean_specificity"),
            "mean_brier_score": metrics.get("mean_brier_score"),
            "mean_best_threshold": metrics.get("mean_best_threshold"),
            "auroc_ci_95_lo": auroc.get("ci_95_lo"),
            "auroc_ci_95_hi": auroc.get("ci_95_hi"),
            "auroc_seed_var": auroc.get("seed_var"),
            "auroc_fold_var": auroc.get("fold_var"),
            "auroc_stability_ratio": auroc.get("stability_ratio"),
        }
        row["ranking_score"] = monte_carlo_rank(row)
        rows.append(row)
        write_json(run_root / "approaches" / approach / "monte_carlo_summary.json", monte)

    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    payload = {
        "bundle_id": run_root.name,
        "approach_count": len(rows),
        "ranking_order": [row["approach_label"] for row in rows],
        "approaches": rows,
    }
    write_json(output_json, payload)
    print(output_json)


if __name__ == "__main__":
    main()
