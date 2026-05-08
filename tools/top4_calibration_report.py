from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from top4_analysis_common import (
    DEFAULT_TOP4,
    bootstrap_mean_ci,
    isotonic_cross_seed_summary,
    load_fold_metrics,
    load_per_slide_predictions,
    reliability_diagram_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--approaches", default=",".join(DEFAULT_TOP4))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    approaches = [item.strip() for item in args.approaches.split(",") if item.strip()]

    for approach in approaches:
        fold_df = load_fold_metrics(run_root, approach)
        pred_df = load_per_slide_predictions(run_root, approach)
        threshold_values = fold_df["best_threshold"].dropna().astype(float).tolist()
        ci_low, ci_high = bootstrap_mean_ci(threshold_values)
        reliability_df = reliability_diagram_frame(
            pred_df["y_true"].astype(int).to_numpy(),
            pred_df["predicted_prob"].astype(float).to_numpy(),
        )
        threshold_df = fold_df[[col for col in ("repeat", "fold", "repeat_seed", "best_threshold") if col in fold_df.columns]].copy()
        threshold_df = threshold_df.rename(columns={"best_threshold": "threshold"})
        threshold_df.to_csv(run_root / "approaches" / approach / "threshold_distribution.csv", index=False)
        reliability_df.to_csv(run_root / "approaches" / approach / "reliability_diagram.csv", index=False)
        payload = {
            "approach_label": approach,
            "threshold_count": int(len(threshold_values)),
            "mean_threshold": float(pd.Series(threshold_values).mean()) if threshold_values else None,
            "std_threshold": float(pd.Series(threshold_values).std(ddof=0)) if threshold_values else None,
            "ci_95_lo": ci_low,
            "ci_95_hi": ci_high,
            "unstable_threshold_flag": bool(threshold_values and pd.Series(threshold_values).std(ddof=0) > 0.05),
            "isotonic_recalibration": isotonic_cross_seed_summary(pred_df),
        }
        write_json(run_root / "approaches" / approach / "threshold_calibration_summary.json", payload)
        print(run_root / "approaches" / approach / "threshold_calibration_summary.json")


if __name__ == "__main__":
    main()
