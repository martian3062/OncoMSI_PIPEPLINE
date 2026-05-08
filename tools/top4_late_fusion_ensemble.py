from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from top4_analysis_common import (
    DEFAULT_TOP4,
    compute_binary_metrics,
    load_per_slide_predictions,
    stacking_oof_predictions,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, help="Bundle root with approaches/ underneath it")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_frames: list[pd.DataFrame] = []
    merge_keys = ["slide_id", "true_label", "y_true", "repeat", "fold", "repeat_seed"]
    for approach in DEFAULT_TOP4:
        merged = load_per_slide_predictions(run_root, approach)
        merged = merged.rename(columns={"predicted_prob": f"score_{approach}"})
        keep_cols = [key for key in merge_keys if key in merged.columns] + [f"score_{approach}"]
        model_frames.append(merged[keep_cols].copy())

    ensemble = model_frames[0]
    for frame in model_frames[1:]:
        join_keys = [key for key in merge_keys if key in ensemble.columns and key in frame.columns]
        ensemble = ensemble.merge(frame, on=join_keys, how="inner")

    score_cols = [col for col in ensemble.columns if col.startswith("score_")]
    ensemble["equal_weight_score"] = ensemble[score_cols].mean(axis=1)
    stacked = stacking_oof_predictions(ensemble.copy(), score_cols)

    equal_metrics = compute_binary_metrics(
        ensemble["y_true"].astype(int).to_numpy(),
        ensemble["equal_weight_score"].astype(float).to_numpy(),
    )
    stacked_ready = stacked.dropna(subset=["stacked_score"]).copy()
    stacked_metrics = compute_binary_metrics(
        stacked_ready["y_true"].astype(int).to_numpy(),
        stacked_ready["stacked_score"].astype(float).to_numpy(),
    )
    metrics = {
        "approaches": DEFAULT_TOP4,
        "equal_weight": equal_metrics,
        "stacking": stacked_metrics,
        "fusion": ["equal_weight_mean", "logistic_stacking_leave_one_fold_out"],
    }

    ensemble[["slide_id", "true_label", "y_true", "repeat", "fold", "equal_weight_score", *score_cols]].to_csv(
        output_dir / "ensemble_predictions.csv",
        index=False,
    )
    stacked_ready[["slide_id", "true_label", "y_true", "repeat", "fold", "stacked_score", *score_cols]].to_csv(
        output_dir / "stacked_predictions.csv",
        index=False,
    )
    write_json(output_dir / "ensemble_metrics.json", metrics)

    print(output_dir / "ensemble_metrics.json")


if __name__ == "__main__":
    main()
