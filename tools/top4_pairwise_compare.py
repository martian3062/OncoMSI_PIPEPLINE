from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from top4_analysis_common import (
    DEFAULT_TOP4,
    delong_roc_test,
    load_fold_metrics,
    load_per_slide_predictions,
    read_json,
    paired_bootstrap_p_value,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--approaches", default=",".join(DEFAULT_TOP4))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def pairwise_fold_test(run_root: Path, left: str, right: str) -> dict:
    left_df = load_fold_metrics(run_root, left)
    right_df = load_fold_metrics(run_root, right)
    shared = left_df.merge(
        right_df,
        on=["repeat", "fold"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if shared.empty:
        raise ValueError(f"No shared repeat/fold rows found for {left} vs {right}")
    return paired_bootstrap_p_value(
        shared["auroc_left"].astype(float).to_numpy(),
        shared["auroc_right"].astype(float).to_numpy(),
    )


def pairwise_delong_test(run_root: Path, left: str, right: str) -> dict:
    left_df = load_per_slide_predictions(run_root, left)
    right_df = load_per_slide_predictions(run_root, right)
    shared = left_df.merge(
        right_df,
        on=["slide_id", "repeat", "fold", "y_true"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if shared.empty:
        raise ValueError(f"No aligned per-slide predictions found for {left} vs {right}")
    return delong_roc_test(
        shared["y_true"].astype(int).to_numpy(),
        shared["predicted_prob_left"].astype(float).to_numpy(),
        shared["predicted_prob_right"].astype(float).to_numpy(),
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    approaches = [item.strip() for item in args.approaches.split(",") if item.strip()]
    output_json = Path(args.output_json) if args.output_json else run_root / "pairwise_compare.json"

    means = {}
    for approach in approaches:
        metrics = read_json(run_root / "approaches" / approach / "metrics.json")
        means[approach] = float(metrics.get("mean_auroc") or 0.0)

    matrix: dict[str, dict] = {}
    for left in approaches:
        matrix[left] = {}
        for right in approaches:
            if left == right:
                matrix[left][right] = {"same_model": True}
                continue
            bootstrap = pairwise_fold_test(run_root, left, right)
            delong = pairwise_delong_test(run_root, left, right)
            matrix[left][right] = {
                "mean_auroc_left": means[left],
                "mean_auroc_right": means[right],
                "bootstrap": bootstrap,
                "delong": delong,
            }

    best = max(approaches, key=lambda item: means[item])
    winner_cluster = [best]
    separable = []
    for approach in approaches:
        if approach == best:
            continue
        pair = matrix[best][approach]
        tied = (
            pair["bootstrap"]["p_value_a_greater_b"] >= args.alpha
            and pair["delong"]["p_value_a_greater_b"] >= args.alpha
        )
        if tied:
            winner_cluster.append(approach)
        else:
            separable.append(approach)

    payload = {
        "bundle_id": run_root.name,
        "alpha": args.alpha,
        "approaches": approaches,
        "mean_auroc": means,
        "pairwise_matrix": matrix,
        "winner_cluster": winner_cluster,
        "separable_from_best": separable,
        "best_by_mean_auroc": best,
    }
    write_json(output_json, payload)
    print(output_json)


if __name__ == "__main__":
    main()
