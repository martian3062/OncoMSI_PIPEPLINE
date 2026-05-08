from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POS_LABEL = "MSI-H"
NEG_LABEL = "MSS"
DEFAULT_TOP4 = [
    "Approach1-UNI2-h",
    "Approach2-Virchow2",
    "Approach5-H-Optimus-0",
    "Approach6-Midnight-12k",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns were present: {list(candidates)}")


def parse_repeat_fold(path_like: str | Path) -> tuple[int | None, int | None]:
    match = re.search(r"_repeat_(\d+)_fold_(\d+)", str(path_like))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def normalize_binary_labels(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    if pd.api.types.is_numeric_dtype(series):
        values = series.fillna(0).astype(int).to_numpy()
        labels = [POS_LABEL if int(value) == 1 else NEG_LABEL for value in values]
        return values, labels

    raw = series.astype(str).str.strip()
    values = raw.str.upper().eq(POS_LABEL.upper()).astype(int).to_numpy()
    labels = [POS_LABEL if int(value) == 1 else NEG_LABEL for value in values]
    return values, labels


def normalize_prediction_frame(path: Path) -> pd.DataFrame:
    df = read_table(path)
    slide_col = first_present(df, ["slide_id", "slide", "slide_name", "patient", "sample_id", "case_id"])
    score_col = first_present(df, ["predicted_prob", "y_pred1", "score", "prob_msi_h", "probability", "prediction"])

    if "true_label" in df.columns:
        label_col = "true_label"
    elif "label" in df.columns:
        label_col = "label"
    elif "msi_status" in df.columns:
        label_col = "msi_status"
    elif "y_true" in df.columns:
        label_col = "y_true"
    else:
        raise KeyError(f"No label column found in {path}")

    y_true, label_names = normalize_binary_labels(df[label_col])
    repeat, fold = parse_repeat_fold(path)
    out = pd.DataFrame(
        {
            "slide_id": df[slide_col].astype(str),
            "true_label": label_names,
            "y_true": y_true.astype(int),
            "predicted_prob": df[score_col].astype(float),
            "repeat": df["repeat"].astype(int) if "repeat" in df.columns else repeat,
            "fold": df["fold"].astype(int) if "fold" in df.columns else fold,
            "repeat_seed": df["repeat_seed"].astype(int) if "repeat_seed" in df.columns else None,
            "prediction_file": str(path),
        }
    )
    return out


def bootstrap_mean_ci(values: Iterable[float], *, seed: int = 310, rounds: int = 10000) -> tuple[float | None, float | None]:
    clean = np.array([float(value) for value in values if value is not None], dtype=float)
    if clean.size == 0:
        return None, None
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    sample_means = np.array(
        [rng.choice(clean, size=clean.size, replace=True).mean() for _ in range(rounds)],
        dtype=float,
    )
    return float(np.percentile(sample_means, 2.5)), float(np.percentile(sample_means, 97.5))


def summarize_metric_distribution(
    metrics_df: pd.DataFrame,
    column: str,
    *,
    seed: int = 310,
) -> dict[str, Any]:
    values = metrics_df[column].dropna().astype(float)
    if values.empty:
        return {}
    ci_low, ci_high = bootstrap_mean_ci(values.tolist(), seed=seed)
    seed_rows: list[dict[str, Any]] = []
    fold_vars: list[float] = []
    if "repeat" in metrics_df.columns:
        for repeat, frame in metrics_df.dropna(subset=[column]).groupby("repeat", sort=True):
            group_values = frame[column].astype(float).to_numpy()
            repeat_seed = None
            if "repeat_seed" in frame.columns and not frame["repeat_seed"].dropna().empty:
                repeat_seed = int(frame["repeat_seed"].dropna().iloc[0])
            seed_rows.append(
                {
                    "repeat": int(repeat),
                    "repeat_seed": repeat_seed,
                    "mean": float(group_values.mean()),
                }
            )
            fold_vars.append(float(np.var(group_values, ddof=0)))
    seed_means = [row["mean"] for row in seed_rows]
    seed_var = float(np.var(seed_means, ddof=0)) if seed_means else 0.0
    fold_var = float(np.mean(fold_vars)) if fold_vars else 0.0
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "sem": float(values.std(ddof=0) / np.sqrt(values.size)),
        "ci_95_lo": ci_low,
        "ci_95_hi": ci_high,
        "seed_mean": seed_rows,
        "seed_var": seed_var,
        "fold_var": fold_var,
        "stability_ratio": float(seed_var / fold_var) if fold_var > 0 else None,
    }


def summarize_monte_carlo_frame(metrics_df: pd.DataFrame, *, seed: int = 310) -> dict[str, Any]:
    metric_map = {
        "auroc": "AUROC",
        "f1_macro": "F1",
        "auprc": "AUPRC",
        "balanced_accuracy": "Bal Acc",
        "recall_msi_h": "MSI-H Recall",
        "specificity": "Specificity",
        "brier_score": "Brier",
    }
    summary = {
        "measurement_count": int(len(metrics_df)),
        "repeat_count": int(metrics_df["repeat"].nunique()) if "repeat" in metrics_df.columns else 1,
        "fold_count": int(metrics_df["fold"].nunique()) if "fold" in metrics_df.columns else int(len(metrics_df)),
        "metrics": {},
    }
    for column, label in metric_map.items():
        if column in metrics_df.columns:
            metric_summary = summarize_metric_distribution(metrics_df, column, seed=seed)
            if metric_summary:
                summary["metrics"][label] = metric_summary
    return summary


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.clip(np.round(y_score, 6), 0, 1))
    candidates = np.unique(np.concatenate(([0.0], thresholds, [0.5, 1.0])))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        y_pred = (y_score >= float(threshold)).astype(int)
        score = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    if threshold is None:
        threshold, tuned_f1 = best_f1_threshold(y_true, y_score)
    else:
        tuned_f1 = float(f1_score(y_true, (y_score >= threshold).astype(int), average="macro", zero_division=0))
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    recall_msi_h = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1_macro": tuned_f1,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall_msi_h": recall_msi_h,
        "specificity": specificity,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def reliability_diagram_frame(y_true: np.ndarray, y_score: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for idx in range(bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx == bins - 1:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)
        rows.append(
            {
                "bin": idx + 1,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_predicted_prob": float(y_score[mask].mean()) if np.any(mask) else None,
                "observed_positive_rate": float(y_true[mask].mean()) if np.any(mask) else None,
            }
        )
    return pd.DataFrame(rows)


def isotonic_cross_seed_summary(pred_df: pd.DataFrame) -> dict[str, Any]:
    if "repeat_seed" not in pred_df.columns or pred_df["repeat_seed"].dropna().nunique() < 2:
        return {"evaluated": False, "per_seed": [], "mean_brier_improvement": None}
    rows: list[dict[str, Any]] = []
    for repeat_seed in sorted(int(value) for value in pred_df["repeat_seed"].dropna().unique().tolist()):
        train_df = pred_df.loc[pred_df["repeat_seed"] != repeat_seed].copy()
        test_df = pred_df.loc[pred_df["repeat_seed"] == repeat_seed].copy()
        if train_df.empty or test_df.empty or train_df["y_true"].nunique() < 2 or test_df["y_true"].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_df["predicted_prob"].astype(float), train_df["y_true"].astype(int))
        calibrated = iso.transform(test_df["predicted_prob"].astype(float))
        raw_brier = float(brier_score_loss(test_df["y_true"].astype(int), test_df["predicted_prob"].astype(float)))
        calibrated_brier = float(brier_score_loss(test_df["y_true"].astype(int), calibrated))
        rows.append(
            {
                "held_out_repeat_seed": repeat_seed,
                "raw_brier_score": raw_brier,
                "calibrated_brier_score": calibrated_brier,
                "brier_improvement": raw_brier - calibrated_brier,
                "raw_auroc": float(roc_auc_score(test_df["y_true"].astype(int), test_df["predicted_prob"].astype(float))),
                "calibrated_auroc": float(roc_auc_score(test_df["y_true"].astype(int), calibrated)),
            }
        )
    return {
        "evaluated": bool(rows),
        "per_seed": rows,
        "mean_brier_improvement": float(np.mean([row["brier_improvement"] for row in rows])) if rows else None,
    }


def paired_bootstrap_p_value(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    rounds: int = 10000,
    seed: int = 310,
) -> dict[str, Any]:
    if values_a.shape != values_b.shape:
        raise ValueError("Bootstrap inputs must have the same shape.")
    diff = values_a - values_b
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = np.arange(diff.size)
    boot_means = np.array(
        [diff[rng.choice(idx, size=diff.size, replace=True)].mean() for _ in range(rounds)],
        dtype=float,
    )
    p_greater = float((np.sum(boot_means <= 0) + 1) / (rounds + 1))
    p_less = float((np.sum(boot_means >= 0) + 1) / (rounds + 1))
    return {
        "observed_mean_diff": observed,
        "p_value_a_greater_b": p_greater,
        "p_value_b_greater_a": p_less,
        "ci_95_lo": float(np.percentile(boot_means, 2.5)),
        "ci_95_hi": float(np.percentile(boot_means, 97.5)),
    }


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(values.shape[0], dtype=float)
    start = 0
    while start < sorted_values.shape[0]:
        end = start
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    out = np.empty(values.shape[0], dtype=float)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty(positive_examples.shape, dtype=float)
    ty = np.empty(negative_examples.shape, dtype=float)
    tz = np.empty(predictions_sorted_transposed.shape, dtype=float)
    for idx in range(predictions_sorted_transposed.shape[0]):
        tx[idx, :] = _compute_midrank(positive_examples[idx, :])
        ty[idx, :] = _compute_midrank(negative_examples[idx, :])
        tz[idx, :] = _compute_midrank(predictions_sorted_transposed[idx, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-y_true)
    sorted_labels = y_true[order]
    label_1_count = int(sorted_labels.sum())
    predictions = np.vstack([pred_a[order], pred_b[order]])
    aucs, covariance = _fast_delong(predictions, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0:
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "auc_diff": float(aucs[0] - aucs[1]),
            "z_score": None,
            "p_value_two_sided": 1.0,
            "p_value_a_greater_b": 0.5,
        }
    z_score = float(abs(aucs[0] - aucs[1]) / math.sqrt(variance))
    p_two_sided = float(math.erfc(z_score / math.sqrt(2.0)))
    cdf = 0.5 * (1.0 + math.erf(((aucs[0] - aucs[1]) / math.sqrt(variance)) / math.sqrt(2.0)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(aucs[0] - aucs[1]),
        "z_score": z_score,
        "p_value_two_sided": p_two_sided,
        "p_value_a_greater_b": float(1.0 - cdf if aucs[0] < aucs[1] else 1.0 - (1.0 - cdf)),
    }


def load_fold_metrics(run_root: Path, approach: str) -> pd.DataFrame:
    path = run_root / "approaches" / approach / "fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_per_slide_predictions(run_root: Path, approach: str) -> pd.DataFrame:
    direct = run_root / "approaches" / approach / "per_slide_predictions.csv"
    if direct.exists():
        return pd.read_csv(direct)

    metrics_path = run_root / "approaches" / approach / "metrics.json"
    metrics = read_json(metrics_path)
    files = [Path(str(item)) for item in metrics.get("artifacts", {}).get("prediction_files", [])]
    if not files:
        raise FileNotFoundError(f"No prediction files were listed for {approach}")
    frames = [normalize_prediction_frame(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def monte_carlo_rank(metrics: dict[str, Any]) -> float:
    auroc = float(metrics.get("mean_auroc") or 0.0)
    auprc = float(metrics.get("mean_auprc") or 0.0)
    bal_acc = float(metrics.get("mean_balanced_accuracy") or 0.0)
    recall = float(metrics.get("mean_recall_msi_h") or 0.0)
    brier = float(metrics.get("mean_brier_score") or 1.0)
    stability_ratio = metrics.get("auroc_stability_ratio")
    penalty = float(stability_ratio) if stability_ratio is not None else 0.0
    return 0.40 * auroc + 0.25 * auprc + 0.15 * bal_acc + 0.10 * recall + 0.10 * (1.0 - brier) - 0.05 * penalty


def stacking_oof_predictions(merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    required = ["y_true", "repeat", "fold", *feature_cols]
    frame = merged.dropna(subset=required).copy()
    frame["stacked_score"] = np.nan
    group_keys = frame[["repeat", "fold"]].drop_duplicates().to_dict("records")
    for group in group_keys:
        mask = (frame["repeat"] == group["repeat"]) & (frame["fold"] == group["fold"])
        train = frame.loc[~mask]
        test = frame.loc[mask]
        if train.empty or test.empty or train["y_true"].nunique() < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(train[feature_cols], train["y_true"].astype(int))
        frame.loc[mask, "stacked_score"] = model.predict_proba(test[feature_cols])[:, 1]
    return frame
