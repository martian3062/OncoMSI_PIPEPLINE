from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from top4_analysis_common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--approach", required=True)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    return parser.parse_args()


def choose_target_fold(fold_df: pd.DataFrame, repeat: int | None, fold: int | None) -> pd.Series:
    selected = fold_df.copy()
    if repeat is not None:
        selected = selected.loc[selected["repeat"] == repeat]
    if fold is not None:
        selected = selected.loc[selected["fold"] == fold]
    if selected.empty:
        raise ValueError("No fold rows matched the requested repeat/fold.")
    return selected.sort_values(["auroc", "f1_macro"], ascending=False).iloc[0]


def find_checkpoint(model_dir: Path) -> Path | None:
    candidates = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(sorted(model_dir.rglob(pattern)))
    return candidates[0] if candidates else None


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    approach = args.approach
    output_zip = Path(args.output_zip)
    bundle_dir = output_zip.with_suffix("")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_root = bundle_dir / "inference_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    metrics = read_json(run_root / "approaches" / approach / "metrics.json")
    fold_df = pd.read_csv(run_root / "approaches" / approach / "fold_metrics.csv")
    target = choose_target_fold(fold_df, args.repeat, args.fold)
    repeat = int(target["repeat"])
    fold = int(target["fold"])
    metrics_artifacts = metrics.get("artifacts", {})
    prediction_files = [Path(item) for item in metrics_artifacts.get("prediction_files", [])]
    target_fragment = f"_repeat_{repeat}_fold_{fold}"
    model_dir = next((path.parent for path in prediction_files if target_fragment in str(path.parent)), None)
    if model_dir is None:
        raise FileNotFoundError(f"Could not locate the model directory for repeat={repeat}, fold={fold}")
    checkpoint = find_checkpoint(model_dir)
    if checkpoint is None and not args.allow_missing_checkpoint:
        raise FileNotFoundError(f"No checkpoint file found under {model_dir}")

    bundle_config = read_json(run_root / "bundle_config.json") if (run_root / "bundle_config.json").exists() else {}
    calibration = {}
    calibration_path = run_root / "approaches" / approach / "threshold_calibration_summary.json"
    if calibration_path.exists():
        calibration = read_json(calibration_path)

    model_card = {
        "approach_label": approach,
        "mil_model": metrics.get("mil_model"),
        "feature_extractor_used": metrics.get("feature_extractor_used"),
        "repeat": repeat,
        "fold": fold,
        "checkpoint_source": str(checkpoint) if checkpoint else None,
        "model_dir": str(model_dir),
    }
    preprocessing = {
        "tile_px": bundle_config.get("request", {}).get("tile_px"),
        "tile_um": bundle_config.get("request", {}).get("tile_um"),
        "max_tiles_per_slide": bundle_config.get("request", {}).get("max_tiles_per_slide"),
        "mpp_override": bundle_config.get("request", {}).get("mpp_override"),
        "qc_method": bundle_config.get("request", {}).get("qc_method"),
    }
    threshold_payload = {
        "deployment_threshold": calibration.get("mean_threshold", metrics.get("mean_best_threshold")),
        "ci_95_lo": calibration.get("ci_95_lo"),
        "ci_95_hi": calibration.get("ci_95_hi"),
    }
    metrics_summary = {
        "internal": metrics,
        "external": metrics.get("external_metrics", {}),
    }

    write_json(bundle_root / "model_card.json", model_card)
    write_json(bundle_root / "preprocessing.json", preprocessing)
    write_json(bundle_root / "threshold.json", threshold_payload)
    write_json(bundle_root / "label_map.json", {"0": "MSS", "1": "MSI-H"})
    write_json(bundle_root / "metrics_summary.json", metrics_summary)
    readme_lines = [
        "# Inference Bundle",
        "",
        f"- approach: `{approach}`",
        f"- repeat: `{repeat}`",
        f"- fold: `{fold}`",
        "",
        "## Runner",
        "",
        "```bash",
        "python run_inference.py \\",
        f"  --checkpoint checkpoint{checkpoint.suffix if checkpoint else '.pth'} \\",
        "  --preprocessing preprocessing.json \\",
        "  --threshold threshold.json \\",
        "  --label-map label_map.json \\",
        "  --slide /path/to/new_slide.svs",
        "```",
        "",
        f"Source model dir: `{model_dir}`",
    ]
    (bundle_root / "README_inference.md").write_text("\n".join(readme_lines), encoding="utf-8")
    if checkpoint is not None:
        shutil.copy2(checkpoint, bundle_root / f"checkpoint{checkpoint.suffix}")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_root.rglob("*"):
            archive.write(path, path.relative_to(bundle_dir))
    print(output_zip)


if __name__ == "__main__":
    main()
