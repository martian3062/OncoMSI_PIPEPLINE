from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


def load_runner_module(project_root: Path):
    runner_path = project_root / "scripts" / "run_tcga_coad_automated_triad.py"
    spec = importlib.util.spec_from_file_location("top4_vm_runner", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load runner module from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--approaches", default="")
    parser.add_argument("--cohort", default="CPTAC-COAD")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def locate_external_cohort(prepared: dict[str, Any], cohort_name: str) -> dict[str, Any]:
    for cohort in prepared.get("external_cohorts_prepared", []):
        if str(cohort.get("name")) == cohort_name:
            return cohort
    raise KeyError(f"External cohort {cohort_name!r} was not found in prepared_bundle.json")


def unique_model_dirs(metrics: dict[str, Any]) -> list[Path]:
    dirs = []
    seen = set()
    for item in metrics.get("artifacts", {}).get("prediction_files", []):
        parent = Path(str(item)).parent
        key = str(parent)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(parent)
    return dirs


def locate_prediction_outputs(path: Path) -> list[Path]:
    return sorted(list(path.rglob("predictions.parquet")) + list(path.rglob("predictions.csv")))


def ensure_external_bags(runner, config: dict[str, Any], spec: dict[str, Any], prepared: dict[str, Any], cohort: dict[str, Any], run_root: Path) -> Path:
    sf, project = runner.load_project(run_root, Path(cohort["annotations_csv"]), Path(cohort["slides_dir"]))
    dataset = runner.make_dataset(project)
    approach_payload = prepared.get("approaches", {}).get(spec["approach_label"], {})
    feature_candidates = approach_payload.get("feature_extractor_candidates") or [spec.get("feature_extractor")]
    extractor, extractor_name, _backend, _resolved = runner.build_extractor(
        sf,
        feature_candidates,
        runner.requested_virchow_weights(config),
        hf_token=runner.requested_hf_token(config),
        requested_backend=str(spec.get("extractor_backend", "auto")),
    )
    cohort_root = Path(cohort["cohort_root"])
    bags_dir = cohort_root / "bags" / f"{extractor_name}_{runner.TILE_PX}px_{runner.TILE_UM}um"
    if not bags_dir.exists() or not any(path.is_file() for path in bags_dir.rglob("*")):
        bags_dir.mkdir(parents=True, exist_ok=True)
        project.generate_feature_bags(extractor, dataset, outdir=str(bags_dir))
    return bags_dir


def run_external_inference(runner, project, dataset, bags_dir: Path, model_dir: Path, outdir: Path) -> list[Path]:
    import slideflow.mil as mil  # type: ignore

    outdir.mkdir(parents=True, exist_ok=True)
    attempts = [
        lambda: getattr(project, "eval_mil")(model=str(model_dir), dataset=dataset, outcomes=runner.OUTCOME, bags=str(bags_dir), outdir=str(outdir)),
        lambda: getattr(project, "eval_mil")(str(model_dir), dataset=dataset, outcomes=runner.OUTCOME, bags=str(bags_dir), outdir=str(outdir)),
        lambda: getattr(project, "predict_mil")(model=str(model_dir), dataset=dataset, outcomes=runner.OUTCOME, bags=str(bags_dir), outdir=str(outdir)),
        lambda: getattr(project, "predict_mil")(str(model_dir), dataset=dataset, outcomes=runner.OUTCOME, bags=str(bags_dir), outdir=str(outdir)),
        lambda: getattr(mil, "eval_mil")(model=str(model_dir), dataset=dataset, outcomes=runner.OUTCOME, bags=str(bags_dir), outdir=str(outdir)),
        lambda: getattr(mil, "predict_mil")(model=str(model_dir), dataset=dataset, outcomes=runner.OUTCOME, bags=str(bags_dir), outdir=str(outdir)),
    ]
    errors = []
    for attempt in attempts:
        try:
            attempt()
        except AttributeError as exc:
            errors.append(f"missing_api:{exc}")
        except TypeError as exc:
            errors.append(f"bad_signature:{exc}")
        except Exception as exc:  # pragma: no cover - runtime adapter
            errors.append(f"runtime:{type(exc).__name__}:{exc}")
        predictions = locate_prediction_outputs(outdir)
        if predictions:
            return predictions
    raise RuntimeError(f"Unable to produce external predictions for {model_dir}. Attempts: {errors}")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    project_root = run_root.parent.parent.parent
    runner = load_runner_module(project_root)

    config = runner.read_json(run_root / "bundle_config.json")
    prepared = runner.read_json(run_root / "prepared_bundle.json")
    cohort = locate_external_cohort(prepared, args.cohort)
    approaches = [item.strip() for item in args.approaches.split(",") if item.strip()]
    if not approaches:
        approaches = [str(spec["approach_label"]) for spec in config.get("specs", [])]

    sf, project = runner.load_project(run_root, Path(cohort["annotations_csv"]), Path(cohort["slides_dir"]))
    dataset = runner.make_dataset(project)
    top_level_rows = []
    for spec in config.get("specs", []):
        approach_label = str(spec["approach_label"])
        if approach_label not in approaches:
            continue
        metrics_path = run_root / "approaches" / approach_label / "metrics.json"
        metrics = runner.read_json(metrics_path)
        bags_dir = ensure_external_bags(runner, config, spec, prepared, cohort, run_root)
        prediction_files = []
        for model_dir in unique_model_dirs(metrics):
            repeat, fold = runner.parse_repeat_fold_from_path(model_dir)
            outdir = run_root / "external_eval" / cohort["slug"] / approach_label / f"repeat_{repeat}_fold_{fold}"
            prediction_files.extend(run_external_inference(runner, project, dataset, bags_dir, model_dir, outdir))

        external = runner.aggregate_prediction_files(
            prediction_files,
            approach_label=approach_label,
            mil_model=str(metrics.get("mil_model") or spec.get("mil_model")),
            epochs=int(metrics.get("epochs") or spec.get("epochs") or 0),
            seed=int(spec.get("seed", 310)),
            repeat_seeds=prepared.get("repeat_seeds", []),
            artifact_label=f"{approach_label}-{args.cohort}",
        )
        external_metrics = {key: value for key, value in external.items() if not key.startswith("_")}
        external_dir = run_root / "approaches" / approach_label / "external_eval"
        external_dir.mkdir(parents=True, exist_ok=True)
        runner.write_json(external_dir / f"{cohort['slug']}_metrics.json", external_metrics)
        metrics.setdefault("external_metrics", {})
        metrics["external_metrics"][args.cohort] = external_metrics
        runner.write_json(metrics_path, metrics)
        top_level_rows.append(
            {
                "approach_label": approach_label,
                "mean_auroc": external_metrics.get("mean_auroc"),
                "mean_auprc": external_metrics.get("mean_auprc"),
                "mean_f1_macro": external_metrics.get("mean_f1_macro"),
                "cohort": args.cohort,
            }
        )

    output_json = Path(args.output_json) if args.output_json else run_root / "external_metrics.json"
    runner.write_json(
        output_json,
        {
            "bundle_id": run_root.name,
            "cohort": args.cohort,
            "approaches": top_level_rows,
        },
    )
    print(output_json)


if __name__ == "__main__":
    main()
