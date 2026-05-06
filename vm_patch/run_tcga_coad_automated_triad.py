from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from hybrid_extractors import hybrid_backend_for_name, normalize_extractor_name, register_hybrid_extractors

TILE_PX = 256
TILE_UM = 128
OUTCOME = "msi_status"
POS_LABEL = "MSI-H"
NEG_LABEL = "MSS"
# The TCGA SVS cohort on the VM is more stable with a single extraction
# process. Multi-process tile extraction intermittently leaves behind zero-byte
# TFRecords / unfinished markers for these slides.
EXTRACTION_WORKERS = 1
DEFAULT_SLIDE_LIMIT = 18
DEFAULT_FOLD_COUNT = 3
DEFAULT_FEATURE_EXTRACTOR_CANDIDATES = (
    "virchow",
    "ctranspath",
)
GENERIC_FEATURE_EXTRACTORS = {"resnet50_imagenet", "resnet50"}
DEFAULT_MIL_FALLBACKS = {
    "Approach1": ("transmil",),
    "Approach2": ("attention_mil",),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_bundle_config_path(config: dict[str, Any]) -> Path:
    explicit = config.get("_bundle_config_path")
    if explicit:
        return Path(str(explicit))
    return Path(config["bundle_root"]) / "bundle_config.json"


def normalize_stem(value: str) -> str:
    return value.split(".", 1)[0].strip().upper()


def parse_candidate_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [str(value)]

    candidates: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
    return candidates


def requested_slide_limit(config: dict[str, Any]) -> int:
    return max(2, int(config["request"].get("slide_limit", DEFAULT_SLIDE_LIMIT)))


def requested_n_folds(config: dict[str, Any]) -> int:
    return max(2, int(config["request"].get("n_folds", DEFAULT_FOLD_COUNT)))


def requested_feature_extractors(config: dict[str, Any]) -> list[str]:
    request = config["request"]
    requested = parse_candidate_list(request.get("feature_extractors"))
    requested.extend(parse_candidate_list(request.get("feature_extractor")))
    requested.extend(DEFAULT_FEATURE_EXTRACTOR_CANDIDATES)
    requested = parse_candidate_list(requested)
    if bool(request.get("allow_generic_fallback", False)):
        return requested
    return [name for name in requested if name.lower() not in GENERIC_FEATURE_EXTRACTORS]


def requested_feature_extractors_for_spec(
    config: dict[str, Any],
    spec: dict[str, Any],
    default_candidates: list[str] | None = None,
) -> list[str]:
    requested = parse_candidate_list(spec.get("feature_extractors"))
    requested.extend(parse_candidate_list(spec.get("feature_extractor")))
    requested.extend(default_candidates or requested_feature_extractors(config))
    requested = parse_candidate_list(requested)
    if bool(spec.get("allow_generic_fallback", config["request"].get("allow_generic_fallback", False))):
        return requested
    return [name for name in requested if name.lower() not in GENERIC_FEATURE_EXTRACTORS]


def requested_max_tiles_per_slide(config: dict[str, Any]) -> int | None:
    value = config["request"].get("max_tiles_per_slide")
    if value in (None, "", 0, "0"):
        return None
    return max(32, int(value))


def requested_mpp_override(config: dict[str, Any]) -> float | None:
    value = config["request"].get("mpp_override")
    if value in (None, "", 0, "0"):
        return None
    return float(value)


def requested_max_negative_multiplier(config: dict[str, Any]) -> float | None:
    value = config["request"].get("max_negative_multiplier")
    if value in (None, "", 0, "0"):
        return None
    return max(1.0, float(value))


def requested_qc_method(config: dict[str, Any]) -> str | None:
    value = config["request"].get("qc_method")
    if value in (None, "", "none", "None"):
        return None
    return str(value)


def requested_n_repeats(config: dict[str, Any]) -> int:
    return max(1, int(config["request"].get("n_repeats", 1)))


def requested_virchow_weights(config: dict[str, Any]) -> str | None:
    value = config["request"].get("virchow_weights")
    if value in (None, "", "None"):
        return None
    return str(value)


def requested_hf_token(config: dict[str, Any]) -> str | None:
    value = config["request"].get("hf_token")
    if value in (None, "", "None"):
        return None
    return str(value)


def requested_extractor_backend(config: dict[str, Any], spec: dict[str, Any]) -> str:
    explicit = spec.get("extractor_backend")
    if explicit not in (None, "", "None"):
        return str(explicit)
    request_default = config["request"].get("extractor_backend")
    if request_default not in (None, "", "None"):
        return str(request_default)
    candidates = requested_feature_extractors_for_spec(config, spec, requested_feature_extractors(config))
    if any(hybrid_backend_for_name(name) == "hybrid" for name in candidates):
        return "hybrid"
    return "slideflow"


def requested_approach_execution_mode(config: dict[str, Any]) -> str:
    value = str(config["request"].get("approach_execution_mode", "parallel")).strip().lower()
    if value not in {"parallel", "sequential"}:
        value = "parallel"
    if int(config["request"].get("max_parallel_approaches", 2)) <= 1:
        return "sequential"
    return value


def requested_mil_models(spec: dict[str, Any]) -> list[str]:
    requested = parse_candidate_list(spec.get("mil_model_candidates"))
    requested.extend(parse_candidate_list(spec.get("mil_model")))
    requested.extend(DEFAULT_MIL_FALLBACKS.get(str(spec.get("approach_label")), ()))
    return parse_candidate_list(requested)


def patient_barcode_from_slide(slide_name: str) -> str:
    parts = normalize_stem(slide_name).split("-")
    return "-".join(parts[:3])


def update_bundle_status(config: dict[str, Any], state: str, **extra: Any) -> None:
    payload = {
        "bundle_id": config["bundle_id"],
        "state": state,
        "updated_at_epoch": time.time(),
        **extra,
    }
    write_json(Path(config["status_path"]), payload)


def update_approach_status(config: dict[str, Any], approach_label: str, state: str, **extra: Any) -> None:
    status_path = Path(config["bundle_root"]) / "approaches" / approach_label / "status.json"
    payload = {
        "bundle_id": config["bundle_id"],
        "approach_label": approach_label,
        "state": state,
        "updated_at_epoch": time.time(),
        **extra,
    }
    write_json(status_path, payload)


def normalized_approach_result(approach_label: str, payload: dict[str, Any]) -> dict[str, Any]:
    mean_auroc = payload.get("mean_auroc")
    mean_f1_macro = payload.get("mean_f1_macro")
    state = str(payload.get("state") or "")
    if mean_auroc is not None or mean_f1_macro is not None:
        state = "completed"
    elif not state:
        state = "failed" if payload.get("error") else "pending"
    result = {
        **payload,
        "approach_label": payload.get("approach_label", approach_label),
        "state": state,
        "mean_auroc": mean_auroc,
        "mean_f1_macro": mean_f1_macro,
        "mean_f1_macro_default_threshold": payload.get("mean_f1_macro_default_threshold"),
        "feature_extractor_used": payload.get("feature_extractor_used"),
    }
    return result


def summarize_approach_payloads(
    config: dict[str, Any],
    prepared: dict[str, Any],
    approach_payloads: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for spec in config["specs"]:
        approach_label = str(spec["approach_label"])
        payload = approach_payloads.get(approach_label)
        if not payload:
            continue
        normalized[approach_label] = normalized_approach_result(approach_label, payload)

    completed = [item for item in normalized.values() if item.get("state") == "completed"]
    failed = [item for item in normalized.values() if item.get("state") == "failed"]
    running = [item for item in normalized.values() if item.get("state") in {"training", "spawned"}]

    best_payload = None
    if completed:
        best_payload = max(
            completed,
            key=lambda item: (
                float(item.get("mean_auroc") or float("-inf")),
                float(item.get("mean_f1_macro") or float("-inf")),
            ),
        )

    if running:
        overall_state = "training_parallel"
    elif failed:
        overall_state = "failed"
    elif normalized and len(completed) == len(normalized):
        overall_state = "completed"
    else:
        overall_state = "prepared"

    return {
        "approaches": normalized,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "running_count": len(running),
        "overall_state": overall_state,
        "best_payload": best_payload,
        "best_approach": best_payload.get("approach_label") if best_payload else None,
    }


def count_valid_tfrecords(dataset) -> int:
    valid = 0
    for path in dataset.tfrecords():
        tfrecord_path = Path(path)
        if tfrecord_path.is_file() and tfrecord_path.stat().st_size > 0:
            valid += 1
    return valid


def clear_extraction_outputs(project_root: Path) -> None:
    for rel in ("tfrecords", "tiles"):
        target = project_root / rel
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def cleanup_experiment_artifacts(sf_root: Path, exp_label: str) -> None:
    mil_root = sf_root / "mil"
    if not mil_root.exists():
        return
    for path in mil_root.rglob("*"):
        if exp_label not in str(path):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def available_bag_slide_stems(bags_dir: Path) -> set[str]:
    return {path.stem for path in bags_dir.glob("*.pt")}


def normalize_slide_identifier(value: str) -> str:
    text = str(value).strip()
    if text.lower().endswith(".svs"):
        return text[:-4]
    return text


def filtered_split_plan_for_bags(split_plan: Path, bags_dir: Path, out_path: Path) -> tuple[Path, list[str]]:
    split_df = pd.read_csv(split_plan)
    if "slide" not in split_df.columns:
        raise ValueError("Subset annotations must include 'slide' column.")
    available = available_bag_slide_stems(bags_dir)
    slide_ids = split_df["slide"].astype(str).map(normalize_slide_identifier)
    filtered = split_df.loc[slide_ids.isin(available)].copy()
    missing = sorted(set(split_df["slide"].astype(str)) - set(filtered["slide"].astype(str)))
    if filtered.empty:
        raise RuntimeError(f"No slides remain after filtering split plan for available bags in {bags_dir}.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_path, index=False)
    return out_path, missing


def mil_output_dir(sf_root: Path, exp_label: str, repeat: int, fold: int) -> Path:
    return sf_root / "mil" / f"{exp_label}_repeat_{repeat}_fold_{fold}"


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)


def list_bucket_files(bucket_uri: str) -> list[dict[str, str]]:
    result = run_command(["gcloud", "storage", "ls", bucket_uri])
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or f"Unable to list bucket: {bucket_uri}")
    files: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        uri = line.strip()
        if not uri or not uri.endswith(".svs"):
            continue
        name = uri.rsplit("/", 1)[-1]
        files.append(
            {
                "uri": uri,
                "name": name,
                "stem": normalize_stem(name),
                "patient": patient_barcode_from_slide(name),
                "suffix": normalize_stem(name).split("-")[-1],
            }
        )
    return files


def choose_balanced_subset(
    rows: list[dict[str, Any]],
    slide_limit: int,
    max_negative_multiplier: float | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[OUTCOME])].append(row)

    for value in grouped.values():
        value.sort(key=lambda item: (str(item.get("patient", "")), str(item.get("slide", ""))))

    positive = grouped.get(POS_LABEL, [])
    negative = grouped.get(NEG_LABEL, [])
    if not positive or not negative:
        raise ValueError(f"Need both {POS_LABEL} and {NEG_LABEL} matches; found {len(positive)} and {len(negative)}.")

    if max_negative_multiplier is not None:
        negative_cap = max(1, int(len(positive) * max_negative_multiplier))
        negative = negative[:negative_cap]

    desired_limit = min(slide_limit, len(positive) + len(negative))
    half = max(1, desired_limit // 2)
    pos_target = min(len(positive), half)
    neg_target = min(len(negative), half)
    selected = positive[:pos_target] + negative[:neg_target]

    if len(selected) < desired_limit:
        leftovers = positive[pos_target:] + negative[neg_target:]
        leftovers.sort(key=lambda item: (str(item.get(OUTCOME, "")), str(item.get("patient", ""))))
        selected.extend(leftovers[: desired_limit - len(selected)])

    if len(selected) < 2:
        raise ValueError(f"Only {len(selected)} matched annotated slides are available after balancing.")

    return selected[:desired_limit]


def assign_patient_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 310,
    n_repeats: int = 1,
) -> pd.DataFrame:
    labels = (
        df[["patient", OUTCOME]]
        .drop_duplicates()
        .sort_values("patient")
        .reset_index(drop=True)
    )
    counts = labels[OUTCOME].value_counts()
    if counts.min() < n_folds:
        raise ValueError(f"Need at least {n_folds} patients in each class for folds; got {counts.to_dict()}.")

    repeated_frames: list[pd.DataFrame] = []
    for repeat_idx in range(1, n_repeats + 1):
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed + (repeat_idx - 1) * 97,
        )
        repeat_labels = labels.copy()
        repeat_labels["fold"] = 0
        repeat_labels["repeat"] = repeat_idx
        for fold_idx, (_, val_idx) in enumerate(
            splitter.split(repeat_labels["patient"], repeat_labels[OUTCOME]),
            start=1,
        ):
            repeat_labels.loc[val_idx, "fold"] = fold_idx
        repeated_frames.append(repeat_labels[["patient", "repeat", "fold"]])

    fold_plan = pd.concat(repeated_frames, ignore_index=True)
    out = df.merge(fold_plan, on="patient", how="inner")
    out["fold"] = out["fold"].astype(int)
    out["repeat"] = out["repeat"].astype(int)
    return out


def materialize_subset(config: dict[str, Any]) -> dict[str, Any]:
    request = config["request"]
    bundle_root = Path(config["bundle_root"])
    annotations_src = Path(request.get("annotations_csv") or "annotations/tcga_crc_msi_annotations.csv")
    if not annotations_src.is_absolute():
        annotations_src = Path(bundle_root).parents[2] / str(annotations_src)
    if not annotations_src.exists():
        raise FileNotFoundError(f"Missing annotations CSV: {annotations_src}")

    annotations_df = pd.read_csv(annotations_src)
    required = {"slide", "patient", OUTCOME}
    missing = required.difference(annotations_df.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    bucket_files = list_bucket_files(str(request["bucket_uri"]))
    exact_map = {row["stem"]: row for row in bucket_files}
    patient_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bucket_files:
        patient_candidates[row["patient"]].append(row)

    preferred_suffix = str(request.get("preferred_exact_suffix") or "DX1").upper()
    preferred_pattern = str(request.get("preferred_slide_pattern") or "DX").upper()

    matched_rows: list[dict[str, Any]] = []
    for row in annotations_df.to_dict("records"):
        slide_stem = normalize_stem(str(row["slide"]))
        match = exact_map.get(slide_stem)
        if match is None:
            patient = str(row["patient"]).upper()
            options = patient_candidates.get(patient, [])
            preferred = [item for item in options if item["suffix"] == preferred_suffix]
            if not preferred:
                preferred = [item for item in options if preferred_pattern in item["suffix"]]
            if preferred:
                match = sorted(preferred, key=lambda item: item["name"])[0]
        if match is None:
            continue
        merged = dict(row)
        # Slideflow annotations should use the slide stem, while the full
        # filename stays available in gdc_filename for download/provenance.
        merged["slide"] = match["name"].removesuffix(".svs")
        if "gdc_filename" in merged:
            merged["gdc_filename"] = match["name"]
        merged["bucket_uri"] = match["uri"]
        merged["bucket_name"] = match["name"]
        merged["bucket_suffix"] = match["suffix"]
        matched_rows.append(merged)

    if not matched_rows:
        raise ValueError("No bucket slides matched the existing annotation file.")

    requested_limit = requested_slide_limit(config)
    requested_folds = requested_n_folds(config)
    max_negative_multiplier = requested_max_negative_multiplier(config)
    selected_rows = choose_balanced_subset(
        matched_rows,
        requested_limit,
        max_negative_multiplier=max_negative_multiplier,
    )
    subset_df = pd.DataFrame(selected_rows)
    patient_counts = (
        subset_df[["patient", OUTCOME]]
        .drop_duplicates()[OUTCOME]
        .value_counts()
    )
    if patient_counts.min() < 2:
        raise ValueError(f"Need at least 2 patients in each class after selection; got {patient_counts.to_dict()}.")
    n_folds = min(requested_folds, int(patient_counts.min()))
    n_repeats = requested_n_repeats(config)
    split_plan_df = assign_patient_folds(subset_df, n_folds=n_folds, seed=310, n_repeats=n_repeats)

    annotations_dir = bundle_root / "annotations"
    slides_dir = bundle_root / "slideflow_project" / "data" / "slides"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    subset_annotations = annotations_dir / "tcga_crc_msi_annotations.csv"
    subset_df.to_csv(subset_annotations, index=False)
    split_plan_path = annotations_dir / "tcga_crc_msi_split_plan.csv"
    split_plan_df.to_csv(split_plan_path, index=False)
    write_json(
        annotations_dir / "selected_slides.json",
        {
            "bundle_id": config["bundle_id"],
            "requested_slide_limit": requested_limit,
            "selected_slide_limit": int(len(subset_df)),
            "requested_n_folds": requested_folds,
            "n_folds": n_folds,
            "n_repeats": n_repeats,
            "matched_slides": len(matched_rows),
            "selected_slides": subset_df.to_dict("records"),
            "max_negative_multiplier": max_negative_multiplier,
        },
    )
    return {
        "subset_annotations": str(subset_annotations),
        "split_plan": str(split_plan_path),
        "slides_dir": str(slides_dir),
        "selected_slide_names": subset_df["bucket_name"].tolist(),
        "selected_slide_uris": subset_df["bucket_uri"].tolist(),
        "label_counts": subset_df[OUTCOME].value_counts().to_dict(),
        "matched_slide_count": len(matched_rows),
        "requested_slide_limit": requested_limit,
        "selected_slide_limit": int(len(subset_df)),
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "max_negative_multiplier": max_negative_multiplier,
    }


def download_subset_slides(config: dict[str, Any], selected_slide_uris: list[str], slides_dir: Path) -> None:
    pending = []
    for uri in selected_slide_uris:
        target = slides_dir / uri.rsplit("/", 1)[-1]
        if target.exists():
            continue
        pending.append(uri)

    if not pending:
        return

    batch_size = 8
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]

        if len(batch) > 1:
            gsutil_result = run_command(["gsutil", "-m", "cp", *batch, str(slides_dir)])
            if gsutil_result.returncode == 0:
                continue

        gcloud_result = run_command(["gcloud", "storage", "cp", *batch, str(slides_dir)])
        if gcloud_result.returncode == 0:
            continue

        for uri in batch:
            target = slides_dir / uri.rsplit("/", 1)[-1]
            single_result = run_command(["gcloud", "storage", "cp", uri, str(target)])
            if single_result.returncode != 0:
                raise RuntimeError(single_result.stderr or single_result.stdout or f"Failed to copy {uri}")


def import_slideflow():
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("VIRTUAL_ENV", None)
    preferred_conda_prefix = Path("/opt/miniforge3/envs/pathology310")
    if preferred_conda_prefix.exists():
        os.environ["CONDA_PREFIX"] = str(preferred_conda_prefix)
    elif not os.environ.get("CONDA_PREFIX"):
        os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parents[1])
    os.environ["SF_BACKEND"] = "torch"
    os.environ["SF_SLIDE_BACKEND"] = "cucim"
    import slideflow as sf
    return sf


def load_project(bundle_root: Path, annotations_csv: Path, slides_dir: Path):
    sf = import_slideflow()
    register_hybrid_extractors(sf)
    sf_root = bundle_root / "slideflow_project"
    sf_root.mkdir(parents=True, exist_ok=True)
    dataset_config = sf_root / "datasets.json"
    try:
        project = sf.Project(str(sf_root))
    except Exception:
        project = sf.Project(
            str(sf_root),
            name=f"TCGA_COAD_{bundle_root.name}",
            annotations=str(annotations_csv),
            sources=["tcga_coad_subset"],
            create=True,
        )
    if not dataset_config.exists():
        project.add_source(
            "tcga_coad_subset",
            slides=str(slides_dir),
            tfrecords=str(sf_root / "tfrecords"),
            tiles=str(sf_root / "tiles"),
        )
    return sf, project


def make_dataset(project):
    return project.dataset(tile_px=TILE_PX, tile_um=TILE_UM)


def build_extractor(
    sf,
    candidates: list[str],
    virchow_weights: str | None = None,
    hf_token: str | None = None,
    requested_backend: str = "auto",
):
    errors: dict[str, str] = {}
    normalized_candidates = parse_candidate_list([normalize_extractor_name(name) for name in candidates])
    for name in normalized_candidates:
        try:
            backend = requested_backend
            inferred = hybrid_backend_for_name(name)
            if backend == "auto":
                backend = inferred
            kwargs = {"resize": True, "mixed_precision": True}
            if backend == "hybrid":
                kwargs["hf_token"] = hf_token
            elif name == "virchow" and virchow_weights:
                kwargs["weights"] = virchow_weights
            if name == "resnet50_imagenet":
                kwargs["tile_px"] = TILE_PX
            return sf.build_feature_extractor(name, **kwargs), name, backend
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
    raise RuntimeError(f"Unable to initialize any requested feature extractor. Tried: {errors}")


def prepare_bundle_for_training(config: dict[str, Any]) -> dict[str, Any]:
    feature_candidates = requested_feature_extractors(config)
    write_json(resolved_bundle_config_path(config), config)
    update_bundle_status(
        config,
        "matching_annotations",
        bucket_uri=config["request"]["bucket_uri"],
        slide_limit=requested_slide_limit(config),
        n_folds=requested_n_folds(config),
        n_repeats=requested_n_repeats(config),
        feature_extractor_candidates=feature_candidates,
        allow_generic_fallback=bool(config["request"].get("allow_generic_fallback", False)),
        virchow_weights=requested_virchow_weights(config),
        max_tiles_per_slide=requested_max_tiles_per_slide(config),
        mpp_override=requested_mpp_override(config),
        qc_method=requested_qc_method(config),
    )
    subset = materialize_subset(config)
    update_bundle_status(
        config,
        "downloading_slides",
        matched_slide_count=int(subset["matched_slide_count"]),
        selected_slide_count=len(subset["selected_slide_names"]),
        selected_slides=subset["selected_slide_names"],
        label_counts=subset["label_counts"],
    )
    slides_dir = Path(subset["slides_dir"])
    download_subset_slides(config, subset["selected_slide_uris"], slides_dir)

    bundle_root = Path(config["bundle_root"])
    annotations_csv = Path(subset["subset_annotations"])
    sf, project = load_project(bundle_root, annotations_csv, slides_dir)
    dataset = make_dataset(project)
    project_root = bundle_root / "slideflow_project"
    clear_extraction_outputs(project_root)
    update_bundle_status(
        config,
        "extracting_tiles",
        downloaded_slide_count=len(subset["selected_slide_names"]),
        selected_slide_count=len(subset["selected_slide_names"]),
        slide_backend=str(getattr(sf, "slide_backend", lambda: "unknown")()),
        extraction_workers=EXTRACTION_WORKERS,
        max_tiles_per_slide=requested_max_tiles_per_slide(config),
        mpp_override=requested_mpp_override(config),
        qc_method=requested_qc_method(config),
    )
    max_tiles_per_slide = requested_max_tiles_per_slide(config)
    mpp_override = requested_mpp_override(config)
    qc_method = requested_qc_method(config)
    dataset.extract_tiles(
        qc=qc_method,
        num_threads=EXTRACTION_WORKERS,
        report=False,
        skip_extracted=False,
        max_tiles=max_tiles_per_slide,
        mpp_override=mpp_override,
    )
    if count_valid_tfrecords(dataset) == 0:
        raise RuntimeError("Tile extraction completed but produced zero TFRecords.")

    update_bundle_status(
        config,
        "generating_features",
        downloaded_slide_count=len(subset["selected_slide_names"]),
        tile_px=TILE_PX,
        tile_um=TILE_UM,
        feature_extractor_candidates=feature_candidates,
        allow_generic_fallback=bool(config["request"].get("allow_generic_fallback", False)),
        virchow_weights=requested_virchow_weights(config),
    )
    bags_by_extractor: dict[str, str] = {}
    approach_payloads: dict[str, dict[str, Any]] = {}
    approach_extractors: dict[str, str] = {}
    approach_backends: dict[str, str] = {}
    default_virchow_weights = requested_virchow_weights(config)
    hf_token = requested_hf_token(config)
    execution_mode = requested_approach_execution_mode(config)
    for spec in config["specs"]:
        approach_label = str(spec["approach_label"])
        spec_candidates = requested_feature_extractors_for_spec(config, spec, feature_candidates)
        extractor, extractor_name, extractor_backend = build_extractor(
            sf,
            spec_candidates,
            str(spec.get("virchow_weights") or default_virchow_weights or ""),
            hf_token=hf_token,
            requested_backend=requested_extractor_backend(config, spec),
        )
        if extractor_name not in bags_by_extractor:
            bags_dir = bundle_root / "slideflow_project" / "bags" / f"{extractor_name}_{TILE_PX}px_{TILE_UM}um"
            bags_by_extractor[extractor_name] = str(bags_dir)
            if execution_mode != "sequential":
                bags_dir.mkdir(parents=True, exist_ok=True)
                project.generate_feature_bags(extractor, dataset, outdir=str(bags_dir))
                bag_files = list(bags_dir.rglob("*"))
                if not any(path.is_file() for path in bag_files):
                    raise RuntimeError(f"Feature bag generation completed but no bag files were written for {extractor_name}.")
        approach_payloads[approach_label] = {
            "feature_extractor_candidates": spec_candidates,
            "feature_extractor_used": extractor_name,
            "extractor_backend": extractor_backend,
            "bags_dir": bags_by_extractor[extractor_name],
        }
        approach_extractors[approach_label] = extractor_name
        approach_backends[approach_label] = extractor_backend

    prepared = {
        **subset,
        "bags_dir": next(iter(bags_by_extractor.values())),
        "feature_extractor_used": next(iter(bags_by_extractor)) if len(bags_by_extractor) == 1 else "multiple",
        "feature_extractor_candidates": feature_candidates,
        "feature_extractors_by_approach": approach_extractors,
        "extractor_backends_by_approach": approach_backends,
        "approach_execution_mode": execution_mode,
        "approaches": approach_payloads,
        "n_folds": int(subset["n_folds"]),
    }
    write_json(bundle_root / "prepared_bundle.json", prepared)
    return prepared


def split_dataset(dataset, split_plan_csv: Path, fold: int, repeat: int = 1):
    split_df = pd.read_csv(split_plan_csv)
    if "slide" not in split_df.columns or "fold" not in split_df.columns:
        raise ValueError("Subset annotations must include 'slide' and 'fold' columns.")

    if "repeat" not in split_df.columns:
        split_df["repeat"] = 1

    slide_folds = (
        split_df[["slide", "repeat", "fold"]]
        .drop_duplicates()
        .copy()
    )
    slide_folds["slide"] = slide_folds["slide"].astype(str)
    slide_folds["fold"] = slide_folds["fold"].astype(int)
    slide_folds["repeat"] = slide_folds["repeat"].astype(int)
    slide_folds = slide_folds.loc[slide_folds["repeat"] == int(repeat)]

    train_slides = slide_folds.loc[slide_folds["fold"] != int(fold), "slide"].tolist()
    val_slides = slide_folds.loc[slide_folds["fold"] == int(fold), "slide"].tolist()
    if not train_slides or not val_slides:
        raise ValueError(
            f"Fold {fold} produced an empty split. "
            f"train={len(train_slides)} val={len(val_slides)}"
        )

    train_dataset = dataset.filter(filters={"slide": train_slides})
    val_dataset = dataset.filter(filters={"slide": val_slides})
    return train_dataset, val_dataset


def infer_score_column(df: pd.DataFrame) -> str:
    for column in [POS_LABEL, f"y_pred_{POS_LABEL}", "y_pred1", "prob_MSI-H", "prediction"]:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            return column
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    raise ValueError(f"Could not infer score column. Numeric columns: {numeric}")


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    thresholds = np.unique(np.clip(np.round(y_score, 6), 0, 1))
    candidates = np.concatenate(([0.0], thresholds, [0.5, 1.0]))
    best_threshold = 0.5
    best_f1 = -1.0
    best_pos_rate = 0.0
    for threshold in np.unique(candidates):
        y_pred = (y_score >= float(threshold)).astype(int)
        score = float(f1_score(y_true, y_pred, average="macro"))
        pos_rate = float(y_pred.mean()) if len(y_pred) else 0.0
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = score
            best_pos_rate = pos_rate
    return best_threshold, best_f1, best_pos_rate


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def aggregate_approach(bundle_root: Path, spec: dict[str, Any], exp_label: str) -> dict[str, Any]:
    sf_root = bundle_root / "slideflow_project"
    prediction_files = sorted(
        p
        for p in list((sf_root / "mil").rglob("predictions.parquet"))
        + list((sf_root / "mil").rglob("predictions.csv"))
        if exp_label in str(p.parent)
    )
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found for {exp_label}")

    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    for path in prediction_files:
        df = read_table(path)
        score_col = infer_score_column(df)
        if OUTCOME in df.columns:
            y_true = (df[OUTCOME] == POS_LABEL).astype(int).to_numpy()
        elif "y_true" in df.columns and pd.api.types.is_numeric_dtype(df["y_true"]):
            y_true = df["y_true"].astype(int).to_numpy()
        else:
            continue
        if len(np.unique(y_true)) < 2:
            continue
        y_score = df[score_col].to_numpy()
        fold_auc = float(roc_auc_score(y_true, y_score))
        default_pred = (y_score >= 0.5).astype(int)
        tuned_threshold, tuned_f1, tuned_pos_rate = best_f1_threshold(y_true, y_score)
        y_pred = (y_score >= tuned_threshold).astype(int)
        rows.append(
            {
                "file": str(path),
                "score_column": score_col,
                "n": int(len(df)),
                "auroc": fold_auc,
                "f1_macro": tuned_f1,
                "f1_macro_default_threshold": float(f1_score(y_true, default_pred, average="macro")),
                "best_threshold": tuned_threshold,
                "predicted_positive_rate": tuned_pos_rate,
                "true_positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
            }
        )
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "file": str(path)}))

    if not rows:
        raise FileNotFoundError(f"No usable prediction rows found for {exp_label}")

    metrics_df = pd.DataFrame(rows)
    metrics = {
        "experiment_id": str(spec["experiment_id"]),
        "approach_label": str(spec["approach_label"]),
        "mil_model": str(spec["mil_model"]),
        "mil_model_requested": parse_candidate_list(spec.get("mil_model_candidates")) or [str(spec["mil_model"])],
        "epochs": int(spec["epochs"]),
        "seed": int(spec["seed"]),
        "mean_auroc": float(metrics_df["auroc"].mean()),
        "mean_f1_macro": float(metrics_df["f1_macro"].mean()),
        "mean_f1_macro_default_threshold": float(metrics_df["f1_macro_default_threshold"].mean()),
        "mean_best_threshold": float(metrics_df["best_threshold"].mean()),
        "folds": int(len(metrics_df)),
        "artifacts": {
            "prediction_files": [str(p) for p in prediction_files],
        },
    }

    approach_dir = bundle_root / "approaches" / str(spec["approach_label"])
    approach_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(approach_dir / "fold_metrics.csv", index=False)
    if curves:
        pd.concat(curves).to_csv(approach_dir / "roc_curves.csv", index=False)
    write_json(approach_dir / "metrics.json", metrics)
    return metrics


def build_transmil_ensemble_summary(
    bundle_root: Path,
    prepared: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    approaches = summary.get("approaches", {})
    if "Approach1" not in approaches or "MonteCarlo" not in approaches:
        return None

    def score_column(df: pd.DataFrame) -> str:
        for candidate in ("y_pred1", POS_LABEL, f"y_pred_{POS_LABEL}", "prediction"):
            if candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
                return candidate
        return infer_score_column(df)

    a1_files = [Path(p) for p in approaches["Approach1"].get("artifacts", {}).get("prediction_files", [])]
    mc_files = [Path(p) for p in approaches["MonteCarlo"].get("artifacts", {}).get("prediction_files", [])]
    rows: list[dict[str, Any]] = []
    for a1_path in a1_files:
        suffix = a1_path.name.split("approach1_transmil_", 1)[-1]
        mc_match = next((path for path in mc_files if path.name.endswith(suffix)), None)
        if mc_match is None:
            continue

        a1_full = read_table(a1_path)
        mc_full = read_table(mc_match)
        a1_df = a1_full[["slide", "y_true", score_column(a1_full)]].copy()
        mc_df = mc_full[["slide", "y_true", score_column(mc_full)]].copy()
        a1_df.columns = ["slide", "y_true", "score_a1"]
        mc_df.columns = ["slide", "y_true", "score_mc"]
        merged = a1_df.merge(mc_df, on=["slide", "y_true"], how="inner")
        if merged.empty or merged["y_true"].nunique() < 2:
            continue

        y_true = merged["y_true"].astype(int).to_numpy()
        y_score = ((merged["score_a1"].to_numpy() + merged["score_mc"].to_numpy()) / 2.0)
        tuned_threshold, tuned_f1, tuned_pos_rate = best_f1_threshold(y_true, y_score)
        default_pred = (y_score >= 0.5).astype(int)
        rows.append(
            {
                "file": f"{a1_path.name}|{mc_match.name}",
                "n": int(len(merged)),
                "auroc": float(roc_auc_score(y_true, y_score)),
                "f1_macro": tuned_f1,
                "f1_macro_default_threshold": float(f1_score(y_true, default_pred, average="macro")),
                "best_threshold": tuned_threshold,
                "predicted_positive_rate": tuned_pos_rate,
                "true_positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
            }
        )

    if not rows:
        return None

    metrics_df = pd.DataFrame(rows)
    metrics = {
        "approach_label": "Ensemble_A1_MC",
        "mil_model": "ensemble_mean",
        "epochs": max(
            int(approaches["Approach1"].get("epochs", 0)),
            int(approaches["MonteCarlo"].get("epochs", 0)),
        ),
        "folds": int(len(metrics_df)),
        "mean_auroc": float(metrics_df["auroc"].mean()),
        "mean_f1_macro": float(metrics_df["f1_macro"].mean()),
        "mean_f1_macro_default_threshold": float(metrics_df["f1_macro_default_threshold"].mean()),
        "mean_best_threshold": float(metrics_df["best_threshold"].mean()),
        "n_folds": int(prepared.get("n_folds", 0)),
        "n_repeats": int(prepared.get("n_repeats", 1)),
        "artifacts": {
            "ensemble_of": ["Approach1", "MonteCarlo"],
        },
    }
    ensemble_dir = bundle_root / "approaches" / "Ensemble_A1_MC"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(ensemble_dir / "fold_metrics.csv", index=False)
    write_json(ensemble_dir / "metrics.json", metrics)
    return metrics


def build_mil_config(mil, candidate_models: list[str], spec: dict[str, Any]):
    errors: dict[str, str] = {}
    for model_name in candidate_models:
        try:
            trainer_kwargs = {
                "lr": float(spec.get("learning_rate", 5e-5 if str(model_name) == "transmil" else 1e-4)),
                "epochs": int(spec["epochs"]),
                "batch_size": int(spec.get("mil_batch_size", 16)),
                "bag_size": int(spec.get("mil_bag_size", 64)),
                "drop_last": bool(spec.get("drop_last", False)),
                "wd": float(spec.get("weight_decay", 1e-4)),
                "fit_one_cycle": bool(spec.get("fit_one_cycle", True)),
                "weighted_loss": bool(spec.get("weighted_loss", True)),
            }
            if spec.get("max_val_bag_size") not in (None, ""):
                trainer_kwargs["max_val_bag_size"] = int(spec["max_val_bag_size"])
            config_obj = mil.mil_config(
                str(model_name),
                **trainer_kwargs,
            )
            return config_obj, str(model_name), errors
        except Exception as exc:
            errors[str(model_name)] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"Unable to initialize MIL config. Tried: {errors}")


def runner_python_executable() -> str:
    hybrid = Path("/home/pardeep/.venvs/pathology310-hybrid/bin/python")
    if hybrid.exists():
        return str(hybrid)
    preferred = Path("/opt/miniforge3/envs/pathology310/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def ensure_feature_bags_for_approach(
    config: dict[str, Any],
    prepared: dict[str, Any],
    spec: dict[str, Any],
    approach_payload: dict[str, Any],
    sf,
    project,
    dataset,
) -> Path:
    bags_dir = Path(str(approach_payload.get("bags_dir") or prepared["bags_dir"]))
    existing_files = [path for path in bags_dir.rglob("*") if path.is_file()] if bags_dir.exists() else []
    if existing_files:
        return bags_dir

    feature_candidates = approach_payload.get("feature_extractor_candidates") or requested_feature_extractors_for_spec(
        config,
        spec,
        prepared.get("feature_extractor_candidates", requested_feature_extractors(config)),
    )
    extractor, extractor_name, _extractor_backend = build_extractor(
        sf,
        feature_candidates,
        requested_virchow_weights(config),
        hf_token=requested_hf_token(config),
        requested_backend=requested_extractor_backend(config, spec),
    )
    bags_dir.parent.mkdir(parents=True, exist_ok=True)
    bags_dir.mkdir(parents=True, exist_ok=True)
    project.generate_feature_bags(extractor, dataset, outdir=str(bags_dir))
    bag_files = [path for path in bags_dir.rglob("*") if path.is_file()]
    if not bag_files:
        raise RuntimeError(f"Feature bag generation completed but no bag files were written for {extractor_name}.")
    return bags_dir


def should_cleanup_bags_after_approach(prepared: dict[str, Any], approach_label: str) -> bool:
    if prepared.get("approach_execution_mode") != "sequential":
        return False
    current = prepared.get("approaches", {}).get(approach_label, {})
    current_extractor = current.get("feature_extractor_used")
    if not current_extractor:
        return False
    for other_label, payload in prepared.get("approaches", {}).items():
        if other_label == approach_label:
            continue
        if payload.get("feature_extractor_used") == current_extractor:
            return False
    return True


def train_one_approach(config: dict[str, Any], approach_label: str) -> None:
    bundle_root = Path(config["bundle_root"])
    prepared = read_json(bundle_root / "prepared_bundle.json")
    spec = next(spec for spec in config["specs"] if spec["approach_label"] == approach_label)

    subset_annotations = Path(prepared["subset_annotations"])
    split_plan = Path(prepared.get("split_plan", prepared["subset_annotations"]))
    slides_dir = Path(prepared["slides_dir"])
    sf, project = load_project(bundle_root, subset_annotations, slides_dir)
    import slideflow.mil as mil
    dataset = make_dataset(project)
    n_folds = requested_n_folds(config)
    n_repeats = int(prepared.get("n_repeats", requested_n_repeats(config)))
    mil_candidates = requested_mil_models(spec)

    update_approach_status(
        config,
        approach_label,
        "training",
        experiment_id=spec["experiment_id"],
        mil_model=spec["mil_model"],
        mil_model_candidates=mil_candidates,
        n_folds=n_folds,
        n_repeats=n_repeats,
    )
    config_obj, configured_model, config_errors = build_mil_config(mil, mil_candidates, spec)
    last_error: Exception | None = None
    runtime_errors: dict[str, str] = {}
    approach_payload = prepared.get("approaches", {}).get(approach_label, {})
    bags_dir = ensure_feature_bags_for_approach(
        config,
        prepared,
        spec,
        approach_payload,
        sf,
        project,
        dataset,
    )
    filtered_split_plan, missing_bag_slides = filtered_split_plan_for_bags(
        split_plan,
        bags_dir,
        bundle_root / "approaches" / approach_label / "available_split_plan.csv",
    )
    sf_root = bundle_root / "slideflow_project"

    for model_name in [configured_model, *[name for name in mil_candidates if name != configured_model]]:
        exp_label: str | None = None
        try:
            if model_name != configured_model:
                config_obj, configured_model, model_init_errors = build_mil_config(mil, [model_name], spec)
                config_errors.update(model_init_errors)
            spec["mil_model"] = model_name
            exp_label = f"{config['bundle_id']}_{spec['experiment_id']}_{approach_label.lower()}_{model_name}"
            cleanup_experiment_artifacts(sf_root, exp_label)
            for repeat in range(1, n_repeats + 1):
                for fold in range(1, n_folds + 1):
                    train_ds, val_ds = split_dataset(dataset, filtered_split_plan, fold, repeat=repeat)
                    outdir = mil_output_dir(sf_root, exp_label, repeat, fold)
                    if outdir.exists():
                        shutil.rmtree(outdir, ignore_errors=True)
                    project.train_mil(
                        config=config_obj,
                        train_dataset=train_ds,
                        val_dataset=val_ds,
                        outcomes=OUTCOME,
                        bags=str(bags_dir),
                        outdir=str(outdir),
                        exp_label=f"{exp_label}_repeat_{repeat}_fold_{fold}",
                        attention_heatmaps=False,
                    )

            metrics = aggregate_approach(bundle_root, spec, exp_label)
            metrics["mil_model_candidates"] = mil_candidates
            metrics["mil_model_config_errors"] = config_errors
            metrics["n_folds"] = n_folds
            metrics["n_repeats"] = n_repeats
            metrics["feature_extractor_used"] = approach_payload.get("feature_extractor_used", prepared.get("feature_extractor_used"))
            metrics["available_bag_slide_count"] = len(available_bag_slide_stems(bags_dir))
            metrics["missing_bag_slides"] = missing_bag_slides
            metrics_path = bundle_root / "approaches" / approach_label / "metrics.json"
            write_json(metrics_path, metrics)
            update_approach_status(config, approach_label, "completed", metrics=metrics)
            if should_cleanup_bags_after_approach(prepared, approach_label) and bags_dir.exists():
                shutil.rmtree(bags_dir, ignore_errors=True)
            return
        except Exception as exc:
            runtime_errors[model_name] = f"{type(exc).__name__}: {exc}"
            if exp_label:
                cleanup_experiment_artifacts(sf_root, exp_label)
            last_error = exc
            continue

    raise RuntimeError(
        f"All MIL candidates failed for {approach_label}. "
        f"Config errors: {config_errors}. Runtime errors: {runtime_errors}"
    ) from last_error


def launch_approach_process(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(config["bundle_root"])
    python_executable = runner_python_executable()
    bundle_config_path = resolved_bundle_config_path(config)
    approach_label = str(spec["approach_label"])
    log_path = bundle_root / "approaches" / approach_label / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        child_env = os.environ.copy()
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env.pop("PYTHONPATH", None)
        child_env.pop("VIRTUAL_ENV", None)
        process = subprocess.Popen(
            [
                python_executable,
                str(Path(__file__).resolve()),
                "--bundle-config",
                str(bundle_config_path),
                "--stage",
                "train-approach",
                "--approach-label",
                approach_label,
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(bundle_root),
            env=child_env,
        )
    update_approach_status(config, approach_label, "spawned", pid=int(process.pid))
    return {"approach_label": approach_label, "pid": int(process.pid)}


def launch_parallel_approaches(config: dict[str, Any]) -> list[dict[str, Any]]:
    limit = max(1, int(config["request"].get("max_parallel_approaches", 2)))
    return [launch_approach_process(config, spec) for spec in config["specs"][:limit]]


def wait_for_parallel_approaches(config: dict[str, Any], processes: list[dict[str, Any]]) -> dict[str, Any]:
    bundle_root = Path(config["bundle_root"])
    active = {entry["pid"]: entry for entry in processes}
    limit = max(1, int(config["request"].get("max_parallel_approaches", 2)))
    launched_labels = {str(entry["approach_label"]) for entry in processes}
    pending_specs = [spec for spec in config["specs"] if str(spec["approach_label"]) not in launched_labels]
    completed: dict[str, Any] = {}
    while active:
        for pid, entry in list(active.items()):
            result = run_command(["bash", "-lc", f"ps -p {pid} >/dev/null 2>&1"])
            if result.returncode == 0:
                continue
            approach_label = str(entry["approach_label"])
            metrics_path = bundle_root / "approaches" / approach_label / "metrics.json"
            status_path = bundle_root / "approaches" / approach_label / "status.json"
            completed[approach_label] = read_json(metrics_path) if metrics_path.exists() else read_json(status_path)
            del active[pid]

        while pending_specs and len(active) < limit:
            next_spec = pending_specs.pop(0)
            next_process = launch_approach_process(config, next_spec)
            active[next_process["pid"]] = next_process

        update_bundle_status(
            config,
            "training_parallel",
            running_approaches=[entry["approach_label"] for entry in active.values()],
            completed_approaches=list(completed),
            pending_approaches=[str(spec["approach_label"]) for spec in pending_specs],
        )
        if active:
            time.sleep(15)
    return completed


def collect_existing_approach_payloads(config: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(config["bundle_root"])
    payloads: dict[str, Any] = {}
    for spec in config["specs"]:
        approach_label = str(spec["approach_label"])
        metrics_path = bundle_root / "approaches" / approach_label / "metrics.json"
        status_path = bundle_root / "approaches" / approach_label / "status.json"
        if metrics_path.exists():
            payloads[approach_label] = read_json(metrics_path)
        elif status_path.exists():
            payloads[approach_label] = read_json(status_path)
    return payloads


def finalize_bundle(config: dict[str, Any], prepared: dict[str, Any], approach_payloads: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(config["bundle_root"])
    summary_view = summarize_approach_payloads(config, prepared, approach_payloads)
    final_summary = {
        "bundle_id": config["bundle_id"],
        "bucket_uri": config["request"]["bucket_uri"],
        "requested_slide_limit": requested_slide_limit(config),
        "slide_limit": int(prepared.get("selected_slide_limit", len(prepared["selected_slide_names"]))),
        "n_folds": int(prepared.get("n_folds", requested_n_folds(config))),
        "n_repeats": int(prepared.get("n_repeats", requested_n_repeats(config))),
        "selected_slide_count": len(prepared["selected_slide_names"]),
        "selected_slides": prepared["selected_slide_names"],
        "label_counts": prepared["label_counts"],
        "feature_extractor_used": prepared["feature_extractor_used"],
        "feature_extractor_candidates": prepared.get("feature_extractor_candidates", []),
        "feature_extractors_by_approach": prepared.get("feature_extractors_by_approach", {}),
        "approaches": summary_view["approaches"],
        "state": summary_view["overall_state"],
        "best_approach": summary_view["best_approach"],
        "completed_approach_count": summary_view["completed_count"],
        "failed_approach_count": summary_view["failed_count"],
        "running_approach_count": summary_view["running_count"],
    }
    ensemble_metrics = build_transmil_ensemble_summary(bundle_root, prepared, final_summary)
    if ensemble_metrics:
        final_summary["approaches"]["Ensemble_A1_MC"] = ensemble_metrics
    write_json(bundle_root / "final_summary.json", final_summary)
    status_extra = {
        "selected_slide_count": len(prepared["selected_slide_names"]),
        "label_counts": prepared["label_counts"],
        "feature_extractor_used": prepared["feature_extractor_used"],
        "best_approach": summary_view["best_approach"],
        "approach_states": {
            label: payload.get("state")
            for label, payload in summary_view["approaches"].items()
        },
        "completed_approach_count": summary_view["completed_count"],
        "failed_approach_count": summary_view["failed_count"],
        "running_approach_count": summary_view["running_count"],
        "summary": final_summary,
    }
    update_bundle_status(config, summary_view["overall_state"], **status_extra)
    return final_summary


def rebuild_existing_bundle(config: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(config["bundle_root"])
    prepared_path = bundle_root / "prepared_bundle.json"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Missing prepared bundle metadata: {prepared_path}")
    prepared = read_json(prepared_path)
    approach_payloads = collect_existing_approach_payloads(config)
    if not approach_payloads:
        raise FileNotFoundError(f"No approach metrics or statuses found under {bundle_root / 'approaches'}")
    return finalize_bundle(config, prepared, approach_payloads)


def run_all(config: dict[str, Any]) -> None:
    prepared = prepare_bundle_for_training(config)
    update_bundle_status(
        config,
        "prepared",
        matched_slide_count=int(prepared.get("matched_slide_count", 0)),
        selected_slide_count=len(prepared["selected_slide_names"]),
        selected_slides=prepared["selected_slide_names"],
        label_counts=prepared["label_counts"],
        feature_extractor_used=prepared["feature_extractor_used"],
        n_folds=int(prepared.get("n_folds", requested_n_folds(config))),
        n_repeats=int(prepared.get("n_repeats", requested_n_repeats(config))),
    )
    processes = launch_parallel_approaches(config)
    update_bundle_status(config, "training_parallel", processes=processes)
    completed = wait_for_parallel_approaches(config, processes)
    finalize_bundle(config, prepared, completed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an automated TCGA COAD adaptive triad bundle.")
    parser.add_argument("--bundle-config", required=True)
    parser.add_argument("--stage", choices=["all", "train-approach", "finalize-existing"], default="all")
    parser.add_argument("--approach-label")
    args = parser.parse_args()

    config = read_json(Path(args.bundle_config))
    config["_bundle_config_path"] = str(Path(args.bundle_config).resolve())
    try:
        if args.stage == "all":
            run_all(config)
        elif args.stage == "finalize-existing":
            rebuild_existing_bundle(config)
        else:
            if not args.approach_label:
                raise ValueError("--approach-label is required for train-approach stage.")
            train_one_approach(config, args.approach_label)
    except Exception as exc:
        if args.stage == "train-approach" and args.approach_label:
            update_approach_status(
                config,
                args.approach_label,
                "failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        update_bundle_status(
            config,
            "failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
