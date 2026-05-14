import json
import shlex
from pathlib import PurePosixPath
from typing import Any

from django.conf import settings

from apps.approaches.models import ApproachTemplate
from apps.archives.models import BatchArchive
from apps.vm.registry import ensure_default_vm_target
from apps.vm.services import default_vm_target, glob_latest_json, read_json_file, run_shell, upload_text

from .models import Run, RunApproachLink


def build_bundle_config(run: Run, target=None) -> dict[str, Any]:
    target = target or default_vm_target()
    bundle_root = PurePosixPath(target.project_root) / "automation" / "tcga_slide_triads" / run.run_id
    status_path = bundle_root / "status.json"
    links = list(run.approach_links.select_related("approach_template").all())
    requested_parallel = max(1, int(getattr(settings, "VM_MAX_PARALLEL_APPROACHES", 2)))
    max_parallel_approaches = min(requested_parallel, max(1, len(links) or 1))
    approach_execution_mode = "parallel" if max_parallel_approaches > 1 else "sequential"
    feature_extractor = ",".join(run.feature_extractor_candidates or [run.feature_extractor_used]).strip(",")
    shared = {
        "bundle_id": run.run_id,
        "bucket_uri": run.source_uri,
        "slide_limit": run.requested_slide_limit,
        "n_folds": run.n_folds,
        "n_repeats": run.n_repeats,
        "repeat_seeds": run.repeat_seeds,
        "external_cohorts": run.external_cohorts,
        "preferred_slide_pattern": "DX",
        "preferred_exact_suffix": "DX1",
        "annotations_csv": run.annotations_csv or settings.VM_DEFAULT_ANNOTATIONS,
        "feature_extractor": feature_extractor,
        "virchow_weights": settings.VM_VIRCHOW_WEIGHTS,
        "allow_generic_fallback": False,
        "hf_token": settings.HF_TOKEN,
        "tile_px": run.tile_px,
        "tile_um": run.tile_um,
        "max_parallel_approaches": max_parallel_approaches,
        "approach_execution_mode": approach_execution_mode,
        "max_tiles_per_slide": run.max_tiles_per_slide,
        "mpp_override": 0.25,
        "qc_method": "otsu",
    }
    specs = []
    if not links:
        for template in ApproachTemplate.objects.filter(is_active=True).order_by("key"):
            links.append(
                RunApproachLink(
                    run=run,
                    approach_template=template,
                    trainer_params=template.default_params,
                )
            )
    for index, link in enumerate(links, start=1):
        params = dict(link.trainer_params or {})
        label = link.approach_template.label.replace(" ", "")
        model_family = link.approach_template.model_family
        specs.append(
            {
                "experiment_name": f"{run.experiment_name}-{link.approach_template.key}-{model_family}-seed{params.get('seed', 310)}",
                **shared,
                "experiment_id": f"{run.run_id}_a{index}",
                "training_mode": str(params.get("training_mode", "mil")),
                "approach_label": label,
                "feature_extractor": str(params.get("feature_extractor", feature_extractor)),
                "extractor_backend": str(params.get("extractor_backend", "auto")),
                "mil_model": model_family,
                "mil_model_candidates": [model_family],
                "epochs": int(params.get("epochs", 20)),
                "learning_rate": float(params.get("learning_rate", 5e-5)),
                "weight_decay": float(params.get("weight_decay", 1e-4)),
                "mil_batch_size": int(params.get("mil_batch_size", 12)),
                "mil_bag_size": int(params.get("bag_size", 128)),
                "max_val_bag_size": int(params.get("max_val_bag_size", params.get("bag_size", 128))),
                "fit_one_cycle": bool(params.get("fit_one_cycle", True)),
                "weighted_loss": bool(params.get("weighted_loss", True)),
                "seed": int(params.get("seed", 310)),
            }
        )
    return {
        "bundle_id": run.run_id,
        "bundle_root": str(bundle_root),
        "status_path": str(status_path),
        "request": {"experiment_name": run.experiment_name, **shared},
        "specs": specs,
    }


def launch_run_on_vm(run: Run) -> dict[str, Any]:
    target = ensure_default_vm_target()
    bundle_config = build_bundle_config(run, target=target)
    runtime_root = PurePosixPath(target.project_root) / "django_rebuild_cleaned_msi" / "runtime"
    bundle_root = PurePosixPath(bundle_config["bundle_root"])
    config_path = runtime_root / "bundle_configs" / f"{run.run_id}.json"
    bundle_root_config_path = bundle_root / "bundle_config.json"
    launch_log_path = runtime_root / "launch_logs" / f"{run.run_id}.log"
    config_payload = json.dumps(bundle_config, indent=2)
    upload_text(target, str(config_path), config_payload)
    upload_text(target, str(bundle_root_config_path), config_payload)
    run.vm_target = target
    run.bundle_config_path = str(config_path)
    run.remote_status_path = str(bundle_config["status_path"])
    run.remote_launch_log_path = str(launch_log_path)
    run.archive_path = str(PurePosixPath(target.project_root) / "automation" / "tcga_batch_archives*")
    run.save(update_fields=["vm_target", "bundle_config_path", "remote_status_path", "remote_launch_log_path", "archive_path", "updated_at"])
    spawn_code = (
        "import os, subprocess; "
        f"os.makedirs({str(launch_log_path.parent)!r}, exist_ok=True); "
        + (
            f"os.environ['MSI_STUDENT_ENCODER_DIR'] = {settings.VM_STUDENT_ENCODER_DIR!r}; "
            if getattr(settings, "VM_STUDENT_ENCODER_DIR", "")
            else ""
        )
        + 
        f"log = open({str(launch_log_path)!r}, 'ab'); "
        "proc = subprocess.Popen("
        f"[{target.runner_python!r}, {settings.VM_RUNNER_SCRIPT!r}, '--bundle-config', {str(config_path)!r}], "
        f"cwd={target.project_root!r}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True"
        "); "
        "print(proc.pid)"
    )
    command = f"/usr/bin/python3 -c {shlex.quote(spawn_code)}"
    result = run_shell(target, command, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Failed to launch VM runner.")
    run.remote_pid = result.stdout.strip().splitlines()[-1].strip()
    run.save(update_fields=["remote_pid", "updated_at"])
    return {
        "run_id": run.run_id,
        "remote_pid": run.remote_pid,
        "bundle_config_path": run.bundle_config_path,
        "remote_status_path": run.remote_status_path,
        "remote_launch_log_path": run.remote_launch_log_path,
    }


def _bundle_root_for_run(run: Run) -> PurePosixPath | None:
    if run.remote_status_path and run.remote_status_path.startswith("/"):
        return PurePosixPath(run.remote_status_path).parent
    if run.bundle_config_path and run.bundle_config_path.startswith("/"):
        config_root = PurePosixPath(run.bundle_config_path).parent.parent.parent
        return config_root / "automation" / "tcga_slide_triads" / run.run_id
    return None


def _read_live_runtime_snapshot(run: Run) -> dict[str, Any]:
    target = default_vm_target()
    bundle_root = _bundle_root_for_run(run)
    if bundle_root is None:
        return {}

    payload = {
        "bundle_root": str(bundle_root),
        "config_path": run.bundle_config_path,
        "launch_log_path": run.remote_launch_log_path,
    }
    remote_script = f"""
import json
import re
from pathlib import Path

payload = json.loads({json.dumps(json.dumps(payload))})
bundle_root = Path(payload["bundle_root"])
config_path = Path(payload["config_path"]) if payload.get("config_path") else None
launch_log_path = Path(payload["launch_log_path"]) if payload.get("launch_log_path") else None
data = {{
    "bundle_root": str(bundle_root),
    "spec_count": 0,
    "selected_slide_count": None,
    "current_extractor": "",
    "extractor_sequence": [],
    "bag_counts": {{}},
    "approach_status_count": 0,
    "approach_metrics_count": 0,
    "missing_bag_slides": [],
    "missing_bag_count": 0,
}}

if config_path and config_path.exists():
    config = json.loads(config_path.read_text())
    data["spec_count"] = len(config.get("specs", []))
    request = config.get("request", {{}})
    data["selected_slide_count"] = request.get("slide_limit") or request.get("requested_slide_limit")

log_pattern = re.compile(r"\\[(\\d{{2}}:\\d{{2}}:\\d{{2}})\\].*Using feature extractor:\\s+([^\\s]+)")
if launch_log_path and launch_log_path.exists():
    for line in launch_log_path.read_text(errors="ignore").splitlines():
        match = log_pattern.search(line)
        if not match:
            continue
        data["extractor_sequence"].append({{"time": match.group(1), "extractor": match.group(2)}})
    if data["extractor_sequence"]:
        data["current_extractor"] = data["extractor_sequence"][-1]["extractor"]

manifest_path = bundle_root / "slideflow_project" / "tfrecords" / "256px_128um" / "manifest.json"
manifest_stems = set()
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    manifest_stems = {{name[:-10] if name.endswith(".tfrecords") else name for name in manifest.keys()}}
    if not data["selected_slide_count"]:
        data["selected_slide_count"] = len(manifest_stems)

bags_root = bundle_root / "slideflow_project" / "bags"
current_bag_stems = set()
if bags_root.exists():
    for bag_dir in sorted([path for path in bags_root.iterdir() if path.is_dir()]):
        pt_count = sum(1 for _ in bag_dir.glob("*.pt"))
        idx_count = sum(1 for _ in bag_dir.glob("*.index.npz"))
        data["bag_counts"][bag_dir.name] = {{"pt": pt_count, "index": idx_count}}
        if data["current_extractor"] and bag_dir.name.startswith(data["current_extractor"] + "_"):
            current_bag_stems = {{path.stem for path in bag_dir.glob("*.pt")}}

if manifest_stems and current_bag_stems:
    missing = sorted(manifest_stems - current_bag_stems)
    data["missing_bag_slides"] = missing[:10]
    data["missing_bag_count"] = len(missing)
elif manifest_stems and data["current_extractor"]:
    data["missing_bag_slides"] = sorted(manifest_stems)[:10]
    data["missing_bag_count"] = len(manifest_stems)

approaches_root = bundle_root / "approaches"
if approaches_root.exists():
    data["approach_status_count"] = sum(1 for _ in approaches_root.rglob("status.json"))
    data["approach_metrics_count"] = sum(1 for _ in approaches_root.rglob("metrics.json"))

print(json.dumps(data))
"""
    command = f"/usr/bin/python3 - <<'PY'\n{remote_script}\nPY"
    result = run_shell(target, command, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def sync_run_status(run: Run) -> dict[str, Any]:
    target = default_vm_target()
    if (not run.remote_status_path or "<bundle_id>" in run.remote_status_path) and run.bundle_config_path:
        config_payload = read_json_file(target, run.bundle_config_path)
        run.remote_status_path = config_payload["status_path"]
        run.save(update_fields=["remote_status_path", "updated_at"])
    if not run.remote_status_path:
        raise ValueError("Run does not have a remote status path yet.")
    status_payload = read_json_file(target, run.remote_status_path)
    run.state = status_payload.get("state", run.state)
    if "selected_slide_count" in status_payload:
        run.selected_slide_count = int(status_payload["selected_slide_count"])
    if "label_counts" in status_payload:
        run.label_counts = status_payload["label_counts"]
    run.save(update_fields=["state", "selected_slide_count", "label_counts", "updated_at"])

    live_runtime = _read_live_runtime_snapshot(run)
    current_extractor = str(live_runtime.get("current_extractor") or "").strip()
    if current_extractor:
        run.feature_extractor_used = current_extractor
        run.save(update_fields=["feature_extractor_used", "updated_at"])

    approach_states: list[str] = []
    for link in run.approach_links.select_related("approach_template"):
        approach_label = link.approach_template.label.replace(" ", "")
        status_path = PurePosixPath(run.remote_status_path).parent / "approaches" / approach_label / "status.json"
        metrics_path = PurePosixPath(run.remote_status_path).parent / "approaches" / approach_label / "metrics.json"
        try:
            payload = read_json_file(target, str(metrics_path))
            link.state = "completed"
            link.metrics_path = str(metrics_path)
            link.mean_auroc = payload.get("mean_auroc")
            link.mean_f1_macro = payload.get("mean_f1_macro")
            link.mean_f1_macro_default_threshold = payload.get("mean_f1_macro_default_threshold")
            link.mean_auprc = payload.get("mean_auprc")
            link.mean_balanced_accuracy = payload.get("mean_balanced_accuracy")
            link.mean_precision = payload.get("mean_precision")
            link.mean_recall_msi_h = payload.get("mean_recall_msi_h")
            link.mean_specificity = payload.get("mean_specificity")
            link.mean_best_threshold = payload.get("mean_best_threshold")
            link.mean_brier_score = payload.get("mean_brier_score")
            link.auroc_std = payload.get("auroc_std")
            link.auroc_ci_low = payload.get("auroc_ci_low")
            link.auroc_ci_high = payload.get("auroc_ci_high")
            link.auroc_per_fold = payload.get("auroc_per_fold", link.auroc_per_fold)
            link.fold_metrics = payload.get("fold_metrics", link.fold_metrics)
            link.aggregate_confusion_matrix = payload.get(
                "aggregate_confusion_matrix",
                link.aggregate_confusion_matrix,
            )
            link.available_bag_slide_count = payload.get("available_bag_slide_count")
            link.missing_bag_slides = payload.get("missing_bag_slides", link.missing_bag_slides)
            link.external_metrics = payload.get("external_metrics", link.external_metrics)
            link.prediction_artifacts = payload.get("artifacts", link.prediction_artifacts)
        except Exception:
            try:
                payload = read_json_file(target, str(status_path))
                link.state = payload.get("state", link.state)
            except Exception:
                continue
        approach_states.append(link.state)
        link.save(
            update_fields=[
                "state",
                "metrics_path",
                "mean_auroc",
                "mean_f1_macro",
                "mean_f1_macro_default_threshold",
                "mean_auprc",
                "mean_balanced_accuracy",
                "mean_precision",
                "mean_recall_msi_h",
                "mean_specificity",
                "mean_best_threshold",
                "mean_brier_score",
                "auroc_std",
                "auroc_ci_low",
                "auroc_ci_high",
                "auroc_per_fold",
                "fold_metrics",
                "aggregate_confusion_matrix",
                "available_bag_slide_count",
                "missing_bag_slides",
                "external_metrics",
                "prediction_artifacts",
                "updated_at",
            ]
        )

    final_summary_path = PurePosixPath(run.remote_status_path).parent / "final_summary.json"
    try:
        summary = read_json_file(target, str(final_summary_path))
        if summary.get("feature_extractor_used"):
            run.feature_extractor_used = summary["feature_extractor_used"]
        if summary.get("selected_slide_count") is not None:
            run.selected_slide_count = int(summary["selected_slide_count"])
        if summary.get("label_counts"):
            run.label_counts = summary["label_counts"]
        if summary.get("external_cohorts") is not None:
            run.external_cohorts = summary["external_cohorts"]
        run.save(
            update_fields=[
                "feature_extractor_used",
                "selected_slide_count",
                "label_counts",
                "external_cohorts",
                "updated_at",
            ]
        )
    except Exception:
        summary = None

    if approach_states:
        if all(state == "completed" for state in approach_states):
            run.state = "completed"
        elif any(state == "failed" for state in approach_states):
            run.state = "failed"
        elif any(state == "training" for state in approach_states):
            run.state = "training_parallel"
        elif any(state == "spawned" for state in approach_states):
            run.state = "training_parallel"
        run.save(update_fields=["state", "updated_at"])
    elif live_runtime.get("approach_status_count") or live_runtime.get("approach_metrics_count"):
        run.state = "training_parallel"
        run.save(update_fields=["state", "updated_at"])

    if not run.feature_extractor_used:
        used = [item for item in (run.feature_extractor_candidates or []) if item]
        if used:
            run.feature_extractor_used = ",".join(used)
            run.save(update_fields=["feature_extractor_used", "updated_at"])

    return {
        "run_id": run.run_id,
        "status": status_payload,
        "final_summary": summary,
        "live_runtime": live_runtime,
    }


def sync_latest_archive() -> dict[str, Any]:
    hit = glob_latest_json(default_vm_target(), settings.VM_ARCHIVE_GLOB, "orchestration_status.json")
    source_kind = "archive"
    if hit is None:
        hit = glob_latest_json(default_vm_target(), settings.VM_LIVE_BUNDLE_GLOB, "final_summary.json")
        source_kind = "live_bundle_summary"
    if hit is None:
        hit = glob_latest_json(default_vm_target(), settings.VM_LIVE_BUNDLE_GLOB, "status.json")
        source_kind = "live_bundle_status"
    if hit is None:
        raise FileNotFoundError("No archive or live bundle summary/status file found on the VM.")
    archive_status_path, payload = hit
    archive_root = str(PurePosixPath(archive_status_path).parent)
    archive, _ = BatchArchive.objects.update_or_create(
        archive_root=archive_root,
        defaults={
            "state": payload.get("state", "completed"),
            "completed_batches": int(payload.get("completed_batches", 0)),
            "aggregate_label_counts": payload.get("aggregate_label_counts", {}),
            "aggregate_approaches": payload.get("aggregate_approaches", {}),
            "orchestration_status_path": archive_status_path,
        },
    )
    return {
        "archive_root": archive.archive_root,
        "state": archive.state,
        "completed_batches": archive.completed_batches,
        "orchestration_status_path": archive.orchestration_status_path,
        "source_kind": source_kind,
    }
