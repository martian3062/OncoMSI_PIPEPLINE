import json
import shlex
from pathlib import PurePosixPath
from typing import Any

from django.conf import settings

from apps.approaches.models import ApproachTemplate
from apps.archives.models import BatchArchive
from apps.vm.services import default_vm_target, glob_latest_json, read_json_file, run_shell, upload_text

from .models import Run, RunApproachLink


def build_bundle_config(run: Run) -> dict[str, Any]:
    target = default_vm_target()
    bundle_root = PurePosixPath(target.project_root) / "automation" / "tcga_slide_triads" / run.run_id
    status_path = bundle_root / "status.json"
    links = list(run.approach_links.select_related("approach_template").all())
    feature_extractor = ",".join(run.feature_extractor_candidates or [run.feature_extractor_used]).strip(",")
    shared = {
        "bundle_id": run.run_id,
        "bucket_uri": run.source_uri,
        "slide_limit": run.requested_slide_limit,
        "n_folds": run.n_folds,
        "n_repeats": run.n_repeats,
        "preferred_slide_pattern": "DX",
        "preferred_exact_suffix": "DX1",
        "annotations_csv": run.annotations_csv or settings.VM_DEFAULT_ANNOTATIONS,
        "feature_extractor": feature_extractor,
        "virchow_weights": settings.VM_VIRCHOW_WEIGHTS,
        "allow_generic_fallback": False,
        "hf_token": settings.HF_TOKEN,
        "tile_px": run.tile_px,
        "tile_um": run.tile_um,
        "max_parallel_approaches": max(1, len(links) or 1),
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
                "training_mode": "mil",
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
    target = default_vm_target()
    bundle_config = build_bundle_config(run)
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
        except Exception:
            try:
                payload = read_json_file(target, str(status_path))
                link.state = payload.get("state", link.state)
            except Exception:
                continue
        approach_states.append(link.state)
        link.save(update_fields=["state", "metrics_path", "mean_auroc", "mean_f1_macro", "updated_at"])

    final_summary_path = PurePosixPath(run.remote_status_path).parent / "final_summary.json"
    try:
        summary = read_json_file(target, str(final_summary_path))
        run.feature_extractor_used = summary.get("feature_extractor_used", run.feature_extractor_used)
        run.selected_slide_count = int(summary.get("selected_slide_count", run.selected_slide_count))
        run.label_counts = summary.get("label_counts", run.label_counts)
        run.save(update_fields=["feature_extractor_used", "selected_slide_count", "label_counts", "updated_at"])
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

    return {
        "run_id": run.run_id,
        "status": status_payload,
        "final_summary": summary,
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
