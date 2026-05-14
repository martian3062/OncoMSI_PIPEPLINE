import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings

from .inference import predict_upload, temporary_pipeline_mode
from .library_data import append_prediction_history


_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_CAPACITY_CONDITION = threading.Condition()
_CAPACITY_IN_USE = 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    for key in ("created_at", "started_at", "finished_at"):
        value = payload.get(key)
        payload[key] = value.isoformat() if isinstance(value, datetime) else None
    return payload


def _set_job(job_id: str, **changes: Any) -> None:
    with _JOB_LOCK:
        if job_id not in _JOBS:
            return
        _JOBS[job_id].update(changes)


def _job_weight_for_mode(pipeline_mode: str) -> int:
    if str(pipeline_mode).strip().lower() == "manager1":
        return max(1, int(getattr(settings, "MSI_EXACT_JOB_WEIGHT", 2) or 2))
    return max(1, int(getattr(settings, "MSI_FAST_JOB_WEIGHT", 1) or 1))


def _job_capacity_limit() -> int:
    return max(1, int(getattr(settings, "MSI_JOB_CAPACITY_LIMIT", 2) or 2))


def _queue_position_locked(job_id: str, pipeline_mode: str) -> int:
    target_weight = _job_weight_for_mode(pipeline_mode)
    ahead = 0
    for queued_job in _JOBS.values():
        if queued_job.get("job_id") == job_id:
            break
        if queued_job.get("status") == "queued":
            ahead += _job_weight_for_mode(str(queued_job.get("pipeline_mode") or "manager2"))
    if ahead <= 0:
        return 1
    return 1 + (ahead // max(target_weight, 1))


def _update_waiting_jobs_locked() -> None:
    for job in _JOBS.values():
        if job.get("status") != "queued":
            continue
        queue_position = _queue_position_locked(str(job.get("job_id") or ""), str(job.get("pipeline_mode") or "manager2"))
        job["progress"] = {
            "stage": "queued",
            "label": "Queued",
            "detail": f"The predictor queue is active. This request is waiting for a safe VM slot. Queue position {queue_position}.",
            "percent": 0,
        }


def _acquire_job_capacity(job_id: str, pipeline_mode: str) -> None:
    global _CAPACITY_IN_USE
    weight = _job_weight_for_mode(pipeline_mode)
    limit = _job_capacity_limit()
    with _CAPACITY_CONDITION:
        while _CAPACITY_IN_USE + weight > limit:
            with _JOB_LOCK:
                _update_waiting_jobs_locked()
            _CAPACITY_CONDITION.wait(timeout=1.0)
        _CAPACITY_IN_USE += weight
        with _JOB_LOCK:
            _update_waiting_jobs_locked()


def _release_job_capacity(pipeline_mode: str) -> None:
    global _CAPACITY_IN_USE
    weight = _job_weight_for_mode(pipeline_mode)
    with _CAPACITY_CONDITION:
        _CAPACITY_IN_USE = max(0, _CAPACITY_IN_USE - weight)
        _CAPACITY_CONDITION.notify_all()
        with _JOB_LOCK:
            _update_waiting_jobs_locked()


def _run_prediction_job(
    job_id: str,
    upload_path: Path,
    uploaded_name: str,
    pipeline_mode: str,
    *,
    delete_upload: bool = True,
) -> None:
    _acquire_job_capacity(job_id, pipeline_mode)
    started_at = _utc_now()
    _set_job(
        job_id,
        status="running",
        started_at=started_at,
        progress={
            "stage": "request_received",
            "label": "Upload received",
            "detail": "The backend accepted the upload and is validating the runtime bundle.",
            "percent": 6,
        },
        elapsed_seconds=0,
        eta_seconds=None,
    )

    stage_labels = {
        "request_received": "Upload received",
        "preview_decode": "Preview decode",
        "exact_init": "Exact runtime init",
        "read_slide": "Slide read",
        "sample_tiles": "Tile sampling",
        "staging_slide": "Slide staging",
        "load_project": "Slideflow project init",
        "extract_tiles": "Slideflow tile extraction",
        "build_extractor": "Extractor init",
        "generate_bag": "Feature bag generation",
        "load_bag": "Feature bag load",
        "decode_image": "Image decode",
        "encode_tiles": "Virchow2 encoding",
        "features_ready": "Features ready",
        "ensemble_scoring": "TransMIL ensemble scoring",
        "completed": "Completed",
    }

    def progress_callback(*, stage: str, detail: str, percent: int) -> None:
        now = _utc_now()
        elapsed_seconds = max(0, int((now - started_at).total_seconds()))
        eta_seconds = None
        if percent > 0 and percent < 100:
            projected_total = int(elapsed_seconds * (100.0 / max(percent, 1)))
            eta_seconds = max(0, projected_total - elapsed_seconds)
        _set_job(
            job_id,
            progress={
                "stage": stage,
                "label": stage_labels.get(stage, stage.replace("_", " ").title()),
                "detail": detail,
                "percent": percent,
            },
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
        )

    try:
        with temporary_pipeline_mode(pipeline_mode):
            result = predict_upload(upload_path, progress_callback=progress_callback)
        finished_at = _utc_now()
        _set_job(
            job_id,
            status="completed",
            result={
                **result,
                "uploaded_name": uploaded_name,
            },
            pipeline_mode=pipeline_mode,
            error="",
            finished_at=finished_at,
            elapsed_seconds=max(0, int((finished_at - started_at).total_seconds())),
            eta_seconds=0,
            progress={
                "stage": "completed",
                "label": "Completed",
                "detail": "Prediction is complete and the response payload is ready.",
                "percent": 100,
            },
        )
        history_context = {}
        with _JOB_LOCK:
            job_snapshot = dict(_JOBS.get(job_id) or {})
            history_context = dict(job_snapshot.get("history_context") or {})
        append_prediction_history(
            {
                "job_id": job_id,
                "uploaded_name": uploaded_name,
                "pipeline_mode": pipeline_mode,
                "result_payload": {
                    **result,
                    "uploaded_name": uploaded_name,
                },
                "label": result.get("label"),
                "confidence_level": result.get("confidence_level"),
                "confidence_percent": result.get("confidence_percent"),
                "probability": result.get("probability"),
                "threshold": result.get("threshold"),
                "tile_count": result.get("tile_count"),
                "checkpoint_count": result.get("checkpoint_count"),
                "feature_dim": result.get("feature_dim"),
                "input_kind": result.get("input_kind"),
                "input_kind_display": result.get("input_kind_display"),
                "encoder_label": result.get("encoder_label"),
                "encoder_backbone": result.get("encoder_backbone"),
                "encoder_type": result.get("encoder_type"),
                "specimen_preview_data_url": result.get("specimen_preview_data_url"),
                "tile_preview_data_url": result.get("tile_preview_data_url"),
                "confidence_score": result.get("confidence_score"),
                "model_quality_score": result.get("model_quality_score"),
                "vote_strength_score": result.get("vote_strength_score"),
                "per_checkpoint": result.get("per_checkpoint") or [],
                "inference": result.get("inference") or {},
                "system": result.get("system") or {},
                "elapsed_seconds": max(0, int((finished_at - started_at).total_seconds())),
                **history_context,
            }
        )
    except Exception as exc:
        finished_at = _utc_now()
        _set_job(
            job_id,
            status="failed",
            error=str(exc).strip() or "Prediction failed.",
            finished_at=finished_at,
            elapsed_seconds=max(0, int((finished_at - started_at).total_seconds())),
            eta_seconds=0,
            progress={
                "stage": "failed",
                "label": "Failed",
                "detail": str(exc).strip() or "Prediction failed.",
                "percent": 100,
            },
        )
    finally:
        _release_job_capacity(pipeline_mode)
        if delete_upload:
            upload_path.unlink(missing_ok=True)


def create_prediction_job(uploaded, pipeline_mode: str, *, history_context: dict[str, Any] | None = None) -> dict[str, Any]:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        for chunk in uploaded.chunks():
            handle.write(chunk)
        upload_path = Path(handle.name)

    job_id = uuid.uuid4().hex
    job_payload = {
        "job_id": job_id,
        "status": "queued",
        "uploaded_name": uploaded.name,
        "pipeline_mode": pipeline_mode,
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "error": "",
        "result": None,
        "elapsed_seconds": 0,
        "eta_seconds": None,
        "progress": {
            "stage": "queued",
            "label": "Queued",
            "detail": "The prediction job is waiting for a safe VM slot.",
            "percent": 0,
        },
        "history_context": dict(history_context or {}),
    }
    with _JOB_LOCK:
        _JOBS[job_id] = job_payload
        _update_waiting_jobs_locked()

    worker = threading.Thread(
        target=_run_prediction_job,
        args=(job_id, upload_path, uploaded.name, pipeline_mode),
        kwargs={"delete_upload": True},
        daemon=True,
        name=f"predict-job-{job_id[:8]}",
    )
    worker.start()
    return _serialize_job(job_payload)


def create_prediction_job_from_path(
    upload_path: Path,
    uploaded_name: str,
    pipeline_mode: str,
    *,
    history_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue a prediction job from an already-present file on disk, such as a VM storage sample."""
    if not upload_path.exists():
        raise FileNotFoundError(f"Stored sample not found: {upload_path}")

    job_id = uuid.uuid4().hex
    job_payload = {
        "job_id": job_id,
        "status": "queued",
        "uploaded_name": uploaded_name,
        "pipeline_mode": pipeline_mode,
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "error": "",
        "result": None,
        "elapsed_seconds": 0,
        "eta_seconds": None,
        "progress": {
            "stage": "queued",
            "label": "Queued",
            "detail": "The stored test file is waiting for a safe VM slot.",
            "percent": 0,
        },
        "history_context": dict(history_context or {}),
    }
    with _JOB_LOCK:
        _JOBS[job_id] = job_payload
        _update_waiting_jobs_locked()

    worker = threading.Thread(
        target=_run_prediction_job,
        args=(job_id, upload_path, uploaded_name, pipeline_mode),
        kwargs={"delete_upload": False},
        daemon=True,
        name=f"predict-job-{job_id[:8]}",
    )
    worker.start()
    return _serialize_job(job_payload)


def get_prediction_job(job_id: str) -> dict[str, Any] | None:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return _serialize_job(job)


def list_prediction_jobs(limit: int = 24) -> list[dict[str, Any]]:
    with _JOB_LOCK:
        jobs = [_serialize_job(job) for job in _JOBS.values()]
    jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return jobs[: max(1, limit)]
