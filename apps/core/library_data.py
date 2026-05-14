import json
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings


def storage_manifest_path() -> Path:
    """Return the repo-local manifest that tracks extra non-training SVS files."""
    return Path(settings.BASE_DIR) / "runtime" / "storage_samples" / "storage_manifest.json"


def prediction_history_path() -> Path:
    """Return the append-only JSONL file for completed upload predictions."""
    return Path(settings.BASE_DIR) / "runtime" / "prediction_history" / "history.jsonl"


@lru_cache(maxsize=64)
def _cached_preview_payload(path_str: str, mtime_ns: int, size_bytes: int) -> dict[str, Any]:
    """Cache specimen/tile preview payloads for stored VM slides and saved history items."""
    del mtime_ns, size_bytes
    path = Path(path_str)
    if not path.exists():
        return {
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
        }
    try:
        from .inference import _build_preview_payload_from_upload, _inference_tile_count

        payload = _build_preview_payload_from_upload(path, tile_count=_inference_tile_count())
        return {
            "specimen_preview_data_url": str(payload.get("specimen_preview_data_url") or ""),
            "tile_preview_data_url": str(payload.get("tile_preview_data_url") or ""),
        }
    except Exception:
        return {
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
        }


def preview_payload_for_path(path_value: str | Path | None) -> dict[str, Any]:
    """Return cached preview payloads for a local slide path when possible."""
    if not path_value:
        return {
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
        }
    path = Path(path_value).expanduser()
    if not path.exists():
        return {
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
        }
    stat = path.stat()
    return _cached_preview_payload(str(path), int(stat.st_mtime_ns), int(stat.st_size))


def load_storage_manifest() -> dict[str, Any]:
    """Load the curated storage manifest that is shown in the frontend storage tab."""
    path = storage_manifest_path()
    if not path.exists():
        return {
            "title": "Extra SVS library",
            "summary": {
                "requested_total": 0,
                "available_total": 0,
                "msi_h_total": 0,
                "mss_total": 0,
            },
            "files": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"title": "Extra SVS library", "summary": {}, "files": []}
    enriched_files = []
    for item in payload.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        local_vm_path = str(item.get("local_vm_path") or "").strip()
        enriched_files.append(
            {
                **item,
                **preview_payload_for_path(local_vm_path),
            }
        )
    payload["files"] = enriched_files
    return payload


def compact_storage_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    compact_files: list[dict[str, Any]] = []
    for item in payload.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        compact_files.append(
            {
                "patient": item.get("patient"),
                "msi_status": item.get("msi_status"),
                "bucket_name": item.get("bucket_name"),
                "available_on_vm": item.get("available_on_vm"),
                "source_group": item.get("source_group"),
            }
        )
    return {
        "title": payload.get("title", "Extra SVS library"),
        "summary": payload.get("summary", {}),
        "files": compact_files,
    }


def find_storage_sample(bucket_name: str) -> dict[str, Any] | None:
    """Find a single stored sample entry by its bucket filename."""
    manifest = load_storage_manifest()
    for item in manifest.get("files", []) or []:
        if str(item.get("bucket_name") or "") == bucket_name:
            return item
    return None


def append_prediction_history(entry: dict[str, Any]) -> None:
    """Persist a compact completed-result record for the history tab."""
    path = prediction_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"saved_at": datetime.now(timezone.utc).isoformat(), **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_prediction_history(limit: int | None = None) -> list[dict[str, Any]]:
    """Return the newest saved prediction history records first."""
    path = prediction_history_path()
    if not path.exists():
        return []
    manifest_index = {
        str(item.get("bucket_name") or ""): item
        for item in (load_storage_manifest().get("files", []) or [])
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            uploaded_name = str(payload.get("uploaded_name") or payload.get("sample_bucket_name") or "")
            manifest_item = manifest_index.get(uploaded_name)
            if manifest_item:
                payload.setdefault("expected_label", manifest_item.get("msi_status"))
                payload.setdefault("patient", manifest_item.get("patient"))
                payload.setdefault("source_group", manifest_item.get("source_group"))
                payload.setdefault("sample_bucket_name", manifest_item.get("bucket_name"))
                payload.setdefault("specimen_preview_data_url", manifest_item.get("specimen_preview_data_url") or "")
                payload.setdefault("tile_preview_data_url", manifest_item.get("tile_preview_data_url") or "")
            rows.append(payload)
    rows.sort(key=lambda item: str(item.get("saved_at") or ""), reverse=True)
    if limit is None:
        return rows
    return rows[: max(1, limit)]


def compact_prediction_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for item in rows:
        compact_rows.append(
            {
                "job_id": item.get("job_id"),
                "saved_at": item.get("saved_at"),
                "uploaded_name": item.get("uploaded_name"),
                "pipeline_mode": item.get("pipeline_mode"),
                "label": item.get("label"),
                "expected_label": item.get("expected_label"),
                "patient": item.get("patient"),
                "source_group": item.get("source_group"),
                "confidence_level": item.get("confidence_level"),
                "confidence_percent": item.get("confidence_percent"),
                "probability": item.get("probability"),
                "threshold": item.get("threshold"),
                "tile_count": item.get("tile_count"),
                "checkpoint_count": item.get("checkpoint_count"),
                "feature_dim": item.get("feature_dim"),
                "input_kind": item.get("input_kind"),
                "input_kind_display": item.get("input_kind_display"),
                "encoder_label": item.get("encoder_label"),
                "encoder_backbone": item.get("encoder_backbone"),
                "encoder_type": item.get("encoder_type"),
                "confidence_score": item.get("confidence_score"),
                "model_quality_score": item.get("model_quality_score"),
                "vote_strength_score": item.get("vote_strength_score"),
                "elapsed_seconds": item.get("elapsed_seconds"),
            }
        )
    return compact_rows


def delete_prediction_history_entry(*, job_id: str = "", saved_at: str = "", uploaded_name: str = "") -> bool:
    """Delete one saved history row by stable identifiers and rewrite the JSONL file."""
    path = prediction_history_path()
    if not path.exists():
        return False
    kept_lines: list[str] = []
    deleted = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(raw_line)
            continue
        if not isinstance(payload, dict):
            kept_lines.append(raw_line)
            continue
        payload_job_id = str(payload.get("job_id") or "")
        payload_saved_at = str(payload.get("saved_at") or "")
        payload_uploaded_name = str(payload.get("uploaded_name") or "")
        matches = False
        if job_id and payload_job_id == job_id:
            matches = True
        elif saved_at and uploaded_name and payload_saved_at == saved_at and payload_uploaded_name == uploaded_name:
            matches = True
        if matches and not deleted:
            deleted = True
            continue
        kept_lines.append(raw_line)
    if deleted:
        rewritten = "\n".join(kept_lines)
        if rewritten:
            rewritten += "\n"
        path.write_text(rewritten, encoding="utf-8")
    return deleted
