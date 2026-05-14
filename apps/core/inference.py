import csv
import base64
import concurrent.futures
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import torch
import torch.nn as nn
from django.conf import settings
from nystrom_attention import NystromAttention
from PIL import Image, ImageDraw
import tifffile

from local_encoder_package import EncoderPackageError, load_encoder_package


DEFAULT_FEATURE_DIM = 2560
LATENT_DIM = 512
ENSEMBLE_SIZE = 8
MAX_CHECKPOINTS_PER_REPEAT = 2
RAW_TILE_SIZE = 256
DEFAULT_TILE_COUNT = 24
TRUSTED_TENSOR_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz"}
RAW_SLIDE_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FEATURE_KEYS = ("features", "feats", "embeddings", "x", "bag", "data")

MANAGER1_MODE = "manager1"
MANAGER2_MODE = "manager2"
PARALLEL_MODE = "parallel"
ProgressCallback = Any
_PIPELINE_OVERRIDE = threading.local()


@dataclass(frozen=True)
class EnsembleCheckpoint:
    checkpoint_path: Path
    repeat: int
    fold: int
    score: float
    threshold: float
    auroc: float
    f1_macro: float
    auprc: float
    balanced_accuracy: float
    approach_label: str = ""
    extractor: str = ""


@dataclass(frozen=True)
class ParallelApproachSpec:
    approach_label: str
    extractor: str
    checkpoint_dir: Path
    fold_metrics_path: Path
    feature_dim: int
    quality_weight: float
    mean_auroc: float
    mean_f1_macro: float
    mean_auprc: float
    mean_balanced_accuracy: float
    mean_best_threshold: float


@dataclass(frozen=True)
class TileSample:
    left: int
    top: int


def _emit_progress(progress_callback: ProgressCallback | None, stage: str, detail: str, percent: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage=stage, detail=detail, percent=max(0, min(100, int(percent))))
    except Exception:
        return


@contextmanager
def temporary_pipeline_mode(mode: str | None):
    previous = getattr(_PIPELINE_OVERRIDE, "mode", None)
    if mode:
        _PIPELINE_OVERRIDE.mode = str(mode).strip().lower()
    else:
        _PIPELINE_OVERRIDE.mode = None
    try:
        yield
    finally:
        _PIPELINE_OVERRIDE.mode = previous


class TransLayer(nn.Module):
    def __init__(self, dim: int = LATENT_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim: int = LATENT_DIM):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        batch_size, _, channels = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(batch_size, channels, h, w)
        cnn_feat = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        feat_token = cnn_feat.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), feat_token), dim=1)


class TransMIL(nn.Module):
    def __init__(self, input_dim: int = DEFAULT_FEATURE_DIM, n_classes: int = 2):
        super().__init__()
        self.pos_layer = PPEG(dim=LATENT_DIM)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, LATENT_DIM), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, LATENT_DIM))
        self.layer1 = TransLayer(dim=LATENT_DIM)
        self.layer2 = TransLayer(dim=LATENT_DIM)
        self.norm = nn.LayerNorm(LATENT_DIM)
        self._fc2 = nn.Linear(LATENT_DIM, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        h = self._fc1(h.float())
        batch_size, token_count, _ = h.shape
        grid_size = int(math.ceil(math.sqrt(token_count)))
        add_length = grid_size * grid_size - token_count
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, grid_size, grid_size)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        return self._fc2(h)


def _bundle_root() -> Path:
    configured = str(getattr(settings, "MSI_INFERENCE_BUNDLE_DIR", "") or "").strip() if settings.configured else ""
    if configured:
        return Path(configured).expanduser()
    if settings.configured:
        return Path(settings.BASE_DIR) / "eraya" / "latest_approach_virchow2"
    return Path(__file__).resolve().parents[2] / "eraya" / "latest_approach_virchow2"


def _read_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parallel_source_root() -> Path:
    base_dir = Path(settings.BASE_DIR) if settings.configured else Path(__file__).resolve().parents[2]
    return base_dir / "eraya" / "latest_approach_virchow2"


def _historical_bundle_config_path() -> Path:
    root = _bundle_root()
    candidates = [
        root / "bundle_config.historical.json",
        root / "inference_bundle" / "bundle_config.historical.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_encoder_dir() -> Path:
    base_dir = Path(settings.BASE_DIR) if settings.configured else Path(__file__).resolve().parents[2]
    configured = str(getattr(settings, "MSI_LOCAL_ENCODER_DIR", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    student_dir = base_dir / "models" / "student_virchow2"
    if student_dir.exists():
        return student_dir
    return base_dir / "models" / "virchow2"


def _feature_dim() -> int:
    try:
        return int(_current_encoder_metadata().get("embedding_dim") or DEFAULT_FEATURE_DIM)
    except Exception:
        return DEFAULT_FEATURE_DIM


def _pipeline_mode() -> str:
    override = str(getattr(_PIPELINE_OVERRIDE, "mode", "") or "").strip().lower()
    if override == PARALLEL_MODE:
        return PARALLEL_MODE
    if override == MANAGER1_MODE:
        return MANAGER1_MODE
    raw_value = str(getattr(settings, "MSI_PIPELINE_MODE", MANAGER1_MODE) or MANAGER1_MODE).strip().lower()
    if raw_value == PARALLEL_MODE:
        return PARALLEL_MODE
    return MANAGER1_MODE


def _is_manager1_mode() -> bool:
    return _pipeline_mode() == MANAGER1_MODE


def _is_parallel_mode() -> bool:
    return _pipeline_mode() == PARALLEL_MODE


def _fold_metrics_path() -> Path:
    root = _bundle_root()
    candidates = [
        root / "approach_files" / "fold_metrics.csv",
        root / "metrics" / "fold_metrics.csv",
        root / "inference_bundle" / "metrics" / "fold_metrics.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _checkpoint_dir() -> Path:
    root = _bundle_root()
    candidates = [
        root / "checkpoints",
        root / "inference_bundle" / "checkpoints",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _score_row(row: dict[str, str]) -> float:
    return (
        float(row["auroc"]) * 0.45
        + float(row["f1_macro"]) * 0.30
        + float(row["auprc"]) * 0.20
        + float(row["balanced_accuracy"]) * 0.10
        - float(row["brier_score"]) * 0.05
    )


def _preferred_device_name() -> str:
    preferred = str(getattr(settings, "MSI_PREFER_DEVICE", "auto") or "auto").strip().lower()
    if preferred not in {"auto", "cpu", "cuda"}:
        return "auto"
    return preferred


def _safe_worker_count(value: int, *, upper_bound: int | None = None) -> int:
    if upper_bound is None:
        upper_bound = os.cpu_count() or 4
    return max(1, min(int(value), int(max(1, upper_bound))))


def _device() -> torch.device:
    preferred = _preferred_device_name()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fast_tile_read_workers() -> int:
    configured = int(getattr(settings, "MSI_FAST_TILE_READ_WORKERS", 0) or 0)
    default = min(max(2, (os.cpu_count() or 4) // 2), 8)
    return _safe_worker_count(configured or default, upper_bound=16)


def _image_tile_workers() -> int:
    configured = int(getattr(settings, "MSI_IMAGE_TILE_WORKERS", 0) or 0)
    default = min(max(2, (os.cpu_count() or 4) // 2), 8)
    return _safe_worker_count(configured or default, upper_bound=16)


def _encode_preprocess_workers() -> int:
    configured = int(getattr(settings, "MSI_ENCODE_PREPROCESS_WORKERS", 0) or 0)
    default = min(max(2, (os.cpu_count() or 4) // 2), 8)
    return _safe_worker_count(configured or default, upper_bound=16)


def _encode_batch_size() -> int:
    configured = int(getattr(settings, "MSI_ENCODE_BATCH_SIZE", 0) or 0)
    if configured > 0:
        return max(1, configured)
    return 48 if _device().type == "cuda" else 16


def _score_worker_count() -> int:
    configured = int(getattr(settings, "MSI_SCORE_WORKERS", 0) or 0)
    available = max(1, len(tuple(_checkpoint_dir().glob("repeat_*_fold_*_best_valid.pth"))))
    default = min(max(2, os.cpu_count() or 4), available)
    return _safe_worker_count(configured or default, upper_bound=available)


def _bundle_metrics_path() -> Path:
    root = _bundle_root()
    candidates = [
        root / "approach_files" / "metrics.json",
        root / "metrics" / "metrics.json",
        root / "inference_bundle" / "metrics" / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parallel_bundle_id() -> str:
    bundle_config = _read_json_payload(_parallel_source_root() / "bundle_config.json")
    return str(bundle_config.get("bundle_id") or "").strip() or "run-65512be1f9c4"


def _parallel_export_root() -> Path:
    return (Path(settings.BASE_DIR) if settings.configured else Path(__file__).resolve().parents[2]) / "eraya" / f"{_parallel_bundle_id()}_eraya_export"


def _parallel_flat_checkpoint_root() -> Path:
    return (Path(settings.BASE_DIR) if settings.configured else Path(__file__).resolve().parents[2]) / "eraya" / f"{_parallel_bundle_id()}_flat_pth"


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _slug_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value)).strip("-")


def _parallel_quality_score(payload: dict[str, Any]) -> float:
    return (
        float(payload.get("mean_auroc") or 0.0) * 0.45
        + float(payload.get("mean_f1_macro") or 0.0) * 0.30
        + float(payload.get("mean_auprc") or 0.0) * 0.20
        + float(payload.get("mean_balanced_accuracy") or 0.0) * 0.10
        - float(payload.get("mean_brier_score") or 0.0) * 0.05
    )


def _feature_dim_from_checkpoint_dir(checkpoint_dir: Path) -> int:
    first_checkpoint = next(checkpoint_dir.glob("repeat_*_fold_*_best_valid.pth"), None)
    if first_checkpoint is None:
        raise FileNotFoundError(f"No preserved checkpoints were found in {checkpoint_dir}.")
    state_dict = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
    weight = state_dict.get("_fc1.0.weight")
    if weight is None or getattr(weight, "shape", None) is None or len(weight.shape) != 2:
        raise ValueError(f"Could not infer feature dimension from {first_checkpoint}.")
    return int(weight.shape[1])


@lru_cache(maxsize=1)
def _parallel_aggregate_summary() -> dict[str, Any]:
    root = _parallel_source_root()
    candidates = [
        root / "top4_montecarlo_aggregate.json",
        _parallel_export_root() / "top4_montecarlo_aggregate.json",
    ]
    for candidate in candidates:
        payload = _read_json_payload(candidate)
        if payload:
            return payload
    return {}


@lru_cache(maxsize=1)
def _parallel_bundle_summary() -> dict[str, Any]:
    root = _parallel_source_root()
    candidates = [
        _parallel_export_root() / "final_summary.json",
        root / "final_summary.json",
    ]
    for candidate in candidates:
        payload = _read_json_payload(candidate)
        if payload:
            return payload
    return {}


@lru_cache(maxsize=1)
def _parallel_bundle_config() -> dict[str, Any]:
    return _read_json_payload(_parallel_source_root() / "bundle_config.json")


@lru_cache(maxsize=1)
def _parallel_approach_specs() -> tuple[ParallelApproachSpec, ...]:
    bundle_config = _parallel_bundle_config()
    final_summary = _parallel_bundle_summary()
    aggregate = _parallel_aggregate_summary()
    summary_lookup = final_summary.get("approaches") or {}
    aggregate_lookup = {
        str(item.get("approach_label") or "").strip(): item
        for item in (aggregate.get("approaches") or [])
        if isinstance(item, dict)
    }
    specs: list[ParallelApproachSpec] = []
    for raw_spec in bundle_config.get("specs") or []:
        if not isinstance(raw_spec, dict):
            continue
        approach_label = str(raw_spec.get("approach_label") or "").strip()
        extractor = str(raw_spec.get("feature_extractor") or "").strip()
        if not approach_label or not extractor:
            continue
        summary_payload = summary_lookup.get(approach_label) or aggregate_lookup.get(approach_label) or {}
        checkpoint_dir = _parallel_flat_checkpoint_root() / _slug_label(approach_label)
        fold_metrics_path = _parallel_export_root() / "approaches" / approach_label / "fold_metrics.csv"
        specs.append(
            ParallelApproachSpec(
                approach_label=approach_label,
                extractor=extractor,
                checkpoint_dir=checkpoint_dir,
                fold_metrics_path=fold_metrics_path,
                feature_dim=_feature_dim_from_checkpoint_dir(checkpoint_dir),
                quality_weight=max(_parallel_quality_score(summary_payload), 1e-6),
                mean_auroc=float(summary_payload.get("mean_auroc") or 0.0),
                mean_f1_macro=float(summary_payload.get("mean_f1_macro") or 0.0),
                mean_auprc=float(summary_payload.get("mean_auprc") or 0.0),
                mean_balanced_accuracy=float(summary_payload.get("mean_balanced_accuracy") or 0.0),
                mean_best_threshold=float(summary_payload.get("mean_best_threshold") or 0.5),
            )
        )
    return tuple(specs)


@lru_cache(maxsize=1)
def _historical_bundle_request() -> dict[str, Any]:
    config_path = _historical_bundle_config_path()
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    request = payload.get("request")
    return request if isinstance(request, dict) else {}


def _exact_extraction_settings() -> dict[str, Any]:
    request = _historical_bundle_request()
    override_tiles = int(getattr(settings, "MSI_EXACT_MAX_TILES", 0) or 0)
    max_tiles = int(override_tiles or request.get("max_tiles_per_slide") or getattr(settings, "MSI_MAX_INFERENCE_TILES", DEFAULT_TILE_COUNT))
    mpp_override = request.get("mpp_override")
    qc_method = request.get("qc_method")
    preview_tiles = int(getattr(settings, "MSI_EXACT_PREVIEW_TILES", 6) or 6)
    tile_threads = int(getattr(settings, "MSI_EXACT_TILE_THREADS", 4) or 4)
    return {
        "max_tiles_per_slide": max(8, max_tiles),
        "preview_tiles": max(1, min(max_tiles, preview_tiles)),
        "tile_threads": max(1, tile_threads),
        "mpp_override": None if mpp_override in (None, "", "None") else float(mpp_override),
        "qc_method": None if qc_method in (None, "", "None", "none") else str(qc_method),
    }


@lru_cache(maxsize=1)
def _bundle_metrics() -> dict[str, Any]:
    metrics_path = _bundle_metrics_path()
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _serving_compatibility() -> tuple[bool, str]:
    try:
        encoder_label = str(_current_encoder_metadata().get("encoder_label") or "")
    except Exception as exc:
        return False, str(exc).strip() or type(exc).__name__
    bundle_extractor = str(
        _bundle_metrics().get("resolved_feature_extractor_used")
        or _bundle_metrics().get("feature_extractor_used")
        or ""
    )
    if encoder_label.startswith("student-") and bundle_extractor != "student-virchow2":
        return (
            False,
            "Direct upload is temporarily disabled because the local student encoder is still paired with the older Virchow2 MIL bundle. "
            "A student-feature MIL retraining run is in progress on the VM, and predictions should not be trusted until that bundle is swapped in.",
        )
    return True, ""


def _inference_tile_count() -> int:
    configured_tile_count = int(getattr(settings, "MSI_MAX_INFERENCE_TILES", DEFAULT_TILE_COUNT) or DEFAULT_TILE_COUNT)
    configured_tile_count = max(8, configured_tile_count)
    if _is_manager1_mode():
        return _exact_extraction_settings()["max_tiles_per_slide"]
    return configured_tile_count


def _active_ensemble_checkpoints() -> tuple[EnsembleCheckpoint, ...]:
    checkpoints = get_ensemble_checkpoints()
    return checkpoints


@lru_cache(maxsize=1)
def get_ensemble_checkpoints() -> tuple[EnsembleCheckpoint, ...]:
    rows: list[dict[str, Any]] = []
    with _fold_metrics_path().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["repeat"] = int(row["repeat"])
            row["fold"] = int(row["fold"])
            row["score"] = _score_row(row)
            rows.append(row)
    rows.sort(key=lambda item: (item["score"], float(item["auroc"]), float(item["f1_macro"])), reverse=True)

    selected: list[EnsembleCheckpoint] = []
    for row in rows:
        repeat = int(row["repeat"])
        fold = int(row["fold"])
        checkpoint_path = _checkpoint_dir() / f"repeat_{repeat}_fold_{fold}_best_valid.pth"
        if not checkpoint_path.exists():
            continue
        selected.append(
            EnsembleCheckpoint(
                checkpoint_path=checkpoint_path,
                repeat=repeat,
                fold=fold,
                score=float(row["score"]),
                threshold=float(row["best_threshold"]),
                auroc=float(row["auroc"]),
                f1_macro=float(row["f1_macro"]),
                auprc=float(row["auprc"]),
                balanced_accuracy=float(row["balanced_accuracy"]),
            )
        )
    return tuple(selected)


@lru_cache(maxsize=1)
def _parallel_checkpoints_by_approach() -> dict[str, tuple[EnsembleCheckpoint, ...]]:
    checkpoint_map: dict[str, tuple[EnsembleCheckpoint, ...]] = {}
    for spec in _parallel_approach_specs():
        rows: list[dict[str, Any]] = []
        with spec.fold_metrics_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["repeat"] = int(row["repeat"])
                row["fold"] = int(row["fold"])
                row["score"] = _score_row(row)
                rows.append(row)
        rows.sort(key=lambda item: (item["score"], float(item["auroc"]), float(item["f1_macro"])), reverse=True)
        selected: list[EnsembleCheckpoint] = []
        for row in rows:
            repeat = int(row["repeat"])
            fold = int(row["fold"])
            checkpoint_path = spec.checkpoint_dir / f"repeat_{repeat}_fold_{fold}_best_valid.pth"
            if not checkpoint_path.exists():
                continue
            selected.append(
                EnsembleCheckpoint(
                    checkpoint_path=checkpoint_path,
                    repeat=repeat,
                    fold=fold,
                    score=float(row["score"]),
                    threshold=float(row["best_threshold"]),
                    auroc=float(row["auroc"]),
                    f1_macro=float(row["f1_macro"]),
                    auprc=float(row["auprc"]),
                    balanced_accuracy=float(row["balanced_accuracy"]),
                    approach_label=spec.approach_label,
                    extractor=spec.extractor,
                )
            )
        checkpoint_map[spec.approach_label] = tuple(selected)
    return checkpoint_map


@lru_cache(maxsize=1)
def _load_encoder() -> tuple[Any, Any, str]:
    package_dir = _default_encoder_dir()
    package = load_encoder_package(package_dir)
    model = package.model.to(_device())
    model.eval()
    return model, package.transform, package.label


@lru_cache(maxsize=4)
def _encoder_metadata(mode: str | None = None) -> dict[str, Any]:
    with temporary_pipeline_mode(mode):
        package_dir = _default_encoder_dir()
        package = load_encoder_package(package_dir)
        training_meta = package.manifest.get("teacher_distillation") or {}
        return {
            "package_dir": str(package_dir),
            "encoder_label": package.label,
            "encoder_type": package.package_type,
            "embedding_dim": package.embedding_dim,
            "tile_count": _inference_tile_count(),
            "backbone_name": package.manifest.get("backbone_name") or "local-dir",
        }


def _current_encoder_metadata() -> dict[str, Any]:
    return _encoder_metadata(_pipeline_mode())


def get_inference_metadata() -> dict[str, Any]:
    if _is_parallel_mode():
        approach_specs = _parallel_approach_specs()
        total_checkpoints = sum(len(items) for items in _parallel_checkpoints_by_approach().values())
        return {
            "bundle_root": str(_parallel_export_root()),
            "approach_label": "Preserved top-4 offline fusion",
            "mil_model": "TransMIL late fusion",
            "feature_dim": 0,
            "feature_dims_by_approach": [
                {
                    "approach_label": spec.approach_label,
                    "extractor": spec.extractor,
                    "feature_dim": spec.feature_dim,
                    "checkpoint_count": len(_parallel_checkpoints_by_approach().get(spec.approach_label, ())),
                    "quality_weight": spec.quality_weight,
                }
                for spec in approach_specs
            ],
            "available_checkpoints": total_checkpoints,
            "selected_checkpoint_count": total_checkpoints,
            "selected_repeats": sorted(
                {
                    checkpoint.repeat
                    for checkpoints in _parallel_checkpoints_by_approach().values()
                    for checkpoint in checkpoints
                }
            ),
            "mean_threshold": float(
                sum(spec.mean_best_threshold for spec in approach_specs) / max(1, len(approach_specs))
            ),
            "accepted_feature_suffixes": [".pt", ".pth", ".bin", ".npz"],
            "accepted_slide_suffixes": [],
            "accepted_upload_suffixes": [".pt", ".pth", ".bin", ".npz"],
            "input_mode": "parallel_feature_bag_package",
            "pipeline_mode": _pipeline_mode(),
            "pipeline_style": "preserved four-model offline fusion",
            "encoder_ready": True,
            "encoder_error": "",
            "encoder": {
                "package_dir": str(_parallel_export_root()),
                "encoder_label": "preserved-top4-package",
                "encoder_type": "offline_multi_bag",
                "embedding_dim": 0,
                "tile_count": 0,
                "backbone_name": ", ".join(spec.extractor for spec in approach_specs),
            },
            "device": str(_device()),
            "preferred_device": _preferred_device_name(),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_status": "available" if torch.cuda.is_available() else "not available",
            "serving_ready": True,
            "serving_message": "",
            "bundle_feature_extractor_used": "multiple",
        }
    checkpoints = _active_ensemble_checkpoints()
    mean_threshold = sum(item.threshold for item in checkpoints) / max(1, len(checkpoints))
    preferred_device = _preferred_device_name()
    cuda_available = bool(torch.cuda.is_available())
    resolved_device = str(_device())
    try:
        encoder_meta = _current_encoder_metadata()
        encoder_ready = True
        encoder_error = ""
    except Exception as exc:
        encoder_meta = {
            "package_dir": str(_default_encoder_dir()),
            "encoder_label": "unavailable",
            "encoder_type": "",
            "embedding_dim": DEFAULT_FEATURE_DIM,
            "tile_count": DEFAULT_TILE_COUNT,
            "backbone_name": "",
        }
        encoder_ready = False
        encoder_error = str(exc).strip() or type(exc).__name__
    serving_ready, serving_message = _serving_compatibility()
    accepted_slide_suffixes = sorted(RAW_SLIDE_SUFFIXES)
    accepted_upload_suffixes = sorted(TRUSTED_TENSOR_SUFFIXES | RAW_SLIDE_SUFFIXES)
    return {
        "bundle_root": str(_bundle_root()),
        "approach_label": str(
            _bundle_metrics().get("approach_label")
            or _bundle_metrics().get("resolved_approach_label")
            or "Preserved bundle"
        ),
        "mil_model": "TransMIL",
        "feature_dim": _feature_dim(),
        "available_checkpoints": len(list(_checkpoint_dir().glob("repeat_*_fold_*_best_valid.pth"))),
        "selected_checkpoint_count": len(checkpoints),
        "selected_repeats": sorted({item.repeat for item in checkpoints}),
        "mean_threshold": mean_threshold,
        "accepted_feature_suffixes": sorted(TRUSTED_TENSOR_SUFFIXES),
        "accepted_slide_suffixes": accepted_slide_suffixes,
        "accepted_upload_suffixes": accepted_upload_suffixes,
        "input_mode": "feature_bag_exact",
        "pipeline_mode": _pipeline_mode(),
        "pipeline_style": "training-matched exact bundle",
        "encoder_ready": encoder_ready,
        "encoder_error": encoder_error,
        "encoder": encoder_meta,
        "device": resolved_device,
        "preferred_device": preferred_device,
        "cuda_available": cuda_available,
        "gpu_status": "available" if cuda_available else "not available",
        "serving_ready": serving_ready,
        "serving_message": serving_message,
        "bundle_feature_extractor_used": str(
            _bundle_metrics().get("resolved_feature_extractor_used")
            or _bundle_metrics().get("feature_extractor_used")
            or ""
        ),
    }


def _unwrap_feature_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in FEATURE_KEYS:
            if key in payload:
                return payload[key]
        for value in payload.values():
            extracted = _unwrap_feature_payload(value)
            if extracted is not None:
                return extracted
        return None
    if isinstance(payload, (list, tuple)):
        for item in payload:
            extracted = _unwrap_feature_payload(item)
            if extracted is not None:
                return extracted
        return None
    if isinstance(payload, (torch.Tensor, np.ndarray)):
        return payload
    return None


def _flatten_named_feature_entries(payload: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            entries.extend(_flatten_named_feature_entries(value, prefix=next_prefix))
        return entries
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}/{index}" if prefix else str(index)
            entries.extend(_flatten_named_feature_entries(value, prefix=next_prefix))
        return entries
    if isinstance(payload, (torch.Tensor, np.ndarray)):
        entries.append((prefix or "tensor", payload))
    return entries


def _coerce_feature_tensor(features: Any, *, expected_feature_dim: int) -> torch.Tensor:
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if not isinstance(features, torch.Tensor):
        raise ValueError("The uploaded feature bag did not decode into a tensor.")
    features = features.detach().cpu()
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    if features.dim() != 2:
        raise ValueError(f"Expected a 2D feature bag, got shape {tuple(features.shape)}.")
    if features.shape[-1] != expected_feature_dim:
        raise ValueError(f"Expected feature dimension {expected_feature_dim}, got {features.shape[-1]}.")
    if features.shape[0] < 1:
        raise ValueError("The uploaded feature bag is empty.")
    return features.float()


def load_feature_bag(upload_path: Path) -> torch.Tensor:
    suffix = upload_path.suffix.lower()
    if suffix not in TRUSTED_TENSOR_SUFFIXES:
        raise ValueError(f"Unsupported upload type: {suffix or 'no extension'}.")

    if suffix == ".npy":
        payload = np.load(upload_path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(upload_path, allow_pickle=False)
        if not archive.files:
            raise ValueError("The .npz upload does not contain any arrays.")
        payload = archive[archive.files[0]]
    else:
        payload = torch.load(upload_path, map_location="cpu", weights_only=False)

    features = _unwrap_feature_payload(payload)
    if features is None:
        raise ValueError("Could not locate a feature tensor in the uploaded bag.")
    return _coerce_feature_tensor(features, expected_feature_dim=_feature_dim())


def _parallel_aliases(spec: ParallelApproachSpec) -> tuple[str, ...]:
    aliases = {
        spec.approach_label,
        _slug_label(spec.approach_label),
        spec.extractor,
        spec.extractor.replace("-", "_"),
        spec.extractor.replace("-", ""),
    }
    approach_num = "".join(ch for ch in spec.approach_label if ch.isdigit())
    if approach_num:
        aliases.add(f"approach{approach_num}")
        aliases.add(f"a{approach_num}")
    return tuple(_normalize_key(item) for item in aliases if item)


def load_parallel_feature_package(upload_path: Path) -> dict[str, torch.Tensor]:
    suffix = upload_path.suffix.lower()
    if suffix not in {".pt", ".pth", ".bin", ".npz"}:
        raise ValueError(
            "Parallel mode expects a packaged multi-bag upload. Use .pt/.pth/.bin dictionaries or a .npz archive "
            "with one bag for each preserved approach."
        )

    if suffix == ".npz":
        archive = np.load(upload_path, allow_pickle=False)
        if not archive.files:
            raise ValueError("The .npz upload does not contain any arrays.")
        payload: Any = {name: archive[name] for name in archive.files}
    else:
        payload = torch.load(upload_path, map_location="cpu", weights_only=False)

    entries = _flatten_named_feature_entries(payload)
    if not entries:
        raise ValueError("Could not locate any named feature tensors in the uploaded parallel package.")

    matched: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for spec in _parallel_approach_specs():
        aliases = _parallel_aliases(spec)
        chosen_payload = None
        for path, candidate in entries:
            normalized_path = _normalize_key(path)
            if any(alias and alias in normalized_path for alias in aliases):
                chosen_payload = candidate
                break
        if chosen_payload is None:
            missing.append(f"{spec.approach_label} ({spec.extractor})")
            continue
        matched[spec.approach_label] = _coerce_feature_tensor(chosen_payload, expected_feature_dim=spec.feature_dim)
    if missing:
        raise ValueError(
            "The parallel package is missing preserved bags for: "
            + ", ".join(missing)
            + ". Include named tensors for all four approaches in one .npz or .pt package."
        )
    return matched


def _coords_from_thumbnail(
    thumb: Image.Image,
    target_width: int,
    target_height: int,
    *,
    max_tiles: int,
) -> list[TileSample]:
    thumb_np = np.asarray(thumb.convert("RGB"))
    gray = thumb_np.mean(axis=2)
    channel_range = thumb_np.max(axis=2) - thumb_np.min(axis=2)
    mask = (gray < 235) & (channel_range > 10)

    aspect = max(thumb.width / max(thumb.height, 1), 0.2)
    cols = max(4, int(round(math.sqrt(max_tiles * aspect))))
    rows = max(4, int(math.ceil(max_tiles / cols)))
    cell_w = max(1, thumb.width // cols)
    cell_h = max(1, thumb.height // rows)
    scale_x = target_width / max(thumb.width, 1)
    scale_y = target_height / max(thumb.height, 1)

    candidates: list[tuple[float, TileSample]] = []
    for row in range(rows):
        for col in range(cols):
            y0 = row * cell_h
            y1 = thumb.height if row == rows - 1 else min(thumb.height, (row + 1) * cell_h)
            x0 = col * cell_w
            x1 = thumb.width if col == cols - 1 else min(thumb.width, (col + 1) * cell_w)
            cell_mask = mask[y0:y1, x0:x1]
            if cell_mask.size == 0 or float(cell_mask.mean()) < 0.15:
                continue
            score = float(cell_mask.mean())
            center_x = int(((x0 + x1) / 2.0) * scale_x) - RAW_TILE_SIZE // 2
            center_y = int(((y0 + y1) / 2.0) * scale_y) - RAW_TILE_SIZE // 2
            candidates.append(
                (
                    score,
                    TileSample(
                        left=min(max(center_x, 0), max(target_width - RAW_TILE_SIZE, 0)),
                        top=min(max(center_y, 0), max(target_height - RAW_TILE_SIZE, 0)),
                    ),
                )
            )
    if not candidates:
        return [TileSample(left=0, top=0)]
    candidates.sort(key=lambda item: item[0], reverse=True)
    coords: list[TileSample] = []
    seen: set[tuple[int, int]] = set()
    for _, sample in candidates:
        key = (sample.left, sample.top)
        if key in seen:
            continue
        seen.add(key)
        coords.append(sample)
        if len(coords) >= max_tiles:
            break
    return coords or [TileSample(left=0, top=0)]


def _sample_image_coords(image: Image.Image, *, tile_count: int) -> list[TileSample]:
    width, height = image.size
    thumb = image.copy()
    thumb.thumbnail((min(1024, width), min(1024, height)))
    return _coords_from_thumbnail(thumb, width, height, max_tiles=tile_count)


def _preview_thumbnail(image: Image.Image, *, max_side: int = 560) -> Image.Image:
    thumb = image.copy().convert("RGB")
    thumb.thumbnail((max_side, max_side))
    return thumb


def _image_to_data_url(image: Image.Image, *, fmt: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt, quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _draw_tile_preview(
    preview_image: Image.Image,
    coords: list[TileSample],
    *,
    source_width: int,
    source_height: int,
) -> Image.Image:
    source_preview = preview_image.copy().convert("RGB")
    dimmed = Image.blend(
        source_preview,
        Image.new("RGB", source_preview.size, (8, 20, 34)),
        0.58,
    )
    canvas = dimmed.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale_x = canvas.width / max(source_width, 1)
    scale_y = canvas.height / max(source_height, 1)
    for idx, sample in enumerate(coords):
        left = int(round(sample.left * scale_x))
        top = int(round(sample.top * scale_y))
        right = int(round((sample.left + RAW_TILE_SIZE) * scale_x))
        bottom = int(round((sample.top + RAW_TILE_SIZE) * scale_y))
        left = max(0, min(left, canvas.width - 1))
        top = max(0, min(top, canvas.height - 1))
        right = max(left + 1, min(max(right, left + 1), canvas.width))
        bottom = max(top + 1, min(max(bottom, top + 1), canvas.height))
        tile_crop = source_preview.crop((left, top, right, bottom))
        canvas.paste(tile_crop, (left, top))
        color = (28, 167, 255, 255) if idx % 2 == 0 else (255, 255, 255, 255)
        fill = (28, 167, 255, 48) if idx % 2 == 0 else (255, 255, 255, 28)
        draw.rectangle((left, top, right, bottom), outline=color, width=5, fill=fill)
        if right - left > 2 and bottom - top > 2:
            draw.rectangle((left + 1, top + 1, right - 1, bottom - 1), outline=(9, 104, 171, 220), width=1)
        badge_size = min(28, max(16, right - left))
        badge_box = (left, top, min(left + badge_size, right), min(top + badge_size, bottom))
        draw.rectangle(badge_box, fill=(8, 24, 38, 220))
        draw.text((badge_box[0] + 6, badge_box[1] + 3), str(idx + 1), fill=(236, 248, 255, 255))
    return canvas.convert("RGB")


def _build_preview_payload(
    image: Image.Image,
    coords: list[TileSample],
    *,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    specimen_thumb = _preview_thumbnail(image)
    tile_thumb = _draw_tile_preview(
        specimen_thumb,
        coords,
        source_width=source_width,
        source_height=source_height,
    )
    return {
        "specimen_preview_data_url": _image_to_data_url(specimen_thumb),
        "tile_preview_data_url": _image_to_data_url(tile_thumb),
    }


def _build_preview_payload_for_image(
    image: Image.Image,
    *,
    tile_count: int,
) -> dict[str, Any]:
    width, height = image.size
    coords = _sample_image_coords(image, tile_count=max(tile_count, 1))
    chosen = coords[: max(tile_count, 1)] or [TileSample(left=0, top=0)]
    return {
        **_build_preview_payload(image, chosen, source_width=width, source_height=height),
        "tile_count": len(chosen),
    }


def _build_preview_payload_for_slide(
    upload_path: Path,
    *,
    tile_count: int,
) -> dict[str, Any]:
    slide = openslide.OpenSlide(str(upload_path))
    try:
        width, height = slide.dimensions
        thumb = slide.get_thumbnail((min(1024, width), min(1024, height))).convert("RGB")
    finally:
        slide.close()
    coords = _coords_from_thumbnail(thumb, width, height, max_tiles=max(tile_count, 1))
    chosen = coords[: max(tile_count, 1)] or [TileSample(left=0, top=0)]
    return {
        **_build_preview_payload(thumb, chosen, source_width=width, source_height=height),
        "tile_count": len(chosen),
    }


def _build_preview_payload_from_upload(
    upload_path: Path,
    *,
    tile_count: int,
) -> dict[str, Any]:
    suffix = upload_path.suffix.lower()
    if suffix in RAW_SLIDE_SUFFIXES:
        try:
            return _build_preview_payload_for_slide(upload_path, tile_count=tile_count)
        except Exception:
            pass
        try:
            tiles, preview_payload = _read_tiff_like_tiles(upload_path, tile_count=max(tile_count, 1))
            return {
                **preview_payload,
                "tile_count": len(tiles),
            }
        except Exception:
            return {
                "specimen_preview_data_url": "",
                "tile_preview_data_url": "",
                "tile_count": max(tile_count, 1),
            }
    try:
        image = Image.open(upload_path).convert("RGB")
    except Exception:
        return {
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
            "tile_count": max(tile_count, 1),
        }
    return _build_preview_payload_for_image(image, tile_count=tile_count)


def _crop_tile_from_image(image: Image.Image, sample: TileSample, *, width: int, height: int) -> Image.Image:
    box = (
        sample.left,
        sample.top,
        min(sample.left + RAW_TILE_SIZE, width),
        min(sample.top + RAW_TILE_SIZE, height),
    )
    tile = image.crop(box)
    if tile.size != (RAW_TILE_SIZE, RAW_TILE_SIZE):
        canvas = Image.new("RGB", (RAW_TILE_SIZE, RAW_TILE_SIZE), "white")
        canvas.paste(tile, (0, 0))
        tile = canvas
    return tile


def _read_region_from_slide(upload_path: Path, sample: TileSample) -> Image.Image:
    slide = openslide.OpenSlide(str(upload_path))
    try:
        return slide.read_region((sample.left, sample.top), 0, (RAW_TILE_SIZE, RAW_TILE_SIZE)).convert("RGB")
    finally:
        slide.close()


def _safe_slide_stem(value: str) -> str:
    raw = Path(value).stem.strip()
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw)
    return cleaned or "uploaded_slide"


def _monitor_exact_tile_extraction(
    tfrecord_dir: Path,
    slide_stem: str,
    progress_callback: ProgressCallback | None,
    stop_event: threading.Event,
) -> None:
    tfrecord_path = tfrecord_dir / f"{slide_stem}.tfrecords"
    index_path = tfrecord_dir / f"{slide_stem}.index.npz"
    manifest_path = tfrecord_dir / "manifest.json"
    last_signature: tuple[bool, bool, int] | None = None
    while not stop_event.wait(2.0):
        tfrecord_exists = tfrecord_path.exists()
        index_exists = index_path.exists()
        size_bytes = tfrecord_path.stat().st_size if tfrecord_exists else 0
        signature = (tfrecord_exists, index_exists, size_bytes)
        if signature == last_signature:
            continue
        last_signature = signature
        size_mib = size_bytes / (1024 * 1024) if size_bytes else 0.0
        if not tfrecord_exists and not index_exists:
            detail = "Slideflow has the slide open and is still cutting training-matched tiles."
            if manifest_path.exists():
                detail = "Slideflow has created the tile workspace and is still cutting training-matched tiles."
            _emit_progress(progress_callback, "extract_tiles", detail, 48)
            continue
        if tfrecord_exists and not index_exists:
            _emit_progress(
                progress_callback,
                "extract_tiles",
                f"Writing Slideflow tile records for the uploaded slide ({size_mib:.1f} MiB so far).",
                53,
            )
            continue
        _emit_progress(
            progress_callback,
            "extract_tiles",
            f"Tile records and index are now on disk ({size_mib:.1f} MiB). Finalizing extraction before feature generation.",
            58,
        )


def _extract_exact_slideflow_bag(upload_path: Path, *, progress_callback: ProgressCallback | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    _emit_progress(progress_callback, "exact_init", "Preparing the training-matched extraction environment.", 14)
    vm_patch_dir = Path(settings.BASE_DIR) / "vm_patch"
    if vm_patch_dir.exists():
        vm_patch_path = str(vm_patch_dir)
        if vm_patch_path not in sys.path:
            sys.path.insert(0, vm_patch_path)
    try:
        from vm_patch.run_tcga_coad_automated_triad import (
            OUTCOME,
            POS_LABEL,
            TILE_PX,
            TILE_UM,
            build_extractor,
            make_dataset,
            load_project,
        )
    except Exception as exc:
        raise RuntimeError(
            "The training-matched Slideflow extraction path is not available in this environment."
        ) from exc

    exact_settings = _exact_extraction_settings()
    try:
        _emit_progress(progress_callback, "preview_decode", "Building a specimen preview before exact extraction starts.", 18)
        preview_payload = _build_preview_payload_from_upload(
            upload_path,
            tile_count=exact_settings["max_tiles_per_slide"],
        )
    except Exception:
        preview_payload = {
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
            "tile_count": exact_settings["max_tiles_per_slide"],
        }

    temp_root = Path(settings.BASE_DIR) / "runtime" / "tmp_exact_inference"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="manager1_exact_", dir=temp_root) as temp_dir:
        bundle_root = Path(temp_dir)
        slides_dir = bundle_root / "slideflow_project" / "data" / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)

        _emit_progress(progress_callback, "staging_slide", "Copying the uploaded whole-slide image into the temporary exact workspace.", 24)
        suffix = upload_path.suffix.lower() or ".svs"
        slide_stem = _safe_slide_stem(upload_path.name)
        local_slide_path = slides_dir / f"{slide_stem}{suffix}"
        shutil.copy2(upload_path, local_slide_path)

        annotations_csv = bundle_root / "annotations.csv"
        with annotations_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient", "slide", OUTCOME])
            writer.writeheader()
            writer.writerow(
                {
                    "patient": slide_stem,
                    "slide": slide_stem,
                    OUTCOME: POS_LABEL,
                }
            )

        hf_token = str(getattr(settings, "HF_TOKEN", "") or "").strip() or None
        _emit_progress(progress_callback, "load_project", "Creating the temporary Slideflow project for this uploaded slide.", 30)
        sf, project = load_project(bundle_root, annotations_csv, slides_dir, source_name="single_upload")
        dataset = make_dataset(project)
        _emit_progress(progress_callback, "extract_tiles", f"Extracting up to {exact_settings['max_tiles_per_slide']} training-matched tiles.", 46)
        tfrecord_dir = bundle_root / "slideflow_project" / "tfrecords" / f"{TILE_PX}px_{TILE_UM}um"
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor_exact_tile_extraction,
            args=(tfrecord_dir, slide_stem, progress_callback, monitor_stop),
            daemon=True,
            name=f"exact-tiles-{slide_stem[:16]}",
        )
        monitor.start()
        try:
            dataset.extract_tiles(
                qc=exact_settings["qc_method"],
                num_threads=exact_settings["tile_threads"],
                report=False,
                skip_extracted=False,
                max_tiles=exact_settings["max_tiles_per_slide"],
                mpp_override=exact_settings["mpp_override"],
            )
        finally:
            monitor_stop.set()
            monitor.join(timeout=1.0)
        _emit_progress(progress_callback, "build_extractor", "Loading the preserved Virchow2 extractor for exact feature generation.", 60)
        extractor, extractor_name, _backend, _resolved = build_extractor(
            sf,
            ["virchow2"],
            hf_token=hf_token,
            requested_backend="hybrid",
        )
        bags_dir = bundle_root / "slideflow_project" / "bags" / f"{extractor_name}_{TILE_PX}px_{TILE_UM}um"
        bags_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(progress_callback, "generate_bag", "Generating the exact feature bag from extracted pathology tiles.", 76)
        project.generate_feature_bags(extractor, dataset, outdir=str(bags_dir))

        bag_path = bags_dir / f"{slide_stem}.pt"
        if not bag_path.exists():
            bag_candidates = sorted(bags_dir.glob("*.pt"))
            if not bag_candidates:
                raise RuntimeError("Exact Slideflow extraction completed, but no feature bag was written for the uploaded slide.")
            bag_path = bag_candidates[0]

        _emit_progress(progress_callback, "load_bag", "Loading the generated feature bag for ensemble scoring.", 88)
        features = load_feature_bag(bag_path)
        return features, {
            "input_kind": "raw_slide_exact",
            "tile_count": int(features.shape[0]),
            **preview_payload,
        }


def _pil_from_array(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim != 3 or array.shape[2] not in (3,):
        raise ValueError(f"Unsupported TIFF page shape: {tuple(array.shape)}")
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        max_value = float(array.max()) if array.size else 0.0
        if max_value > 0:
            array = np.clip((array / max_value) * 255.0, 0, 255)
        array = array.astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _read_tiff_like_tiles(upload_path: Path, *, tile_count: int) -> tuple[list[Image.Image], dict[str, Any]]:
    with tifffile.TiffFile(str(upload_path)) as tif:
        series = tif.series[0] if tif.series else None
        if series is None:
            raise ValueError("No TIFF image series found in upload.")
        page = None
        best_area = -1
        for level in getattr(series, "levels", []) or [series]:
            shape = getattr(level, "shape", ())
            if len(shape) < 2:
                continue
            area = int(shape[0]) * int(shape[1])
            if area > best_area:
                best_area = area
                page = level
        if page is None:
            raise ValueError("Could not select a readable TIFF level from upload.")
        image = _pil_from_array(page.asarray())
    width, height = image.size
    coords = _sample_image_coords(image, tile_count=max(tile_count, 1))
    if len(coords) < tile_count:
        coords.extend([coords[-1]] * (tile_count - len(coords)))
    chosen = coords[:tile_count]
    worker_count = min(len(chosen), _image_tile_workers())
    if worker_count <= 1:
        tiles = [_crop_tile_from_image(image, sample, width=width, height=height) for sample in chosen]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            tiles = list(pool.map(lambda sample: _crop_tile_from_image(image, sample, width=width, height=height), chosen))
    return tiles, _build_preview_payload(image, chosen, source_width=width, source_height=height)


def _read_slide_tiles(upload_path: Path, *, tile_count: int, progress_callback: ProgressCallback | None = None) -> tuple[list[Image.Image], dict[str, Any]]:
    _emit_progress(progress_callback, "read_slide", "Reading the uploaded slide and probing for a supported slide backend.", 22)
    try:
        slide = openslide.OpenSlide(str(upload_path))
    except openslide.OpenSlideUnsupportedFormatError as exc:
        try:
            return _read_tiff_like_tiles(upload_path, tile_count=tile_count)
        except Exception as fallback_exc:
            raise ValueError(
                "This file could not be opened by OpenSlide, and the TIFF fallback reader also failed. "
                "If it is a normal image, upload it as .png/.jpg. "
                f"Fallback detail: {fallback_exc}"
            ) from exc
    except openslide.OpenSlideError as exc:
        try:
            return _read_tiff_like_tiles(upload_path, tile_count=tile_count)
        except Exception as fallback_exc:
            raise ValueError(
                "The uploaded slide could not be read by OpenSlide on this machine, and the TIFF fallback reader also failed. "
                f"Fallback detail: {fallback_exc}"
            ) from exc
    try:
        width, height = slide.dimensions
        _emit_progress(progress_callback, "sample_tiles", "Sampling tissue-rich regions from the slide preview.", 34)
        thumb = slide.get_thumbnail((min(1024, width), min(1024, height))).convert("RGB")
        coords = _coords_from_thumbnail(thumb, width, height, max_tiles=max(tile_count * 3, tile_count))
        chosen = coords[:tile_count]
        if len(chosen) < tile_count:
            chosen.extend([chosen[-1]] * (tile_count - len(chosen)))
        worker_count = min(len(chosen), _fast_tile_read_workers())
        if worker_count <= 1:
            tiles = [
                slide.read_region((sample.left, sample.top), 0, (RAW_TILE_SIZE, RAW_TILE_SIZE)).convert("RGB")
                for sample in chosen
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                tiles = list(pool.map(lambda sample: _read_region_from_slide(upload_path, sample), chosen))
        return tiles, _build_preview_payload(thumb, chosen, source_width=width, source_height=height)
    finally:
        slide.close()


def _read_image_tiles(upload_path: Path, *, tile_count: int) -> tuple[list[Image.Image], dict[str, Any]]:
    try:
        image = Image.open(upload_path).convert("RGB")
    except Exception as exc:
        raise ValueError("The uploaded image file could not be opened.") from exc
    width, height = image.size
    coords = _sample_image_coords(image, tile_count=max(tile_count, 1))
    if len(coords) < tile_count:
        coords.extend([coords[-1]] * (tile_count - len(coords)))
    chosen = coords[:tile_count]
    worker_count = min(len(chosen), _image_tile_workers())
    if worker_count <= 1:
        tiles = [_crop_tile_from_image(image, sample, width=width, height=height) for sample in chosen]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            tiles = list(pool.map(lambda sample: _crop_tile_from_image(image, sample, width=width, height=height), chosen))
    return tiles, _build_preview_payload(image, chosen, source_width=width, source_height=height)


@torch.inference_mode()
def _encode_tiles(tiles: list[Image.Image], *, progress_callback: ProgressCallback | None = None) -> torch.Tensor:
    _emit_progress(progress_callback, "encode_tiles", "Encoding the sampled tiles into Virchow2 feature vectors.", 72)
    model, transform, _ = _load_encoder()
    preprocess_workers = min(len(tiles), _encode_preprocess_workers())
    if preprocess_workers <= 1:
        encoded_inputs = [transform(tile) for tile in tiles]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=preprocess_workers) as pool:
            encoded_inputs = list(pool.map(transform, tiles))

    device = _device()
    batch_size = _encode_batch_size()
    feature_batches: list[torch.Tensor] = []
    for start in range(0, len(encoded_inputs), batch_size):
        batch = torch.stack(encoded_inputs[start : start + batch_size]).to(device, non_blocking=device.type == "cuda")
        feature_batches.append(model(batch).detach().cpu().float())
    return torch.cat(feature_batches, dim=0) if len(feature_batches) > 1 else feature_batches[0]


def extract_feature_bag(upload_path: Path, *, progress_callback: ProgressCallback | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    suffix = upload_path.suffix.lower()
    tile_count = _current_encoder_metadata()["tile_count"]
    if suffix in TRUSTED_TENSOR_SUFFIXES:
        _emit_progress(progress_callback, "load_feature_bag", "Loading the trusted feature bag directly from disk.", 26)
        return load_feature_bag(upload_path), {
            "input_kind": "feature_bag",
            "specimen_preview_data_url": "",
            "tile_preview_data_url": "",
        }
    if _is_manager1_mode():
        if suffix in RAW_SLIDE_SUFFIXES:
            return _extract_exact_slideflow_bag(upload_path, progress_callback=progress_callback)
        raise ValueError(
            "Manager1 exact mode is active. This high-fidelity path supports trusted feature bags and real whole-slide files only. "
            "Standard image uploads are disabled because they do not follow the preserved Slideflow + libvips extraction path."
        )
    if suffix in RAW_SLIDE_SUFFIXES:
        tiles, preview_payload = _read_slide_tiles(upload_path, tile_count=tile_count, progress_callback=progress_callback)
        return _encode_tiles(tiles, progress_callback=progress_callback), {"input_kind": "raw_slide", "tile_count": len(tiles), **preview_payload}
    if suffix in IMAGE_SUFFIXES:
        _emit_progress(progress_callback, "decode_image", "Opening the uploaded image and preparing tile sampling.", 24)
        tiles, preview_payload = _read_image_tiles(upload_path, tile_count=tile_count)
        return _encode_tiles(tiles, progress_callback=progress_callback), {"input_kind": "image", "tile_count": len(tiles), **preview_payload}
    raise ValueError(
        f"Unsupported upload type: {suffix or 'no extension'}. "
        f"Use a slide/image ({', '.join(sorted(RAW_SLIDE_SUFFIXES | IMAGE_SUFFIXES))}) or a feature bag ({', '.join(sorted(TRUSTED_TENSOR_SUFFIXES))})."
    )


@lru_cache(maxsize=4)
def _load_ensemble_models(mode: str | None = None) -> tuple[tuple[EnsembleCheckpoint, TransMIL], ...]:
    loaded = []
    device = _device()
    with temporary_pipeline_mode(mode):
        for item in _active_ensemble_checkpoints():
            model = TransMIL(input_dim=_feature_dim())
            state_dict = torch.load(item.checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            loaded.append((item, model))
    return tuple(loaded)


@lru_cache(maxsize=1)
def _load_parallel_models() -> dict[str, tuple[tuple[EnsembleCheckpoint, TransMIL], ...]]:
    loaded: dict[str, tuple[tuple[EnsembleCheckpoint, TransMIL], ...]] = {}
    device = _device()
    for spec in _parallel_approach_specs():
        approach_loaded: list[tuple[EnsembleCheckpoint, TransMIL]] = []
        for item in _parallel_checkpoints_by_approach().get(spec.approach_label, ()):
            model = TransMIL(input_dim=spec.feature_dim)
            state_dict = torch.load(item.checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            approach_loaded.append((item, model))
        loaded[spec.approach_label] = tuple(approach_loaded)
    return loaded


def _score_loaded_models(features: torch.Tensor, loaded_models: tuple[tuple[EnsembleCheckpoint, TransMIL], ...]) -> dict[str, Any]:
    per_checkpoint = []
    bundle_positive_probabilities = []
    quality_scores = []
    device = _device()
    features_on_device = features.to(device, non_blocking=device.type == "cuda").clone()

    if device.type == "cuda":
        for item, model in loaded_models:
            logits = model(features_on_device)
            bundle_positive_probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
            bundle_positive_probabilities.append(bundle_positive_probability)
            quality_score = (
                float(item.auroc) * 0.40
                + float(item.f1_macro) * 0.25
                + float(item.auprc) * 0.20
                + float(item.balanced_accuracy) * 0.15
            )
            quality_scores.append(quality_score)
            per_checkpoint.append(
                {
                    "checkpoint": item.checkpoint_path.name,
                    "repeat": item.repeat,
                    "fold": item.fold,
                    "probability": 1.0 - bundle_positive_probability,
                    "threshold": 1.0 - float(item.threshold),
                    "auroc": item.auroc,
                    "f1_macro": item.f1_macro,
                    "auprc": item.auprc,
                    "balanced_accuracy": item.balanced_accuracy,
                    "quality_score": quality_score,
                    "approach_label": item.approach_label,
                    "extractor": item.extractor,
                }
            )
    else:
        for item, model in loaded_models:
            logits = model(features_on_device)
            bundle_positive_probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
            quality_score = (
                float(item.auroc) * 0.40
                + float(item.f1_macro) * 0.25
                + float(item.auprc) * 0.20
                + float(item.balanced_accuracy) * 0.15
            )
            per_checkpoint.append(
                {
                    "checkpoint": item.checkpoint_path.name,
                    "repeat": item.repeat,
                    "fold": item.fold,
                    "probability": 1.0 - bundle_positive_probability,
                    "threshold": 1.0 - float(item.threshold),
                    "auroc": item.auroc,
                    "f1_macro": item.f1_macro,
                    "auprc": item.auprc,
                    "balanced_accuracy": item.balanced_accuracy,
                    "quality_score": quality_score,
                    "approach_label": item.approach_label,
                    "extractor": item.extractor,
                }
            )
            bundle_positive_probabilities.append(bundle_positive_probability)
            quality_scores.append(quality_score)
    bundle_positive_probability = float(sum(bundle_positive_probabilities) / max(1, len(bundle_positive_probabilities)))
    bundle_positive_threshold = float(sum(item.threshold for item, _ in loaded_models) / max(1, len(bundle_positive_probabilities)))
    ensemble_probability = 1.0 - bundle_positive_probability
    ensemble_threshold = 1.0 - bundle_positive_threshold
    vote_spread = abs(bundle_positive_probability - bundle_positive_threshold)
    vote_score = max(0.0, min(1.0, vote_spread * 2.0))
    model_quality = float(sum(quality_scores) / max(1, len(quality_scores)))
    blended_confidence = (model_quality * 0.65) + (vote_score * 0.35)
    if blended_confidence >= 0.82:
        confidence_level = "High"
    elif blended_confidence >= 0.64:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    return {
        "label": "MSI-H" if ensemble_probability >= ensemble_threshold else "MSS",
        "probability": ensemble_probability,
        "threshold": ensemble_threshold,
        "confidence": vote_spread,
        "confidence_score": blended_confidence,
        "confidence_percent": blended_confidence * 100.0,
        "confidence_level": confidence_level,
        "model_quality_score": model_quality,
        "vote_strength_score": vote_score,
        "tile_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "checkpoint_count": len(bundle_positive_probabilities),
        "per_checkpoint": per_checkpoint,
    }


@torch.inference_mode()
def _predict_feature_tensor(features: torch.Tensor, *, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    _emit_progress(progress_callback, "ensemble_scoring", "Running the preserved TransMIL ensemble across the extracted feature bag.", 90)
    loaded_models = _load_ensemble_models(_pipeline_mode())
    return _score_loaded_models(features, loaded_models)


@torch.inference_mode()
def predict_parallel_feature_package(upload_path: Path, *, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    _emit_progress(progress_callback, "parallel_package", "Loading the preserved four-model feature package.", 18)
    feature_bags = load_parallel_feature_package(upload_path)
    loaded_models = _load_parallel_models()
    approach_results: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    total_tiles = 0
    total_checkpoints = 0
    for index, spec in enumerate(_parallel_approach_specs(), start=1):
        _emit_progress(
            progress_callback,
            "parallel_scoring",
            f"Scoring {spec.approach_label} ({index}/{len(_parallel_approach_specs())}) with its preserved checkpoints.",
            30 + int(index * 12),
        )
        result = _score_loaded_models(feature_bags[spec.approach_label], loaded_models.get(spec.approach_label, ()))
        result["approach_label"] = spec.approach_label
        result["extractor"] = spec.extractor
        result["quality_weight"] = spec.quality_weight
        result["mean_auroc"] = spec.mean_auroc
        result["mean_f1_macro"] = spec.mean_f1_macro
        result["mean_auprc"] = spec.mean_auprc
        result["mean_balanced_accuracy"] = spec.mean_balanced_accuracy
        approach_results.append(result)
        checkpoint_rows.extend(result["per_checkpoint"])
        total_tiles += int(result["tile_count"])
        total_checkpoints += int(result["checkpoint_count"])

    total_weight = sum(max(float(item["quality_weight"]), 1e-6) for item in approach_results)
    fused_probability = sum(float(item["probability"]) * float(item["quality_weight"]) for item in approach_results) / max(total_weight, 1e-6)
    fused_threshold = sum(float(item["threshold"]) * float(item["quality_weight"]) for item in approach_results) / max(total_weight, 1e-6)
    equal_weight_probability = sum(float(item["probability"]) for item in approach_results) / max(1, len(approach_results))
    vote_spread = abs(fused_probability - fused_threshold)
    vote_score = max(0.0, min(1.0, vote_spread * 2.0))
    model_quality = sum(float(item["model_quality_score"]) * float(item["quality_weight"]) for item in approach_results) / max(total_weight, 1e-6)
    blended_confidence = (model_quality * 0.65) + (vote_score * 0.35)
    if blended_confidence >= 0.82:
        confidence_level = "High"
    elif blended_confidence >= 0.64:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    _emit_progress(progress_callback, "completed", "Parallel offline fusion is complete and the response payload is being returned.", 100)
    return {
        "label": "MSI-H" if fused_probability >= fused_threshold else "MSS",
        "probability": float(fused_probability),
        "equal_weight_probability": float(equal_weight_probability),
        "threshold": float(fused_threshold),
        "confidence": vote_spread,
        "confidence_score": blended_confidence,
        "confidence_percent": blended_confidence * 100.0,
        "confidence_level": confidence_level,
        "model_quality_score": model_quality,
        "vote_strength_score": vote_score,
        "tile_count": total_tiles,
        "feature_dim": 0,
        "checkpoint_count": total_checkpoints,
        "input_kind": "parallel_feature_package",
        "input_kind_display": "Parallel feature package",
        "encoder_label": "preserved-top4-package",
        "encoder_backbone": ", ".join(spec.extractor for spec in _parallel_approach_specs()),
        "encoder_type": "offline_multi_bag",
        "per_checkpoint": checkpoint_rows,
        "per_approach": approach_results,
        "feature_bag_count": len(approach_results),
        "fusion_method": "quality_weighted_mean",
    }


@torch.inference_mode()
def predict_feature_bag(upload_path: Path, *, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    features = load_feature_bag(upload_path)
    result = _predict_feature_tensor(features, progress_callback=progress_callback)
    result["input_kind"] = "feature_bag"
    return result


def predict_upload(upload_path: Path, *, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    _emit_progress(progress_callback, "request_received", "The backend accepted the upload and is validating the runtime bundle.", 6)
    if _is_parallel_mode():
        result = predict_parallel_feature_package(upload_path, progress_callback=progress_callback)
        return result
    serving_ready, serving_message = _serving_compatibility()
    if not serving_ready:
        raise RuntimeError(serving_message)
    try:
        features, payload_meta = extract_feature_bag(upload_path, progress_callback=progress_callback)
    except EncoderPackageError as exc:
        raise RuntimeError(f"Local encoder package is not ready: {exc}") from exc
    _emit_progress(progress_callback, "features_ready", "Feature preparation is complete; final ensemble scoring is starting.", 86)
    result = _predict_feature_tensor(features, progress_callback=progress_callback)
    result["input_kind"] = payload_meta["input_kind"]
    result["input_kind_display"] = payload_meta["input_kind"].replace("_", " ").title()
    encoder_meta = _current_encoder_metadata()
    result["encoder_label"] = encoder_meta["encoder_label"]
    result["encoder_type"] = encoder_meta["encoder_type"]
    result["encoder_backbone"] = encoder_meta["backbone_name"]
    result["specimen_preview_data_url"] = payload_meta.get("specimen_preview_data_url", "")
    result["tile_preview_data_url"] = payload_meta.get("tile_preview_data_url", "")
    _emit_progress(progress_callback, "completed", "Prediction is complete and the response payload is being returned.", 100)
    return result
