from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from timm import create_model
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import transforms


class EncoderPackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncoderPackage:
    label: str
    package_type: str
    root: Path
    embedding_dim: int
    model: nn.Module
    transform: Any
    manifest: dict[str, Any]


class LocalDirVisionEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        pool: str = "cls_mean",
        patch_token_offset: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.pool = pool
        self.patch_token_offset = patch_token_offset

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(batch)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        if features.ndim != 3:
            raise RuntimeError(f"Unexpected encoder output shape: {tuple(features.shape)}")
        cls_token = features[:, 0]
        patch_tokens = features[:, self.patch_token_offset :] if features.shape[1] > self.patch_token_offset else features[:, 1:]
        if patch_tokens.numel() == 0 or self.pool == "cls":
            return cls_token
        patch_mean = patch_tokens.mean(dim=1)
        if self.pool == "mean":
            return patch_mean
        if self.pool == "cls_mean":
            return torch.cat([cls_token, patch_mean], dim=-1)
        raise RuntimeError(f"Unsupported local-dir pooling mode: {self.pool}")


class TimmProjectionEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(backbone_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.backbone(batch)
        return self.projection(_reduce_backbone_features(features))


def _reduce_backbone_features(features: Any) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    if hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state
    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    if features.ndim == 3:
        features = features.mean(dim=1)
    if features.ndim != 2:
        raise RuntimeError(f"Unexpected student backbone output shape: {tuple(features.shape)}")
    return features


def infer_backbone_output_dim(backbone: nn.Module, input_size: int = 224) -> int:
    backbone_device = next(backbone.parameters(), None)
    probe_device = backbone_device.device if backbone_device is not None else torch.device("cpu")
    was_training = backbone.training
    backbone.eval()
    with torch.inference_mode():
        probe = torch.zeros(1, 3, input_size, input_size, device=probe_device)
        reduced = _reduce_backbone_features(backbone(probe))
    if was_training:
        backbone.train()
    return int(reduced.shape[1])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_encoder_manifest(package_dir: str | Path) -> dict[str, Any]:
    package_dir = Path(package_dir)
    manifest_path = package_dir / "PACKAGE_MANIFEST.json"
    if manifest_path.exists():
        return _read_json(manifest_path)
    config_path = package_dir / "config.json"
    safetensor_path = package_dir / "model.safetensors"
    if config_path.exists() and safetensor_path.exists():
        return {
            "package_type": "timm_local_dir_encoder",
            "encoder_label": package_dir.name,
            "embedding_dim": 2560,
            "pool": "cls_mean",
            "patch_token_offset": 5,
            "extra_create_kwargs": {
                "num_classes": 0,
                "mlp_layer": "SwiGLUPacked",
                "act_layer": "SiLU",
            },
        }
    raise EncoderPackageError(f"No PACKAGE_MANIFEST.json or local-dir config found in {package_dir}")


def _projection_transform(manifest: dict[str, Any]):
    input_size = int(manifest.get("input_size", 224))
    mean = tuple(manifest.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(manifest.get("std", (0.229, 0.224, 0.225)))
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _build_local_dir_package(package_dir: Path, manifest: dict[str, Any]) -> EncoderPackage:
    extra = dict(manifest.get("extra_create_kwargs") or {})
    if extra.get("mlp_layer") == "SwiGLUPacked":
        extra["mlp_layer"] = SwiGLUPacked
    if extra.get("act_layer") == "SiLU":
        extra["act_layer"] = torch.nn.SiLU
    model = create_model(f"local-dir:{package_dir}", pretrained=True, **extra)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    encoder = LocalDirVisionEncoder(
        model,
        pool=str(manifest.get("pool", "cls_mean")),
        patch_token_offset=int(manifest.get("patch_token_offset", 1)),
    )
    return EncoderPackage(
        label=str(manifest.get("encoder_label") or package_dir.name),
        package_type="timm_local_dir_encoder",
        root=package_dir,
        embedding_dim=int(manifest.get("embedding_dim", 2560)),
        model=encoder,
        transform=transform,
        manifest=manifest,
    )


def _build_projection_package(package_dir: Path, manifest: dict[str, Any]) -> EncoderPackage:
    backbone_name = str(manifest.get("backbone_name") or "")
    checkpoint_file = str(manifest.get("checkpoint_file") or "student_encoder.pt")
    if not backbone_name:
        raise EncoderPackageError("Projection encoder package is missing backbone_name")
    checkpoint_path = package_dir / checkpoint_file
    if not checkpoint_path.exists():
        raise EncoderPackageError(f"Projection encoder checkpoint not found: {checkpoint_path}")
    backbone = create_model(
        backbone_name,
        pretrained=False,
        num_classes=0,
        global_pool=str(manifest.get("global_pool", "avg")),
    )
    input_size = int(manifest.get("input_size", 224))
    backbone_dim = int(manifest.get("backbone_dim", 0))
    if backbone_dim <= 0:
        backbone_dim = infer_backbone_output_dim(backbone, input_size=input_size)
    embedding_dim = int(manifest.get("embedding_dim", 2560))
    if backbone_dim <= 0:
        raise EncoderPackageError(f"Could not infer backbone feature dim for {backbone_name}")
    encoder = TimmProjectionEncoder(backbone, backbone_dim=backbone_dim, embedding_dim=embedding_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    encoder.load_state_dict(state_dict, strict=True)
    return EncoderPackage(
        label=str(manifest.get("encoder_label") or package_dir.name),
        package_type="timm_projection_encoder",
        root=package_dir,
        embedding_dim=embedding_dim,
        model=encoder,
        transform=_projection_transform(manifest),
        manifest=manifest,
    )


def load_encoder_package(package_dir: str | Path) -> EncoderPackage:
    package_dir = Path(package_dir)
    manifest = load_encoder_manifest(package_dir)
    package_type = str(manifest.get("package_type") or "")
    if package_type == "timm_projection_encoder":
        return _build_projection_package(package_dir, manifest)
    if package_type == "timm_local_dir_encoder":
        return _build_local_dir_package(package_dir, manifest)
    raise EncoderPackageError(f"Unsupported encoder package type: {package_type}")
