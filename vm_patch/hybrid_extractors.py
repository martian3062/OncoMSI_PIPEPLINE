from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from huggingface_hub import login
from timm import create_model
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import transforms


HYBRID_EXTRACTOR_ALIASES = {
    "conch": "conch",
    "virchow2": "virchow2",
    "uni2-h": "uni2-h",
    "uni2h": "uni2-h",
    "h-optimus-0": "h-optimus-0",
    "h_optimus_0": "h-optimus-0",
}

HYBRID_EXTRACTOR_NAMES = {
    "conch",
    "virchow2",
    "uni2-h",
    "h-optimus-0",
}

_REGISTERED = False


@dataclass
class ExtractorSpec:
    name: str
    num_features: int | None
    model_builder: Callable[[str | None], tuple[nn.Module, Callable[..., Any]]]
    aliases: tuple[str, ...] = ()


class _ConchImageEncoder(nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image, proj_contrast=False, normalize=False)


class _Virchow2ImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 2:
            return output
        if output.ndim != 3:
            raise RuntimeError(f"Virchow2 encoder returned unexpected shape: {tuple(output.shape)}")
        class_token = output[:, 0]
        patch_tokens = output[:, 5:] if output.shape[1] > 5 else output[:, 1:]
        if patch_tokens.numel() == 0:
            features = class_token
        else:
            features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        if features.ndim != 2:
            raise RuntimeError(f"Virchow2 pooled features must be 2D, got {tuple(features.shape)}")
        return features


class _TensorFriendlyTransform:
    def __init__(self, preprocess: Callable[..., Any]):
        self.preprocess = preprocess

    def __repr__(self) -> str:
        return repr(self.preprocess)

    def __call__(self, image: Any):
        if not isinstance(image, torch.Tensor):
            return self.preprocess(image)

        output = image
        if not torch.is_floating_point(output):
            output = output.float()
        if output.max().item() > 1.0:
            output = output / 255.0

        if isinstance(self.preprocess, transforms.Compose):
            for step in self.preprocess.transforms:
                if isinstance(step, transforms.ToTensor):
                    continue
                if getattr(step, "__name__", "") == "_convert_to_rgb":
                    continue
                output = step(output)
            return output
        return self.preprocess(output)


def normalize_extractor_name(name: str) -> str:
    key = str(name).strip().lower()
    return HYBRID_EXTRACTOR_ALIASES.get(key, key)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_hf_token(hf_token: str | None) -> str | None:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    try:
        login(token=token, add_to_git_credential=False)
    except Exception:
        pass
    return token


def _build_virchow2(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        num_classes=0,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return _Virchow2ImageEncoder(model), _TensorFriendlyTransform(transform)


def _build_uni2_h(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, _TensorFriendlyTransform(transform)


def _build_h_optimus_0(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        num_classes=0,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ]
    )
    return model, _TensorFriendlyTransform(transform)


def _build_conch(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    token = _configure_hf_token(hf_token)
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError(
            "CONCH support requires the optional package from "
            "git+https://github.com/Mahmoodlab/CONCH.git"
        ) from exc
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        "hf_hub:MahmoodLab/CONCH",
        hf_auth_token=token,
    )
    return _ConchImageEncoder(model), _TensorFriendlyTransform(preprocess)


SPECS = {
    "virchow2": ExtractorSpec("virchow2", 2560, _build_virchow2),
    "uni2-h": ExtractorSpec("uni2-h", 1536, _build_uni2_h, aliases=("uni2h",)),
    "h-optimus-0": ExtractorSpec("h-optimus-0", 1536, _build_h_optimus_0, aliases=("h_optimus_0",)),
    "conch": ExtractorSpec("conch", 512, _build_conch),
}


def hybrid_backend_for_name(name: str) -> str:
    return "hybrid" if normalize_extractor_name(name) in HYBRID_EXTRACTOR_NAMES else "slideflow"


def register_hybrid_extractors(sf, hf_token: str | None = None) -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from slideflow.model.extractors import register_torch
    from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

    def _build_extractor_class(spec: ExtractorSpec):
        class HybridFeatureExtractor(TorchFeatureExtractor):
            tag = spec.name

            def __init__(self, hf_token: str | None = None):
                super().__init__()
                model, transform = spec.model_builder(hf_token or os.environ.get("HF_TOKEN"))
                self.model = model.to(_device())
                self.model.eval()
                self.transform = transform
                self.preprocess_kwargs = {"standardize": False}
                self.num_features = spec.num_features

            def dump_config(self):
                return self._dump_config(
                    class_name=f"hybrid_extractors.{self.__class__.__name__}",
                    kwargs={},
                )

        HybridFeatureExtractor.__name__ = f"{spec.name.replace('-', '_').title().replace('_', '')}Extractor"
        return HybridFeatureExtractor

    for spec in SPECS.values():
        extractor_cls = _build_extractor_class(spec)

        def _make_factory(extractor_cls=extractor_cls):
            def _factory(hf_token: str | None = None, **_: Any):
                return extractor_cls(hf_token=hf_token)

            return _factory

        for tag in {spec.name, *spec.aliases}:
            register_torch(tag)(_make_factory())

    _REGISTERED = True
