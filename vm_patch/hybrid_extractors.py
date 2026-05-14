from __future__ import annotations

import os
import sys
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, login
from timm import create_model
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
APP_ROOT = ROOT_DIR / "django_rebuild_cleaned_msi"
if APP_ROOT.exists() and str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from local_encoder_package import load_encoder_package


HYBRID_EXTRACTOR_ALIASES = {
    "conch": "conch",
    "conch-v1.5": "conchv1_5",
    "conch-v15": "conchv1_5",
    "conch_v1_5": "conchv1_5",
    "conchv1.5": "conchv1_5",
    "conchv15": "conchv1_5",
    "conchv1_5": "conchv1_5",
    "virchow2": "virchow2",
    "uni2-h": "uni2-h",
    "uni2h": "uni2-h",
    "h-optimus-0": "h-optimus-0",
    "h_optimus_0": "h-optimus-0",
    "phikon-v2": "phikon-v2",
    "phikon_v2": "phikon-v2",
    "prov-gigapath": "prov-gigapath",
    "prov_gigapath": "prov-gigapath",
    "prism": "prism-virchow",
    "prism-virchow": "prism-virchow",
    "prism_virchow": "prism-virchow",
    "dinov2-large": "dinov2-large",
    "dinov2_large": "dinov2-large",
    "dinov2": "dinov2-large",
    "dino-v2-large": "dinov2-large",
    "dinov3-vitb16": "dinov3-vitb16",
    "dinov3": "dinov3-vitb16",
    "dino-v3": "dinov3-vitb16",
    "dino_v3": "dinov3-vitb16",
    "chief": "chief",
    "chief-ctranspath": "chief",
    "chief_ctp": "chief",
    "midnight": "midnight",
    "midnight-12k": "midnight",
    "student": "student-virchow2",
    "student-virchow2": "student-virchow2",
    "student_virchow2": "student-virchow2",
}

HYBRID_EXTRACTOR_NAMES = {
    "conch",
    "conchv1_5",
    "virchow2",
    "uni2-h",
    "h-optimus-0",
    "phikon-v2",
    "prov-gigapath",
    "prism-virchow",
    "dinov2-large",
    "dinov3-vitb16",
    "chief",
    "midnight",
    "student-virchow2",
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


class _ForwardImageEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        forward_kind: str = "tensor",
        pool: str = "cls",
        patch_token_offset: int = 1,
    ):
        super().__init__()
        self.model = model
        self.forward_kind = forward_kind
        self.pool = pool
        self.patch_token_offset = patch_token_offset

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.forward_kind == "pixel_values":
            output = self.model(pixel_values=image)
        else:
            output = self.model(image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        if output.ndim == 2:
            return output
        if output.ndim == 4:
            if self.pool == "spatial_mean":
                return output.mean(dim=(1, 2))
            raise RuntimeError(f"Unsupported 4D pooling mode: {self.pool}")
        if output.ndim != 3:
            raise RuntimeError(f"Extractor returned unexpected shape: {tuple(output.shape)}")

        cls_token = output[:, 0]
        patch_tokens = output[:, self.patch_token_offset :] if output.shape[1] > self.patch_token_offset else output[:, 1:]
        if self.pool == "cls":
            return cls_token
        if patch_tokens.numel() == 0:
            return cls_token
        if self.pool == "mean":
            return patch_tokens.mean(1)
        if self.pool == "cls_mean":
            return torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)
        raise RuntimeError(f"Unsupported pooling mode: {self.pool}")


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


def _install_timm_compat_shims() -> None:
    """Backfill timm module paths expected by older third-party repos."""
    try:
        import timm.layers.helpers as timm_layers_helpers
    except Exception:
        return
    sys.modules.setdefault("timm.models.layers.helpers", timm_layers_helpers)


def _build_processor_transform(
    model_id: str,
    *,
    trust_remote_code: bool = False,
) -> _TensorFriendlyTransform:
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    size = getattr(processor, "size", {}) or {}
    crop_size = int(size.get("height") or size.get("width") or size.get("shortest_edge") or 224)
    resize_size = int(size.get("shortest_edge") or crop_size)
    mean = tuple(getattr(processor, "image_mean", None) or (0.5, 0.5, 0.5))
    std = tuple(getattr(processor, "image_std", None) or (0.5, 0.5, 0.5))
    pipeline = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return _TensorFriendlyTransform(pipeline)


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
    return _ForwardImageEncoder(model, pool="cls_mean", patch_token_offset=5), _TensorFriendlyTransform(transform)


def _build_prism_virchow(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return _ForwardImageEncoder(model, pool="cls_mean", patch_token_offset=1), _TensorFriendlyTransform(transform)


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
    return _ForwardImageEncoder(model, pool="spatial_mean"), _TensorFriendlyTransform(transform)


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
    return _ForwardImageEncoder(model, pool="spatial_mean"), _TensorFriendlyTransform(transform)


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


def _build_conchv1_5(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    weights_path = hf_hub_download(
        repo_id="MahmoodLab/conchv1_5",
        filename="pytorch_model_vision.bin",
        token=hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    raw_state = torch.load(weights_path, map_location="cpu")
    state_dict = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    cleaned_state = {
        key.removeprefix("trunk."): value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("trunk.")
    }
    model = create_model(
        "vit_large_patch16_224",
        pretrained=False,
        img_size=448,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
    )
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    allowed_missing = {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
    real_missing = [item for item in missing if item not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            f"CONCHv1.5 manual load mismatch. Missing={real_missing} Unexpected={unexpected}"
        )
    transform = transforms.Compose(
        [
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return _ForwardImageEncoder(model, pool="cls"), _TensorFriendlyTransform(transform)


def _build_phikon_v2(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    transform = _build_processor_transform("owkin/phikon-v2")
    return _ForwardImageEncoder(model, forward_kind="pixel_values", pool="cls"), transform


def _build_prov_gigapath(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = create_model(
        "hf-hub:prov-gigapath/prov-gigapath",
        pretrained=True,
        num_classes=0,
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return _ForwardImageEncoder(model, pool="cls"), _TensorFriendlyTransform(transform)


def _build_dinov2_large(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model_id = "facebook/dinov2-large"
    model = AutoModel.from_pretrained(model_id)
    transform = _build_processor_transform(model_id)
    return _ForwardImageEncoder(model, forward_kind="pixel_values", pool="cls"), transform


def _build_dinov3_vitb16(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    model = AutoModel.from_pretrained(model_id)
    transform = _build_processor_transform(model_id)
    return _ForwardImageEncoder(model, forward_kind="pixel_values", pool="cls"), transform


def _build_midnight(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    _configure_hf_token(hf_token)
    model = AutoModel.from_pretrained("kaiko-ai/midnight")
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return _ForwardImageEncoder(model, forward_kind="pixel_values", pool="cls_mean"), _TensorFriendlyTransform(transform)


def _build_local_student(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    del hf_token
    package_dir = os.environ.get("MSI_STUDENT_ENCODER_DIR") or os.environ.get("MSI_LOCAL_ENCODER_DIR")
    if not package_dir:
        raise FileNotFoundError("Set MSI_STUDENT_ENCODER_DIR to a distilled encoder package directory before using student-virchow2.")
    package = load_encoder_package(package_dir)
    return package.model, _TensorFriendlyTransform(package.transform)


def _build_chief(hf_token: str | None) -> tuple[nn.Module, Callable[..., Any]]:
    repo_path = os.environ.get("CHIEF_REPO_PATH", "/home/pardeep/models/CHIEF")
    weights_path = os.environ.get("CHIEF_CTRANSPATH_WEIGHTS", "/home/pardeep/models/CHIEF/model_weight/CHIEF_CTransPath.pth")
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"CHIEF repo not found at {repo_path}. Clone https://github.com/hms-dbmi/CHIEF.git first."
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"CHIEF CTransPath weights not found at {weights_path}. Install CHIEF_CTransPath.pth first."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    _install_timm_compat_shims()
    import models.ctran as chief_ctran

    if not getattr(chief_ctran, "_codex_convstem_compat", False):
        original_convstem = chief_ctran.ConvStem

        class CompatConvStem(original_convstem):
            def __init__(
                self,
                img_size: int = 224,
                patch_size: int = 4,
                in_chans: int = 3,
                embed_dim: int = 768,
                norm_layer: Callable[..., nn.Module] | None = None,
                flatten: bool = True,
                output_fmt: str | None = None,
                strict_img_size: bool | None = None,
                **kwargs: Any,
            ) -> None:
                del strict_img_size, kwargs
                super().__init__(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    norm_layer=norm_layer,
                    flatten=flatten if output_fmt is None else False,
                )
                self.output_fmt = output_fmt
                self.flatten = flatten if output_fmt is None else False

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch, channels, height, width = x.shape
                assert height == self.img_size[0] and width == self.img_size[1], (
                    f"Input image size ({height}*{width}) doesn't match model "
                    f"({self.img_size[0]}*{self.img_size[1]})."
                )
                x = self.proj(x)
                if self.flatten:
                    x = x.flatten(2).transpose(1, 2)
                elif self.output_fmt and self.output_fmt.upper() == "NHWC":
                    x = x.permute(0, 2, 3, 1).contiguous()
                x = self.norm(x)
                return x

        chief_ctran.ConvStem = CompatConvStem
        chief_ctran._codex_convstem_compat = True

    ctranspath = chief_ctran.ctranspath

    model = ctranspath()
    model.head = nn.Identity()
    state = torch.load(weights_path, map_location="cpu")
    raw_state = state.get("model", state)
    remapped_state = {}
    downsample_pattern = re.compile(r"^layers\.(\d+)\.downsample\.(.+)$")
    for key, value in raw_state.items():
        if key.endswith("attn.relative_position_index") or key.endswith("attn_mask"):
            continue
        match = downsample_pattern.match(key)
        if match:
            key = f"layers.{int(match.group(1)) + 1}.downsample.{match.group(2)}"
        remapped_state[key] = value
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    real_missing = [item for item in missing if item not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            f"CHIEF CTransPath load mismatch. Missing={real_missing} Unexpected={unexpected}"
        )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return _ForwardImageEncoder(model, pool="spatial_mean"), _TensorFriendlyTransform(transform)


SPECS = {
    "conch": ExtractorSpec("conch", 512, _build_conch),
    "conchv1_5": ExtractorSpec("conchv1_5", None, _build_conchv1_5, aliases=("conch-v1.5", "conch-v15", "conchv1.5")),
    "virchow2": ExtractorSpec("virchow2", 2560, _build_virchow2),
    "prism-virchow": ExtractorSpec("prism-virchow", 2560, _build_prism_virchow, aliases=("prism",)),
    "uni2-h": ExtractorSpec("uni2-h", 1536, _build_uni2_h, aliases=("uni2h",)),
    "h-optimus-0": ExtractorSpec("h-optimus-0", 1536, _build_h_optimus_0, aliases=("h_optimus_0",)),
    "phikon-v2": ExtractorSpec("phikon-v2", 1024, _build_phikon_v2, aliases=("phikon_v2",)),
    "prov-gigapath": ExtractorSpec("prov-gigapath", None, _build_prov_gigapath, aliases=("prov_gigapath",)),
    "dinov2-large": ExtractorSpec("dinov2-large", 1024, _build_dinov2_large, aliases=("dinov2_large", "dinov2", "dino-v2-large")),
    "dinov3-vitb16": ExtractorSpec("dinov3-vitb16", 768, _build_dinov3_vitb16, aliases=("dinov3", "dino-v3", "dino_v3")),
    "chief": ExtractorSpec("chief", 768, _build_chief, aliases=("chief-ctranspath", "chief_ctp")),
    "midnight": ExtractorSpec("midnight", None, _build_midnight, aliases=("midnight-12k",)),
    "student-virchow2": ExtractorSpec("student-virchow2", 2560, _build_local_student, aliases=("student", "student_virchow2")),
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
                self.num_features = spec.num_features or getattr(model, "num_features", None)

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
