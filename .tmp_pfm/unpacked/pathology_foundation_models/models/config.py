import torch


from enum import Enum
from torch import nn
from typing import Callable

from pathology_foundation_models.models.adapters.loader import (
    __default_hf_loader,
    __load_uni,
    __load_uni2h,
    __load_h_optimus_0,
    __load_hibou_b,
    __load_hibou_L,
    __load_virchow,
    __load_virchow_v2,
)
from pathology_foundation_models.models.adapters.inference import (
    __extract_features_vit_default,
    __extract_features_resnet_default,
    __extract_features_virchow_default,
    __extract_features_uni,
    __extract_features_uni2h,
    __extract_features_h_optimus_0,
)


class FoundationModelEnum(Enum):
    """
    Enum for foundation model types.
    """

    UNI = "MahmoodLab/uni"
    UNI2H = "MahmoodLab/UNI2-h"
    PHIKON = "owkin/phikon"
    PHIKON_V2 = "owkin/phikon-v2"
    H_OPTIMUS_0 = "bioptimus/H-optimus-0"
    HIBOU_B = "histai/hibou-b"
    HIBOU_L = "histai/hibou-L"
    VIRCHOW = "paige-ai/Virchow"
    VIRCHOW_V2 = "paige-ai/Virchow2"
    DINO_V1_S16 = "facebook/dino-vits16"
    DINO_V1_B16 = "facebook/dino-vitb16"
    DINO_V1_S8 = "facebook/dino-vits8"
    DINO_V1_B8 = "facebook/dino-vitb8"
    DINO_V2_S = "facebook/dinov2-small"
    DINO_V2_B = "facebook/dinov2-base"
    DINO_V2_L = "facebook/dinov2-large"
    DINO_V2_G = "facebook/dinov2-giant"
    VIT_B16 = "google/vit-base-patch16-224"
    VIT_L16 = "google/vit-large-patch16-224"
    RESNET_18 = "microsoft/resnet-18"
    RESNET_34 = "microsoft/resnet-34"
    RESNET_50 = "microsoft/resnet-50"


_embedding_dims = {
    FoundationModelEnum.UNI: 1024,
    FoundationModelEnum.UNI2H: 1536,
    FoundationModelEnum.PHIKON: 768,
    FoundationModelEnum.PHIKON_V2: 1024,
    FoundationModelEnum.H_OPTIMUS_0: 1536,
    FoundationModelEnum.HIBOU_B: 768,
    FoundationModelEnum.HIBOU_L: 1024,
    FoundationModelEnum.VIRCHOW: 2560,
    FoundationModelEnum.VIRCHOW_V2: 2560,
    # TODO find out embedding dims
    FoundationModelEnum.DINO_V1_S16: 384,
    FoundationModelEnum.DINO_V1_B16: 768,
    FoundationModelEnum.DINO_V1_S8: 384,
    FoundationModelEnum.DINO_V1_B8: 768,
    FoundationModelEnum.DINO_V2_S: 384,
    FoundationModelEnum.DINO_V2_B: 768,
    FoundationModelEnum.DINO_V2_L: 1024,
    FoundationModelEnum.DINO_V2_G: 1536,
    FoundationModelEnum.VIT_B16: 768,
    FoundationModelEnum.VIT_L16: 1024,
    FoundationModelEnum.RESNET_18: 512,
    FoundationModelEnum.RESNET_34: 512,
    FoundationModelEnum.RESNET_50: 2048,
}


def list_models() -> list[str]:
    """
    Lists all available models
    """
    return [model.value for model in FoundationModelEnum]


def get_embedding_dim(model_type: FoundationModelEnum | str) -> int:
    """
    Returns the embedding dimension for the model type.
    """
    model_type = FoundationModelEnum(model_type)
    try:
        return _embedding_dims[model_type]
    except KeyError:
        raise NotImplementedError(f"Unknown model type: {model_type.value}")


def get_loader_fn(
    model_type: FoundationModelEnum,
) -> Callable[[], tuple[nn.Module, nn.Module]]:
    """
    Returns the model loading function for the model type.
    """
    if model_type == FoundationModelEnum.UNI:
        return __load_uni
    elif model_type == FoundationModelEnum.UNI2H:
        return __load_uni2h
    elif model_type == FoundationModelEnum.H_OPTIMUS_0:
        return __load_h_optimus_0
    elif model_type == FoundationModelEnum.HIBOU_B:
        return __load_hibou_b
    elif model_type == FoundationModelEnum.HIBOU_L:
        return __load_hibou_L
    elif model_type == FoundationModelEnum.VIRCHOW:
        return __load_virchow
    elif model_type == FoundationModelEnum.VIRCHOW_V2:
        return __load_virchow_v2
    elif model_type in [
        FoundationModelEnum.PHIKON,
        FoundationModelEnum.PHIKON_V2,
        FoundationModelEnum.VIT_B16,
        FoundationModelEnum.VIT_L16,
        FoundationModelEnum.DINO_V1_S8,
        FoundationModelEnum.DINO_V1_B8,
        FoundationModelEnum.DINO_V1_S16,
        FoundationModelEnum.DINO_V1_B16,
        FoundationModelEnum.DINO_V2_S,
        FoundationModelEnum.DINO_V2_B,
        FoundationModelEnum.DINO_V2_L,
        FoundationModelEnum.DINO_V2_G,
        FoundationModelEnum.RESNET_18,
        FoundationModelEnum.RESNET_34,
        FoundationModelEnum.RESNET_50,
    ]:
        return __default_hf_loader(model_type.value)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type.value}")


def get_inference_fn(
    model_type: FoundationModelEnum,
) -> Callable[[torch.Tensor, nn.Module, nn.Module], torch.Tensor]:
    """
    Returns the inference function for the model type.
    """
    if model_type == FoundationModelEnum.UNI:
        return __extract_features_uni
    elif model_type == FoundationModelEnum.UNI2H:
        return __extract_features_uni2h
    elif model_type == FoundationModelEnum.H_OPTIMUS_0:
        return __extract_features_h_optimus_0
    elif model_type in [FoundationModelEnum.VIRCHOW, FoundationModelEnum.VIRCHOW_V2]:
        return __extract_features_virchow_default
    elif model_type in [
        FoundationModelEnum.HIBOU_B,
        FoundationModelEnum.HIBOU_L,
        FoundationModelEnum.PHIKON,
        FoundationModelEnum.PHIKON_V2,
        FoundationModelEnum.DINO_V1_S8,
        FoundationModelEnum.DINO_V1_B8,
        FoundationModelEnum.DINO_V1_S16,
        FoundationModelEnum.DINO_V1_B16,
        FoundationModelEnum.DINO_V2_S,
        FoundationModelEnum.DINO_V2_B,
        FoundationModelEnum.DINO_V2_L,
        FoundationModelEnum.DINO_V2_G,
        FoundationModelEnum.VIT_B16,
        FoundationModelEnum.VIT_L16,
    ]:
        return __extract_features_vit_default
    elif model_type in [
        FoundationModelEnum.RESNET_18,
        FoundationModelEnum.RESNET_34,
        FoundationModelEnum.RESNET_50,
    ]:
        return __extract_features_resnet_default
    else:
        raise NotImplementedError(f"Unknown model type: {model_type.value}")
