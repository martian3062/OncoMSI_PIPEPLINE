"""
Implement model-specific inference functions for foundation models.
"""

import torch

from torch import nn


def __extract_features_vit_default(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    Default inference function for ViT-like models:
        extracts features from images using ViT-like model.
    Assumes image and model in the same device.

    Features are the class token (first output token)

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The UNI model
    :param transform: the preprocessing transform
    :return: Extracted features
    """
    # process the image
    inputs = transform(images, return_tensors="pt")
    # cast back to original device
    inputs = {k: v.to(images.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
    return features


def __extract_features_resnet_default(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    Default inference function for Resnet models
        extracts features from images using Resnet-like model.
    Assumes image and model in the same device.

    Key difference is that Resnet models apply final avg pooling layer
    over last_hidden_state

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The UNI model
    :param transform: the preprocessing transform
    :return: Extracted features
    """
    # process the image
    inputs = transform(images, return_tensors="pt")
    # cast back to original device
    inputs = {k: v.to(images.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.pooler_output  # (N, D, 1, 1)
    return features.reshape((len(images), -1))


def __extract_features_uni(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/MahmoodLab/UNI

    Extracts features from images using the UNI model.
    Assumes image and model in the same device.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The UNI model
    :param transform: the preprocessing transform
    :return: Extracted features
    """
    image_tensor = transform(images)
    with torch.inference_mode():
        features = model(image_tensor)
    return features


def __extract_features_uni2h(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/MahmoodLab/UNI2-h

    Extracts features from images using the UNI2-h model.
    Assumes image and model in the same device.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The UNI2-h model
    :param transform: the preprocessing transform
    :return: Extracted features
    """
    image_tensor = transform(images)
    with torch.inference_mode():
        features = model(image_tensor)
    return features


def __extract_features_h_optimus_0(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/bioptimus/H-optimus-0 (adapted to work with tensors)

    Extracts features from images using the H-Optimus-0 model.
    Assumes image and model in the same device.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The H-Optimus-0 model
    :param transform: the preprocessing transform
    :return: Extracted features
    """
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            # (C, H, W) -> (H, W, C)
            numpy_images = [image.permute(1, 2, 0).cpu().numpy() for image in images]
            preprocessed_images = torch.stack(
                [transform(image) for image in numpy_images]
            )  # (N, C, H, W)
            features = model(preprocessed_images.to(images.device))
    return features


def __extract_features_virchow_default(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/paige-ai/Virchow
    and https://huggingface.co/paige-ai/Virchow2

    Extracts features from images using models from the Virchow family.
    Assumes image and model in the same device.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The Virchow model
    :param transform: the preprocessing transform
    :return: Extracted features
    """
    images = transform(images)
    # model deck recommends running inference in mixed precision (autocast mode, f16)
    # so we do that
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        output = model(images)
    class_tokens = output[:, 0]  # (N, 1280)
    patch_tokens = output[:, 1:]  # (N, 256, 1280)
    # concatenate class token and average pool of patch tokens
    features = torch.cat([class_tokens, patch_tokens.mean(dim=1)], dim=-1)  # (N, 2560)
    return features
