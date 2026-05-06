"""
Implements model-specific loading functions for foundation models.
"""

import timm
import torch

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch import nn
from torchvision import transforms as T
from transformers import AutoModel, AutoImageProcessor


def __default_hf_loader(model_name: str) -> tuple[nn.Module, nn.Module]:
    """
    Various models use the default HF loading code,
    like phikon-v2 and the DINO family

    Loads model from Hugging Face to CPU

    DON'T use this function directly

    :param model_name: HF model path
    :return: model, transform
    """

    def __loader_fn():
        transform = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return model, transform

    return __loader_fn


def __load_uni() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/MahmoodLab/UNI

    Loads the UNI model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval()
    return model, transform


def __load_uni2h() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/MahmoodLab/UNI2-h

    Loads the UNI2-h model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval()
    return model, transform


def __load_h_optimus_0() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/bioptimus/H-optimus-0

    Loads the H-Optimus-0 model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model.eval()
    return model, transform


def __load_hibou_b() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/histai/hibou-b

    Loads the Hibou-B model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    transform = AutoImageProcessor.from_pretrained(
        "histai/hibou-b", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
    model.eval()
    return model, transform


def __load_hibou_L() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/histai/hibou-L

    Loads the Hibou-L model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    transform = AutoImageProcessor.from_pretrained(
        "histai/hibou-L", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model.eval()
    return model, transform


def __load_virchow() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/paige-ai/Virchow

    Loads the Virchow model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    # need to specify MLP layer and activation function for proper init
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval()
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform


def __load_virchow_v2() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/paige-ai/Virchow2

    Loads the Virchow2 model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    # same as Virchow v1, just different model path
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval()

    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform
