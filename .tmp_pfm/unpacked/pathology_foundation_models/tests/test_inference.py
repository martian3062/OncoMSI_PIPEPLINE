import PIL
import torch
import pytest

from pathology_foundation_models.models.inference import extract_features_from_dataset

# NOTE can't import enums twice with different import statements,
# see https://stackoverflow.com/questions/40371360/imported-enum-class-is-not-comparing-equal-to-itself
# so we import from models package
from pathology_foundation_models.models import (
    FoundationModelEnum,
    load_foundation_model,
    get_embedding_dim,
)
from pathology_foundation_models.tests.fixtures import hf_token, image_dataset
from pathology_foundation_models.models.utils import convert_to_batch_tensor


@pytest.mark.parametrize(
    "model_type",
    [
        FoundationModelEnum.UNI,
        FoundationModelEnum.UNI2H,
        FoundationModelEnum.PHIKON,
        FoundationModelEnum.PHIKON_V2,
        FoundationModelEnum.H_OPTIMUS_0,
        FoundationModelEnum.HIBOU_B,
        FoundationModelEnum.HIBOU_L,
        FoundationModelEnum.VIRCHOW,
        FoundationModelEnum.VIRCHOW_V2,
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
    ],
)
def test_inference_models_from_PIL(model_type, hf_token):
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model(model_type, device="cuda", token=hf_token)
    features = model(image)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, get_embedding_dim(model_type))


def test_batch_inference(hf_token, image_dataset):
    # inference was already tested, any model is fine for this test
    model = load_foundation_model(
        FoundationModelEnum.UNI, device="cuda", token=hf_token
    )
    batch_tensor = extract_features_from_dataset(
        image_dataset, model, batch_size=3, num_workers=2, display_progress=True
    )

    assert isinstance(batch_tensor, torch.Tensor)
    assert batch_tensor.shape == (len(image_dataset), 1024)


def test_convert_to_batch_tensor():
    image = PIL.Image.new("RGB", (224, 224), color="blue")
    batch_tensor = convert_to_batch_tensor(image)
    assert batch_tensor.shape == (1, 3, 224, 224)

    images = [image, PIL.Image.new("RGB", (224, 224), color="green")]
    batch_tensor = convert_to_batch_tensor(images)
    assert batch_tensor.shape == (2, 3, 224, 224)

    tensor_image = torch.rand(3, 224, 224)
    batch_tensor = convert_to_batch_tensor(tensor_image)
    assert batch_tensor.shape == (1, 3, 224, 224)
