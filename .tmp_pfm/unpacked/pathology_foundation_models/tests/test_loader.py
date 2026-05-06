from pathology_foundation_models.models import (
    FoundationModelEnum,
    load_foundation_model,
)
from pathology_foundation_models.tests.fixtures import hf_token


def test_load_no_device(hf_token):
    """
    Test loading a foundation model without specifying a device. Should load to CPU
    """
    model_type = FoundationModelEnum.UNI
    model = load_foundation_model(model_type, token=hf_token)
    assert model.model_type == model_type
    assert model.device == "cpu"
    assert all(param.device.type == "cpu" for param in model.model.parameters())


def test_load_cpu(hf_token):
    model_type = FoundationModelEnum.UNI
    model = load_foundation_model(model_type, device="cpu", token=hf_token)
    assert model.model_type == model_type
    assert model.device == "cpu"
    assert all(param.device.type == "cpu" for param in model.model.parameters())


def test_load_gpu(hf_token):
    model_type = FoundationModelEnum.UNI
    model = load_foundation_model(model_type, device="cuda", token=hf_token)
    assert model.model_type == model_type
    assert model.device == "cuda"
    assert all(param.device.type == "cuda" for param in model.model.parameters())
