import os
import pytest
import tempfile
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms


@pytest.fixture
def hf_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")
    return token


@pytest.fixture
def image_dataset():
    # Create a temporary directory for the image dataset
    temp_dir = tempfile.TemporaryDirectory()
    # Create a simple image dataset with dummy images
    os.makedirs(os.path.join(temp_dir.name, "class1"))
    os.makedirs(os.path.join(temp_dir.name, "class2"))
    # Create dummy images
    for i in range(5):
        img = transforms.ToPILImage()(torch.rand(3, 224, 224))
        img.save(os.path.join(temp_dir.name, "class1", f"image_{i}.jpg"))
        img.save(os.path.join(temp_dir.name, "class2", f"image_{i}.jpg"))

    yield ImageFolder(temp_dir.name, transform=transforms.ToTensor())
    # teardown
    temp_dir.cleanup()
