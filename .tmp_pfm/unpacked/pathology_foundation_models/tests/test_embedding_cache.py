import os
import tempfile
import torch

from torchvision.datasets import ImageFolder
from PIL import Image

from pathology_foundation_models.dataset.cached_embedding import EmbeddingCache
from pathology_foundation_models.models.inference import extract_features
from pathology_foundation_models.models.fm_model import (
    FoundationModelEnum,
    load_foundation_model,
)
from pathology_foundation_models.tests.fixtures import hf_token, image_dataset


def test_make_embedding_cache(hf_token: str, image_dataset: ImageFolder):
    model = load_foundation_model(
        FoundationModelEnum.UNI, device="cuda", token=hf_token
    )
    cache = EmbeddingCache.init_from_image_dataset(image_dataset, model, batch_size=32)
    dataset_dir = image_dataset.root

    sample_img = Image.open(os.path.join(dataset_dir, "class1", "image_0.jpg"))
    sample_embedding = extract_features(sample_img, model)
    embed_dim = sample_embedding.shape[-1]

    # internal tensors from TensorDataset are ok
    assert cache.tensors[0].shape == (len(image_dataset), embed_dim)
    assert cache.tensors[1].shape == (len(image_dataset),)
    # custom attrs are ok
    assert torch.equal(cache.embeddings, cache.tensors[0])
    assert torch.equal(cache.labels, cache.tensors[1])
    expected_paths = [
        os.path.join(dataset_dir, "class1", f"image_{i}.jpg") for i in range(5)
    ] + [os.path.join(dataset_dir, "class2", f"image_{i}.jpg") for i in range(5)]
    assert cache.image_paths == expected_paths


def test_load_embedding_cache(hf_token: str, image_dataset: ImageFolder):
    model = load_foundation_model(
        FoundationModelEnum.UNI, device="cuda", token=hf_token
    )
    cache = EmbeddingCache.init_from_image_dataset(image_dataset, model, batch_size=32)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        cache.save(temp_file.name)
        temp_file_path = temp_file.name

    loaded_cache = EmbeddingCache.load_from_file(temp_file_path, device="cuda")
    os.remove(temp_file_path)
    assert isinstance(loaded_cache, EmbeddingCache)
    assert torch.equal(loaded_cache.embeddings, cache.embeddings)
    assert torch.equal(loaded_cache.labels, cache.labels)
    assert loaded_cache.image_paths == cache.image_paths
