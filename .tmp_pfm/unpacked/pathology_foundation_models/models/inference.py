import torch
import torchvision.transforms as T

from PIL.Image import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from pathology_foundation_models.models import FoundationModel


def extract_features_from_dataset(
    images: ImageFolder,
    model: FoundationModel,
    batch_size: int,
    num_workers: int = 4,
    display_progress: bool = False,
) -> torch.Tensor:
    """
    Extracts features from a collection images using the specified model.
    Performs batching to handle large datasets efficiently.

    **Note: images must be of the same size, since images are batched before each inference pass**

    :param images: ImageFolder dataset containing images
    :param model: FoundationModel instance containing the model and processor
    :param batch_size: Batch size for processing images
    :param num_workers: Number of workers for DataLoader
    :param display_progress: Whether to display a progress bar
    :return: Extracted features as a torch.Tensor of shape (N, D) where
        N is the number of images and D is the embedding dimension
    """
    dataloader = torch.utils.data.DataLoader(
        images, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    _embeddings = []
    if display_progress:
        dataloader = tqdm(dataloader, desc="Extracting features")
    for batch, _ in dataloader:  # ignore labels
        embedding_batch = model(batch)
        _embeddings.append(embedding_batch)
    return torch.cat(_embeddings, dim=0)
