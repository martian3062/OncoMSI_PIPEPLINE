import torch

from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset

from pathology_foundation_models.models import (
    FoundationModel,
    extract_features_from_dataset,
)


class EmbeddingCache(TensorDataset):
    """
    A dataset that stores embeddings and labels for images.
    Inherits from TensorDataset to allow easy access to embeddings and labels.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        image_paths: list[str] | None = None,
        device: str = None,
    ):
        """
        Initializes the EmbeddingCache with embeddings and labels.

        :param embeddings: Tensor of shape (N, D) where N is the number of images and D is the embedding dimension
        :param labels: Optional tensor of shape (N,) containing labels for the images
        :param image_paths: Optional list of image paths corresponding to the embeddings
        :param device: Device to move the tensors to (e.g., "cuda" or "cpu"). If None, tensors will not be moved.
        """
        labels = (
            labels
            if labels is not None
            else torch.tensor([], dtype=torch.long, device=embeddings.device)
        )  # TODO review this, might be bad idea
        if device:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
        super().__init__(embeddings, labels)
        self.embeddings = embeddings
        self.labels = labels
        self.image_paths = image_paths

    @staticmethod
    def init_from_image_dataset(
        image_dataset: ImageFolder,
        model: FoundationModel,
        batch_size: int,
        num_workers: int = 4,
        display_progress: bool = True,
    ) -> "EmbeddingCache":
        """
        Initializes the EmbeddingCache with embeddings and labels.

        :param image_dataset: ImageFolder dataset containing images
        :param model: FoundationModel instance containing the model and processor
        :param batch_size: Batch size for processing images
        :param num_workers: Number of workers for data loading
        :param display_progress: Whether to display progress bar during feature extraction
        """
        embeddings = extract_features_from_dataset(
            image_dataset,
            model,
            batch_size,
            num_workers=num_workers,
            display_progress=display_progress,
        )  # is on model device
        image_paths = [img_path for img_path, _ in image_dataset.imgs]
        if hasattr(image_dataset, "targets"):
            return EmbeddingCache(
                embeddings,
                torch.tensor(image_dataset.targets).to(model.device),
                image_paths,
            )
        else:
            return EmbeddingCache(embeddings, image_paths=image_paths)

    def __getitem__(self, index):
        """
        Returns the image path, embedding and label for the given index.

        :param index: Index of the item to retrieve
        :return: Tuple of (image_path, embedding, label)
        """
        embedding, label = super().__getitem__(index)
        image_path = self.image_paths[index]
        return image_path, embedding, label

    def save(self, cache_path: str) -> None:
        """
        Saves the embedding cache to a file.

        :param embedding_cache: Dictionary mapping image paths to their corresponding embeddings
        :param cache_path: Path to save the cache file
        """
        torch.save(
            {
                "embeddings": self.embeddings,
                "labels": self.labels,
                "image_paths": self.image_paths,
            },
            cache_path,
        )

    @staticmethod
    def load_from_file(cache_path: str, device: str = "cpu") -> "EmbeddingCache":
        """
        Loads the embedding cache from a file.

        :param cache_path: Path to the cache file
        :return: EmbeddingCache instance
        """
        dataset_dict = torch.load(cache_path)
        return EmbeddingCache(
            embeddings=dataset_dict["embeddings"],
            labels=dataset_dict.get("labels", None),
            image_paths=dataset_dict.get("image_paths", None),
            device=device,
        )
