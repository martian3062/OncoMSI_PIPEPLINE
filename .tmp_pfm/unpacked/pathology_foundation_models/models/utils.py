import torch
import torchvision.transforms as T

from PIL.Image import Image


def convert_to_batch_tensor(images: Image | list[Image] | torch.Tensor) -> torch.Tensor:
    """
    Converts a single PIL image or a torch tensor of shape (3, H, W) to a batch tensor of shape (1, 3, H, W).

    If the input is already a batch tensor of shape (N, 3, H, W), it will be returned as is.

    :param image: PIL Image or torch.Tensor of shape (3, H, W) or torch Tensor of shape (N, 3, H, W)
    :return: Batch tensor of shape (1, 3, H, W)
    """
    if isinstance(images, list):
        if not all(isinstance(img, Image) for img in images):
            raise TypeError("All items in the list must be PIL Images")
    elif not isinstance(images, Image) and not isinstance(images, torch.Tensor):
        raise TypeError(
            f"Input must be a PIL Image or a torch Tensor, got {type(images)}"
        )

    if isinstance(images, list):
        transform = T.ToTensor()
        image_list = [transform(img) for img in images]
        images = torch.stack(image_list, dim=0)
    if isinstance(images, Image):
        transform = T.ToTensor()
        images = transform(images).unsqueeze(0)  # Add batch dimension
    elif isinstance(images, torch.Tensor):
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        elif images.dim() != 4:
            raise ValueError(
                f"Input tensor must be of shape (3, H, W) or (N, 3, H, W). Got {images.shape}"
            )
    else:
        raise TypeError("Input must be a PIL Image or a torch Tensor")

    assert (
        images.dim() == 4
    ), "Unexpected return shape, expected (1, 3, H, W) or (N, 3, H, W)"
    return images
