"""Image loading primitives, kept apart from `datasets` so `companion_selection` can reuse them."""

from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Generic dataset for image loading with configurable color mode and labels."""

    def __init__(
        self,
        image_paths: list[str],
        color_mode: str = "RGB",
        labels: list[list[int]] | list[int] | None = None,
        transform: Any = None,
    ) -> None:
        """Initialize the dataset with image paths, color mode, optional labels, and transform."""
        self.image_paths = image_paths
        self.color_mode = color_mode
        self.labels = labels if labels is not None else [[0] for _ in image_paths]
        self.transform = transform

        # Dummy data attribute, for compatibility
        if image_paths:
            sample_img = Image.open(image_paths[0]).convert(color_mode)
            sample_array = np.array(sample_img)
            self.data = np.zeros((len(image_paths), *sample_array.shape), dtype=sample_array.dtype)
        else:
            self.data = np.zeros((0, 28, 28) if color_mode == "L" else (0, 128, 128, 3), dtype=np.uint8)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, Any]:
        """Load and return image and label at given index; a tensor only once a transform is set."""
        image = Image.open(self.image_paths[idx]).convert(self.color_mode)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

    @property
    def targets(self) -> torch.Tensor:
        """Return targets as torch tensor. Base implementation returns zeros."""
        return torch.zeros(len(self.image_paths), dtype=torch.long)


def get_companion_transform(dataset_name: str) -> torchvision.transforms.Compose:
    """Build the transform a companion dataset's estimator was trained with, back to [-1, 1]."""
    if dataset_name in {"chest-xray", "chest_xray"}:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(128),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if dataset_name not in {"mnist", "fashion-mnist", "fashion_mnist"}:
        raise ValueError(f"No companion transform defined for dataset '{dataset_name}'")

    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
