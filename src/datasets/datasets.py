"""Module for retrieving the datasets."""

import logging
import os
from typing import Any

import numpy as np
import torch
import torchvision
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from src.models import DatasetParams

logger = logging.getLogger(__name__)


def get_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the MNIST dataset."""
    dataset = torchvision.datasets.MNIST(
        root=params.dataroot,
        download=True,
        train=params.train,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    return dataset


def get_fashion_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the FASHION-MNIST dataset."""
    dataset = torchvision.datasets.FashionMNIST(
        root=params.dataroot,
        download=True,
        train=params.train,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    return dataset


def get_cifar10(params: DatasetParams) -> Dataset:
    """Retrieve the CIFAR-10 dataset."""
    dataset = torchvision.datasets.CIFAR10(
        root=params.dataroot,
        download=True,
        train=params.train,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    return dataset


def get_chest_xray(params: DatasetParams) -> Dataset:
    """Retrieve the CHEST-XRAY dataset."""
    split = "train" if params.train else "test"

    # If the `pytesting` flag is set to True, download only 10% of the data
    if params.train and params.pytesting:
        split = "test[:30%]"

    ds = load_dataset("keremberke/chest-xray-classification", name="full", split=split)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    class ChestXrayDataset(Dataset):
        """Custom dataset class for handling the Chest X-ray dataset."""

        def __init__(self, hf_dataset: Dataset, transform: Any = None) -> None:
            """Initialize the Chest X-ray dataset with given HuggingFace dataset and transform."""
            self.hf_dataset = hf_dataset
            self.transform = transform

        def __len__(self) -> int:
            """Return the number of samples in the dataset."""
            return len(self.hf_dataset)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            """Retrieve the image and label for the given index."""
            sample = self.hf_dataset[idx]
            image = sample["image"]

            if self.transform:
                image = self.transform(image)

            label = sample["labels"]
            return image, label

        @property
        def data(self) -> torch.Tensor:
            """Return all images in the dataset as a tensor stack."""
            return torch.stack([self.transform(sample["image"]) for sample in self.hf_dataset])

        @property
        def targets(self) -> torch.Tensor:
            """Return all labels in the dataset as a tensor."""
            return torch.tensor([sample["labels"] for sample in self.hf_dataset])

    return ChestXrayDataset(ds, transform=transform)


def get_ambiguous_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the AmbiguousMNIST dataset."""
    # TODO
    raise NotImplementedError(
        "AmbiguousMNIST loading is not implemented yet. " f"Requested dataroot: {params.dataroot}"
    )


def get_ambiguess_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the Ambiguess MNIST dataset."""
    # TODO
    raise NotImplementedError(
        "Ambiguess MNIST loading is not implemented yet. " f"Requested dataroot: {params.dataroot}"
    )


def get_ambiguess_fmnist(params: DatasetParams) -> Dataset:
    """Retrieve the Ambiguess FMNIST dataset."""
    # TODO
    raise NotImplementedError(
        "Ambiguess FMNIST loading is not implemented yet. " f"Requested dataroot: {params.dataroot}"
    )


class CompanionDataset(Dataset):
    """Custom dataset class for handling companion datasets from multiple GAN subsets."""

    def __init__(self, image_paths: list[str], transform: Any = None) -> None:
        """
        Initialize the companion dataset with image paths and optional transform.

        Args:
            image_paths: List of full paths to image files.
            transform: Optional torchvision transform to apply to images.

        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return the image at the given index.

        Args:
            idx: Index of the image to load.

        Returns:
            Transformed image tensor.

        """
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    @property
    def data(self) -> torch.Tensor:
        """Return all images in the dataset as a tensor stack."""
        images = []
        for path in self.image_paths:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    @property
    def targets(self) -> torch.Tensor:
        """Return dummy targets as torch tensor with all zeros for compatibility."""
        return torch.zeros(len(self.image_paths), dtype=torch.long)


def _find_companion_dataset_images(dataroot: str, dataset_name: str, n_samples: int = 200) -> list[str]:
    """
    Find and collect companion dataset images from all GAN subsets of a dataset.

    Args:
        dataroot: Root directory containing the dataset and AmbiGAN outputs.
        dataset_name: Name of the dataset (e.g., 'mnist', 'fashion_mnist', 'chest_xray').
        n_samples: Number of random images to select (default: 200).

    Returns:
        List of paths to randomly selected companion dataset images.

    Raises:
        ValueError: If no companion datasets are found.

    """
    # Find the out_dir by looking for the AmbiGAN directory
    # The dataroot is typically: {out_dir}/data
    # The AmbiGAN root should be at: {out_dir}/AmbiGAN/{dataset_name}-*
    out_dir = os.path.dirname(dataroot)
    gan_root = os.path.join(out_dir, "AmbiGAN")

    if not os.path.exists(gan_root):
        raise ValueError(f"AmbiGAN root directory not found at {gan_root}")

    all_images = []

    # Find all subdirectories matching the pattern {dataset_name}-*
    for entry in os.listdir(gan_root):
        subset_dir = os.path.join(gan_root, entry)

        # Check if this is a subdirectory for the correct dataset
        if not os.path.isdir(subset_dir) or not entry.startswith(f"{dataset_name}-"):
            continue

        # Find the most recent run (subdirectory) within this subset
        try:
            run_dirs = [
                os.path.join(subset_dir, d)
                for d in os.listdir(subset_dir)
                if os.path.isdir(os.path.join(subset_dir, d))
            ]

            if not run_dirs:
                logger.warning(f"No run directories found in {subset_dir}")
                continue

            # Select the most recently modified run
            latest_run = max(run_dirs, key=os.path.getmtime)

            # Look for the companion dataset
            companion_dir = os.path.join(latest_run, "companion_dataset", "ambi")

            if not os.path.exists(companion_dir):
                logger.warning(f"Companion dataset not found for {entry} at {companion_dir}")
                continue

            # Collect all image files from this companion dataset
            image_files = sorted(
                [
                    os.path.join(companion_dir, f)
                    for f in os.listdir(companion_dir)
                    if f.endswith((".png", ".jpg", ".jpeg"))
                ]
            )

            all_images.extend(image_files)
            logger.info(f"Found {len(image_files)} images in {companion_dir}")

        except (OSError, ValueError) as e:
            logger.warning(f"Error processing subset {entry}: {e}")
            continue

    if not all_images:
        raise ValueError(f"No companion dataset images found for dataset '{dataset_name}' in {gan_root}")

    logger.info(f"Total companion images found: {len(all_images)}")

    # Randomly select n_samples images
    if len(all_images) < n_samples:
        logger.warning(f"Requested {n_samples} samples but only {len(all_images)} available. Using all available.")
        selected_images = all_images
    else:
        selected_images = list(np.random.choice(all_images, size=n_samples, replace=False))

    return selected_images


def get_companion_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the Companion MNIST dataset."""
    image_paths = _find_companion_dataset_images(params.dataroot, "mnist")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return CompanionDataset(image_paths, transform=transform)


def get_companion_fmnist(params: DatasetParams) -> Dataset:
    """Retrieve the Companion FMNIST dataset."""
    image_paths = _find_companion_dataset_images(params.dataroot, "fashion_mnist")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return CompanionDataset(image_paths, transform=transform)


def get_companion_chest_xray(params: DatasetParams) -> Dataset:
    """Retrieve the Companion Chest X-ray dataset."""
    image_paths = _find_companion_dataset_images(params.dataroot, "chest_xray")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return CompanionDataset(image_paths, transform=transform)
