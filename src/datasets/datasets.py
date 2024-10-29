"""Module for retrieving the datasets."""

from typing import Any

import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import Dataset

from src.models import DatasetParams


def get_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the MNIST dataset."""
    dataset = torchvision.datasets.MNIST(
        root=params.dataroot,
        download=True,
        train=params.train,
        transform=torchvision.transforms.Compose(
            [
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
        split = f"{split}[:10%]"

    ds = load_dataset("keremberke/chest-xray-classification", name="full", split=split)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
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
