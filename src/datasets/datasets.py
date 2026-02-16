"""Module for retrieving the datasets."""

import logging
import os
from typing import Any

import ddu_dirty_mnist
import numpy as np
import torch
import torchvision
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from src.models import DatasetParams

logger = logging.getLogger(__name__)


class EmpiricalNormalizeToRange(torch.nn.Module):
    """
    Custom transform that applies z-score normalization then rescales to [-1, 1] range.

    This handles empirical normalization for datasets with limited dynamic range
    (e.g., ambiguess-mnist) while maintaining compatibility with FID which expects [-1, 1] input.
    """

    def __init__(self, mean: float, std: float, clamp_range: float = 3.0) -> None:
        """
        Initialize with empirical mean and std.

        Args:
            mean: Empirical mean for z-score normalization
            std: Empirical std for z-score normalization
            clamp_range: Values are clamped to [-clamp_range, clamp_range] before rescaling

        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.clamp_range = clamp_range

        # Pre-compute rescaling parameters based on [0, 1] input range
        # This avoids recalculating on every forward pass
        min_normalized = (0.0 - self.mean) / self.std
        max_normalized = (1.0 - self.mean) / self.std
        min_clamped = max(-self.clamp_range, min(min_normalized, self.clamp_range))
        max_clamped = max(-self.clamp_range, min(max_normalized, self.clamp_range))

        self.mid_point = (min_clamped + max_clamped) / 2.0
        self.scale_factor = 2.0 / (max_clamped - min_clamped)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization and rescale to [-1, 1].

        Args:
            tensor: Input tensor in [0, 1] range

        Returns:
            Normalized tensor in [-1, 1] range

        """
        # Apply z-score normalization: (x - mean) / std
        normalized = (tensor - self.mean) / self.std

        # Clamp to [-clamp_range, clamp_range] to handle outliers
        clamped = torch.clamp(normalized, -self.clamp_range, self.clamp_range)

        # Rescale from [min_clamped, max_clamped] to [-1, 1] using pre-computed parameters
        rescaled = (clamped - self.mid_point) * self.scale_factor

        return rescaled


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
    del params  # Not used: ambiguous datasets don't use dataroot or train flags

    # Get FILESDIR from environment, default to current directory
    filesdir = os.environ.get("FILESDIR", ".")
    data_dir = os.path.join(filesdir, "data")

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    ambiguous_mnist_test = ddu_dirty_mnist.AmbiguousMNIST(data_dir, train=False, download=True, device="cpu")

    # Wrapper class to handle deduplication (select one every 10 samples)
    class _AmbiguousMNISTDataset(Dataset):
        """
        Wrapper for AmbiguousMNIST that handles deduplication.

        Rescales z-score normalized data to [-1, 1] range (matching standard PyTorch
        normalization for MNIST and other datasets). This ensures compatibility with
        feature extractors and evaluation metrics that expect [-1, 1] normalized images.
        """

        def __init__(self, dataset: Dataset, step: int = 10) -> None:
            """Initialize with the dataset and sampling step."""
            self.dataset = dataset
            self.step = step
            # Create indices for every 10th sample
            self.indices = list(range(0, len(dataset), step))

            # Empirical stats from ddu_dirty_mnist z-normalized data
            # These values are observed from actual data samples
            self.data_min = -0.57
            self.data_max = 2.61

        def _normalize_to_minus1_1(self, tensor: torch.Tensor) -> torch.Tensor:
            """
            Normalize z-score normalized tensor to [-1, 1] range using linear scaling.

            Maps the empirical data range to [-1, 1] to match standard PyTorch
            normalization (Normalize((0.5,), (0.5,))) used for MNIST and other datasets.
            This ensures features extracted from ambiguous-mnist are comparable to
            those from standard MNIST.
            """
            # Linear rescaling from [data_min, data_max] to [0, 1]
            normalized_01 = (tensor - self.data_min) / (self.data_max - self.data_min)
            # Clip to [0, 1] in case of outliers
            normalized_01 = torch.clamp(normalized_01, 0.0, 1.0)
            # Convert from [0, 1] to [-1, 1]: 2*x - 1
            return 2.0 * normalized_01 - 1.0

        def __len__(self) -> int:
            """Return the number of deduplicated samples."""
            return len(self.indices)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            """Retrieve the image and label for the given index."""
            actual_idx = self.indices[idx]
            image, label = self.dataset[actual_idx]
            # Normalize image to [-1, 1] to match standard MNIST normalization
            return self._normalize_to_minus1_1(image), label

        @property
        def data(self) -> torch.Tensor:
            """Return all images in the dataset as a tensor stack."""
            images = []
            for idx in self.indices:
                image, _ = self.dataset[idx]
                if isinstance(image, torch.Tensor):
                    # Normalize to [-1, 1]
                    images.append(self._normalize_to_minus1_1(image))
                else:
                    # Convert to tensor if not already
                    tensor = torch.from_numpy(np.array(image))
                    images.append(self._normalize_to_minus1_1(tensor))
            return torch.stack(images)

        @property
        def targets(self) -> torch.Tensor:
            """Return all labels in the dataset as a tensor."""
            labels = []
            for idx in self.indices:
                _, label = self.dataset[idx]
                labels.append(label)
            return torch.tensor(labels)

    return _AmbiguousMNISTDataset(ambiguous_mnist_test, step=10)


class _AmbiguousHFDataset(Dataset):
    """Generic wrapper for ambiguous datasets from HuggingFace."""

    def __init__(self, hf_dataset: Dataset, transform: Any = None) -> None:
        """Initialize with HuggingFace dataset and transform."""
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieve the image and label for the given index."""
        sample = self.hf_dataset[idx]
        image = sample["image"]

        # Ensure image is a PIL Image and convert to RGB (same as CompanionDataset)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.new("RGB", (28, 28))
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = sample["label"]
        return image, label

    @property
    def data(self) -> torch.Tensor:
        """Return all images in the dataset as a tensor stack."""
        images = []
        for sample in self.hf_dataset:
            image = sample["image"]
            # Ensure image is a PIL Image and convert to RGB
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.new("RGB", (28, 28))
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    @property
    def targets(self) -> torch.Tensor:
        """Return all labels in the dataset as a tensor."""
        return torch.tensor([sample["label"] for sample in self.hf_dataset])


def _load_ambiguous_hf_dataset(
    hf_dataset_id: str,
    transform: torchvision.transforms.Compose,
    pytesting: bool = False,
) -> Dataset:
    """
    Load an ambiguous dataset from HuggingFace Hub.

    Args:
        hf_dataset_id: HuggingFace dataset ID (e.g., 'mweiss/mnist_ambiguous')
        transform: Transform to apply to images
        pytesting: If True, load only a subset for testing purposes

    Returns:
        Dataset wrapper with the images and labels

    """
    # Always use test split for ambiguous datasets
    # Use trust_remote_code=True for script-based datasets
    split = "test"
    if pytesting:
        split = "test[:10%]"  # Load only 10% for testing

    ds = load_dataset(hf_dataset_id, split=split, trust_remote_code=True)
    return _AmbiguousHFDataset(ds, transform=transform)


def get_ambiguess_mnist(params: DatasetParams) -> Dataset:
    """
    Retrieve the Ambiguess MNIST dataset from HuggingFace Hub.

    Uses empirically calculated mean and std (0.1214, 0.2219) to account for
    the lower contrast and dynamic range of the ambiguous dataset compared to
    standard MNIST (mean=0.1255, std=0.3030).

    Empirical normalization is rescaled to [-1, 1] for FID compatibility.
    """
    # Empirical mean and std for ambiguess-mnist (calculated from test set on [0, 1] range)
    # These account for the inherently lower contrast of ambiguous images
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            EmpiricalNormalizeToRange(mean=0.121400, std=0.221853, clamp_range=3.0),
        ]
    )
    return _load_ambiguous_hf_dataset("mweiss/mnist_ambiguous", transform, params.pytesting)


def get_ambiguess_fmnist(params: DatasetParams) -> Dataset:
    """
    Retrieve the Ambiguess FMNIST dataset from HuggingFace Hub.

    Uses empirically calculated mean and std (0.2241, 0.2910) to account for
    the specific characteristics of the ambiguous fashion-MNIST dataset.

    Empirical normalization is rescaled to [-1, 1] for FID compatibility.
    """
    # Empirical mean and std for ambiguess-fmnist (calculated from test set on [0, 1] range)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            EmpiricalNormalizeToRange(mean=0.224149, std=0.291023, clamp_range=3.0),
        ]
    )
    return _load_ambiguous_hf_dataset("mweiss/fashion_mnist_ambiguous", transform, params.pytesting)


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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load and return the image at the given index.

        Args:
            idx: Index of the image to load.

        Returns:
            Tuple of (image tensor, dummy label).

        """
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0

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


def _find_companion_dataset_images(dataroot: str, dataset_name: str, n_samples_per_subset: int = 200) -> list[str]:
    """
    Find and collect companion dataset images from all GAN subsets of a dataset.

    Args:
        dataroot: Root directory containing the dataset and AmbiGAN outputs.
        dataset_name: Name of the dataset (e.g., 'mnist', 'fashion_mnist', 'chest_xray').
        n_samples_per_subset: Number of random images to select per class subset (default: 200).

    Returns:
        List of paths to randomly selected companion dataset images (200 per subset).

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

            logger.info(f"Found {len(image_files)} images in {companion_dir}")

            # Randomly select n_samples_per_subset images from this subset
            if len(image_files) < n_samples_per_subset:
                logger.warning(
                    f"Requested {n_samples_per_subset} samples from {entry} but only {len(image_files)} available. Using all available."
                )
                selected_from_subset = image_files
            else:
                selected_from_subset = list(np.random.choice(image_files, size=n_samples_per_subset, replace=False))

            all_images.extend(selected_from_subset)

        except (OSError, ValueError) as e:
            logger.warning(f"Error processing subset {entry}: {e}")
            continue

    if not all_images:
        raise ValueError(f"No companion dataset images found for dataset '{dataset_name}' in {gan_root}")

    logger.info(f"Total companion images collected: {len(all_images)}")

    return all_images


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
