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

    # Use same transform pipeline as standard MNIST
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load with deduplication wrapper
    ambiguous_mnist_test = ddu_dirty_mnist.AmbiguousMNIST(
        data_dir,
        train=False,
        download=True,
        device="cuda",
        noise_stddev=0.0,
        normalize=False,
        transform=transform,
    )

    # Wrapper class to handle deduplication (select one every 10 samples)
    class _AmbiguousMNISTDataset(Dataset):
        """
        Wrapper for AmbiguousMNIST that handles deduplication.

        Selects every 10th sample to reduce dataset size while maintaining diversity.
        """

        def __init__(self, dataset: Dataset, step: int = 10) -> None:
            """Initialize with the dataset and sampling step."""
            self.dataset = dataset
            self.step = step
            # Create indices for every 10th sample
            self.indices = list(range(0, len(dataset), step))

        def __len__(self) -> int:
            """Return the number of deduplicated samples."""
            return len(self.indices)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            """Retrieve the image and label for the given index."""
            actual_idx = self.indices[idx]
            image, label = self.dataset[actual_idx]
            return image, label

        @property
        def data(self) -> torch.Tensor:
            """Return all images in the dataset as a tensor stack."""
            images = []
            for idx in self.indices:
                image, _ = self.dataset[idx]
                images.append(image)
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


def _extract_ground_truth_from_p_label(p_label: list[float] | np.ndarray) -> list[int]:
    """
    Extract ground truth class labels from probability distribution.

    Args:
        p_label: List or array of probabilities (e.g., [0.5, 0, 0, 0, 0.5, 0]).

    Returns:
        List of class indices with non-zero probability (e.g., [0, 4]).

    """
    try:
        return [int(i) for i, prob in enumerate(p_label) if prob > 0]
    except (TypeError, ValueError) as e:
        logger.warning(f"Could not extract ground truth from p_label: {e}")
        return []


class _AmbiguousHFDataset(Dataset):
    """Generic wrapper for ambiguous datasets from HuggingFace."""

    def __init__(self, hf_dataset: Dataset, transform: Any = None) -> None:
        """Initialize with HuggingFace dataset and transform."""
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        """Retrieve the image and ground truth labels for the given index."""
        sample = self.hf_dataset[idx]
        image = sample["image"]

        # Ensure image is a PIL Image and convert to RGB (same as CompanionDataset)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.new("RGB", (28, 28))
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract ground truth from p_label (indices with non-zero probability)
        ground_truth = _extract_ground_truth_from_p_label(sample.get("p_label", []))
        return image, ground_truth

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
    def targets(self) -> list[list[int]]:
        """Return all ground truth labels in the dataset as a list of lists."""
        return [_extract_ground_truth_from_p_label(sample.get("p_label", [])) for sample in self.hf_dataset]


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

    Uses standard MNIST normalization for consistency with MNIST dataset.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return _load_ambiguous_hf_dataset("mweiss/mnist_ambiguous", transform, params.pytesting)


def get_ambiguess_fmnist(params: DatasetParams) -> Dataset:
    """
    Retrieve the Ambiguess FMNIST dataset from HuggingFace Hub.

    Uses standard FashionMNIST normalization for consistency with FashionMNIST dataset.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return _load_ambiguous_hf_dataset("mweiss/fashion_mnist_ambiguous", transform, params.pytesting)


class CompanionDataset(Dataset):
    """Custom dataset class for handling companion datasets from multiple GAN subsets."""

    def __init__(self, image_paths: list[str], labels: list[list[int]] | None = None, transform: Any = None) -> None:
        """
        Initialize the companion dataset with image paths, optional labels, and optional transform.

        Args:
            image_paths: List of full paths to image files.
            labels: List of ground truth labels corresponding to each image.
            transform: Optional torchvision transform to apply to images.

        """
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [[0] for _ in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        """
        Load and return the image at the given index.

        Args:
            idx: Index of the image to load.

        Returns:
            Tuple of (image tensor, ground truth label list).

        """
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

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
        """Return targets as torch tensor with ground truth labels."""
        return torch.tensor(self.labels, dtype=torch.long)


def _parse_ground_truth_from_dirname(dirname: str, dataset_name: str) -> list[int]:
    """
    Extract ground truth class labels from directory name.

    Args:
        dirname: Directory name (e.g., 'mnist-4v9' or 'mnist.4v9').
        dataset_name: Name of the dataset (e.g., 'mnist').

    Returns:
        List of class labels (e.g., [4, 9]).

    """
    # Remove dataset prefix (e.g., 'mnist-' or 'mnist.')
    prefix_dash = f"{dataset_name}-"
    prefix_dot = f"{dataset_name}."

    if dirname.startswith(prefix_dash):
        class_str = dirname[len(prefix_dash) :]
    elif dirname.startswith(prefix_dot):
        class_str = dirname[len(prefix_dot) :]
    else:
        logger.warning(f"Could not parse ground truth from directory: {dirname}")
        return []

    # Split by 'v' to get individual classes (e.g., '4v9' -> ['4', '9'])
    try:
        classes = [int(c) for c in class_str.split("v")]
        return classes
    except ValueError:
        logger.warning(f"Could not parse class labels from: {class_str}")
        return []


def _find_companion_dataset_images(  # pylint: disable=too-many-branches,too-many-statements  # noqa: C901
    dataroot: str, dataset_name: str, n_samples_per_subset: int = 200
) -> tuple[list[str], list[list[int]]]:
    """
    Find and collect companion dataset images from all GAN subsets of a dataset.

    Args:
        dataroot: Root directory containing the dataset and AmbiGAN outputs.
        dataset_name: Name of the dataset (e.g., 'mnist', 'fashion_mnist', 'chest_xray').
        n_samples_per_subset: Number of random images to select per class subset (default: 200).

    Returns:
        Tuple of (list of image paths, list of ground truth labels for each image).
        Where each element in the labels list corresponds to the classes in the companion subset.

    Raises:
        ValueError: If no companion datasets are found.

    """
    # Convert underscores to hyphens for directory names
    dataset_dir_name = dataset_name.replace("_", "-")

    # Find the out_dir by looking for the AmbiGAN directory
    # The dataroot is typically: {out_dir}/data
    # The AmbiGAN root should be at: {out_dir}/AmbiGAN/{dataset_name}-*
    out_dir = os.path.dirname(dataroot)
    gan_root = os.path.join(out_dir, "AmbiGAN")

    if not os.path.exists(gan_root):
        raise ValueError(f"AmbiGAN root directory not found at {gan_root}")

    all_images = []
    all_labels = []

    # Special case for inherently binary datasets (chest-xray) without -1v0 suffix
    if dataset_name == "chest_xray":
        subset_dir = os.path.join(gan_root, dataset_dir_name)
        if os.path.isdir(subset_dir):
            # Find the most recent run
            run_dirs = [
                os.path.join(subset_dir, d)
                for d in os.listdir(subset_dir)
                if os.path.isdir(os.path.join(subset_dir, d))
            ]
            if run_dirs:
                latest_run = max(run_dirs, key=os.path.getmtime)
                companion_dir = os.path.join(latest_run, "companion_dataset", "ambi")
                if os.path.exists(companion_dir):
                    image_files = sorted(
                        [
                            os.path.join(companion_dir, f)
                            for f in os.listdir(companion_dir)
                            if f.endswith((".png", ".jpg", ".jpeg"))
                        ]
                    )
                    if image_files:
                        all_images.extend(image_files)
                        # For chest-xray binary: ground truth is [1, 0]
                        all_labels.extend([[1, 0]] * len(image_files))
                        logger.info(f"Found {len(image_files)} companion images for chest-xray")

        if all_images:
            return all_images, all_labels

    # Find all subdirectories matching the pattern {dataset_dir_name}-*
    for entry in os.listdir(gan_root):
        subset_dir = os.path.join(gan_root, entry)

        # Check if this is a subdirectory for the correct dataset
        if not os.path.isdir(subset_dir) or not entry.startswith(f"{dataset_dir_name}-"):
            continue

        # Parse ground truth from directory name
        ground_truth = _parse_ground_truth_from_dirname(entry, dataset_name)
        if not ground_truth:
            logger.warning(f"Skipping {entry}: could not parse ground truth labels")
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

            # Add images and their corresponding ground truth labels
            all_images.extend(selected_from_subset)
            all_labels.extend([ground_truth] * len(selected_from_subset))

        except (OSError, ValueError) as e:
            logger.warning(f"Error processing subset {entry}: {e}")
            continue

    if not all_images:
        raise ValueError(f"No companion dataset images found for dataset '{dataset_name}' in {gan_root}")

    logger.info(f"Total companion images collected: {len(all_images)}")

    return all_images, all_labels


def get_companion_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the Companion MNIST dataset with ground truth labels."""
    image_paths, labels = _find_companion_dataset_images(params.dataroot, "mnist")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return CompanionDataset(image_paths, labels=labels, transform=transform)


def get_companion_fmnist(params: DatasetParams) -> Dataset:
    """Retrieve the Companion FMNIST dataset with ground truth labels."""
    image_paths, labels = _find_companion_dataset_images(params.dataroot, "fashion_mnist")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return CompanionDataset(image_paths, labels=labels, transform=transform)


def get_companion_chest_xray(params: DatasetParams) -> Dataset:
    """Retrieve the Companion Chest X-ray dataset with ground truth labels."""
    image_paths, labels = _find_companion_dataset_images(params.dataroot, "chest_xray")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return CompanionDataset(image_paths, labels=labels, transform=transform)


def _find_synthetic_dataset_images(dataroot: str, dataset_name: str) -> list[str]:
    """
    Find and collect synthetic dataset images from the latest GAN run.

    Args:
        dataroot: Root directory containing the dataset and AmbiGAN outputs.
        dataset_name: Name of the dataset (e.g., 'mnist', 'fashion_mnist', 'chest_xray').

    Returns:
        List of paths to synthetic dataset images.

    Raises:
        ValueError: If no synthetic dataset is found.

    """
    # Convert underscores to hyphens for directory names
    dataset_dir_name = dataset_name.replace("_", "-")

    # Find the out_dir by looking for the AmbiGAN directory
    out_dir = os.path.dirname(dataroot)
    gan_root = os.path.join(out_dir, "AmbiGAN")

    if not os.path.exists(gan_root):
        raise ValueError(f"AmbiGAN root directory not found at {gan_root}")

    # Find the directory matching the dataset
    subset_dir = os.path.join(gan_root, dataset_dir_name)

    if not os.path.isdir(subset_dir):
        raise ValueError(f"Dataset directory not found at {subset_dir}")

    # Find the most recent run
    run_dirs = [
        os.path.join(subset_dir, d) for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))
    ]

    if not run_dirs:
        raise ValueError(f"No run directories found in {subset_dir}")

    # Select the most recently modified run
    latest_run = max(run_dirs, key=os.path.getmtime)

    # Look for the synthetic dataset in the latest run
    synthetic_dir = os.path.join(latest_run, "synthetic")

    if not os.path.exists(synthetic_dir):
        raise ValueError(f"Synthetic dataset not found at {synthetic_dir}")

    # Collect all image files
    image_files = sorted(
        [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    )

    if not image_files:
        raise ValueError(f"No images found in synthetic dataset directory {synthetic_dir}")

    logger.info(f"Found {len(image_files)} synthetic images in {synthetic_dir}")

    return image_files


def get_synthetic_mnist(params: DatasetParams) -> Dataset:
    """Retrieve the Synthetic MNIST dataset."""
    image_paths = _find_synthetic_dataset_images(params.dataroot, "mnist")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    class SyntheticDataset(Dataset):
        """Simple dataset for synthetic images without labels."""

        def __init__(self, image_paths: list[str], transform: Any = None) -> None:
            """Initialize the dataset."""
            self.image_paths = image_paths
            self.transform = transform
            # Create dummy data attribute for compatibility
            self.data = np.array([np.array(Image.open(p)) for p in image_paths])
            self.targets = np.zeros(len(image_paths))

        def __len__(self) -> int:
            """Return the number of samples."""
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            """Retrieve image and dummy label."""
            image = Image.open(self.image_paths[idx]).convert("L")
            if self.transform:
                image = self.transform(image)
            return image, 0

    return SyntheticDataset(image_paths, transform=transform)


def get_synthetic_fmnist(params: DatasetParams) -> Dataset:
    """Retrieve the Synthetic Fashion-MNIST dataset."""
    image_paths = _find_synthetic_dataset_images(params.dataroot, "fashion_mnist")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    class SyntheticDataset(Dataset):
        """Simple dataset for synthetic images without labels."""

        def __init__(self, image_paths: list[str], transform: Any = None) -> None:
            """Initialize the dataset."""
            self.image_paths = image_paths
            self.transform = transform
            # Create dummy data attribute for compatibility
            self.data = np.array([np.array(Image.open(p)) for p in image_paths])
            self.targets = np.zeros(len(image_paths))

        def __len__(self) -> int:
            """Return the number of samples."""
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            """Retrieve image and dummy label."""
            image = Image.open(self.image_paths[idx]).convert("L")
            if self.transform:
                image = self.transform(image)
            return image, 0

    return SyntheticDataset(image_paths, transform=transform)


def get_synthetic_chest_xray(params: DatasetParams) -> Dataset:
    """Retrieve the Synthetic Chest X-ray dataset."""
    image_paths = _find_synthetic_dataset_images(params.dataroot, "chest_xray")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    class SyntheticDataset(Dataset):
        """Simple dataset for synthetic images without labels."""

        def __init__(self, image_paths: list[str], transform: Any = None) -> None:
            """Initialize the dataset."""
            self.image_paths = image_paths
            self.transform = transform
            # Create dummy data attribute for compatibility
            self.data = np.array([np.array(Image.open(p)) for p in image_paths])
            self.targets = np.zeros(len(image_paths))

        def __len__(self) -> int:
            """Return the number of samples."""
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            """Retrieve image and dummy label."""
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, 0

    return SyntheticDataset(image_paths, transform=transform)
