"""Module for loading the datasets."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.datasets import (
    CompanionDataset,
    get_ambiguess_fmnist,
    get_ambiguess_mnist,
    get_ambiguous_mnist,
    get_chest_xray,
    get_cifar10,
    get_companion_ambiguous_chest_xray,
    get_companion_ambiguous_fmnist,
    get_companion_ambiguous_mnist,
    get_companion_chest_xray,
    get_companion_fmnist,
    get_companion_mnist,
    get_fashion_mnist,
    get_mnist,
    get_synthetic_chest_xray,
    get_synthetic_fmnist,
    get_synthetic_mnist,
)
from src.datasets.utils import BinaryDataset
from src.enums import DatasetNames
from src.models import DatasetParams, ImageParams, LoadDatasetParams

logger = logging.getLogger(__name__)


def get_function(dataset_name: DatasetNames) -> Callable[[DatasetParams], Any]:
    """Retrieve the function to load the dataset."""
    mapping: dict[DatasetNames, Callable[[DatasetParams], Any]] = {
        DatasetNames.mnist: get_mnist,
        DatasetNames.fashion_mnist: get_fashion_mnist,
        DatasetNames.cifar10: get_cifar10,
        DatasetNames.chest_xray: get_chest_xray,
        DatasetNames.ambiguous_mnist: get_ambiguous_mnist,
        DatasetNames.ambiguess_mnist: get_ambiguess_mnist,
        DatasetNames.ambiguess_fmnist: get_ambiguess_fmnist,
        DatasetNames.companion_mnist: get_companion_mnist,
        DatasetNames.companion_fmnist: get_companion_fmnist,
        DatasetNames.companion_chest_xray: get_companion_chest_xray,
        DatasetNames.companion_ambiguous_mnist: get_companion_ambiguous_mnist,
        DatasetNames.companion_ambiguous_fmnist: get_companion_ambiguous_fmnist,
        DatasetNames.companion_ambiguous_chest_xray: get_companion_ambiguous_chest_xray,
        DatasetNames.synthetic_mnist: get_synthetic_mnist,
        DatasetNames.synthetic_fmnist: get_synthetic_fmnist,
        DatasetNames.synthetic_chest_xray: get_synthetic_chest_xray,
    }
    return mapping[dataset_name]


def load_dataset(params: LoadDatasetParams) -> tuple[Dataset, int, ImageParams]:
    """
    Load dataset, optionally modify it for binary classification.
    Return  with class and image size information.
    """
    try:
        DatasetNames.valid_dataset(params.dataset_name)
        logger.info(f"{params.dataset_name.value} is a valid dataset.")
    except ValueError as e:
        logger.error(e)
        raise e

    download_function = get_function(params.dataset_name)
    download_params = DatasetParams(dataroot=params.dataroot, split=params.split, pytesting=params.pytesting)
    dataset = download_function(download_params)

    # Check if the dataset is empty and log an error
    if len(dataset) == 0:
        error = f"The dataset '{params.dataset_name}' is empty. Please verify the data availability."
        logger.error(error)
        raise ValueError(error)

    image_size = tuple(dataset.data.shape[1:])
    if len(image_size) == 2:
        image_size = (1, *image_size)

    elif len(image_size) == 3 and image_size[2] == 3:
        image_size = (image_size[2], image_size[0], image_size[1])

    targets = dataset.targets if torch.is_tensor(dataset.targets) else torch.tensor(dataset.targets)
    num_classes = targets.unique().size(0)

    if params.pos_class is not None and params.neg_class is not None:
        num_classes = 2
        dataset = BinaryDataset(dataset, params)

    return dataset, num_classes, ImageParams(image_size=image_size)


def _enum_to_str(value: Any) -> str:
    """Convert enum to string value, handling both enum and string inputs."""
    return value.value if hasattr(value, "value") else str(value)


def get_binary_classes(
    balanced: bool,
    dataset_name: DatasetNames | str,
) -> tuple[int | None, int | None]:
    """Get binary classes if dataset supports balancing and it's enabled."""
    if not balanced:
        return None, None
    binary_datasets = {
        "chest-xray": (1, 0),
        "synthetic-chest-xray": (1, 0),
    }
    dataset_str = _enum_to_str(dataset_name)
    return binary_datasets.get(dataset_str, (None, None))


def save_companion_dataset_metadata(dataset: Any, dataset_name: str, dataroot: str) -> None:
    """
    Save companion dataset metadata (image paths and labels) to a JSON file.

    Args:
        dataset: The loaded dataset object (should be CompanionDataset if applicable)
        dataset_name: Name of the dataset (e.g., 'companion-mnist')
        dataroot: Root data directory

    """
    # Only save if this is a CompanionDataset
    if not isinstance(dataset, CompanionDataset):
        return

    # Create data directory if it doesn't exist
    data_dir = Path(dataroot) / ".." / "data"
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    metadata = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset.image_paths),
        "image_paths": dataset.image_paths,
        "labels": dataset.labels,
    }

    output_file = data_dir / f"companion-{dataset_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved companion dataset metadata to {output_file}")


def collate_with_ground_truth(batch: list[tuple[Any, Any]]) -> tuple[torch.Tensor, list]:
    """
    Preserve ground truth labels as lists in batch collation.

    For ambiguous datasets (ambiguess, companion), labels are lists of class indices.
    This collate function stacks images into a tensor but keeps labels as a list of lists.

    Args:
        batch: List of (image, label) tuples from the dataset.

    Returns:
        Tuple of (stacked_images, list_of_labels).

    """
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    # Stack images into a tensor
    stacked_images = torch.stack(images)

    # Keep labels as a list (don't convert to tensor)
    # This preserves list[list[int]] for ambiguous datasets
    return stacked_images, labels


def load_datasets_for_evaluation(
    dataroot: str,
    dataset_name: DatasetNames,
    balanced: bool = False,
    batch_size: int = 32,
) -> tuple[DataLoader, dict[str, Any]]:
    """
    Load dataset for evaluating ambiguity metrics.

    For most datasets: Uses test set split into 50/10/40, returns only held-out 40% eval portion.
    For chest x-ray: Uses validation set split into 80/20, returns only held-out 20% eval portion.

    Args:
        dataroot: Root directory where datasets are stored
        dataset_name: Name of the dataset to load (using DatasetNames enum)
        balanced: Whether to load balanced binary classification version
        seed: Random seed for reproducibility
        batch_size: Batch size for DataLoader

    Returns:
        Tuple of (Test DataLoader, metadata dict with companion metrics if applicable)

    """
    dataset_str = _enum_to_str(dataset_name)
    logger.info("Loading %s dataset...", dataset_str)

    # Load dataset
    pos_class, neg_class = get_binary_classes(balanced, dataset_name)

    test_dataset, num_classes, _ = load_dataset(
        LoadDatasetParams(
            dataroot=dataroot,
            dataset_name=dataset_name,
            pos_class=pos_class,
            neg_class=neg_class,
            split="test",
            pytesting=False,
        )
    )

    # Use custom collate function for datasets with ground truth labels (ambiguess, companion)
    # This preserves labels as lists instead of converting to tensors
    collate_fn = None
    if dataset_name in (
        DatasetNames.ambiguess_mnist,
        DatasetNames.ambiguess_fmnist,
        DatasetNames.companion_mnist,
        DatasetNames.companion_fmnist,
        DatasetNames.companion_chest_xray,
        DatasetNames.companion_ambiguous_mnist,
        DatasetNames.companion_ambiguous_fmnist,
        DatasetNames.companion_ambiguous_chest_xray,
    ):
        collate_fn = collate_with_ground_truth

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info("  ✓ Loaded %s with %d classes", dataset_str, num_classes)

    # Save companion dataset metadata if applicable
    save_companion_dataset_metadata(test_dataset, dataset_str, dataroot)

    dataset_metadata: dict[str, Any] = {}

    return test_dataloader, dataset_metadata


def extract_ground_truth_labels(dataloader: DataLoader) -> list | None:
    """
    Extract ground truth labels from a dataloader.

    Returns flattened ground truth labels if available (for ambiguous/companion datasets),
    None otherwise.

    Args:
        dataloader: DataLoader to extract labels from

    Returns:
        Flattened list of ground truth labels, or None if labels are not lists

    """
    all_labels = []
    for _, labels in dataloader:
        all_labels.append(labels)

    if not all_labels or not isinstance(all_labels[0], list):
        return None

    # Flatten all ground truth labels
    ground_truth = []
    for label_batch in all_labels:
        if isinstance(label_batch, torch.Tensor):
            ground_truth.extend(label_batch.cpu().numpy().tolist())
        else:
            ground_truth.extend(label_batch)

    return ground_truth
