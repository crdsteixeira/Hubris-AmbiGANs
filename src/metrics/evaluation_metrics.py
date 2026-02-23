"""Evaluation metrics computation utilities."""

import logging
from typing import Any

import numpy as np
import torch
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader

from src.datasets.evaluation import extract_ground_truth_labels
from src.datasets.load import load_dataset
from src.enums import DatasetNames, DeviceType
from src.metrics.ambiguity import compute_entropy, compute_top_pairs
from src.metrics.image_quality import (
    extract_features,
)
from src.models import LoadDatasetParams

logger = logging.getLogger(__name__)

# Pymdma metrics to compute
PYMDMA_METRIC_NAMES = [
    "improved_precision",
    "improved_recall",
    "giqa_qs",
    "giqa_ds",
    "density",
    "coverage",
]


def _enum_to_str(value: Any) -> str:
    """Convert enum to string value, handling both enum and string inputs."""
    return value.value if hasattr(value, "value") else str(value)


def get_model_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: DeviceType | str,
) -> torch.Tensor:
    """
    Get softmax predictions from a model on a dataloader.

    Handles various output formats from different models:
    - SimpleCNN/MLP binary: 1D sigmoid probabilities
    - SimpleCNN/MLP multiclass: raw logits
    - Other models: raw logits
    - Pretrained models: raw logits

    Converts all outputs to [N, n_classes] softmax probabilities.

    Args:
        model: Model to get predictions from (must be in eval mode)
        dataloader: DataLoader with images
        device: Device to use for computation

    Returns:
        Tensor of shape (n_samples, n_classes) with softmax probabilities

    """
    device_str = device.value if isinstance(device, DeviceType) else str(device)

    all_preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device_str)
            outputs = model(images)
            all_preds.append(outputs.cpu())

    all_preds = torch.cat(all_preds)

    # Reshape 1D output to [N, 1] if needed
    if all_preds.dim() == 1:
        all_preds = all_preds.unsqueeze(1)

    # Handle binary classification case [N, 1] (typically sigmoid output)
    # Convert to [N, 2] with proper probability distribution: [1-p, p]
    if all_preds.shape[1] == 1:
        p = torch.clamp(all_preds, 0.0, 1.0)
        all_preds = torch.cat([1 - p, p], dim=1)
        return all_preds

    # For multiclass: check if already probabilities or need softmax
    # If all values in [0,1] and sum close to 1, treat as probabilities; else apply softmax
    if all_preds.max().item() <= 1.0 and all_preds.min().item() >= 0.0:
        # Likely probabilities - verify by checking row sums
        row_sums = all_preds.sum(dim=1)
        if (row_sums > 0.9).all() and (row_sums < 1.1).all():
            return all_preds  # Already probabilities

    # Otherwise treat as logits and apply softmax
    return torch.softmax(all_preds, dim=1)


def extract_training_features(
    dataroot: str,
    training_dataset: str,
    device: DeviceType | str,
    eval_dataset_name: str | None = None,
) -> tuple[np.ndarray, Any]:
    """
    Extract features from the real training dataset to use as reference for pymdma.

    Args:
        dataroot: Root data directory
        training_dataset: Name of the training dataset
        device: Device to use for computation
        eval_dataset_name: Name of the evaluation dataset. If provided and different from training dataset,
                          the number of training samples will match the evaluation dataset size.
                          If None or same as training dataset, defaults to 10k samples.

    Returns:
        Tuple of (feature array from training dataset, reusable extractor model)

    """
    # Determine number of training samples to extract
    n_samples = 10000  # Default
    if eval_dataset_name is not None and eval_dataset_name != training_dataset:
        # Load evaluation dataset to determine its size
        try:
            eval_dataset, _, _ = load_dataset(
                LoadDatasetParams(
                    dataroot=dataroot,
                    dataset_name=DatasetNames(eval_dataset_name),
                    split="test",
                    pytesting=False,
                    pos_class=None,
                    neg_class=None,
                )
            )
            if hasattr(eval_dataset, "__len__"):
                n_samples = len(eval_dataset)
                logger.info(f"Using {n_samples} training samples to match evaluation dataset {eval_dataset_name}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to determine evaluation dataset {eval_dataset_name} size, using 10k: {e}")

    logger.info("Extracting features from training dataset (%s) for pymdma reference...", training_dataset)

    # Load training dataset
    train_dataset, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=dataroot,
            dataset_name=DatasetNames(training_dataset),
            split="train",  # Load training set
            pytesting=False,
            pos_class=None,
            neg_class=None,
        )
    )

    # Sample down to the determined number of images
    if hasattr(train_dataset, "__len__"):
        total_samples: int | float = len(train_dataset)
    else:
        # If dataset doesn't have len, we'll load all
        total_samples = float("inf")

    if total_samples > n_samples:
        logger.info(f"Sampling {n_samples} images from {total_samples} total")
        indices = np.random.choice(int(total_samples), size=n_samples, replace=False)
        # Convert numpy indices to Python ints (HuggingFace datasets don't accept numpy.int64)
        indices = indices.tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Collect training images
    all_train_images = []
    with torch.no_grad():
        for images, _ in train_dataloader:
            all_train_images.append(images)

    all_train_images = torch.cat(all_train_images)
    logger.info("  ✓ Loaded %d training images for reference", len(all_train_images))

    # Create and reuse extractor
    device_str = device.value if isinstance(device, DeviceType) else str(device)
    extractor = ExtractorFactory.model_from_name(name="dino_vits8")
    extractor = extractor.to(device_str)
    extractor.eval()

    # Extract features
    real_features = extract_features(all_train_images, device, extractor)
    logger.info("  ✓ Extracted features: shape %s", real_features.shape)

    return real_features, extractor


def compute_classifier_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: DeviceType | str,
) -> tuple[float, dict[str, float]]:
    """
    Compute classifier-specific metrics (entropy, top_pairs).

    These metrics depend on model predictions and vary across classifiers.

    Args:
        model: Trained classifier model
        dataloader: DataLoader with test data
        device: Device to use for computation

    Returns:
        Tuple of (entropy, metrics_dict with optional top_pairs)

    """
    softmax_preds = get_model_predictions(model, dataloader, device)

    # Compute entropy metric
    entropy = compute_entropy(softmax_preds)

    metrics = {"entropy": entropy}

    ground_truth = extract_ground_truth_labels(dataloader)
    if ground_truth is not None:
        top_pairs = compute_top_pairs(softmax_preds, ground_truth)
        metrics["top_pairs"] = top_pairs

    return entropy, metrics
