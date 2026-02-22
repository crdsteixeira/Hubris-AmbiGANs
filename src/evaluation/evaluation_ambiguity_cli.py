"""CLI for ambiguity evaluation."""  # pylint: disable=too-many-lines

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifier.multiclass_train_utils import (
    split_test_set_for_classifier_training,
    split_validation_set_for_ambiguity_evaluation,
    train_single_multiclass_classifier,
)
from src.datasets.datasets import CompanionDataset
from src.datasets.load import load_dataset
from src.enums import ClassifierType, DatasetNames, DeviceType
from src.metrics.ambiguity import compute_entropy, compute_top_pairs
from src.metrics.fid.fid import FID
from src.metrics.image_quality import (
    compute_fid_metric,
    compute_pymdma_metrics_from_images,
    extract_features,
)
from src.models import CLAmbiguityArgs, ConfigTrainingParams, LoadDatasetParams
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import setup_reprod

configure_logging()
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


# Datasets that support binary classification with hardcoded class indices
BINARY_DATASETS = {
    "chest-xray": (1, 0),
    "synthetic-chest-xray": (1, 0),
}


def _enum_to_str(value: Any) -> str:
    """Convert enum to string value, handling both enum and string inputs."""
    return value.value if hasattr(value, "value") else str(value)


def _get_binary_classes(
    config: "CLAmbiguityArgs",
    dataset_name: DatasetNames | str,
) -> tuple[int | None, int | None]:
    """Get binary classes if dataset supports balancing and it's enabled."""
    if not config.balanced:
        return None, None

    dataset_str = _enum_to_str(dataset_name)
    return BINARY_DATASETS.get(dataset_str, (None, None))


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


def parse_args() -> CLAmbiguityArgs:
    """Parse and validate command-line arguments from config file."""
    parser = argparse.ArgumentParser(description="Run ambiguity evaluation")
    parser.add_argument(
        "config", help="Path to YAML config file (e.g., ambiguity-evaluation/advanced_mnist_config.yaml)"
    )

    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Extract values from config
    # Handle both old format (dataset: string) and new format (datasets.training: string)
    if isinstance(config_dict.get("datasets"), dict):
        training_dataset = config_dict["datasets"].get("training")
        eval_datasets = config_dict["datasets"].get("evaluation", [])
        balanced = config_dict["datasets"].get("balanced", False)
    else:
        training_dataset = config_dict.get("dataset")
        eval_datasets = config_dict.get("datasets", [])
        balanced = config_dict.get("balanced", False)

    data_dir = config_dict.get("data_dir", "data")
    out_dir = config_dict.get("out_dir", "models")

    # If FILESDIR is set and paths are relative, prepend FILESDIR
    filesdir = os.environ.get("FILESDIR")
    if filesdir:
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(filesdir, data_dir)
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(filesdir, out_dir)

    # Extract training parameters, using defaults if not specified
    training_config = config_dict.get("training", {})
    training_params = ConfigTrainingParams(
        batch_size=training_config.get("batch_size", 64),
        epochs=training_config.get("epochs", 30),
        lr=training_config.get("lr", 0.001),
    )

    args_dict = {
        "dataroot": data_dir,
        "out_dir": out_dir,
        "models": [ClassifierType(c) for c in config_dict.get("classifiers", [])],
        "datasets": [DatasetNames(d) if isinstance(d, str) else d for d in eval_datasets],
        "device": DeviceType(config_dict.get("device", "cpu")),
        "seed": config_dict.get("seed"),
        "training_dataset": training_dataset,
        "balanced": balanced,
        "training_params": training_params,
    }

    return CLAmbiguityArgs(**args_dict)


def find_checkpoint(out_dir: str, training_dataset: str, classifier_type: str, seed: int) -> Path | None:
    """
    Find checkpoint file for a trained classifier.

    Args:
        out_dir: Output directory where models are stored
        training_dataset: Training dataset name (subdirectory)
        classifier_type: Type of classifier (e.g., 'cnn', 'vgg16')
        seed: Random seed used for training

    Returns:
        Path to checkpoint if found, None otherwise

    """
    checkpoint_dir = Path(out_dir) / training_dataset

    if not checkpoint_dir.exists():
        return None

    # Look for checkpoint files: {classifier_type}_{seed} or {classifier_type}_{seed}.pt
    for pattern in [f"{classifier_type}_{seed}.pt", f"{classifier_type}_{seed}"]:
        for checkpoint_file in checkpoint_dir.glob(pattern):
            return checkpoint_file

    return None


def ensure_models_trained(config: "CLAmbiguityArgs", training_dataset: str) -> None:
    """
    Ensure all models are trained (but don't keep them in memory).

    Args:
        config: Configuration containing models and seed info
        training_dataset: Dataset to train on

    """
    seed = config.seed if config.seed is not None else 42

    training_dataset_enum = DatasetNames(training_dataset)
    pos_class, neg_class = _get_binary_classes(config, training_dataset_enum)

    for classifier in config.models:
        classifier_str = _enum_to_str(classifier)

        # Check if checkpoint exists
        checkpoint_path = find_checkpoint(config.out_dir, training_dataset, classifier_str, seed)

        if checkpoint_path is None:
            logger.info(
                "Training %s on %s with epochs=%d, lr=%f...",
                classifier_str,
                training_dataset,
                config.training_params.epochs,
                config.training_params.lr,
            )
            train_single_multiclass_classifier(
                dataset_name=training_dataset,
                classifier_type=classifier,
                data_dir=config.dataroot,
                out_dir=config.out_dir,
                batch_size=config.training_params.batch_size,
                epochs=config.training_params.epochs,
                lr=config.training_params.lr,
                device=config.device,
                seed=config.seed,
                pos_class=pos_class,
                neg_class=neg_class,
            )
        else:
            logger.info("Found trained %s on %s: %s", classifier_str, training_dataset, checkpoint_path)


def load_model_for_evaluation(
    config: "CLAmbiguityArgs",
    classifier_type: str,
    training_dataset: str,
) -> torch.nn.Module:
    """
    Load a single model checkpoint for evaluation.

    Args:
        config: Configuration containing output directory and device info
        classifier_type: Type of classifier to load
        training_dataset: Training dataset name

    Returns:
        Loaded model

    """
    seed = config.seed if config.seed is not None else 42
    checkpoint_path = find_checkpoint(config.out_dir, training_dataset, classifier_type, seed)

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for {classifier_type} with seed {seed} in {config.out_dir}/{training_dataset}"
        )

    # construct_classifier_from_checkpoint returns a tuple: (model, params, stats, args, optimizer)
    model, _, _, _, _ = construct_classifier_from_checkpoint(str(checkpoint_path), config.device)
    return model


def load_datasets_for_evaluation(
    config: "CLAmbiguityArgs",
    dataset_name: DatasetNames,
    batch_size: int = 32,
) -> tuple[DataLoader, dict[str, Any]]:
    """
    Load dataset for evaluating ambiguity metrics.

    For most datasets: Uses test set split into 50/10/40, returns only held-out 40% eval portion.
    For chest x-ray: Uses validation set split into 80/20, returns only held-out 20% eval portion.

    Args:
        config: Configuration containing dataroot, device info, and seed
        dataset_name: Name of the dataset to load (using DatasetNames enum)
        batch_size: Batch size for DataLoader

    Returns:
        Tuple of (Test DataLoader, metadata dict with companion metrics if applicable)

    """
    dataset_str = _enum_to_str(dataset_name)
    logger.info("Loading %s dataset...", dataset_str)

    # Load dataset (validation split for chest x-ray, test split for others)
    pos_class, neg_class = _get_binary_classes(config, dataset_name)

    test_dataset, num_classes, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=dataset_name,
            pos_class=pos_class,
            neg_class=neg_class,
            split="test",
            pytesting=False,
        )
    )

    # Split dataset based on dataset type
    # For chest x-ray: validation split into 80/20 (train/eval for ambiguity metrics)
    # For others: test split into 50/10/40 (train/val/eval for ambiguity metrics)
    seed = config.seed if config.seed is not None else 42
    if dataset_name == DatasetNames.chest_xray:
        # Use 80/20 split for chest x-ray, return 20% held-out eval portion
        _, eval_split = split_validation_set_for_ambiguity_evaluation(test_dataset, seed=seed)
    else:
        # Use 50/10/40 split for other datasets, return 40% held-out eval portion
        _, _, eval_split = split_test_set_for_classifier_training(test_dataset, seed=seed)

    # Use custom collate function for datasets with ground truth labels (ambiguess, companion)
    # This preserves labels as lists instead of converting to tensors
    collate_fn = None
    if dataset_name in (
        DatasetNames.ambiguess_mnist,
        DatasetNames.ambiguess_fmnist,
        DatasetNames.companion_mnist,
        DatasetNames.companion_fmnist,
        DatasetNames.companion_chest_xray,
    ):
        collate_fn = collate_with_ground_truth

    test_dataloader = DataLoader(eval_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info("  ✓ Loaded %s with %d classes", dataset_str, num_classes)

    # Save companion dataset metadata if applicable
    save_companion_dataset_metadata(test_dataset, dataset_str, config.dataroot)

    dataset_metadata: dict[str, Any] = {}

    return test_dataloader, dataset_metadata


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


def get_model_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: DeviceType | str,
) -> torch.Tensor:
    """
    Get softmax predictions from a model on a dataloader.

    TODO: review this function

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


def generate_fid_stats(  # pylint: disable=too-many-statements
    dataroot: str,
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 6,
    device: DeviceType | str = "cpu",
    use_test_set: bool = True,
    n_samples: int = 10000,
) -> str:
    """
    Generate and save FID statistics for a dataset.

    IMPORTANT: This uses InceptionV3-based feature extraction (inception_fid) which is compatible
    with FrechetInceptionDistance. Do NOT use DINO or other extractors for FID statistics.

    Args:
        dataroot: Root directory where datasets are stored
        dataset_name: Name of the dataset (e.g., 'mnist', 'companion-mnist')
        batch_size: Batch size for processing
        num_workers: Number of worker processes for data loading
        device: Device to use ('cpu' or 'cuda:X')
        use_test_set: If True, use test set; if False, use training set
        n_samples: Maximum number of samples to use for statistics (None for all)

    Returns:
        Path to the generated FID statistics file

    """
    stats_dir = Path(dataroot) / "fid-stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    stats_file = stats_dir / f"stats.{dataset_name}.npz"

    # If stats already exist, return the path
    if stats_file.exists():
        logger.info(f"FID statistics already exist at {stats_file}")
        return str(stats_file)

    logger.info(f"Generating FID statistics for {dataset_name}...")

    split = "test" if use_test_set else "train"
    # Load dataset
    try:
        dataset, _, _ = load_dataset(
            LoadDatasetParams(
                dataroot=dataroot,
                dataset_name=DatasetNames(dataset_name),
                pos_class=None,
                neg_class=None,
                split=split,
                pytesting=False,
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

    logger.info(f"Dataset size: {len(dataset)}")

    # Sample down if necessary
    if n_samples is not None and len(dataset) > n_samples:
        logger.info(f"Sampling {n_samples} images from {len(dataset)} total")
        indices = np.random.choice(len(dataset), size=n_samples, replace=False)
        # Convert numpy indices to Python ints (HuggingFace datasets don't accept numpy.int64)
        indices = indices.tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    else:
        if n_samples is not None:
            logger.info(f"Using all {len(dataset)} images (less than requested {n_samples})")

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize FID metric (use repo's FID class to compute reference stats)
    device_obj = DeviceType(device) if isinstance(device, str) else device
    device_str = device_obj.value if isinstance(device_obj, DeviceType) else str(device_obj)
    fid = FID(fid_stats_file=None, dims=2048, n_images=len(dataset), device=device_obj)

    # Initialize feature extractor for FID (use InceptionV3-based FID extractor, not DINO)
    # inception_fid produces 2048-dimensional features compatible with FID
    logger.info("Loading InceptionV3 extractor for FID computation...")
    inception_extractor = ExtractorFactory.model_from_name(name="inception_fid")
    inception_extractor = inception_extractor.to(device_str)
    inception_extractor.eval()

    # Also initialize DINO extractor for pymdma metrics reference
    logger.info("Loading DINO ViT extractor for pymdma metrics...")
    dino_extractor = ExtractorFactory.model_from_name(name="dino_vits8")
    dino_extractor = dino_extractor.to(device_str)
    dino_extractor.eval()
    all_features = []

    # Calculate FID statistics
    logger.info("Computing FID statistics...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing FID stats"):
            images = batch[0]

            if images.ndim < 2:
                raise ValueError(
                    f"Images must have at least two dimensions (batch size and channel), got {images.ndim}D tensor."
                )

            # Convert to RGB by repeating across the channel dimension if needed
            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)

            images = images.to(device_str)

            # NOTE: Directly update fid.fid with is_real=True to accumulate reference statistics
            # We bypass FID.update() because it always uses is_real=False
            # Convert from [-1, 1] to [0, 1] for InceptionV3
            images_normalized = (images + 1.0) / 2.0
            fid.fid.update(images_normalized, is_real=True)

            # Extract DINO features for pymdma metrics reference (on normalized images)
            features = dino_extractor(images_normalized).detach().cpu().numpy()
            all_features.append(features)

    # Extract statistics from FID instance
    # Compute mean and covariance from accumulated statistics
    m = fid.fid.real_sum / fid.fid.num_real_images
    s = fid.fid.real_cov_sum - fid.fid.num_real_images * torch.outer(m, m)

    # Save statistics
    logger.info(f"Saving FID statistics to {stats_file}...")
    with open(f"{stats_file}", "wb") as f:
        np.savez(
            f,
            mu=m.cpu().numpy(),
            sigma=s.cpu().numpy(),
            real_sum=fid.fid.real_sum.cpu().numpy(),
            real_cov_sum=fid.fid.real_cov_sum.cpu().numpy(),
            num_real_images=fid.fid.num_real_images.cpu().numpy(),
            all_features=np.concatenate(all_features, axis=0),
        )

    logger.info(f"FID statistics saved to {stats_file}")
    return str(stats_file)


def find_fid_stats(dataroot: str, dataset_name: str) -> str | None:
    """
    Find FID statistics file for a dataset.

    Args:
        dataroot: Root directory where datasets are stored
        dataset_name: Name of the dataset

    Returns:
        Path to FID stats file if found, None otherwise

    """
    fid_stats_dir = Path(dataroot) / "fid-stats"
    if not fid_stats_dir.exists():
        return None

    # Look for stats file matching the dataset name
    stats_file = fid_stats_dir / f"stats.{dataset_name}.npz"
    if stats_file.exists():
        return str(stats_file)

    return None


def compute_evaluation_metrics(  # noqa: C901
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: DeviceType | str,
    fid_stats_path: str | None = None,
    real_features: np.ndarray | None = None,
    extractor: Any = None,
    eval_sample_size: int | None = None,
) -> dict[str, float]:
    """
    Compute evaluation metrics for a model and dataset.

    Args:
        model: Trained classifier model
        dataloader: DataLoader with test data
        device: Device to use for computation
        fid_stats_path: Optional path to FID statistics file
        real_features: Optional reference features from real training dataset for pymdma comparison
        extractor: Optional cached feature extractor to avoid recreating it
        eval_sample_size: Optional limit on evaluation images for faster metrics computation

    Returns:
        Dictionary with metric names and values

    """
    # Convert device to string for PyTorch operations
    device_str = device.value if isinstance(device, DeviceType) else str(device)

    model.eval()
    all_preds = []
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device_str)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_images.append(images.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_images = torch.cat(all_images)

    # Compute entropy metric
    entropy = compute_entropy(all_preds)

    metrics = {
        "entropy": entropy,
    }

    # Compute top_pairs metric if ground truth labels are available (ambiguess/companion datasets)
    # Check if labels are lists (ground truth class lists) rather than single integers
    if all_labels and isinstance(all_labels[0], list):
        # Flatten all ground truth labels
        ground_truth = []
        for label_batch in all_labels:
            if isinstance(label_batch, torch.Tensor):
                ground_truth.extend(label_batch.cpu().numpy().tolist())
            else:
                ground_truth.extend(label_batch)

        top_pairs = compute_top_pairs(all_preds, ground_truth)
        metrics["top_pairs"] = top_pairs

    # Compute FID if stats file is available
    if fid_stats_path is not None:
        try:
            fid_score = compute_fid_metric(model, dataloader, device, fid_stats_path)
            metrics["fid"] = fid_score
            logger.info(f"✓ FID score: {fid_score:.4f}")
        except FileNotFoundError:
            logger.error(f"✗✗✗ CRITICAL: FID stats file missing: {fid_stats_path}")
            logger.error("✗✗✗ Did Phase 1.5 fail to generate stats? Check logs above!")
            raise
        except (RuntimeError, ValueError) as e:
            logger.error(f"✗✗✗ CRITICAL: Failed to compute FID: {e}")
            logger.error(f"✗✗✗ Exception type: {type(e).__name__}")
            import traceback  # pylint: disable=import-outside-toplevel

            logger.error(f"✗✗✗ Traceback: {traceback.format_exc()}")
            raise

    # Compute pymdma metrics if reference features available
    if real_features is not None:
        try:
            pymdma_metrics = compute_pymdma_metrics_from_images(
                all_images, real_features, device, extractor, eval_sample_size
            )
            metrics.update(pymdma_metrics)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to compute pymdma metrics: %s", e)

    return metrics


def extract_training_features(
    config: "CLAmbiguityArgs", device: DeviceType | str, eval_dataset_name: str | None = None
) -> tuple[np.ndarray, object]:
    """
    Extract features from the real training dataset to use as reference for pymdma.

    Args:
        config: Configuration containing dataroot and training dataset info
        device: Device to use for computation
        eval_dataset_name: Name of the evaluation dataset. If provided and different from training dataset,
                          the number of training samples will match the evaluation dataset size.
                          If None or same as training dataset, defaults to 10k samples.

    Returns:
        Tuple of (feature array from training dataset, reusable extractor model)

    """
    training_dataset_str = config.training_dataset

    # Determine number of training samples to extract
    n_samples = 10000  # Default
    if eval_dataset_name is not None and eval_dataset_name != training_dataset_str:
        # Load evaluation dataset to determine its size
        try:
            eval_dataset, _, _ = load_dataset(
                LoadDatasetParams(
                    dataroot=config.dataroot,
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

    logger.info("Extracting features from training dataset (%s) for pymdma reference...", training_dataset_str)

    # Load training dataset
    train_dataset, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=DatasetNames(training_dataset_str),
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


def compute_dataset_metrics(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches  # noqa: C901
    config: "CLAmbiguityArgs",
    dataset: DatasetNames,
    fid_stats_path: str | None = None,
    real_features: np.ndarray | None = None,
    extractor: Any = None,
) -> tuple[float | None, dict[str, float]]:
    """
    Compute FID, pymdma, and confusion distance metrics for a dataset (dataset-level, independent of classifier).

    These metrics measure image quality/distribution and should be constant across all classifiers.

    Args:
        config: Configuration
        dataset: Dataset to compute metrics for
        fid_stats_path: Path to FID reference stats
        real_features: Reference features for pymdma
        extractor: Cached feature extractor

    Returns:
        Tuple of (fid_score, dataset_metrics_dict) where dataset_metrics_dict includes pymdma and confusion_distance metrics

    """
    dataset_str = _enum_to_str(dataset)
    logger.info("  Computing dataset-level metrics for %s...", dataset_str)

    # Load dataset
    try:
        test_dataset, _, _ = load_dataset(
            LoadDatasetParams(
                dataroot=config.dataroot,
                dataset_name=dataset,
                pos_class=None,
                neg_class=None,
                split="test",
                pytesting=False,
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to load dataset %s: %s", dataset_str, e)
        return None, {}

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Save companion dataset metadata if applicable
    save_companion_dataset_metadata(test_dataset, dataset_str, config.dataroot)

    # Generate FID statistics using the actual evaluation dataset size
    if fid_stats_path is None or not os.path.exists(fid_stats_path):
        if dataset_str != config.training_dataset:
            try:
                logger.info(f"  Generating FID statistics for {dataset_str} (size: {len(test_dataset)})...")
                fid_stats_path = generate_fid_stats(
                    dataroot=config.dataroot,
                    dataset_name=config.training_dataset,
                    batch_size=64,
                    num_workers=6,
                    device=config.device,
                    use_test_set=False,  # Use training set as reference
                    n_samples=len(test_dataset),  # Match evaluation dataset size
                )
                logger.info(f"  ✓ FID statistics generated: {fid_stats_path}")

                # Create symlink if this is a synthetic/companion/ambiguous dataset
                if dataset_str != config.training_dataset:
                    stats_dir = Path(config.dataroot) / "fid-stats"
                    ref_stats_file = stats_dir / f"stats.{config.training_dataset}.npz"
                    eval_stats_file = stats_dir / f"stats.{dataset_str}.npz"
                    if ref_stats_file.exists() and not eval_stats_file.exists():
                        logger.info(f"  Linking stats.{config.training_dataset}.npz -> stats.{dataset_str}.npz")
                        eval_stats_file.symlink_to(ref_stats_file)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"Failed to generate FID statistics for {dataset_str}: {e}")
                fid_stats_path = None

    # Compute FID
    fid_score = None
    if fid_stats_path is not None:
        try:
            fid_score = compute_fid_metric(None, test_dataloader, config.device, fid_stats_path)
            logger.info("    ✓ FID score: %.4f", fid_score)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to compute FID: %s", e)

    # Compute pymdma metrics
    dataset_metrics = {}
    if real_features is not None:
        try:
            # Load all images from dataset without sampling
            all_images = []
            with torch.no_grad():
                for images, _ in test_dataloader:
                    all_images.append(images)
            all_images = torch.cat(all_images)

            # Compute metrics without sampling (deterministic)
            pymdma_metrics = compute_pymdma_metrics_from_images(
                all_images, real_features, config.device, extractor, sample_size=None
            )
            dataset_metrics.update(pymdma_metrics)
            for metric_name in PYMDMA_METRIC_NAMES:
                if metric_name in dataset_metrics:
                    logger.info("    ✓ %s: %.4f", metric_name, dataset_metrics[metric_name])
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to compute pymdma metrics: %s", e)

    return fid_score, dataset_metrics


def evaluate_single_model(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches  # noqa: C901
    config: "CLAmbiguityArgs",
    classifier: ClassifierType,
    dataset: DatasetNames,
    dataset_fid: float | None = None,
    dataset_metrics: dict | None = None,
) -> bool:
    """
    Evaluate a single model on a single dataset.

    Computes classifier-specific metrics (entropy, uncertainty).
    Uses pre-computed dataset-level metrics (FID, pymdma, confusion distance) which are identical across classifiers.

    Args:
        config: Configuration with models and dataset info
        classifier: Classifier type to evaluate
        dataset: Dataset to evaluate on
        dataset_fid: Pre-computed FID score for this dataset (dataset-level metric, same for all classifiers)
        dataset_metrics: Pre-computed dataset-level metrics for this dataset (pymdma, confusion distance, etc.)

    Returns:
        Boolean indicating if evaluation succeeded (True) or failed (False)

    """
    if dataset_metrics is None:
        dataset_metrics = {}

    classifier_str = _enum_to_str(classifier)
    dataset_str = _enum_to_str(dataset)

    # Initialize wandb run for this specific model-dataset pair
    wandb.init(
        project="ambiguity-evaluation",
        name=f"{config.training_dataset}-{classifier_str}-{dataset_str}",
        config={
            "training_dataset": config.training_dataset,
            "model": classifier_str,
            "eval_dataset": dataset_str,
            "seed": config.seed,
        },
        reinit=True,
    )

    try:
        # Load model (memory-efficient: one at a time)
        logger.info("  Loading %s model...", classifier_str)
        model = load_model_for_evaluation(config, classifier_str, config.training_dataset)
        model.eval()
        logger.info("  ✓ Model loaded - checking device placement")

        # Verify model is on correct device
        device_str = config.device.value if isinstance(config.device, DeviceType) else str(config.device)
        first_param = next(model.parameters())
        logger.info(f"    Model parameters are on: {first_param.device}")
        logger.info(f"    Expected device: {device_str}")

        # Load dataset for entropy computation
        logger.info("  Loading %s dataset (different from training for varied entropy)...", dataset_str)
        test_dataloader, dataset_companion_metrics = load_datasets_for_evaluation(config, dataset, batch_size=32)
        logger.info("  ✓ Dataset %s loaded - %d samples", dataset_str, len(test_dataloader.dataset))

        # Compute classifier-specific metrics (entropy - depends on model predictions)
        logger.info("  Computing classifier-specific metrics...")
        softmax_preds = get_model_predictions(model, test_dataloader, config.device)
        logger.info(
            f"    Probabilities shape: {softmax_preds.shape}, min: {softmax_preds.min():.4f}, max: {softmax_preds.max():.4f}"
        )

        # IMPORTANT: Entropy MUST be computed fresh for each classifier-dataset pair
        # Do NOT reuse entropy from previous evaluations
        entropy = compute_entropy(softmax_preds)
        logger.info(f"    ✓ Computed entropy for {classifier_str} on {dataset_str}: {entropy:.6f}")

        # Combine metrics: classifier-specific entropy + dataset-level FID/pymdma/confusion_distance
        metrics = {
            "entropy": entropy,
        }

        # Compute top_pairs metric if ground truth is available (ambiguess/companion datasets)
        # Collect ground truth from dataloader
        all_labels = []
        for _, labels in test_dataloader:
            all_labels.append(labels)

        ground_truth = None
        if all_labels and isinstance(all_labels[0], list):
            # Flatten all ground truth labels
            ground_truth = []
            for label_batch in all_labels:
                if isinstance(label_batch, torch.Tensor):
                    ground_truth.extend(label_batch.cpu().numpy().tolist())
                else:
                    ground_truth.extend(label_batch)

            top_pairs = compute_top_pairs(softmax_preds, ground_truth)
            metrics["top_pairs"] = top_pairs
            logger.info(f"    ✓ Computed top_pairs for {classifier_str} on {dataset_str}: {top_pairs:.6f}")

        # Add pre-computed dataset-level metrics (same for all classifiers)
        if dataset_fid is not None:
            metrics["fid"] = dataset_fid
        metrics.update(dataset_metrics)

        # Add companion dataset metrics if available
        metrics.update(dataset_companion_metrics)

        # Log results
        log_msg = f"  ✓ Metrics computed - Entropy: {metrics['entropy']:.4f}"
        if "top_pairs" in metrics:
            log_msg += f", Top Pairs: {metrics['top_pairs']:.4f}"
        if "fid" in metrics:
            log_msg += f", FID: {metrics['fid']:.4f}"

        for metric_name in PYMDMA_METRIC_NAMES:
            if metric_name in metrics:
                log_msg += f", {metric_name}: {metrics[metric_name]:.4f}"
        logger.info(log_msg)

        # Log metrics to wandb
        log_data = {
            "classifier": classifier_str,
            "entropy": metrics["entropy"],
        }
        if "top_pairs" in metrics:
            log_data["top_pairs"] = metrics["top_pairs"]
        if "fid" in metrics:
            log_data["fid"] = metrics["fid"]
        for metric_name in PYMDMA_METRIC_NAMES:
            if metric_name in metrics:
                log_data[metric_name] = metrics[metric_name]
        wandb.log(log_data)

        # Unload model to free memory
        del model
        torch.cuda.empty_cache()
        logger.info("  ✓ Evaluation complete")
        wandb.finish(exit_code=0)
        return True

    except (OSError, FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("  ✗ Evaluation failed: %s", e)
        wandb.log(
            {
                "classifier": classifier_str,
                "dataset": dataset_str,
                "error": str(e),
            }
        )
        wandb.alert(  # type: ignore[attr-defined]
            title=f"Evaluation Failed: {classifier_str} on {dataset_str}",
            text=f"Error: {str(e)}",
            level="ERROR",
        )
        wandb.finish(exit_code=1)
        return False


def run_evaluation_loop(
    config: "CLAmbiguityArgs",
    eval_datasets: list,
) -> None:
    """
    Run the evaluation loop for all models on all datasets.

    Computes dataset-level metrics (FID, pymdma) once per dataset, then evaluates all classifiers
    on that dataset using the cached metrics.

    Args:
        config: Configuration with models and dataset info
        eval_datasets: List of datasets to evaluate on

    """
    total_steps = len(eval_datasets) * (1 + len(config.models))  # 1 dataset-level + N classifiers
    current_step = 0

    # Create feature extractor once and reuse it across datasets
    extractor = None

    for dataset in eval_datasets:
        dataset_str = _enum_to_str(dataset)

        # Skip computing FID/pymdma metrics if evaluating on training dataset
        if dataset_str == config.training_dataset:
            logger.info(f"\nEvaluating on training dataset {dataset_str} (skipping FID/pymdma metrics)")
            dataset_fid = None
            dataset_metrics: dict[str, Any] = {}
            real_features = None
        else:
            # Extract training features for this specific evaluation dataset
            # Number of training samples will match the evaluation dataset size
            try:
                real_features, extractor = extract_training_features(config, config.device, dataset_str)
            except (RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to extract training features for %s: %s", dataset_str, e)
                real_features = None

            # Step 1: Compute dataset-level metrics once (FID + pymdma)
            current_step += 1
            logger.info(f"\n[{current_step}/{total_steps}] Computing dataset-level metrics for {dataset_str}")

            fid_stats_path = find_fid_stats(config.dataroot, dataset_str)
            dataset_fid, dataset_metrics = compute_dataset_metrics(
                config, dataset, fid_stats_path=fid_stats_path, real_features=real_features, extractor=extractor
            )

        # Step 2: Evaluate each classifier on this dataset using cached metrics
        for classifier in config.models:
            current_step += 1
            classifier_str = _enum_to_str(classifier)
            logger.info(f"\n[{current_step}/{total_steps}] Evaluating {classifier_str} on {dataset_str}")
            evaluate_single_model(config, classifier, dataset, dataset_fid, dataset_metrics)


def main() -> None:  # pylint: disable=too-many-statements
    """Entry point for ambiguity evaluation."""
    logger.info("Ambiguity evaluation is starting...")

    config = parse_args()
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    # Determine evaluation datasets
    eval_datasets = config.datasets if config.datasets else [DatasetNames(config.training_dataset)]

    # Check and train models if needed
    separator = "=" * 80
    logger.info(separator)
    logger.info("PHASE 1: Checking/Training Models")
    logger.info(separator)

    ensure_models_trained(config, config.training_dataset)
    logger.info("All models ready for evaluation")

    # Phase 2: Evaluate each model on each dataset
    logger.info(separator)
    logger.info("PHASE 2: Evaluation")
    logger.info(separator)

    # Run evaluation loop - training features will be extracted per evaluation dataset
    # with size matching the evaluation dataset (or 10k if evaluating on training dataset itself)
    run_evaluation_loop(config, eval_datasets)

    logger.info(separator)
    logger.info("Ambiguity evaluation complete")


if __name__ == "__main__":
    main()
