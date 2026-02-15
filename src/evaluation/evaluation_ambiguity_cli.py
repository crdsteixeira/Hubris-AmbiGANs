"""CLI for ambiguity evaluation."""  # pylint: disable=too-many-lines

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from pydantic import ValidationError
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifier.multiclass_train_utils import train_single_multiclass_classifier
from src.datasets.load import load_dataset
from src.enums import ClassifierType, DatasetNames, DeviceType
from src.metrics.fid.fid import FID
from src.models import CLAmbiguityArgs, LoadDatasetParams
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import calculate_pymdma_metrics, setup_reprod

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


def _enum_to_str(value: Any) -> str:
    """Convert enum to string value, handling both enum and string inputs."""
    return value.value if hasattr(value, "value") else str(value)


def _str_to_device(device: DeviceType | str) -> str:
    """Convert DeviceType enum to string, handling both enum and string inputs."""
    return device.value if hasattr(device, "value") else str(device)


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
    else:
        training_dataset = config_dict.get("dataset")
        eval_datasets = config_dict.get("datasets", [])

    # Get FILESDIR from environment, use relative paths if not set
    filesdir = os.environ.get("FILESDIR")
    data_dir = config_dict.get("data_dir", "data")
    out_dir = config_dict.get("out_dir", "models")

    # If FILESDIR is set and paths are relative, prepend FILESDIR
    if filesdir:
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(filesdir, data_dir)
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(filesdir, out_dir)

    args_dict = {
        "dataroot": data_dir,
        "out_dir": out_dir,
        "models": [ClassifierType(c) for c in config_dict.get("classifiers", [])],
        "datasets": [DatasetNames(d) if isinstance(d, str) else d for d in eval_datasets],
        "device": DeviceType(config_dict.get("device", "cpu")),
        "seed": config_dict.get("seed"),
        "training_dataset": training_dataset,
    }

    try:
        return CLAmbiguityArgs.model_validate(args_dict)
    except ValidationError as exc:
        logger.error("Argument validation error: %s", exc)
        raise


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

    for classifier in config.models:
        classifier_str = _enum_to_str(classifier)

        # Check if checkpoint exists
        checkpoint_path = find_checkpoint(config.out_dir, training_dataset, classifier_str, seed)

        if checkpoint_path is None:
            logger.info("Training %s on %s...", classifier_str, training_dataset)
            train_single_multiclass_classifier(
                dataset_name=training_dataset,
                classifier_type=classifier,
                data_dir=config.dataroot,
                out_dir=config.out_dir,
                device=config.device,
                seed=config.seed,
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
) -> DataLoader:
    """
    Load test dataset for evaluation.

    Args:
        config: Configuration containing dataroot and device info
        dataset_name: Name of the dataset to load (using DatasetNames enum)
        batch_size: Batch size for DataLoader

    Returns:
        Test DataLoader

    """
    dataset_str = _enum_to_str(dataset_name)
    logger.info("Loading %s dataset...", dataset_str)

    # Load test dataset
    test_dataset, num_classes, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=dataset_name,
            train=False,
            pytesting=False,
            pos_class=None,
            neg_class=None,
        )
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("  ✓ Loaded %s with %d classes", dataset_str, num_classes)

    return test_dataloader


def compute_entropy(predictions: torch.Tensor) -> float:
    """
    Compute entropy from classifier predictions (softmax probabilities).

    Args:
        predictions: Tensor of shape (n_samples, n_classes) with probabilities

    Returns:
        Mean entropy across samples

    """
    # Ensure predictions are probabilities (sum to 1)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-7
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)

    # Calculate entropy: -sum(p * log(p))
    entropy = -(predictions * torch.log(predictions)).sum(dim=1)

    # Return mean entropy
    return entropy.mean().item()


def compute_top_pairs(_: torch.Tensor, __: torch.Tensor) -> float:
    """
    Compute top pairs from classifier predictions (softmax probabilities).

    Args:
        _: Tensor of shape (n_samples, n_classes) with probabilities
        __: Tensor of shape (n_samples,) with true class labels
    Returns:
        Top pairs metric (e.g., difference between top 2 probabilities)

    """
    # TODO implement

    return 0


def generate_fid_stats(
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

    # Load dataset
    try:
        dataset, _, _ = load_dataset(
            LoadDatasetParams(
                dataroot=dataroot,
                dataset_name=DatasetNames(dataset_name),
                pos_class=None,
                neg_class=None,
                train=not use_test_set,  # train=False means use test set
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

            # Images are in [-1, 1] range. FID.update() normalizes them to [0, 1]
            images = images.to(device_str)

            # NOTE: Directly update fid.fid with is_real=True to accumulate reference statistics
            # We bypass FID.update() because it always uses is_real=False
            # Normalize to [0, 1] for InceptionV3
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


def compute_fid_metric(
    model: torch.nn.Module | None,
    dataloader: DataLoader,
    device: DeviceType | str,
    fid_stats_path: str,
) -> float:
    """
    Compute FID metric from model predictions.

    Args:
        model: Trained classifier model (not used, kept for backward compatibility)
        dataloader: DataLoader with test data
        device: Device to use for computation
        fid_stats_path: Path to FID statistics file

    Returns:
        FID score

    """
    # model is not used - FID only depends on images, not model predictions
    if model is not None:
        model.eval()

    # Initialize FID metric with reference statistics
    device_obj = DeviceType(device) if isinstance(device, str) else device
    device_str = device_obj.value if isinstance(device_obj, DeviceType) else str(device_obj)
    fid = FID(fid_stats_file=fid_stats_path, dims=2048, n_images=len(dataloader.dataset), device=device_obj)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device_str)
            # Convert to RGB if needed
            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)
            # NOTE: Use fid.update() which handles normalization and RGB conversion
            # This calls fid.fid.update() with is_real=False (for synthetic/generated images)
            fid.update(images, (0, 0))  # Second param is ignored, kept for API compatibility

    fid_score = fid.finalize()
    return fid_score


def compute_evaluation_metrics(
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

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device_str)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_images.append(images.cpu())

    all_preds = torch.cat(all_preds)
    all_images = torch.cat(all_images)

    # Compute softmax probabilities for entropy calculation
    softmax_preds = torch.softmax(all_preds, dim=1)

    # Compute entropy metric
    entropy = compute_entropy(softmax_preds)

    metrics = {
        "entropy": entropy,
    }

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


def extract_features(images: torch.Tensor, device: DeviceType | str, extractor: Any = None) -> np.ndarray:
    """
    Extract features from images using DINO ViT S/8.

    Args:
        images: Tensor of images with shape (batch_size, channels, height, width)
        device: Device to use for computation
        extractor: Optional cached extractor model. If None, creates a new one.

    Returns:
        Feature array with shape (batch_size, feature_dim)

    """
    try:
        # Convert device to string for PyTorch operations
        device_str = device.value if isinstance(device, DeviceType) else str(device)

        if extractor is None:
            extractor = ExtractorFactory.model_from_name(name="dino_vits8")
            extractor = extractor.to(device_str)  # Move model to device
            extractor.eval()
        else:
            extractor = extractor.to(device_str)

        features_list = []

        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Convert to RGB if needed
            if batch_images.shape[1] != 3:
                batch_images = batch_images.repeat(1, 3, 1, 1)

            # Normalize to [0, 1]
            batch_images = (batch_images + 1.0) / 2.0
            batch_images = batch_images.clamp(0, 1)

            with torch.no_grad():
                features = extractor(batch_images.to(device_str)).cpu().numpy()
            features_list.append(features)

        return np.concatenate(features_list, axis=0)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Failed to extract features: %s", e)
        raise


def extract_training_features(config: "CLAmbiguityArgs", device: DeviceType | str) -> tuple[np.ndarray, object]:
    """
    Extract features from the real training dataset to use as reference for pymdma.

    Args:
        config: Configuration containing dataroot and training dataset info
        device: Device to use for computation

    Returns:
        Tuple of (feature array from training dataset, reusable extractor model)

    """
    training_dataset_str = config.training_dataset

    logger.info("Extracting features from training dataset (%s) for pymdma reference...", training_dataset_str)

    # Load training dataset
    train_dataset, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=DatasetNames(training_dataset_str),
            train=True,  # Load training set
            pytesting=False,
            pos_class=None,
            neg_class=None,
        )
    )

    # Sample down to 10k images for efficiency
    n_samples = 10000
    if hasattr(train_dataset, "__len__"):
        total_samples: int | float = len(train_dataset)
    else:
        # If dataset doesn't have len, we'll load all
        total_samples = float("inf")

    if total_samples > n_samples:
        logger.info(f"Sampling {n_samples} images from {total_samples} total")
        indices = np.random.choice(int(total_samples), size=n_samples, replace=False)
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


def compute_pymdma_metrics_from_images(
    eval_images: torch.Tensor,
    real_features: np.ndarray,
    device: DeviceType | str,
    extractor: Any = None,
    sample_size: int | None = None,
) -> dict[str, float]:
    """
    Compute pymdma metrics comparing synthetic evaluation dataset against real training features.

    Args:
        eval_images: Synthetic evaluation dataset images
        real_features: Feature array from real training dataset (reference)
        device: Device to use for computation
        extractor: Optional cached extractor model to avoid recreating it
        sample_size: Optional limit on evaluation images for faster metrics (None = use all)

    Returns:
        Dictionary with pymdma metrics

    """
    logger.info("Extracting features from evaluation dataset...")

    # Sample evaluation images if size limit specified
    if sample_size is not None and len(eval_images) > sample_size:
        indices = np.random.choice(len(eval_images), size=sample_size, replace=False)
        eval_images = eval_images[indices]
        logger.info("Sampled %d evaluation images (full set: %d)", sample_size, len(eval_images))

    # Extract features from synthetic evaluation dataset (reuse extractor if provided)
    synt_features = extract_features(eval_images, device, extractor)

    # Calculate pymdma metrics comparing synthetic vs real
    logger.info("Computing pymdma metrics (synthetic vs real training distribution)...")
    pymdma_df = calculate_pymdma_metrics(real_features, synt_features)

    # Convert DataFrame to dictionary
    metrics_dict = {}
    for col in pymdma_df.columns:
        metrics_dict[col] = pymdma_df[col].values[0]

    return metrics_dict


def compute_dataset_metrics(  # pylint: disable=too-many-locals,too-many-statements  # noqa: C901
    config: "CLAmbiguityArgs",
    dataset: DatasetNames,
    fid_stats_path: str | None = None,
    real_features: np.ndarray | None = None,
    extractor: Any = None,
) -> tuple[float | None, dict[str, float]]:
    """
    Compute FID and pymdma metrics for a dataset (dataset-level, independent of classifier).

    These metrics measure image quality/distribution and should be constant across all classifiers.

    Args:
        config: Configuration
        dataset: Dataset to compute metrics for
        fid_stats_path: Path to FID reference stats
        real_features: Reference features for pymdma
        extractor: Cached feature extractor

    Returns:
        Tuple of (fid_score, pymdma_metrics_dict)

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
                train=False,
                pytesting=False,
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to load dataset %s: %s", dataset_str, e)
        return None, {}

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Compute FID
    fid_score = None
    if fid_stats_path is not None:
        try:
            fid_score = compute_fid_metric(None, test_dataloader, config.device, fid_stats_path)
            logger.info("    ✓ FID score: %.4f", fid_score)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to compute FID: %s", e)

    # Compute pymdma metrics
    pymdma_metrics = {}
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
            for metric_name in PYMDMA_METRIC_NAMES:
                if metric_name in pymdma_metrics:
                    logger.info("    ✓ %s: %.4f", metric_name, pymdma_metrics[metric_name])
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to compute pymdma metrics: %s", e)

    return fid_score, pymdma_metrics


def evaluate_single_model(  # pylint: disable=too-many-locals,unused-argument,too-many-statements  # noqa: C901
    config: "CLAmbiguityArgs",
    classifier: ClassifierType,
    dataset: DatasetNames,
    real_features: np.ndarray | None,
    extractor: Any = None,
    dataset_fid: float | None = None,
    dataset_pymdma: dict | None = None,
) -> bool:
    """
    Evaluate a single model on a single dataset.

    Computes classifier-specific metrics (entropy, uncertainty).
    Uses pre-computed dataset-level metrics (FID, pymdma) which are identical across classifiers.

    Args:
        config: Configuration with models and dataset info
        classifier: Classifier type to evaluate
        dataset: Dataset to evaluate on
        real_features: Optional reference features (kept for backward compatibility)
        extractor: Optional cached feature extractor (kept for backward compatibility)
        dataset_fid: Pre-computed FID score for this dataset (dataset-level metric, same for all classifiers)
        dataset_pymdma: Pre-computed pymdma metrics for this dataset (dataset-level metrics, same for all classifiers)

    Returns:
        Boolean indicating if evaluation succeeded (True) or failed (False)

    """
    if dataset_pymdma is None:
        dataset_pymdma = {}

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

        # Load dataset for entropy computation
        logger.info("  Loading %s dataset...", dataset_str)
        test_dataloader = load_datasets_for_evaluation(config, dataset, batch_size=32)

        # Compute classifier-specific metrics (entropy - depends on model predictions)
        logger.info("  Computing classifier-specific metrics...")

        # Convert device to string for PyTorch operations
        device_str = config.device.value if isinstance(config.device, DeviceType) else str(config.device)

        all_preds = []
        with torch.no_grad():
            for images, _ in test_dataloader:
                images = images.to(device_str)
                outputs = model(images)
                all_preds.append(outputs.cpu())

        all_preds = torch.cat(all_preds)
        softmax_preds = torch.softmax(all_preds, dim=1)
        entropy = compute_entropy(softmax_preds)

        # Combine metrics: classifier-specific entropy + dataset-level FID/pymdma
        metrics = {
            "entropy": entropy,
        }

        # Add pre-computed dataset-level metrics (same for all classifiers)
        if dataset_fid is not None:
            metrics["fid"] = dataset_fid
        metrics.update(dataset_pymdma)

        # Log results
        log_msg = f"  ✓ Metrics computed - Entropy: {metrics['entropy']:.4f}"
        if "fid" in metrics:
            log_msg += f", FID: {metrics['fid']:.4f}"

        for metric_name in PYMDMA_METRIC_NAMES:
            if metric_name in metrics:
                log_msg += f", {metric_name}: {metrics[metric_name]:.4f}"
        logger.info(log_msg)

        # Log metrics to wandb
        log_data = {
            "classifier": classifier_str,
            "dataset": dataset_str,
            "entropy": metrics["entropy"],
        }
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
            level="error",
        )
        wandb.finish(exit_code=1)
        return False


def run_evaluation_loop(
    config: "CLAmbiguityArgs",
    eval_datasets: list,
    real_features: np.ndarray | None,
    extractor: Any = None,
) -> None:
    """
    Run the evaluation loop for all models on all datasets.

    Computes dataset-level metrics (FID, pymdma) once per dataset, then evaluates all classifiers
    on that dataset using the cached metrics.

    Args:
        config: Configuration with models and dataset info
        eval_datasets: List of datasets to evaluate on
        real_features: Optional reference features for pymdma metrics
        extractor: Optional cached feature extractor to avoid recreating it

    """
    total_steps = len(eval_datasets) * (1 + len(config.models))  # 1 dataset-level + N classifiers
    current_step = 0

    for dataset in eval_datasets:
        dataset_str = _enum_to_str(dataset)

        # Step 1: Compute dataset-level metrics once (FID + pymdma)
        current_step += 1
        logger.info(f"\n[{current_step}/{total_steps}] Computing dataset-level metrics for {dataset_str}")

        fid_stats_path = find_fid_stats(config.dataroot, dataset_str)
        dataset_fid, dataset_pymdma = compute_dataset_metrics(
            config, dataset, fid_stats_path=fid_stats_path, real_features=real_features, extractor=extractor
        )

        # Step 2: Evaluate each classifier on this dataset using cached metrics
        for classifier in config.models:
            current_step += 1
            classifier_str = _enum_to_str(classifier)
            logger.info(f"\n[{current_step}/{total_steps}] Evaluating {classifier_str} on {dataset_str}")
            evaluate_single_model(config, classifier, dataset, real_features, extractor, dataset_fid, dataset_pymdma)


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

    # Phase 1.5: Generate FID statistics for evaluation datasets if needed
    logger.info(separator)
    logger.info("PHASE 1.5: Generating FID Statistics")
    logger.info(separator)

    for dataset in eval_datasets:
        dataset_str = _enum_to_str(dataset)
        try:
            logger.info(f"Generating FID statistics for {dataset_str}...")

            # Determine the reference dataset for FID stats
            # For synthetic datasets (companion-*, ambiguous-*), use the training dataset as reference
            if "companion-" in dataset_str or "ambiguous-" in dataset_str:
                ref_dataset = config.training_dataset
                use_test_set = True  # Use test set from reference dataset
            else:
                ref_dataset = dataset_str
                use_test_set = False  # For non-synthetic, use training set

            # Always use 10k samples for consistent pymdma metrics comparison
            stats_file = generate_fid_stats(
                dataroot=config.dataroot,
                dataset_name=ref_dataset,
                batch_size=64,
                num_workers=6,
                device=config.device,
                use_test_set=use_test_set,
                n_samples=10000,
            )

            logger.info(f"✓ FID statistics generated: {stats_file}")

            # If using a reference dataset (synthesis case), create a symlink so find_fid_stats finds it
            if ref_dataset != dataset_str:
                stats_dir = Path(config.dataroot) / "fid-stats"
                ref_stats_file = stats_dir / f"stats.{ref_dataset}.npz"
                eval_stats_file = stats_dir / f"stats.{dataset_str}.npz"
                if ref_stats_file.exists() and not eval_stats_file.exists():
                    logger.info(f"Linking {ref_stats_file.name} -> {eval_stats_file.name}")
                    eval_stats_file.symlink_to(ref_stats_file)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"✗✗✗ CRITICAL: Failed to generate FID statistics for {dataset_str}: {e}")
            logger.error("✗✗✗ FID metrics will NOT be computed!")
            logger.error(f"✗✗✗ Exception details: {type(e).__name__}")
            import traceback  # pylint: disable=import-outside-toplevel

            logger.error(f"✗✗✗ Traceback: {traceback.format_exc()}")
            raise  # Fail hard so user knows something is wrong

    # Phase 2: Evaluate each model on each dataset
    logger.info(separator)
    logger.info("PHASE 2: Evaluation")
    logger.info(separator)

    # Extract real training features once for pymdma comparison
    logger.info("Extracting reference features from training dataset...")
    extractor = None
    try:
        real_features, extractor = extract_training_features(config, config.device)
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("Failed to extract training features for pymdma: %s", e)
        real_features = None
        extractor = None

    run_evaluation_loop(config, eval_datasets, real_features, extractor)

    logger.info(separator)
    logger.info("Ambiguity evaluation complete")


if __name__ == "__main__":
    main()
