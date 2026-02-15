"""CLI for ambiguity evaluation."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from pydantic import ValidationError
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader

from src.classifier.multiclass_train_utils import train_single_multiclass_classifier
from src.datasets.load import load_dataset
from src.enums import ClassifierType, DatasetNames, DeviceType
from src.metrics.fid.fid import FID
from src.models import CLAmbiguityArgs, LoadDatasetParams
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.read_config import read_training_config
from src.utils.utility_functions import calculate_pymdma_metrics, setup_reprod

configure_logging()
logger = logging.getLogger(__name__)


def parse_args() -> CLAmbiguityArgs:
    """Parse and validate command-line arguments from config file."""
    parser = argparse.ArgumentParser(description="Run ambiguity evaluation")
    parser.add_argument(
        "config", help="Path to YAML config file (e.g., ambiguity-evaluation/advanced_mnist_config.yaml)"
    )

    args = parser.parse_args()

    # Load config from YAML file
    config_data = read_training_config(args.config)

    # Extract values from config
    args_dict = {
        "dataroot": config_data.data_dir,
        "out_dir": config_data.out_dir,
        "models": config_data.classifiers,
        "datasets": [],  # datasets.evaluation is optional
        "device": config_data.device,
        "seed": config_data.seed,
        "training_dataset": config_data.dataset,
    }

    try:
        return CLAmbiguityArgs.model_validate(args_dict)
    except ValidationError as exc:
        logger.error("Argument validation error: %s", exc)
        raise


def find_checkpoint(out_dir: str, classifier_type: str, seed: int) -> Path | None:
    """
    Find checkpoint file for a trained classifier.

    Args:
        out_dir: Output directory where models are stored
        classifier_type: Type of classifier (e.g., 'cnn', 'vgg16')
        seed: Random seed used for training

    Returns:
        Path to checkpoint if found, None otherwise

    """
    checkpoint_pattern = f"{classifier_type}_{seed}_*.pt"
    checkpoint_dir = Path(out_dir)

    if not checkpoint_dir.exists():
        return None

    for checkpoint_file in checkpoint_dir.glob(checkpoint_pattern):
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
        classifier_str = classifier.value if isinstance(classifier, ClassifierType) else str(classifier)

        # Check if checkpoint exists
        checkpoint_path = find_checkpoint(config.out_dir, classifier_str, seed)

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


def load_model_for_evaluation(
    config: "CLAmbiguityArgs",
    classifier_type: str,
) -> torch.nn.Module:
    """
    Load a single model checkpoint for evaluation.

    Args:
        config: Configuration containing output directory and device info
        classifier_type: Type of classifier to load

    Returns:
        Loaded model

    """
    seed = config.seed if config.seed is not None else 42
    checkpoint_path = find_checkpoint(config.out_dir, classifier_type, seed)

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found for {classifier_type} with seed {seed}")

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
    # Get the string value from enum
    dataset_str = dataset_name.value if isinstance(dataset_name, DatasetNames) else str(dataset_name)

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
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    fid_stats_path: str,
) -> float:
    """
    Compute FID metric from model predictions.

    Args:
        model: Trained classifier model
        dataloader: DataLoader with test data
        device: Device to use for computation
        fid_stats_path: Path to FID statistics file

    Returns:
        FID score

    """
    model.eval()

    # Initialize FID metric with reference statistics
    device_type = DeviceType(device) if isinstance(device, str) else device
    fid = FID(fid_stats_file=fid_stats_path, dims=2048, n_images=len(dataloader.dataset), device=device_type)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            # Convert to RGB if needed
            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)
            # Update FID with generated images
            fid.update(images, (0, 0))

    fid_score = fid.finalize()
    return fid_score


def compute_evaluation_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    fid_stats_path: str | None = None,
    real_features: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute evaluation metrics for a model and dataset.

    Args:
        model: Trained classifier model
        dataloader: DataLoader with test data
        device: Device to use for computation
        fid_stats_path: Optional path to FID statistics file
        real_features: Optional reference features from real training dataset for pymdma comparison

    Returns:
        Dictionary with metric names and values

    """
    model.eval()
    all_preds = []
    all_images = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
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
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.warning("Failed to compute FID: %s", e)

    # Compute pymdma metrics if real features are available
    if real_features is not None:
        try:
            pymdma_metrics = compute_pymdma_metrics_from_images(all_images, real_features, device)
            metrics.update(pymdma_metrics)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to compute pymdma metrics: %s", e)

    return metrics


def extract_features(images: torch.Tensor, device: str) -> np.ndarray:
    """
    Extract features from images using DINO ViT S/8.

    Args:
        images: Tensor of images with shape (batch_size, channels, height, width)
        device: Device to use for computation

    Returns:
        Feature array with shape (batch_size, feature_dim)

    """
    try:
        extractor = ExtractorFactory.model_from_name(name="dino_vits8")
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
                features = extractor(batch_images.to(device)).cpu().numpy()
            features_list.append(features)

        return np.concatenate(features_list, axis=0)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Failed to extract features: %s", e)
        raise


def extract_training_features(config: "CLAmbiguityArgs", device: str) -> np.ndarray:
    """
    Extract features from the real training dataset to use as reference for pymdma.

    Args:
        config: Configuration containing dataroot and training dataset info
        device: Device to use for computation

    Returns:
        Feature array from training dataset

    """
    # Get the string value from enum
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

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Collect all training images
    all_train_images = []
    with torch.no_grad():
        for images, _ in train_dataloader:
            all_train_images.append(images)

    all_train_images = torch.cat(all_train_images)
    logger.info("  ✓ Loaded %d training images", len(all_train_images))

    # Extract features
    real_features = extract_features(all_train_images, device)
    logger.info("  ✓ Extracted features: shape %s", real_features.shape)

    return real_features


def compute_pymdma_metrics_from_images(
    eval_images: torch.Tensor,
    real_features: np.ndarray,
    device: str,
) -> dict[str, float]:
    """
    Compute pymdma metrics comparing synthetic evaluation dataset against real training features.

    Args:
        eval_images: Synthetic evaluation dataset images
        real_features: Feature array from real training dataset (reference)
        device: Device to use for computation

    Returns:
        Dictionary with pymdma metrics

    """
    logger.info("Extracting features from evaluation dataset...")

    # Extract features from synthetic evaluation dataset
    synt_features = extract_features(eval_images, device)

    # Calculate pymdma metrics comparing synthetic vs real
    logger.info("Computing pymdma metrics (synthetic vs real training distribution)...")
    pymdma_df = calculate_pymdma_metrics(real_features, synt_features)

    # Convert DataFrame to dictionary
    metrics_dict = {}
    for col in pymdma_df.columns:
        metrics_dict[col] = pymdma_df[col].values[0]

    return metrics_dict


def evaluate_single_model(  # pylint: disable=too-many-locals
    config: "CLAmbiguityArgs",
    classifier: ClassifierType,
    dataset: DatasetNames,
    real_features: np.ndarray | None,
) -> None:
    """
    Evaluate a single model on a single dataset.

    Args:
        config: Configuration with models and dataset info
        classifier: Classifier type to evaluate
        dataset: Dataset to evaluate on
        real_features: Optional reference features for pymdma metrics

    """
    classifier_str = classifier.value if isinstance(classifier, ClassifierType) else str(classifier)
    dataset_str = dataset.value if isinstance(dataset, DatasetNames) else str(dataset)

    try:
        # Load model (memory-efficient: one at a time)
        logger.info("  Loading %s model...", classifier_str)
        model = load_model_for_evaluation(config, classifier_str)
        model.eval()

        # Load dataset
        logger.info("  Loading %s dataset...", dataset_str)
        test_dataloader = load_datasets_for_evaluation(config, dataset, batch_size=32)

        # Check for FID stats
        fid_stats_path = find_fid_stats(config.dataroot, dataset_str)
        if fid_stats_path is not None:
            logger.info("  Found FID stats: %s", fid_stats_path)
        else:
            logger.debug("  No FID stats found for %s", dataset_str)

        # Compute metrics
        logger.info("  Computing evaluation metrics...")
        metrics = compute_evaluation_metrics(model, test_dataloader, config.device, fid_stats_path, real_features)

        # Log results
        log_msg = f"  ✓ Metrics computed - Entropy: {metrics['entropy']:.4f}"
        if "fid" in metrics:
            log_msg += f", FID: {metrics['fid']:.4f}"

        pymdma_metric_names = [
            "improved_precision",
            "improved_recall",
            "giqa_qs",
            "giqa_ds",
            "density",
            "coverage",
            "msid",
        ]
        for metric_name in pymdma_metric_names:
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
        for metric_name in pymdma_metric_names:
            if metric_name in metrics:
                log_data[metric_name] = metrics[metric_name]
        wandb.log(log_data)

        # Unload model to free memory
        del model
        torch.cuda.empty_cache()
        logger.info("  ✓ Evaluation complete")

    except (OSError, FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("  ✗ Evaluation failed: %s", e)
        wandb.log(
            {
                "classifier": classifier_str,
                "dataset": dataset_str,
                "error": str(e),
            }
        )


def run_evaluation_loop(
    config: "CLAmbiguityArgs",
    eval_datasets: list,
    real_features: np.ndarray | None,
) -> None:
    """
    Run the evaluation loop for all models on all datasets.

    Args:
        config: Configuration with models and dataset info
        eval_datasets: List of datasets to evaluate on
        real_features: Optional reference features for pymdma metrics

    """
    total_evaluations = len(config.models) * len(eval_datasets)
    current_eval = 0

    for dataset in eval_datasets:
        for classifier in config.models:
            current_eval += 1
            classifier_str = classifier.value if isinstance(classifier, ClassifierType) else str(classifier)
            dataset_str = dataset.value if isinstance(dataset, DatasetNames) else str(dataset)

            logger.info("\n[%d/%d] Evaluating %s on %s", current_eval, total_evaluations, classifier_str, dataset_str)
            evaluate_single_model(config, classifier, dataset, real_features)


def main() -> None:
    """Entry point for ambiguity evaluation."""
    logger.info("Ambiguity evaluation is starting...")

    config = parse_args()
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    # Initialize wandb for logging
    wandb.init(
        project="ambiguity-evaluation",
        name=f"eval-models-{config.training_dataset}",
        config={
            "training_dataset": config.training_dataset,
            "models": [str(m) for m in config.models],
            "seed": config.seed,
            "device": str(config.device),
        },
    )

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

    # Extract real training features once for pymdma comparison
    logger.info("Extracting reference features from training dataset...")
    try:
        real_features = extract_training_features(config, config.device)
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("Failed to extract training features for pymdma: %s", e)
        real_features = None

    run_evaluation_loop(config, eval_datasets, real_features)

    logger.info(separator)
    logger.info("Ambiguity evaluation complete")

    wandb.finish()


if __name__ == "__main__":
    main()
