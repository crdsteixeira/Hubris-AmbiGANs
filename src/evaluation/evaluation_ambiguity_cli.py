"""CLI for ambiguity evaluation."""

import argparse
import logging
import os
from typing import Any

import numpy as np
import torch
import wandb
import yaml

from src.classifier.multiclass_train_utils import train_single_multiclass_classifier
from src.datasets.load import extract_ground_truth_labels, load_datasets_for_evaluation
from src.enums import ClassifierType, DatasetNames, DeviceType
from src.metrics.ambiguity import compute_entropy, compute_top_pairs
from src.metrics.evaluation_metrics import (
    PYMDMA_METRIC_NAMES,
    extract_training_features,
    get_model_predictions,
)
from src.metrics.image_quality import (
    compute_fid_metric,
    compute_pymdma_metrics_from_images,
    find_fid_stats,
    generate_fid_stats,
)
from src.models import CLAmbiguityArgs, ConfigTrainingParams
from src.utils.checkpoint import construct_classifier_from_checkpoint, find_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import setup_reprod

configure_logging()
logger = logging.getLogger(__name__)


def _enum_to_str(value: Any) -> str:
    """Convert enum to string value, handling both enum and string inputs."""
    return value.value if hasattr(value, "value") else str(value)


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


def ensure_models_trained(config: "CLAmbiguityArgs", training_dataset: str) -> None:
    """
    Ensure all models are trained (but don't keep them in memory).

    Args:
        config: Configuration containing models and seed info
        training_dataset: Dataset to train on

    """
    seed = config.seed if config.seed is not None else 42

    for classifier in config.models:
        # Check if checkpoint exists
        classifier_str = _enum_to_str(classifier)
        checkpoint_path = find_checkpoint(config.out_dir, training_dataset, classifier_str, seed)

        if checkpoint_path is None:
            logger.info(
                "Training %s on %s with epochs=%d...",
                classifier_str,
                training_dataset,
                config.training_params.epochs,
            )
            train_single_multiclass_classifier(
                dataset_name=training_dataset,
                classifier_type=classifier,
                data_dir=config.dataroot,
                out_dir=config.out_dir,
                batch_size=config.training_params.batch_size,
                epochs=config.training_params.epochs,
                device=config.device,
                seed=config.seed,
            )
        else:
            logger.info("Found trained %s on %s: %s", classifier_str, training_dataset, checkpoint_path)


def _get_or_generate_fid_stats(
    config: "CLAmbiguityArgs",
    dataset_str: str,
    fid_stats_path: str | None,
    dataset_size: int,
) -> str | None:
    """Get FID stats path, generating if needed."""
    if fid_stats_path and os.path.exists(fid_stats_path):
        return fid_stats_path

    if dataset_str == config.training_dataset:
        return None

    try:
        logger.info(f"  Generating FID statistics for {dataset_str}...")
        stats_path = generate_fid_stats(
            dataroot=config.dataroot,
            dataset_name=config.training_dataset,
            batch_size=64,
            num_workers=6,
            device=config.device,
            use_test_set=False,
            n_samples=dataset_size,
        )
        logger.info(f"  ✓ FID statistics generated: {stats_path}")
        return stats_path
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to generate FID statistics for {dataset_str}: {e}")
        return None


def _compute_fid_score(fid_stats_path: str | None, test_dataloader: Any, config: "CLAmbiguityArgs") -> float | None:
    """Compute FID score if stats path is available."""
    if fid_stats_path is None:
        return None

    try:
        fid_score = compute_fid_metric(None, test_dataloader, config.device, fid_stats_path)
        logger.info("    ✓ FID score: %.4f", fid_score)
        return fid_score
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to compute FID: %s", e)
        return None


def _compute_pymdma_metrics(
    test_dataloader: Any, real_features: np.ndarray, config: "CLAmbiguityArgs", extractor: Any
) -> dict[str, float]:
    """Compute pymdma metrics."""
    try:
        all_images = []
        with torch.no_grad():
            for images, _ in test_dataloader:
                all_images.append(images)
        all_images = torch.cat(all_images)

        pymdma_metrics = compute_pymdma_metrics_from_images(
            all_images, real_features, config.device, extractor, sample_size=None
        )
        for metric_name in PYMDMA_METRIC_NAMES:
            if metric_name in pymdma_metrics:
                logger.info("    ✓ %s: %.4f", metric_name, pymdma_metrics[metric_name])
        return pymdma_metrics
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to compute pymdma metrics: %s", e)
        return {}


def compute_dataset_metrics(
    config: "CLAmbiguityArgs",
    dataset: DatasetNames,
    fid_stats_path: str | None = None,
    real_features: np.ndarray | None = None,
    extractor: Any = None,
) -> tuple[float | None, dict[str, float]]:
    """
    Compute FID and pymdma metrics for a dataset (dataset-level, independent of classifier).

    Args:
        config: Configuration
        dataset: Dataset to compute metrics for
        fid_stats_path: Path to FID reference stats
        real_features: Reference features for pymdma
        extractor: Cached feature extractor

    Returns:
        Tuple of (fid_score, dataset_metrics_dict)

    """
    dataset_str = _enum_to_str(dataset)
    logger.info("  Computing dataset-level metrics for %s...", dataset_str)

    # Load dataset
    try:
        test_dataloader, _ = load_datasets_for_evaluation(config.dataroot, dataset, config.balanced, batch_size=32)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to load dataset %s: %s", dataset_str, e)
        return None, {}

    # Get or generate FID statistics
    fid_stats_path = _get_or_generate_fid_stats(config, dataset_str, fid_stats_path, len(test_dataloader.dataset))

    # Compute FID score
    fid_score = _compute_fid_score(fid_stats_path, test_dataloader, config)

    # Compute pymdma metrics
    dataset_metrics = {}
    if real_features is not None:
        dataset_metrics = _compute_pymdma_metrics(test_dataloader, real_features, config, extractor)

    return fid_score, dataset_metrics


def _compute_classifier_metrics(
    model: Any, test_dataloader: Any, config: "CLAmbiguityArgs", classifier_str: str, dataset_str: str
) -> dict[str, float]:
    """Compute entropy and top_pairs metrics for a classifier."""
    metrics = {}
    softmax_preds = get_model_predictions(model, test_dataloader, config.device)

    entropy = compute_entropy(softmax_preds)
    logger.info(f"    ✓ Computed entropy for {classifier_str} on {dataset_str}: {entropy:.6f}")
    metrics["entropy"] = entropy

    ground_truth = extract_ground_truth_labels(test_dataloader)
    if ground_truth is not None:
        top_pairs = compute_top_pairs(softmax_preds, ground_truth)
        metrics["top_pairs"] = top_pairs
        logger.info(f"    ✓ Computed top_pairs for {classifier_str} on {dataset_str}: {top_pairs:.6f}")

    return metrics


def _log_evaluation_results(classifier_str: str, metrics: dict[str, float]) -> None:
    """Log evaluation results to console and wandb."""
    log_msg = f"  ✓ Metrics computed - Entropy: {metrics['entropy']:.4f}"
    if "top_pairs" in metrics:
        log_msg += f", Top Pairs: {metrics['top_pairs']:.4f}"
    if "fid" in metrics:
        log_msg += f", FID: {metrics['fid']:.4f}"

    for metric_name in PYMDMA_METRIC_NAMES:
        if metric_name in metrics:
            log_msg += f", {metric_name}: {metrics[metric_name]:.4f}"
    logger.info(log_msg)

    log_data = {"classifier": classifier_str, "entropy": metrics["entropy"]}
    if "top_pairs" in metrics:
        log_data["top_pairs"] = metrics["top_pairs"]
    if "fid" in metrics:
        log_data["fid"] = metrics["fid"]
    for metric_name in PYMDMA_METRIC_NAMES:
        if metric_name in metrics:
            log_data[metric_name] = metrics[metric_name]
    wandb.log(log_data)


def evaluate_single_model(
    config: "CLAmbiguityArgs",
    classifier: ClassifierType,
    dataset: DatasetNames,
    dataset_fid: float | None = None,
    dataset_metrics: dict | None = None,
) -> bool:
    """
    Evaluate a single model on a single dataset.

    Args:
        config: Configuration with models and dataset info
        classifier: Classifier type to evaluate
        dataset: Dataset to evaluate on
        dataset_fid: Pre-computed FID score for this dataset
        dataset_metrics: Pre-computed dataset-level metrics

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
        # Load model
        logger.info("  Loading %s model...", classifier_str)
        seed = config.seed if config.seed is not None else 42
        checkpoint_path = find_checkpoint(config.out_dir, config.training_dataset, classifier_str, seed)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for {classifier_str} with seed {seed} in {config.out_dir}/{config.training_dataset}"
            )
        model, _, _, _, _ = construct_classifier_from_checkpoint(str(checkpoint_path), config.device)
        model.eval()
        logger.info("  ✓ Model loaded")

        # Load dataset
        logger.info("  Loading %s dataset...", dataset_str)
        test_dataloader, dataset_companion_metrics = load_datasets_for_evaluation(
            config.dataroot, dataset, config.balanced, batch_size=32
        )
        logger.info("  ✓ Dataset %s loaded - %d samples", dataset_str, len(test_dataloader.dataset))

        # Compute classifier-specific metrics
        logger.info("  Computing classifier-specific metrics...")
        metrics = _compute_classifier_metrics(model, test_dataloader, config, classifier_str, dataset_str)

        # Add dataset-level metrics
        if dataset_fid is not None:
            metrics["fid"] = dataset_fid
        metrics.update(dataset_metrics)
        metrics.update(dataset_companion_metrics)

        # Log results
        _log_evaluation_results(classifier_str, metrics)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        logger.info("  ✓ Evaluation complete")
        wandb.finish(exit_code=0)
        return True

    except (OSError, FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("  ✗ Evaluation failed: %s", e)
        wandb.log({"classifier": classifier_str, "dataset": dataset_str, "error": str(e)})
        wandb.alert(  # type: ignore[attr-defined]
            title=f"Evaluation Failed: {classifier_str} on {dataset_str}",
            text=f"Error: {str(e)}",
            level="ERROR",
        )
        wandb.finish(exit_code=1)
        return False


def run_evaluation_loop(config: "CLAmbiguityArgs", eval_datasets: list) -> None:
    """
    Run the evaluation loop for all models on all datasets.

    Args:
        config: Configuration with models and dataset info
        eval_datasets: List of datasets to evaluate on

    """
    total_steps = len(eval_datasets) * (1 + len(config.models))
    current_step = 0
    extractor = None

    for dataset in eval_datasets:
        dataset_str = _enum_to_str(dataset)

        # Skip FID/pymdma for training dataset
        if dataset_str == config.training_dataset:
            logger.info(f"\nEvaluating on test dataset {dataset_str} (skipping FID/pymdma metrics)")
            dataset_fid = None
            dataset_metrics: dict[str, Any] = {}
            real_features = None
        else:
            # Extract training features for evaluation dataset
            try:
                real_features, extractor = extract_training_features(
                    config.dataroot, config.training_dataset, config.device, dataset_str
                )
            except (RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to extract training features for %s: %s", dataset_str, e)
                real_features = None

            # Compute dataset-level metrics
            current_step += 1
            logger.info(f"\n[{current_step}/{total_steps}] Computing dataset-level metrics for {dataset_str}")

            fid_stats_path = find_fid_stats(config.dataroot, dataset_str)
            dataset_fid, dataset_metrics = compute_dataset_metrics(
                config, dataset, fid_stats_path=fid_stats_path, real_features=real_features, extractor=extractor
            )

        # Evaluate each classifier on this dataset
        for classifier in config.models:
            current_step += 1
            classifier_str = _enum_to_str(classifier)
            logger.info(f"\n[{current_step}/{total_steps}] Evaluating {classifier_str} on {dataset_str}")
            evaluate_single_model(config, classifier, dataset, dataset_fid, dataset_metrics)


def main() -> None:
    """Entry point for ambiguity evaluation."""
    logger.info("Ambiguity evaluation is starting...")

    config = parse_args()
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    # Determine evaluation datasets
    eval_datasets = config.datasets if config.datasets else [DatasetNames(config.training_dataset)]

    # Phase 1: Check and train models
    separator = "=" * 80
    logger.info(separator)
    logger.info("PHASE 1: Checking/Training Models")
    logger.info(separator)

    ensure_models_trained(config, config.training_dataset)
    logger.info("All models ready for evaluation")

    # Phase 2: Evaluate models
    logger.info(separator)
    logger.info("PHASE 2: Evaluation")
    logger.info(separator)

    run_evaluation_loop(config, eval_datasets)

    logger.info(separator)
    logger.info("Ambiguity evaluation complete")


if __name__ == "__main__":
    main()
