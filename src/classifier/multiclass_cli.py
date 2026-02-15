"""CLI interface for multiclass classifier training."""

import argparse
import logging
import os
import traceback

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from src.classifier.multiclass_train_utils import train_single_multiclass_classifier
from src.models import CLMulticlassTrain, ConfigTrainingParams
from src.utils.logging import configure_logging
from src.utils.read_config import read_training_config
from src.utils.utility_functions import setup_reprod

configure_logging()
logger = logging.getLogger(__name__)


def parse_args() -> CLMulticlassTrain:
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Train AmbiGAN with a config file")
    parser.add_argument("--config", type=str, dest="config_path", required=True, help="Config file")

    # Parse the arguments from command line
    args = parser.parse_args()
    # Convert argparse Namespace to dictionary for validation
    args_dict = vars(args)

    # Validate parsed arguments using Pydantic model
    try:
        validated_args = CLMulticlassTrain.model_validate(args_dict)
        return validated_args
    except ValidationError as e:
        # Print validation error and exit
        logger.error(f"Validation error: {e}")
        raise


def main() -> None:  # pylint: disable=too-many-statements,broad-exception-caught
    """Train multiple classifier architectures on a single dataset."""
    load_dotenv()

    logger.info("Training classifiers is starting...")

    args = parse_args()

    config = read_training_config(args.config_path)
    logger.info(f"Loaded experiment configuration from {args.config_path}")

    # Load configuration
    dataset = config.dataset
    classifiers = config.classifiers

    # Set random seed
    seed = config.seed
    if seed is None:
        seed = np.random.randint(100000)
    setup_reprod(seed)
    logger.info(f" > Seed: {seed}")

    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)

    # Store results
    results = []
    failed_count = 0

    total_combinations = len(classifiers)

    logger.info(f"\n{'#'*80}")
    logger.info(f"TRAINING {total_combinations} CLASSIFIERS ON {dataset.upper()}")
    logger.info(f"Classifiers: {classifiers}")
    logger.info(f"{'#'*80}\n")

    # Train all classifiers on the dataset
    for classifier in classifiers:
        # Get default training parameters from config
        train_params = ConfigTrainingParams(**config.training.model_dump())

        # Override with classifier-specific config if available
        if config.per_classifier_training and classifier in config.per_classifier_training:
            classifier_specific = config.per_classifier_training[classifier]
            classifier_dict = classifier_specific.model_dump(exclude_none=True)
            train_params = ConfigTrainingParams(**{**config.training.model_dump(), **classifier_dict})

        try:
            metrics = train_single_multiclass_classifier(
                dataset_name=dataset,
                classifier_type=classifier,
                data_dir=config.data_dir,
                out_dir=config.out_dir,
                batch_size=train_params.batch_size,
                epochs=train_params.epochs,
                lr=train_params.lr,
                device=config.device,
                seed=seed,
                entity=config.entity,
                project=config.project,
            )

            has_error = "error" in metrics
            results.append(
                {
                    "classifier": classifier,
                    "top1_accuracy": metrics.get("top1_accuracy", 0.0),
                    "top2_accuracy": metrics.get("top2_accuracy", 0.0),
                    "loss": metrics.get("loss", float("inf")),
                    "error": metrics.get("error", None),
                }
            )

            if has_error:
                failed_count += 1

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to train %s on %s: %s", classifier, dataset, e)
            traceback.print_exc()
            results.append(
                {
                    "classifier": classifier,
                    "top1_accuracy": 0.0,
                    "top2_accuracy": 0.0,
                    "loss": float("inf"),
                    "error": str(e),
                }
            )
            failed_count += 1

    # Print summary table
    logger.info(f"\n\n{'='*80}")
    logger.info(f"TRAINING SUMMARY - {total_combinations} CLASSIFIERS ON {dataset.upper()}")
    logger.info(f"{'='*80}\n")

    # Create DataFrame for nice display
    df = pd.DataFrame(results)

    # Display results
    logger.info(f"{'Classifier':<15} | {'Top-1 Acc':<11} | {'Top-2 Acc':<11} | {'Loss':<10}")
    logger.info(f"{'-'*15}-+-{'-'*11}-+-{'-'*11}-+-{'-'*10}")

    for _, row in df.iterrows():
        logger.info(
            f"{row['classifier']:<15} | "
            f"{row['top1_accuracy']:<10.4f} | {row['top2_accuracy']:<10.4f} | {row['loss']:<10.4f}"
        )

    logger.info(f"\n{'='*80}\n")
    logger.info(f"Total trained: {total_combinations - failed_count}/{total_combinations}")
    logger.info(f"Total failed: {failed_count}/{total_combinations}")
    logger.info(f"Models saved to: {config.out_dir}\n")

    # Save results to CSV
    csv_path = os.path.join(config.out_dir, f"{dataset}_training_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
