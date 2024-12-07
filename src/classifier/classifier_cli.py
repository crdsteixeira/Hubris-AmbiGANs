"""Interface for Classifiers and Ensemble."""

import argparse
import logging
import os
from collections.abc import Callable

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader

from src.classifier.construct_classifier import construct_classifier
from src.classifier.train_classifier import evaluate, save_predictions, train
from src.datasets.load import load_dataset
from src.enums import TrainingStage
from src.metrics.accuracy import binary_accuracy, multiclass_accuracy
from src.models import (
    CLTrainArgs,
    EvaluateParams,
    LoadDatasetParams,
    TrainClassifierArgs,
)
from src.utils.checkpoint import checkpoint, construct_classifier_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import generate_cnn_configs, setup_reprod

configure_logging()

logger = logging.getLogger(__name__)


def parse_args() -> CLTrainArgs:
    """Parse command-line arguments using argparse and validate them with Pydantic."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train a classifier with command-line arguments")

    # Add arguments to the parser (matching the fields in CLTrainArgs)
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to generated files")
    parser.add_argument("--name", type=str, help="Name of the classifier for output files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--c_type", type=str, default="mlp", choices=["cnn", "mlp", "ensemble"], help="Classifier type")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")
    parser.add_argument("--early_stop", type=int, help="Early stopping criteria (optional)")
    parser.add_argument("--early_acc", type=float, default=1.0, help="Early accuracy threshold for backpropagation")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument("--nf", type=int, default=2, help="Number of filters or features in the model")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for computation")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--pos_class", type=int, required=True, help="Positive class for binary classification")
    parser.add_argument("--neg_class", type=int, required=True, help="Negative class for binary classification")
    parser.add_argument("--ensemble_type", type=str, required=False, help="Type of ensemble when applicable")
    parser.add_argument(
        "--ensemble_output_method", type=str, required=False, help="Output method for ensemble when applicable"
    )

    # Parse the arguments from command line
    args = parser.parse_args()
    logger.debug(args)

    # Convert argparse Namespace to dictionary for validation
    args_dict = vars(args)

    # Validate parsed arguments using Pydantic model
    try:
        validated_args = CLTrainArgs.model_validate(args_dict)
        return validated_args
    except ValidationError as e:
        # Print validation error and exit
        logger.error(f"Validation error: {e}")
        raise ValidationError(e) from e


def main() -> None:
    """Run process to train classifier and ensembles."""
    load_dotenv()

    # Parse arguments using Pydantic model
    args = parse_args()

    # Set random seed
    args.seed = np.random.randint(100000) if args.seed is None else args.seed
    setup_reprod(args.seed)
    logger.info(f" > Seed: {args.seed}")

    # Setup classifiers list (if applicable for ensemble)
    classifiers_nf: int | list[int] | list[list[int]] | None = args.nf
    if args.ensemble_type is not None:
        classifiers_nf = generate_cnn_configs(args.nf)
    logger.debug(f"Classifiers Features: {classifiers_nf}")

    # Use the device directly from args (already validated by Pydantic)
    logger.info(f" > Using device: {args.device.value}")

    # Load dataset
    dataset, num_classes, img_size = load_dataset(
        LoadDatasetParams(
            dataroot=args.data_dir,
            dataset_name=args.dataset_name,
            pos_class=args.pos_class,
            neg_class=args.neg_class,
            train=True,
            pytesting=False,
        )
    )
    logger.info(f" > Using dataset: {args.dataset_name}")

    # Determine if binary classification
    binary_classification = num_classes == 2
    if binary_classification:
        logger.info(f"\t> Binary classification between {args.pos_class} and {args.neg_class}")
        binary_dataset_dir = f"{args.dataset_name}.{args.pos_class}v{args.neg_class}"

    # Prepare output directory
    out_dir = os.path.join(args.out_dir, binary_dataset_dir)

    # Split dataset into training and validation sets
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(5 / 6 * len(dataset)), len(dataset) - int(5 / 6 * len(dataset))]
    )

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_set, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=args.data_dir,
            dataset_name=args.dataset_name,
            pos_class=args.pos_class,
            neg_class=args.neg_class,
            train=False,
            pytesting=False,
        )
    )

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    args_train_classifier = TrainClassifierArgs(
        type=args.c_type,
        data_dir=args.data_dir,
        out_dir=out_dir,
        name=args.name,
        dataset_name=args.dataset_name,
        pos_class=args.pos_class,
        neg_class=args.neg_class,
        batch_size=args.batch_size,
        c_type=args.c_type,
        epochs=args.epochs,
        early_stop=args.early_stop,
        early_acc=args.early_acc,
        lr=args.lr,
        seed=args.seed,
        nf=classifiers_nf,
        device=args.device,
        img_size=img_size.image_size,
        n_classes=num_classes,
        ensemble_type=args.ensemble_type,
        output_method=args.ensemble_output_method,
    )

    # Construct classifier
    C = construct_classifier(args_train_classifier)
    logger.info(C)

    acc_fun: Callable
    # Loss function and accuracy function
    if binary_classification:
        criterion = nn.BCELoss()
        acc_fun = binary_accuracy
    else:
        criterion = nn.CrossEntropyLoss()
        acc_fun = multiclass_accuracy

    # Train the model
    stats, cp_path = train(
        C,
        criterion,
        train_loader,
        val_loader,
        acc_fun,
        train_classifier_args=args_train_classifier,  # Use unified TrainClassifierArgs
        cl_args=args,
    )

    # Load the best model checkpoint
    best_C = construct_classifier_from_checkpoint(cp_path, device=args.device)[0]
    logger.info("\n > Loading checkpoint from best epoch for testing ...")

    # Test the model
    evaluate_params = EvaluateParams(
        device=args.device,
        verbose=True,
        desc="Test",
        header="Test",
    )

    test_acc, test_loss = evaluate(
        best_C,
        test_loader,
        criterion,
        acc_fun,
        params=evaluate_params,
    )
    stats.test_acc = test_acc
    stats.test_loss = test_loss
    logger.info(f"Test Accuracy: {test_acc}")
    logger.info(f"Test Loss: {test_loss}")

    # Save checkpoint
    cp_path = checkpoint(
        model=best_C,
        model_name=(
            (
                f"{args_train_classifier.type}-"
                f"{args_train_classifier.nf}-"
                f"{args_train_classifier.epochs}."
                f"{args_train_classifier.seed}"
            )
            if args.name is None
            else args.name
        ),
        model_params=args_train_classifier,
        train_stats=stats,
        args=args,
        output_dir=args_train_classifier.out_dir,
    )

    # Predictions on training data
    save_predictions(best_C, train_loader, args_train_classifier, TrainingStage.train, cp_path)

    # Predictions on testing data
    save_predictions(best_C, test_loader, args_train_classifier, TrainingStage.test, cp_path)

    logger.info(f"\n > Saved checkpoint to {cp_path}")
    logger.info(f" > Test Accuracy: {test_acc}")
    logger.info(f" > Test Loss: {test_loss}")


if __name__ == "__main__":
    main()
