"""Utilities for multiclass classifier training."""

import logging
import os

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from src.classifier.construct_classifier import construct_classifier
from src.classifier.train_classifier import evaluate, save_predictions, train
from src.datasets.load import load_dataset
from src.enums import ClassifierType, DatasetNames, DeviceType, TrainingStage
from src.metrics.accuracy import binary_accuracy, multiclass_accuracy, top_n_accuracy
from src.models import (
    CLTrainArgs,
    EvaluateParams,
    LoadDatasetParams,
    TrainClassifierArgs,
)
from src.utils.checkpoint import construct_classifier_from_checkpoint

logger = logging.getLogger(__name__)


def evaluate_with_top_k_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> dict:
    """
    Evaluate model with top-1 and top-2 accuracies.

    Args:
        model: The classifier model
        dataloader: Data loader for evaluation
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary with top-1, top-2 accuracies and loss

    """
    model.eval()
    model.to(device)

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            X, y = data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X, output_feature_maps=False)
            loss = criterion(outputs, y)

            running_loss += loss.item() * X.shape[0]
            all_preds.append(outputs.detach().cpu())
            all_labels.append(y.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = running_loss / len(dataloader.dataset)
    top1_acc = top_n_accuracy(all_preds, all_labels, n=1)
    top2_acc = top_n_accuracy(all_preds, all_labels, n=2)

    return {
        "loss": avg_loss,
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
    }


def train_single_multiclass_classifier(  # pylint: disable=too-many-positional-arguments,broad-exception-caught,too-many-statements
    dataset_name: DatasetNames | str,
    classifier_type: ClassifierType | str,
    data_dir: str,
    out_dir: str,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 5e-4,
    device: DeviceType = DeviceType.cuda,
    seed: int | None = None,
    entity: str | None = None,
    project: str = "multiclass-classifiers",
    pos_class: int | None = None,
    neg_class: int | None = None,
) -> dict:
    """
    Train a single multiclass classifier on a dataset.

    Args:
        dataset_name: Name of the dataset
        classifier_type: Type of classifier
        data_dir: Path to dataset directory
        out_dir: Output directory for models
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use (cuda or cpu)
        seed: Random seed for reproducibility
        entity: WandB entity name
        project: WandB project name
        pos_class: Index of positive class for binary classification
        neg_class: Index of negative class for binary classification

    Returns:
        Dictionary with metrics (top1_accuracy, top2_accuracy, loss) and 'error' field if failed

    """
    dataset_enum = dataset_name if isinstance(dataset_name, DatasetNames) else DatasetNames(dataset_name)
    classifier_enum = (
        classifier_type if isinstance(classifier_type, ClassifierType) else ClassifierType(classifier_type)
    )
    dataset_name_value = dataset_enum.value
    classifier_type_value = classifier_enum.value

    logger.info("\n%s", "=" * 80)
    logger.info("Training %s on %s", classifier_type_value.upper(), dataset_name_value)
    logger.info("%s\n", "=" * 80)

    # Initialize WandB
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        name=f"{dataset_name_value}-{classifier_type_value}-{seed}",
        config={
            "dataset": dataset_name_value,
            "classifier": classifier_type_value,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
        },
    )

    try:
        # Load dataset
        load_params = LoadDatasetParams(
            dataroot=data_dir,
            dataset_name=dataset_enum,
            pos_class=pos_class,
            neg_class=neg_class,
            train=True,
            pytesting=False,
        )
        dataset, num_classes, img_size = load_dataset(load_params)
        logger.info("Dataset: %s | Classes: %s | Image Size: %s", dataset_name_value, num_classes, img_size.image_size)

        binary_mode = pos_class is not None and neg_class is not None and num_classes == 2

        # Prepare output directory
        dataset_out_dir = os.path.join(out_dir, dataset_name_value)
        os.makedirs(dataset_out_dir, exist_ok=True)

        # Split dataset (70-15-15 train-val-test)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create training arguments
        args = TrainClassifierArgs(
            type=classifier_enum,
            data_dir=data_dir,
            out_dir=dataset_out_dir,
            name=f"{classifier_type_value}_{seed}",
            dataset_name=dataset_enum,
            pos_class=None,
            neg_class=None,
            batch_size=batch_size,
            c_type=classifier_enum,
            epochs=epochs,
            early_stop=None,
            early_acc=1.0,
            lr=lr,
            seed=seed,
            nf=2,  # Base filter size for CNN; hidden dim for others
            device=device,
            img_size=img_size.image_size,
            n_classes=num_classes,
            ensemble_type=None,
            output_method=None,
        )

        cl_args = CLTrainArgs(
            data_dir=data_dir,
            out_dir=dataset_out_dir,
            dataset_name=dataset_enum,
            pos_class=None,
            neg_class=None,
            batch_size=batch_size,
            c_type=classifier_enum,
            epochs=epochs,
            early_stop=None,
            early_acc=1.0,
            lr=lr,
            seed=seed,
            nf=2,
            device=device,
            n_classes=num_classes,
            ensemble_type=None,
            ensemble_output_method=None,
            entity=entity,
            project=project,
            name=f"{classifier_type_value}_{seed}",
        )

        # Loss and accuracy
        if binary_mode:
            criterion = nn.BCELoss()
            acc_fun = binary_accuracy  # type: ignore[assignment]
        else:
            criterion = nn.CrossEntropyLoss()
            acc_fun = multiclass_accuracy  # type: ignore[assignment]

        # Construct classifier
        C = construct_classifier(args)
        logger.info(f"\nModel Architecture:\n{C}")

        # Train the model
        stats, cp_path = train(
            C,
            criterion,
            train_loader,
            val_loader,
            acc_fun,
            train_classifier_args=args,
            cl_args=cl_args,
        )
        logger.info(f"Model saved to: {cp_path}")

        # Load best checkpoint
        best_C = construct_classifier_from_checkpoint(cp_path, device=device)[0]
        logger.info(f"\nLoading best model from checkpoint: {cp_path}")

        # Evaluate on test set
        if binary_mode:
            eval_params = EvaluateParams(
                device=device,
                verbose=False,
                desc="Test",
                header=None,
            )
            test_acc, test_loss = evaluate(
                best_C,
                test_loader,
                criterion,
                acc_fun,
                params=eval_params,
            )
            test_metrics = {
                "loss": test_loss,
                "top1_accuracy": test_acc,
                "top2_accuracy": 0.0,
            }
        else:
            test_metrics = evaluate_with_top_k_accuracy(
                best_C,
                test_loader,
                criterion,
                device=device.value,
            )

        logger.info("\n%s", "=" * 80)
        logger.info("RESULTS: %s on %s", classifier_type_value.upper(), dataset_name_value)
        logger.info("Top-1 Accuracy: %.4f", test_metrics["top1_accuracy"])
        logger.info("Top-2 Accuracy: %.4f", test_metrics["top2_accuracy"])
        logger.info("Test Loss: %.4f", test_metrics["loss"])
        logger.info("%s\n", "=" * 80)

        # Log final test metrics to WandB
        wandb.log(
            {
                "top1_accuracy": test_metrics["top1_accuracy"],
                "top2_accuracy": test_metrics["top2_accuracy"],
                "test_loss": test_metrics["loss"],
                "best_epoch": stats.best_epoch,
            }
        )

        # Save predictions
        save_predictions(best_C, train_loader, args, TrainingStage.train, cp_path)
        save_predictions(best_C, test_loader, args, TrainingStage.test, cp_path)

        wandb_run.finish()

        return test_metrics

    except Exception as e:
        error_msg = f"Training failed for {classifier_type_value} on {dataset_name_value}: {e}"
        logger.error(error_msg)

        # Log error to WandB and mark run as failed
        try:
            wandb.log({"error": str(e)})
            wandb.finish(exit_code=1)  # Mark as failed
        except Exception as wandb_error:
            logger.error(f"Failed to log error to WandB: {wandb_error}")

        return {
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "loss": float("inf"),
            "error": str(e),
        }
