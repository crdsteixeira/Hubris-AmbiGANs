"""Utilities for multiclass classifier training."""

import gc
import logging
import os
from collections.abc import Callable

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
            y = y.to(device).long()

            outputs = model(X)
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
    device: DeviceType = DeviceType.cuda,
    seed: int | None = None,
    entity: str | None = None,
    project: str = "multiclass-classifiers",
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
        pos_class: Positive class for binary classification
        neg_class: Negative class for binary classification

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
            "seed": seed,
        },
    )

    try:
        # Load dataset
        load_params = LoadDatasetParams(
            dataroot=data_dir,
            dataset_name=dataset_enum,
            pos_class=1 if dataset_enum == DatasetNames.chest_xray else None,
            neg_class=0 if dataset_enum == DatasetNames.chest_xray else None,
            split="validation" if dataset_enum == DatasetNames.chest_xray else "train",
            pytesting=False,
        )
        dataset, num_classes, img_size = load_dataset(load_params)
        logger.info("Dataset: %s | Classes: %s | Image Size: %s", dataset_name_value, num_classes, img_size.image_size)

        # Prepare output directory
        dataset_out_dir = os.path.join(out_dir, dataset_name_value)
        os.makedirs(dataset_out_dir, exist_ok=True)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
        # Create data loaders (num_workers=0 for chest-xray to avoid memory issues with large images)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Log dataset sizes
        logger.info("Dataset sizes - Train: %d | Val: %d", len(train_set), len(val_set))

        # Create training arguments
        common_args = {
            "data_dir": data_dir,
            "out_dir": dataset_out_dir,
            "dataset_name": dataset_enum,
            "pos_class": 1 if dataset_enum == DatasetNames.chest_xray else None,
            "neg_class": 0 if dataset_enum == DatasetNames.chest_xray else None,
            "batch_size": batch_size,
            "c_type": classifier_enum,
            "epochs": epochs,
            "early_stop": None,
            "early_acc": 1.0,
            "seed": seed,
            "nf": [32, 64],
            "device": device,
            "n_classes": num_classes,
            "ensemble_type": None,
            "name": f"{classifier_type_value}_{seed}",
        }

        args = TrainClassifierArgs(
            type=classifier_enum,
            img_size=img_size.image_size,
            output_method=None,
            **common_args,  # type: ignore[arg-type]
        )

        cl_args = CLTrainArgs(
            ensemble_output_method=None,
            entity=entity,
            project=project,
            **common_args,  # type: ignore[arg-type]
        )

        # Construct classifier
        C = construct_classifier(args)
        logger.info(f"\nModel Architecture:\n{C}")

        acc_fun: Callable
        # Loss function and accuracy function
        if num_classes == 2:
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
            train_classifier_args=args,
            cl_args=cl_args,
        )
        logger.info(f"Model saved to: {cp_path}")

        del val_loader, train_loader, C
        torch.cuda.empty_cache()
        gc.collect()

        # Load best checkpoint
        best_C = construct_classifier_from_checkpoint(cp_path, device=device)[0]
        logger.info(f"\nLoading best model from checkpoint: {cp_path}")

        # load test set for evaluation
        eval_set, _, _ = load_dataset(
            LoadDatasetParams(
                dataroot=data_dir,
                dataset_name=dataset_enum,
                pos_class=1 if dataset_enum == DatasetNames.chest_xray else None,
                neg_class=0 if dataset_enum == DatasetNames.chest_xray else None,
                split="test" if dataset_enum == DatasetNames.chest_xray else "test[:25%]",
                pytesting=False,
            )
        )
        test_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

        # Evaluate on test set
        if num_classes == 2:
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
        # save_predictions(best_C, train_loader, args, TrainingStage.train, cp_path)
        save_predictions(best_C, test_loader, args, TrainingStage.test, cp_path)

        del test_loader, best_C
        torch.cuda.empty_cache()
        gc.collect()

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

        # Re-raise the exception so the caller knows training failed
        raise
