"""Module to train Classifiers."""

import logging
import os
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.enums import DeviceType, TrainingStage
from src.models import CLTrainArgs, EvaluateParams, TrainClassifierArgs, TrainingStats
from src.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def evaluate(
    C: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    acc_fun: Callable,
    params: EvaluateParams,
) -> tuple[float, float]:
    """Evaluate the classifier and return accuracy and loss."""
    training = C.training
    C.eval()
    C.to(params.device.value)
    running_loss = torch.tensor(0.0, dtype=torch.float32)
    running_accuracy = torch.tensor(0.0, dtype=torch.float32)
    per_C_accuracy = []

    seq = tqdm(dataloader, desc=params.desc) if params.verbose else dataloader

    if params.header:
        logger.info(f"\n --- {params.header} ---\n")

    for data in seq:
        X, y = data
        X = X.to(params.device.value)
        y = y.to(params.device.value)

        with torch.no_grad():
            accuracies = []
            if C.params.output_method == "identity":
                for m in C.models:
                    y_hat = m(X, output_feature_maps=False)
                    loss = criterion(y_hat, y)
                    running_accuracy += acc_fun(y_hat, y, avg=False).cpu()
                    running_loss += loss.item() * X.shape[0]
                    accuracies.append(acc_fun(y_hat, y, avg=True).cpu())
            else:
                y_total = C(X, output_feature_maps=True)
                y_hat = y_total[0]
                y_c_hat = y_total[-1][-1]  # Get features before last layer

                loss = criterion(y_hat, y)

                running_accuracy += acc_fun(y_hat, y, avg=False).cpu()
                running_loss += loss.item() * X.shape[0]

                for j in range(y_c_hat.size(-1)):
                    accuracies.append(acc_fun(y_c_hat[:, j], y, avg=True).cpu())

        per_C_accuracy.append(accuracies)

    acc = running_accuracy / len(dataloader.dataset)
    loss = running_loss / len(dataloader.dataset)

    if training:
        C.train()

    per_C_accuracy = np.array(per_C_accuracy)
    logger.info(f"per classifier accuracy:  {np.mean(per_C_accuracy, axis=0)}.")
    return acc.item(), loss.item()


def default_train_fn(
    C: nn.Module,
    X: Tensor,
    Y: Tensor,
    crit: Callable,
    acc_fun: Callable,
    _: float,
    params: TrainClassifierArgs,
) -> tuple[Tensor, Tensor]:
    """Train a single batch with the default method."""
    X = X.to(params.device.value)
    Y = Y.to(params.device.value)
    y_hat = C(X)
    loss = crit(y_hat, Y)
    acc = acc_fun(y_hat, Y, avg=False).cpu()

    if params.early_acc > (acc / len(Y)):
        loss.backward()

    return loss, acc


def log_stage_header(stage: TrainingStage, epoch: int) -> None:
    """Log the header for the current training stage and epoch."""
    logger.info(f"\n --- {stage.value.capitalize()}: Epoch {epoch + 1} ---\n")


def train(
    C: nn.Module,
    crit: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    acc_fun: Any,
    train_classifier_args: TrainClassifierArgs,
    cl_args: CLTrainArgs,
) -> tuple[TrainingStats, str]:
    """Training loop with validation and checkpointing."""
    stats = TrainingStats()

    # Move model to specified device
    C.to(train_classifier_args.device.value)
    # Optimizer
    opt = optim.Adam(C.parameters(), lr=train_classifier_args.lr)
    C.train()
    cp_path = checkpoint(
        model=C,
        model_name=(
            (
                f"{train_classifier_args.type}-"
                f"{train_classifier_args.nf}-"
                f"{train_classifier_args.epochs}."
                f"{train_classifier_args.seed}"
            )
            if cl_args.name is None
            else cl_args.name
        ),
        model_params=train_classifier_args,
        train_stats=stats,
        args=cl_args,
        output_dir=train_classifier_args.out_dir,
    )
    for stage in [TrainingStage.train, TrainingStage.optimize]:
        C_fn = select_training_function(C, stage)
        if not C_fn:
            break

        for epoch in range(train_classifier_args.epochs):
            stats.cur_epoch = epoch
            log_stage_header(stage, epoch)

            train_loss, train_acc = execute_epoch(C_fn, C, train_loader, opt, crit, acc_fun, train_classifier_args)

            # Record stats
            stats.train_acc.append(train_acc)
            stats.train_loss.append(train_loss)

            logger.info(f"{stage.value.capitalize()}: Loss: {train_loss}")
            logger.info(f"{stage.value.capitalize()}: Accuracy: {train_acc}")

            # Validation step
            val_acc, val_loss = validate(C, val_loader, crit, acc_fun, train_classifier_args.device)
            stats.val_acc.append(val_acc)
            stats.val_loss.append(val_loss)

            logger.info(f"{stage.value.capitalize()}: Loss: {val_loss}")
            logger.info(f"{stage.value.capitalize()}: Accuracy: {val_acc}")
            # Early stopping and checkpointing logic
            cp_path = handle_checkpointing(
                C=C,
                val_loss=val_loss,
                stats=stats,
                epoch=epoch,
                args=train_classifier_args,
                out_dir=train_classifier_args.out_dir,
                cp_path=cp_path,
                cl_args=cl_args,
            )
            if stats.early_stop_tracker == train_classifier_args.early_stop:
                break

    return stats, cp_path


def select_training_function(C: nn.Module, stage: TrainingStage) -> Callable | None:
    """Select the appropriate training function based on the training stage and model attributes."""
    if stage == TrainingStage.optimize and getattr(C, "optimize", False):
        return C.optimize_helper
    if stage == TrainingStage.train:
        if getattr(C, "train_models", False):
            return C.train_helper
        return default_train_fn
    return None


def execute_epoch(
    C_fn: Callable,
    C: nn.Module,
    loader: DataLoader,
    opt: optim.Optimizer,
    crit: Any,
    acc_fun: Any,
    params: TrainClassifierArgs,
) -> tuple[float, float]:
    """Execute one training epoch and return the training loss and accuracy."""
    running_loss = torch.tensor(0.0, dtype=torch.float32)
    running_accuracy = torch.tensor(0.0, dtype=torch.float32)

    for data in tqdm(loader, desc="Training"):
        X, y = data
        X = X.to(params.device.value)
        y = y.to(params.device.value)

        opt.zero_grad()
        loss, acc = C_fn(C, X, y, crit, acc_fun, params.early_acc, params)
        opt.step()

        running_accuracy += acc.cpu()
        running_loss += loss.cpu() * X.shape[0]

    train_loss = running_loss / len(loader.dataset)
    train_acc = running_accuracy / len(loader.dataset)

    return train_loss.item(), train_acc.item()


def validate(
    C: nn.Module, val_loader: DataLoader, crit: nn.Module, acc_fun: Callable, device: DeviceType
) -> tuple[float, float]:
    """Perform validation and return accuracy and loss."""
    evaluate_params = EvaluateParams(
        device=device,
        verbose=True,
        desc="Validate",
        header="Validation",
    )
    return evaluate(C, val_loader, crit, acc_fun, params=evaluate_params)


def handle_checkpointing(
    C: nn.Module,
    val_loss: float,
    stats: TrainingStats,
    epoch: int,
    args: TrainClassifierArgs,
    out_dir: str,
    cp_path: str,
    cl_args: CLTrainArgs,
) -> str:
    """Handle checkpointing and early stopping."""
    if val_loss < stats.best_loss:
        stats.best_loss = val_loss
        stats.best_epoch = epoch
        stats.early_stop_tracker = 0

        cp_path = checkpoint(
            model=C,
            model_name=f"{args.type}-{args.nf}-{args.epochs}.{args.seed}" if cl_args.name is None else cl_args.name,
            model_params=args,
            train_stats=stats,
            args=cl_args,
            output_dir=out_dir,
        )
        logger.info(f" > Saved checkpoint to {cp_path}")
    else:
        if args.early_stop is not None:
            stats.early_stop_tracker += 1
            logger.info(f" > Early stop counter: {stats.early_stop_tracker}/{args.early_stop}")

    return cp_path


def save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    args: TrainClassifierArgs,
    dataset_stage: TrainingStage,
    cp_path: str,
) -> None:
    """
    Generate predictions using the model, save them as a .npy file.
    Also create histograms for visualization.
    """
    model.eval()  # Set model to evaluation mode
    y_hat_full = []  # Store the predictions for all batches

    # Generate predictions
    logger.info(f"\n > Generating predictions on {dataset_stage.value} data ...")
    for X, _ in dataloader:
        X = X.to(args.device.value)
        with torch.no_grad():
            y_hat_batch = model(X)
            y_hat_full.append(y_hat_batch.cpu())

    # Concatenate predictions into a single tensor
    y_hat_final: Tensor = torch.cat(y_hat_full)

    # Save predictions as .npy file
    output_path = os.path.join(cp_path, f"{dataset_stage.value}_y_hat.npy")
    np.save(output_path, y_hat_final.numpy(), allow_pickle=False)

    # Create and save histogram of predictions
    sns.histplot(data=y_hat_final.numpy(), stat="proportion", bins=20)
    plt.savefig(os.path.join(cp_path, f"{dataset_stage.value}_y_hat.svg"), dpi=300)
    plt.clf()

    logger.info(f" > Saved {dataset_stage.value} predictions and histogram to {cp_path}")


def main() -> None:
    """Run the training, validation, and testing of the classifier."""
    load_dotenv()
