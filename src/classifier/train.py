import os
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifier import construct_classifier
from src.datasets import load_dataset
from src.metrics.accuracy import binary_accuracy, multiclass_accuracy
from src.models import (ClassifierParams, CLTrainArgs, DefaultTrainParams,
                        DeviceType, EvaluateParams, TrainArgs, TrainingStage,
                        TrainingStats)
from src.utils import setup_reprod
from src.utils.checkpoint import (checkpoint,
                                  construct_classifier_from_checkpoint)


def evaluate(
    C: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    acc_fun: Callable[[torch.Tensor, torch.Tensor, bool], float],
    params: EvaluateParams,  # Using the new Pydantic model
) -> Tuple[float, float]:
    """Evaluate the classifier and return accuracy and loss."""
    training = C.training
    C.eval()
    C.to(params.device.value)
    running_loss = 0.0
    running_accuracy = 0.0
    per_C_accuracy = []

    seq = tqdm(dataloader, desc=params.desc) if params.verbose else dataloader

    if params.header:
        print(f"\n --- {params.header} ---\n")

    for i, data in enumerate(seq, 0):
        X, y = data
        X = X.to(params.device.value)
        y = y.to(params.device.value)

        with torch.no_grad():
            accuracies = []
            if C.m_val:
                for m in C.models:
                    y_hat = m(X, output_feature_maps=False)
                    loss = criterion(y_hat, y)
                    running_accuracy += acc_fun(y_hat, y, avg=False).cpu()
                    running_loss += loss.item() * X.shape[0]
                    accuracies.append(acc_fun(y_hat, y, avg=True).cpu())
            else:
                y_total = C(X, output_feature_maps=True)
                y_hat = y_total[-1]
                y_c_hat = y_total[0]

                loss = criterion(y_hat, y)

                running_accuracy += acc_fun(y_hat, y, avg=False).cpu()
                running_loss += loss.item() * X.shape[0]

                for j in range(y_c_hat.size(-1)):
                    accuracies.append(
                        acc_fun(y_c_hat[:, j], y, avg=True).cpu())

        per_C_accuracy.append(accuracies)

    acc = running_accuracy / len(dataloader.dataset)
    loss = running_loss / len(dataloader.dataset)

    if training:
        C.train()

    per_C_accuracy = np.array(per_C_accuracy)
    print("per classifier accuracy: ", np.mean(per_C_accuracy, axis=0))
    return acc.item(), loss


# Assume DefaultTrainParams is already defined


def default_train_fn(
    C: nn.Module,
    X: Tensor,
    Y: Tensor,
    crit: Callable[[Tensor, Tensor], Tensor],
    acc_fun: Callable[[Tensor, Tensor, bool], float],
    params: DefaultTrainParams,  # Using the new Pydantic model
) -> Tuple[Tensor, Tensor]:
    """Default training function for a single batch."""
    X = X.to(params.device.value)
    Y = Y.to(params.device.value)
    y_hat = C(X)
    loss = crit(y_hat, Y)
    acc = acc_fun(y_hat, Y, avg=False).cpu()

    if params.early_acc > (acc / len(Y)):
        loss.backward()

    return loss, acc


def train(
    C: nn.Module,
    opt: optim.Optimizer,
    crit: Any,  # Loss function
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    acc_fun: Any,  # Accuracy function
    args: TrainArgs,  # Use Pydantic model for args
    name: str,
    model_params: ClassifierParams,  # Use ClassifierParams for model parameters
    device: DeviceType,
) -> Tuple[TrainingStats, str]:  # Use TrainingStats model for stats
    """Training loop with validation and checkpointing."""

    stats = TrainingStats()  # Initialize the stats using the Pydantic model

    C.to(device.value)
    C.train()
    for stage in [TrainingStage.train, TrainingStage.optimize]:
        if stage == TrainingStage.optimize:
            if C.optimize:
                C_fn = C.optimize_helper
            else:
                break
        elif stage == TrainingStage.train:
            if C.train_models:
                C_fn = C.train_helper
            else:
                params = DefaultTrainParams(
                    early_acc=args.early_acc, device=device)

                def C_fn(*args):
                    return default_train_fn(*args, params=params)

        for epoch in range(args.epochs):
            stats.cur_epoch = epoch
            print(
                f"\n --- {stage.value.capitalize()}: Epoch {epoch + 1} ---\n",
                flush=True,
            )

            running_accuracy = 0.0
            running_loss = 0.0

            for i, data in enumerate(
                tqdm(train_loader, desc=stage.value.capitalize()), 0
            ):
                X, y = data
                X = X.to(device.value)
                y = y.to(device.value)

                opt.zero_grad()
                loss, acc = C_fn(C, X, y, crit, acc_fun)
                opt.step()

                running_accuracy += acc
                running_loss += loss.item() * X.shape[0]

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_accuracy / len(train_loader.dataset)
            stats.train_acc.append(train_acc.item())
            stats.train_loss.append(train_loss)

            print(f"{stage.value.capitalize()}: Loss: {train_loss}", flush=True)
            print(f"{stage.value.capitalize()}: Accuracy: {train_acc}", flush=True)

            # Validation step
            evaluate_params = EvaluateParams(
                # The device can be 'cpu' or 'cuda' (using your Pydantic DeviceType model)
                device=device,
                verbose=True,  # Whether to show progress bar
                desc="Validate",  # Description for progress bar during validation
                header="Validation",  # Optional header for logging
            )
            val_acc, val_loss = evaluate(
                C,
                val_loader,
                crit,
                acc_fun,
                params=evaluate_params,  # Pass the params instance
            )
            stats.val_acc.append(val_acc)
            stats.val_loss.append(val_loss)

            print(f"{stage.value.capitalize()}: Loss: {val_loss}", flush=True)
            print(f"{stage.value.capitalize()}: Accuracy: {val_acc}", flush=True)

            # Early stopping and checkpointing logic
            if val_loss < stats.best_loss:
                stats.best_loss = val_loss
                stats.best_epoch = epoch
                stats.early_stop_tracker = 0

                cp_path = checkpoint(
                    C, name, model_params, stats, args, output_dir=args.out_dir
                )
                print(f" > Saved checkpoint to {cp_path}")
            else:
                if args.early_stop is not None:
                    stats.early_stop_tracker += 1
                    print(
                        f" > Early stop counter: {stats.early_stop_tracker}/{args.early_stop}"
                    )

                    if stats.early_stop_tracker == args.early_stop:
                        break

    return stats, cp_path


def parse_args() -> CLTrainArgs:
    """Parses and returns the training arguments."""
    args = CLTrainArgs()  # Default values are used from Pydantic model.
    return args


def save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: DeviceType,
    dataset_name: TrainingStage,
    cp_path: str,
):
    """
    Generate predictions using the model, save the predicted values as .npy,
    and generate histograms for visualization.

    Args:
        model (nn.Module): The trained model to generate predictions.
        dataloader (DataLoader): The dataloader for which predictions are needed (train or test).
        device (torch.device): Device ('cpu' or 'cuda') to use for computation.
        dataset_name (str): Name to identify the dataset ('train' or 'test').
        cp_path (str): Path to save the predictions and histograms.
    """
    model.eval()  # Set model to evaluation mode
    y_hat_full = []  # Store the predictions for all batches

    # Generate predictions
    print(f"\n > Generating predictions on {dataset_name.value} data ...")
    for X, _ in dataloader:
        X = X.to(device.value)
        with torch.no_grad():
            y_hat_batch = model(X)
            y_hat_full.append(y_hat_batch.cpu())

    # Concatenate predictions into a single tensor
    y_hat_full = torch.cat(y_hat_full)

    # Save predictions as .npy file
    np.save(
        os.path.join(cp_path, f"{dataset_name.value}_y_hat.npy"),
        y_hat_full.numpy(),
        allow_pickle=False,
    )

    # Create and save histogram of predictions
    sns.histplot(data=y_hat_full.numpy(), stat="proportion", bins=20)
    plt.savefig(os.path.join(
        cp_path, f"{dataset_name.value}_y_hat.svg"), dpi=300)
    plt.clf()

    print(
        f" > Saved {dataset_name.value} predictions and histogram to {cp_path}")


def main() -> None:
    """Main function to run the training, validation, and testing of the classifier."""
    load_dotenv()

    # Parse arguments using Pydantic model
    try:
        args = CLTrainArgs.model_validate(parse_args().__dict__)
        print(args)
    except ValidationError as e:
        print("Validation error:", e)
        # Optionally exit or handle the error gracefully
        exit(1)
    # Set random seed
    seed = np.random.randint(100000) if args.seed is None else args.seed
    setup_reprod(seed)
    args.seed = seed
    print(" > Seed", args.seed)

    # Parse number of features (nf)
    args.nf = eval(args.nf)

    # Use the device directly from args (already validated by Pydantic)
    device = args.device
    print(f" > Using device: {device.value}")

    # Name for the classifier based on type, nf, epochs, and seed
    name = (
        f"{args.c_type}-{args.nf}-{args.epochs}.{args.seed}"
        if args.name is None
        else args.name
    )

    # Load dataset
    dataset_name = args.dataset_name
    dataset, num_classes, img_size = load_dataset(
        dataset_name, args.data_dir, pos_class=args.pos_class, neg_class=args.neg_class
    )
    print(" > Using dataset", dataset_name)

    # Determine if binary classification
    binary_classification = num_classes == 2
    if binary_classification:
        print(
            f"\t> Binary classification between {args.pos_class} and {args.neg_class}"
        )
        dataset_name = f"{dataset_name}.{args.pos_class}v{args.neg_class}"

    # Prepare output directory
    out_dir = os.path.join(args.out_dir, dataset_name)

    # Split dataset into training and validation sets
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(5 / 6 * len(dataset)), len(dataset) -
                  int(5 / 6 * len(dataset))]
    )

    # Create data loaders for training, validation, and testing
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False
    )

    test_set = load_dataset(
        args.dataset_name, args.data_dir, args.pos_class, args.neg_class, train=False
    )[0]
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False
    )

    # Create ClassifierParams using the parsed args
    model_params = ClassifierParams(
        type=args.c_type,
        img_size=img_size,
        nf=args.nf,
        n_classes=num_classes,
        device=args.device,  # Using DeviceType from args
    )

    # Construct classifier (pass the Pydantic model directly)
    C = construct_classifier(model_params)
    print(C, flush=True)

    # Optimizer
    opt = optim.Adam(C.parameters(), lr=args.lr)

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
        opt,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        acc_fun,
        args,
        name,
        model_params,
        device,
    )

    # Load the best model checkpoint
    best_C = construct_classifier_from_checkpoint(cp_path, device=device)[0]
    print("\n > Loading checkpoint from best epoch for testing ...")

    # Test the model
    evaluate_params = EvaluateParams(
        # The device can be 'cpu' or 'cuda' (using your Pydantic DeviceType model)
        device=device,
        verbose=True,  # Set to True to show progress bar
        desc="Test",  # Description for progress bar
        header="Test",  # Optional header for logging
    )

    test_acc, test_loss = evaluate(
        best_C,
        test_loader,
        criterion,
        acc_fun,
        params=evaluate_params,  # Pass the params instance
    )
    stats.test_acc = test_acc
    stats.test_loss = test_loss
    print(f"Test acc. = {test_acc}")
    print(f"test loss. = {test_loss}")

    # Save checkpoint
    cp_path = checkpoint(best_C, name, model_params,
                         stats, args, output_dir=out_dir)

    # Predictions on training data
    save_predictions(best_C, train_loader, device,
                     TrainingStage.train, cp_path)

    # Predictions on testing data
    save_predictions(best_C, test_loader, device, TrainingStage.test, cp_path)

    print(f"\n > Saved checkpoint to {cp_path}")
    print(cp_path)
    print(test_acc)
    print(test_loss)


if __name__ == "__main__":
    main()
