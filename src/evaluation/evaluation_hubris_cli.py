"""CL for models evaluation."""

import gc
import glob
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.datasets.companion_selection import (
    MAX_CONFUSION_DISTANCE,
    SelectionOptions,
    select_companion_images,
)
from src.datasets.image_dataset import ImageDataset
from src.datasets.load import DatasetNames, load_dataset
from src.enums import DeviceType, PretrainedModels
from src.evaluation.pretrained_models import (
    CONVNEXT_MODEL_ID,
    EFFICIENTNETV2_MODEL_ID,
    SWIN_MODEL_ID,
    TRAINING_RECIPE_ID,
    VIT_LEGACY_CXR_MODEL_ID,
    VIT_MODEL_ID,
    ConvNext,
    EfficientNetV2,
    Swin,
    ViT,
)
from src.metrics.accuracy import binary_accuracy, binary_precision_recall_f1
from src.metrics.hubris import Hubris
from src.models import CLEvaluationArgs, LoadDatasetParams
from src.utils.checkpoint import checkpoint, construct_classifier_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import setup_reprod

# Load environment variables
load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", dest="dataroot", default=f"{os.environ['FILESDIR']}/data", help="Directory with dataset")
parser.add_argument("--companion-data", dest="companion_dataroot", help="Directory with companion dataset")
parser.add_argument(
    "--models", dest="models", nargs="+", help="Pretrained models to be evaluated (space-separated list)"
)
parser.add_argument("--dataset", dest="dataset_name", default="mnist", help="Dataset (mnist, fashion-mnist, etc.)")
parser.add_argument("--pos", dest="pos_class", default=3, type=int, help="Positive class for binary classification")
parser.add_argument("--neg", dest="neg_class", default=0, type=int, help="Negative class for binary classification")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size to use")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to re-train")
parser.add_argument(
    "--estimator-path",
    dest="estimator_path",
    default=None,
    type=str,
    help="Path to estimator. If none, does not calculate relative Hubris",
)
parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes for data loading")
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda, or cpu)")
parser.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory to save evaluation csv file")
parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
parser.add_argument("--gan-id", dest="gan_id", default=None, help="GAN experiment ID for wandb tracking")


def evaluate(
    config: CLEvaluationArgs,
    model: nn.Module,
    loader: DataLoader,
    name: str,
    compute_prf: bool = True,
) -> pd.DataFrame:
    """Evaluate model using companion dataset with memory-efficient inference."""
    model.eval()
    preds = []
    ref_preds = []
    labels = []

    # Load estimator if needed
    estimator = None
    if config.estimator_path is not None:
        estimator, _, _, _, _ = construct_classifier_from_checkpoint(config.estimator_path, device=config.device)
        estimator.eval()

    with torch.no_grad():
        for images, label in tqdm(loader):
            preds.append(model(images.to(config.device)).cpu())
            labels.append(label)
            if estimator is not None:
                ref_preds.append(estimator(images.to(config.device)).cpu())

            # Cleanup memory after each batch
            del images, label
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        full_preds = torch.cat(preds)
        full_labels = torch.cat(labels)
        full_ref_preds = torch.cat(ref_preds) if estimator is not None else None

    accuracy = binary_accuracy(full_preds, full_labels, avg=True, threshold=0.50).item()
    precision, recall, f1_score = None, None, None
    if compute_prf:
        precision, recall, f1_score = binary_precision_recall_f1(full_preds, full_labels, threshold=0.50)

    hubris = Hubris(C=None, dataset_size=len(full_preds))
    absolute_hubris = hubris.compute(full_preds, ref_preds=None)
    improved_hubris = hubris.compute_improved(full_preds, ref_preds=None)

    df = pd.DataFrame()
    df = df.assign(
        dataset=[name],
        accuracy=[accuracy],
        precision=[precision],
        recall=[recall],
        f1_score=[f1_score],
        absolute_hubris=[absolute_hubris],
        improved_hubris=[improved_hubris],
        acd=[(0.50 - full_preds).abs().mean().item()],
    )

    # Compute relative Hubris if estimator was used
    if estimator is not None and len(ref_preds) > 0:
        relative_hubris = hubris.compute(full_preds, ref_preds=full_ref_preds)
        improved_relative_hubris = hubris.compute_improved(full_preds, ref_preds=full_ref_preds)
        df = df.assign(
            relative_hubris=[relative_hubris],
            improved_relative_hubris=[improved_relative_hubris],
        )

    # Clean up estimator model
    del estimator
    torch.cuda.empty_cache()
    gc.collect()

    return df


def setup_config(args_dict: dict) -> CLEvaluationArgs:
    """Initialize configuration with the provided arguments."""
    try:
        config = CLEvaluationArgs(**args_dict)
    except ValidationError as e:
        logger.error(f"Argument validation error: {e}")
        raise

    # Logging the arguments
    logger.info(config)
    return config


def setup_wandb_for_model(config: CLEvaluationArgs, model: PretrainedModels, gan_id: str, seed: int) -> None:
    """Initialize wandb for a specific model-dataset pair."""
    timestamp = datetime.now().strftime("%b%d_%H%M%S")
    wandb.init(
        project="AmbiGAN-Evaluation",
        name=f"{gan_id}-{model.value}-{timestamp}",
        group=f"{config.dataset_name}.{config.pos_class}v{config.neg_class}",
        resume="allow",
        config={
            "gan_id": gan_id,
            "model": model.value,
            "dataset": config.dataset_name,
            "pos_class": config.pos_class,
            "neg_class": config.neg_class,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "estimator_path": config.estimator_path,
            "seed": seed,
        },
    )


def balance_indices(indices: list[int], targets: torch.Tensor) -> list[int]:
    """Oversample the minority class within `indices` until both classes are equally represented."""
    pos = [i for i in indices if targets[i] == 1]
    neg = [i for i in indices if targets[i] == 0]
    if not pos or not neg or len(pos) == len(neg):
        return indices

    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = torch.randint(len(minority), (len(majority) - len(minority),)).tolist()
    return indices + [minority[i] for i in extra]


def split_train_validation(dataset: Dataset, val_fraction: float = 0.5) -> tuple[Subset, Subset]:
    """
    Split a training set into fine-tuning and validation halves.

    `BinaryDataset` balances classes by appending duplicates of minority-class samples, so
    splitting it directly would put copies of the same image on both sides. Split the
    distinct prefix instead and rebalance each half independently, which keeps the
    validation signal honest. Falls back to a plain random split for other dataset types.

    Args:
        dataset: Dataset to split
        val_fraction: Share of the distinct samples held out for validation

    Returns:
        Tuple of (fine-tuning subset, validation subset).

    """
    num_distinct = getattr(dataset, "num_original", None)
    targets = getattr(dataset, "targets", None)
    if num_distinct is None or targets is None:
        half = len(dataset) // 2
        train_set, val_set = torch.utils.data.random_split(dataset, [half, len(dataset) - half])
        return train_set, val_set

    shuffled = torch.randperm(num_distinct).tolist()
    split_at = int(num_distinct * (1.0 - val_fraction))
    return (
        Subset(dataset, balance_indices(shuffled[:split_at], targets)),
        Subset(dataset, balance_indices(shuffled[split_at:], targets)),
    )


def load_ambiguous_companion(config: CLEvaluationArgs, transform: Any) -> ImageDataset | None:
    """Load the companion images the guiding estimator was undecided about, None if there are none."""
    run_dir = Path(config.companion_dataroot).parent
    options = SelectionOptions(max_confusion_distance=MAX_CONFUSION_DISTANCE, device=config.device)
    # No quota: hubris is computed over every ambiguous image the run holds, and never generates more
    image_paths = select_companion_images(run_dir, n_samples=None, options=options)
    if not image_paths:
        logger.warning("No companion image within confusion distance %s for %s", MAX_CONFUSION_DISTANCE, run_dir.name)
        return None
    # Scalar labels, as ImageFolder gives the unfiltered companion set: hubris reads only the
    # predictions, and a list-valued label would not collate into a tensor
    labels: list[int] = [0] * len(image_paths)
    return ImageDataset(image_paths, color_mode="RGB", labels=labels, transform=transform)


def load_datasets(
    config: CLEvaluationArgs,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader | None]:
    """
    Load training, test, and companion datasets.

    The last loader holds only the companion images within `MAX_CONFUSION_DISTANCE` of the decision
    boundary, and is None when the run has none.

    For fine-tuning pretrained models, uses different splits by dataset:
    - Chest X-ray: Uses VALIDATION because it has more images
    """
    finetune_train_set, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=config.dataset_name,
            pos_class=config.pos_class,
            neg_class=config.neg_class,
            split="train",
            pytesting=False,
        )
    )
    # Half the training data is fine-tuned on, the other half validates for model selection.
    finetune_train_set, finetune_val_set = split_train_validation(finetune_train_set)

    finetune_test_set, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=config.dataset_name,
            pos_class=config.pos_class,
            neg_class=config.neg_class,
            split="validation" if config.dataset_name == DatasetNames.chest_xray.value else "test",
            pytesting=False,
        )
    )

    # Load companion dataset for evaluation
    test_transform: Any = getattr(finetune_test_set, "transform", None)
    ambi_dataset = ImageFolder(root=config.companion_dataroot, transform=test_transform)
    filtered_dataset = load_ambiguous_companion(config, test_transform)

    # Create dataloaders
    train_dataloader = DataLoader(finetune_train_set, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(finetune_val_set, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(finetune_test_set, batch_size=config.batch_size, shuffle=False)
    ambi_dataloader = DataLoader(ambi_dataset, batch_size=config.batch_size, shuffle=False)
    filtered_dataloader = (
        DataLoader(filtered_dataset, batch_size=config.batch_size, shuffle=False)
        if filtered_dataset is not None
        else None
    )

    # Log dataset sizes
    logger.info(f"Training set size: {len(finetune_train_set)}")
    logger.info(f"Validation set size: {len(finetune_val_set)}")
    logger.info(f"Test set size: {len(finetune_test_set)}")
    logger.info(f"Companion/Ambiguity dataset size: {len(ambi_dataset)}")
    filtered_size = len(filtered_dataset) if filtered_dataset is not None else 0
    logger.info(f"Companion subset within confusion distance {MAX_CONFUSION_DISTANCE}: {filtered_size}")

    return train_dataloader, val_dataloader, test_dataloader, ambi_dataloader, filtered_dataloader


def construct_model_path(config: CLEvaluationArgs, model_name: str) -> Path:
    """
    Construct the finetuned model save path.

    Args:
        config: Evaluation configuration
        model_name: Name of the model (e.g., 'convnext', 'vit')

    Returns:
        Path object pointing to the model directory

    """
    filesdir = os.environ.get("FILESDIR", "./data")
    model_dir = (
        Path(filesdir)
        / "models"
        / "finetuned"
        / f"{config.dataset_name}.{config.pos_class}v{config.neg_class}"
        / model_name
    )
    return model_dir


class StaleCheckpointError(Exception):
    """Raised when a stored checkpoint was finetuned from a different backbone than the one now expected."""


EXPECTED_MODEL_IDS = {
    "convnext": CONVNEXT_MODEL_ID,
    "vit": VIT_MODEL_ID,
    "efficientnetv2": EFFICIENTNETV2_MODEL_ID,
    "swin": SWIN_MODEL_ID,
}

# Backbone used before checkpoints started recording their model id. Only `vit` differs
# from EXPECTED_MODEL_IDS: every pre-existing vit checkpoint came from the CXR backbone.
LEGACY_MODEL_IDS = {**EXPECTED_MODEL_IDS, "vit": VIT_LEGACY_CXR_MODEL_ID}


def expected_model_id(model_name: str) -> str:
    """
    Return the HuggingFace backbone id a model should be built from.

    Args:
        model_name: Name of the model (e.g., 'convnext', 'vit')

    Returns:
        HuggingFace model id.

    """
    if model_name not in EXPECTED_MODEL_IDS:
        raise ValueError(f"Unknown pretrained model: {model_name}")
    return EXPECTED_MODEL_IDS[model_name]


def construct_pretrained_model(model_name: str) -> nn.Module:
    """Build an untrained pretrained-model wrapper."""
    if model_name == "convnext":
        return ConvNext()
    if model_name == "vit":
        return ViT()
    if model_name == "efficientnetv2":
        return EfficientNetV2()
    if model_name == "swin":
        return Swin()
    raise ValueError(f"Unknown pretrained model: {model_name}")


def model_exists(config: CLEvaluationArgs, model_name: str) -> bool:
    """Check if a finetuned model already exists."""
    model_dir = construct_model_path(config, model_name)
    # checkpoint() appends model_name, creating model_dir/{model_name}/classifier.pth
    checkpoint_path = model_dir / model_name / "classifier.pth"
    # Check if checkpoint file exists
    exists = checkpoint_path.exists()
    logger.info("Checking for model at: %s (exists: %s)", checkpoint_path, exists)
    return exists


def load_finetuned_model(
    config: CLEvaluationArgs,
    model_name: str,
    device: str,
) -> nn.Module:
    """Load a previously finetuned model from disk."""
    model_dir = construct_model_path(config, model_name)
    # checkpoint() appends model_name, creating model_dir/{model_name}/
    model_checkpoint_dir = model_dir / model_name
    logger.info("Loading finetuned model from: %s", model_checkpoint_dir)

    checkpoint_file = model_checkpoint_dir / "classifier.pth"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_file}")

    device_type = DeviceType(device) if isinstance(device, str) else device

    # Load checkpoint
    checkpoint_data = torch.load(
        checkpoint_file, map_location=str(device_type.value if isinstance(device_type, DeviceType) else device_type)
    )

    # If checkpoint has trainer params (old style), use construct_classifier_from_checkpoint
    if (
        checkpoint_data.get("params")
        and isinstance(checkpoint_data["params"], dict)
        and "type" in checkpoint_data["params"]
    ):
        model, _, _, _, _ = construct_classifier_from_checkpoint(str(model_checkpoint_dir), device=device_type)
    else:
        # Refuse checkpoints finetuned from a different backbone: architectures can match
        # (e.g. both ViT-base/16-224), so strict=False below would silently load wrong weights.
        expected_id = expected_model_id(model_name)
        stored_id = checkpoint_data.get("model_id", LEGACY_MODEL_IDS.get(model_name))
        if stored_id != expected_id:
            raise StaleCheckpointError(
                f"Checkpoint at {checkpoint_file} was finetuned from '{stored_id}', but '{expected_id}' is expected."
            )

        stored_recipe = checkpoint_data.get("recipe_id")
        if stored_recipe != TRAINING_RECIPE_ID:
            raise StaleCheckpointError(
                f"Checkpoint at {checkpoint_file} was finetuned with recipe '{stored_recipe}', "
                f"but '{TRAINING_RECIPE_ID}' is expected."
            )

        # For pretrained models, create new instance and load state_dict
        model = construct_pretrained_model(model_name)

        # Load the state_dict into the underlying model
        device_str = device_type.value if isinstance(device_type, DeviceType) else str(device_type)

        # Try loading with strict=False to handle both old and new checkpoint formats
        state_dict = checkpoint_data["state"]

        # If keys have "model." prefix, we need to remove it or load into wrapper instead
        if any(k.startswith("model.") for k in state_dict.keys()):
            # Old format: load into the wrapper
            model.load_state_dict(state_dict, strict=False)
        else:
            # New format: load into the inner model
            model.model.load_state_dict(state_dict, strict=False)

        model.model.to(device_str)
        model.model.eval()

    return model


def train_model(
    model_type: PretrainedModels,
    epochs: int,
    device: DeviceType,
    train_dataloader: DataLoader,
    config: CLEvaluationArgs,
    val_dataloader: DataLoader | None = None,
) -> nn.Module:
    """
    Train or load pretrained model.

    If a finetuned model already exists for this dataset/class configuration,
    it will be loaded instead of retraining.

    Args:
        model_type: Type of model to train/load
        epochs: Number of epochs for training
        device: Device to use for training
        train_dataloader: DataLoader for training data
        config: Evaluation configuration
        val_dataloader: DataLoader used to pick the best epoch

    Returns:
        The trained or loaded model

    """
    model_name = model_type.value

    # Check if finetuned model already exists
    if model_exists(config, model_name):
        logger.info("Finetuned model found for %s. Loading from disk...", model_name)
        try:
            return load_finetuned_model(config, model_name, device)
        except StaleCheckpointError as e:
            logger.warning("Discarding stale checkpoint for %s: %s Retraining...", model_name, e)

    # Train new model
    logger.info("Retraining model %s...", model_type.value)
    model = construct_pretrained_model(model_name)
    model.retrain(train_dataloader, epochs=epochs, device=device, val_dataloader=val_dataloader)

    # Save the trained model
    model_dir = construct_model_path(config, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # For pretrained models, save state_dict directly
    model_checkpoint_dir = model_dir / model_name
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving finetuned model to: %s", model_checkpoint_dir)

    # Save state dict and metadata (save wrapper state_dict to maintain compatibility)
    save_dict = {
        "name": model_name,
        "model_id": model.model_id,  # backbone the weights were finetuned from
        "recipe_id": TRAINING_RECIPE_ID,  # fine-tuning recipe that produced the weights
        "state": model.state_dict(),  # Save wrapper state_dict for compatibility
    }
    torch.save(save_dict, model_checkpoint_dir / "classifier.pth")

    return model


def _log_evaluation_to_wandb(df: pd.DataFrame, config: "CLEvaluationArgs") -> None:
    """Log evaluation metrics to wandb."""
    filtered = df[df["dataset"] == f"{config.dataset_name} Companion Filtered"]["improved_hubris"]
    wandb.log(
        {
            # Improved absolute hubris over only the companion images the guiding estimator was
            # undecided about, i.e. within MAX_CONFUSION_DISTANCE of the decision boundary
            "hubris_improved_absolute_filter": filtered.values[0] if len(filtered) else None,
            "hubris_a_original": df[df["dataset"] == f"{config.dataset_name} Original"]["absolute_hubris"].values[0],
            "hubris_a_companion": df[df["dataset"] == f"{config.dataset_name} Companion"]["absolute_hubris"].values[0],
            "hubris_improved_original": df[df["dataset"] == f"{config.dataset_name} Original"][
                "improved_hubris"
            ].values[0],
            "hubris_improved_companion": df[df["dataset"] == f"{config.dataset_name} Companion"][
                "improved_hubris"
            ].values[0],
            "hubris_r_original": (
                df[df["dataset"] == f"{config.dataset_name} Original"]["relative_hubris"].values[0]
                if "relative_hubris" in df.columns
                else None
            ),
            "hubris_r_companion": (
                df[df["dataset"] == f"{config.dataset_name} Companion"]["relative_hubris"].values[0]
                if "relative_hubris" in df.columns
                else None
            ),
            "hubris_improved_r_original": (
                df[df["dataset"] == f"{config.dataset_name} Original"]["improved_relative_hubris"].values[0]
                if "improved_relative_hubris" in df.columns
                else None
            ),
            "hubris_improved_r_companion": (
                df[df["dataset"] == f"{config.dataset_name} Companion"]["improved_relative_hubris"].values[0]
                if "improved_relative_hubris" in df.columns
                else None
            ),
            "acd_original": df[df["dataset"] == f"{config.dataset_name} Original"]["acd"].values[0],
            "acd_companion": df[df["dataset"] == f"{config.dataset_name} Companion"]["acd"].values[0],
            "accuracy": df[df["dataset"] == f"{config.dataset_name} Original"]["accuracy"].values[0],
            "precision_original": df[df["dataset"] == f"{config.dataset_name} Original"]["precision"].values[0],
            "recall_original": df[df["dataset"] == f"{config.dataset_name} Original"]["recall"].values[0],
            "f1_original": df[df["dataset"] == f"{config.dataset_name} Original"]["f1_score"].values[0],
        }
    )


def _log_companion_metrics(config: "CLEvaluationArgs") -> None:
    """Log companion dataset metrics to wandb."""
    metrics_csv_pattern = os.path.join(config.companion_dataroot, "ambi", "*_metrics.csv")
    metrics_csv_files = glob.glob(metrics_csv_pattern)

    if not metrics_csv_files:
        logger.info("No metrics CSV found in companion_dataset folder")
        return

    # Read the most recent metrics CSV
    metrics_csv = sorted(metrics_csv_files)[-1]
    logger.info("Reading metrics from: %s", metrics_csv)
    try:
        metrics_df = pd.read_csv(metrics_csv)
        # Log all metrics from CSV to wandb
        for col in metrics_df.columns:
            value = metrics_df[col].iloc[0]
            if pd.notna(value):  # Only log non-null values
                wandb.log({f"{col}_companion": value})
        logger.info("Dataset metrics logged to wandb")
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
        logger.warning("Could not read metrics CSV: %s", e)


def main() -> None:  # pylint: disable=too-many-nested-blocks
    """Calculate and save model statistics based on the provided CLI arguments."""
    logger.info("Model evaluation is starting...")

    args = parser.parse_args()
    logger.debug(args)

    # Convert parsed arguments to dictionary and validate using Pydantic model
    args_dict = vars(args)

    # Setup config
    config = setup_config(args_dict)

    # Set random seed
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)
    logger.info(" > Seed: %s", config.seed)

    # create evaluation folder, if it doesn't exist
    os.makedirs(config.out_dir, exist_ok=True)

    # Load datasets (only once for all models)
    train_dataloader, val_dataloader, test_dataloader, ambi_dataloader, filtered_dataloader = load_datasets(config)

    # Get GAN ID for experiment tracking
    gan_id = args_dict.get("gan_id") or "unknown"

    # Loop through each model and create separate wandb run for each
    for model_enum in config.models:
        logger.info("\n%s", "=" * 60)
        logger.info("Starting evaluation for model: %s", model_enum.value)
        logger.info("%s\n", "=" * 60)

        # Initialize wandb for this model-dataset pair
        setup_wandb_for_model(config, model_enum, gan_id, config.seed)

        # Train or load model
        model = train_model(
            model_enum, config.epochs, config.device, train_dataloader, config, val_dataloader=val_dataloader
        )

        df = pd.DataFrame()
        df = pd.concat((df, evaluate(config, model, test_dataloader, name=f"{config.dataset_name} Original")))
        df = pd.concat(
            (
                df,
                evaluate(
                    config,
                    model,
                    ambi_dataloader,
                    name=f"{config.dataset_name} Companion",
                    compute_prf=False,
                ),
            )
        )

        if filtered_dataloader is not None:
            df = pd.concat(
                (
                    df,
                    evaluate(
                        config,
                        model,
                        filtered_dataloader,
                        name=f"{config.dataset_name} Companion Filtered",
                        compute_prf=False,
                    ),
                )
            )

        # save to CSV for local backup
        csv_path = os.path.join(
            config.out_dir, f"{datetime.now():%Y%m%d_%H%M}_{config.seed}_{model_enum.value}_evaluation.csv"
        )
        df.to_csv(path_or_buf=csv_path, index=False)
        logger.info("Evaluation results saved to CSV: %s", csv_path)

        # log results to wandb
        _log_evaluation_to_wandb(df, config)

        # Try to read existing metrics CSV from companion_dataset folder (only once per run)
        if model_enum == config.models[0]:  # Only log dataset metrics once
            _log_companion_metrics(config)

        checkpoint(model, model_enum.value, None, None, None, output_dir=config.out_dir, optimizer=None)
        logger.info("Model %s evaluation completed", model_enum.value)

        # Cleanup: Delete model and clear CUDA memory before next iteration
        del model
        torch.cuda.empty_cache()
        gc.collect()

        wandb.finish()

    logger.info("%s", "\n" + "=" * 60)
    logger.info("All model evaluations completed")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
