"""CL for models evaluation."""

import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.datasets.load import load_dataset
from src.enums import PretrainedModels
from src.evaluation.pretrained_models import ConvNext, ViT
from src.metrics.accuracy import binary_accuracy
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
parser.add_argument("--model", dest="model", help="Pretrained model to be evaluated")
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


def evaluate(config: CLEvaluationArgs, model: nn.Module, loader: DataLoader, name: str) -> pd.DataFrame:
    """Evaluate model using companion dataset."""
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for images, label in tqdm(loader):
            preds.append(model(images.to(config.device)).cpu())
            labels.append(label)
        full_preds = torch.cat(preds)
        full_labels = torch.cat(labels)

    accuracy = binary_accuracy(full_preds, full_labels, avg=True, threshold=0.50).item()
    hubris = Hubris(C=None, dataset_size=len(full_preds))
    absolute_hubris = hubris.compute(full_preds, ref_preds=None)

    df = pd.DataFrame()
    df = df.assign(
        dataset=[name],
        accuracy=[accuracy],
        absolute_hubris=[absolute_hubris],
        acd=[(0.50 - full_preds).abs().mean().item()],
    )

    # Load estimator if needed, for relative Hubris
    if config.estimator_path is not None:
        C, _, _, _, _ = construct_classifier_from_checkpoint(config.estimator_path, device=config.device)
        preds = []
        with torch.no_grad():
            for images, _ in tqdm(loader):
                preds.append(C(images.to(config.device)).cpu())
            ref_preds = torch.cat(preds)

        relative_hubris = hubris.compute(full_preds, ref_preds=ref_preds)
        df = df.assign(
            relative_hubris=[relative_hubris],
        )

    return df


def main() -> None:
    """Calculate and save model statistics based on the provided CLI arguments."""

    logger.info("Model evaluation is starting...")

    args = parser.parse_args()
    logger.debug(args)

    # Convert parsed arguments to dictionary and validate using Pydantic model
    args_dict = vars(args)

    try:
        config = CLEvaluationArgs(**args_dict)
    except ValidationError as e:
        logger.error(f"Argument validation error: {e}")
        raise

    # Logging the arguments
    logger.info(config)

    # Initialize wandb with GAN ID for experiment tracking
    wandb.init(
        project="AmbiGAN-Evaluation",
        name=f"{config.dataset_name}.{config.pos_class}v{config.neg_class}-{config.model.value}",
        id=args.gan_id,
        resume="allow",
        config={
            "model": config.model.value,
            "dataset": config.dataset_name,
            "pos_class": config.pos_class,
            "neg_class": config.neg_class,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
        },
    )

    # Set random seed
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)
    logger.info(f" > Seed: {config.seed}")
    wandb.config.update({"seed": config.seed})

    # create evaluation folder, if it doesn't exist
    os.makedirs(config.out_dir, exist_ok=True)

    # Load original train dataset for retrain
    dataset, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=config.dataset_name,
            pos_class=config.pos_class,
            neg_class=config.neg_class,
            train=True,
            pytesting=False,
        )
    )

    # Load test dataset
    test_dataset, _, _ = load_dataset(
        LoadDatasetParams(
            dataroot=config.dataroot,
            dataset_name=config.dataset_name,
            pos_class=config.pos_class,
            neg_class=config.neg_class,
            train=False,
            pytesting=False,
        )
    )

    # Load companion dataset
    ambi_dataset = ImageFolder(root=config.companion_dataroot, transform=test_dataset.transform)

    # TODO: Check if retrained model already exists and load it, otherwise retrain
    logger.info(f"Retraining {config.model.value} model...")
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    if config.model == PretrainedModels.convnext:
        model = ConvNext()
        model.retrain(train_dataloader, epochs=config.epochs, device=config.device)
    elif config.model == PretrainedModels.vit:
        model = ViT()
        model.retrain(train_dataloader, epochs=config.epochs, device=config.device)
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    ambi_dataloader = DataLoader(ambi_dataset, batch_size=config.batch_size, shuffle=False)

    df = pd.DataFrame()
    df = pd.concat((df, evaluate(config, model, test_dataloader, name=f"{config.dataset_name} Original")))
    df = pd.concat((df, evaluate(config, model, ambi_dataloader, name=f"{config.dataset_name} Companion")))

    # save to CSV for local backup
    csv_path = os.path.join(config.out_dir, f"{datetime.now():%Y%m%d_%H%M}_{config.seed}_evaluation.csv")
    df.to_csv(path_or_buf=csv_path, index=False)
    logger.info(f"Evaluation results saved to CSV: {csv_path}")

    # log results to wandb
    wandb.log({"hubris_a_original": df[df["dataset"] == f"{config.dataset_name} Original"]["absolute_hubris"].values[0],
               "hubris_a_companion": df[df["dataset"] == f"{config.dataset_name} Companion"]["absolute_hubris"].values[0],
               "hubris_r_original": df[df["dataset"] == f"{config.dataset_name} Original"]["relative_hubris"].values[0] if "relative_hubris" in df.columns else None,
               "hubris_r_companion": df[df["dataset"] == f"{config.dataset_name} Companion"]["relative_hubris"].values[0] if "relative_hubris" in df.columns else None,
               "acd_original": df[df["dataset"] == f"{config.dataset_name} Original"]["acd"].values[0],
               "acd_companion": df[df["dataset"] == f"{config.dataset_name} Companion"]["acd"].values[0],
               "accuracy": df[df["dataset"] == f"{config.dataset_name} Original"]["accuracy"].values[0],
               })
    # wandb.save(csv_path)

    checkpoint(model, config.model.value, None, None, None, output_dir=config.out_dir, optimizer=None)
    logger.info(f"Model evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
