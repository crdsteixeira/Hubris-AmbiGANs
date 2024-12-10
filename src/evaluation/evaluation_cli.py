"""CL for models evaluation."""

import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.datasets.load import load_dataset
from src.enums import DatasetNames
from src.evaluation.pretrained_models import ConvNext, ViT
from src.metrics.hubris import Hubris
from src.models import CLEvaluationArgs, LoadDatasetParams
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import setup_reprod

# Load environment variables
load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", dest="dataroot", default=f"{os.environ['FILESDIR']}/data", help="Directory with dataset")
parser.add_argument("--companion-data", dest="companion_dataroot", help="Directory with companion dataset")
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


def evaluate(config: CLEvaluationArgs, model: nn.Module, loader: DataLoader, name: str) -> pd.DataFrame:
    """Evaluate model using companion dataset."""
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in tqdm(loader):
            preds.append(model(images.to(config.device)).cpu())
        full_preds = torch.cat(preds)

    hubris = Hubris(C=None, dataset_size=len(full_preds))
    absolute_hubris = hubris.compute(full_preds, ref_preds=None)

    df = pd.DataFrame()
    df = df.assign(
        dataset=[name],
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

    # Set random seed
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)
    logger.info(f" > Seed: {config.seed}")

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

    # Retrain models
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    if config.dataset_name in (DatasetNames.mnist, DatasetNames.fashion_mnist):
        model = ConvNext()
        model.retrain(train_dataloader, epochs=config.epochs, device=config.device)
    elif config.dataset_name in (DatasetNames.chest_xray):
        model = ViT()
        model.retrain(train_dataloader, epochs=config.epochs, device=config.device)

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    ambi_dataloader = DataLoader(ambi_dataset, batch_size=config.batch_size, shuffle=False)

    df = pd.DataFrame()
    df = pd.concat((df, evaluate(config, model, test_dataloader, name=f"{config.dataset_name} Original")))
    df = pd.concat((df, evaluate(config, model, ambi_dataloader, name=f"{config.dataset_name} Companion")))
    df.to_csv(
        path_or_buf=os.path.join(config.out_dir, f"{datetime.now():%Y%m%d_%H%M}_{config.seed}_evaluation.csv"),
        index=False,
    )

    logger.info(f"Model evaluation saved to {config.out_dir}")


if __name__ == "__main__":
    main()
