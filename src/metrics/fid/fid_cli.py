"""Module to run FID from CLI."""

import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import ValidationError
from torch import nn
from torcheval.metrics import FrechetInceptionDistance
from tqdm import tqdm

from src.datasets.load import load_dataset
from src.models import FIDArgs, LoadDatasetParams
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.logging import configure_logging

# Load environment variables
load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)
logger.info("FID calculation is starting...")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", dest="dataroot", default=f"{os.environ['FILESDIR']}/data", help="Directory with dataset")
parser.add_argument("--dataset", dest="dataset", default="fashion-mnist", help="Dataset (mnist, fashion-mnist, etc.)")
parser.add_argument("--pos", dest="pos_class", default=3, type=int, help="Positive class for binary classification")
parser.add_argument("--neg", dest="neg_class", default=0, type=int, help="Negative class for binary classification")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size to use")
parser.add_argument(
    "--model-path", dest="model_path", default=None, type=str, help="Path to classifier. If none, uses InceptionV3"
)
parser.add_argument("--num-workers", type=int, default=6, help="Number of worker processes for data loading")
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda, or cpu)")
parser.add_argument("--name", dest="name", default=None, help="Name of generated .npz file")


def get_feature_map_function(config: FIDArgs) -> nn.Module | None:
    """Get the feature map function based on model path."""
    if config.model_path is not None:

        model = construct_classifier_from_checkpoint(config.model_path, config.device)[0]
        model.to(config.device)
        model.eval()
        if hasattr(model, "output_feature_maps"):
            model.output_feature_maps = True
            return model
    return None


def main() -> None:
    """Calculate and save FID statistics based on the provided CLI arguments."""
    args = parser.parse_args()
    logger.debug(args)

    # Convert parsed arguments to dictionary and validate using Pydantic model
    args_dict = vars(args)

    try:
        config = FIDArgs(**args_dict)
    except ValidationError as e:
        logger.error(f"Argument validation error: {e}")
        raise ValidationError(e) from e

    # Logging the arguments
    logger.info(config)

    fid = FrechetInceptionDistance(model=get_feature_map_function(config), feature_dim=2048, device=config.device)

    # Load dataset
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
    logger.info(f" > Using dataset: {config.dataset_name}")

    logger.info(f"Dataset size: {len(dataset)}")

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    # Calculate mu and sigma
    for batch in tqdm(dataloader):
        fid.update(batch[0], is_real=True)
    m = (fid.real_sum / fid.num_real_images).unsqueeze(0)
    s = fid.real_cov_sum - fid.num_real_images * torch.matmul(m.T, m)

    # Save FID statistics
    os.makedirs(os.path.join(config.dataroot, "fid-stats"), exist_ok=True)
    stats_filename = os.path.join(config.dataroot, "fid-stats", f'{config.name or f"stats.{config.dataset_name}"}')
    with open(f"{stats_filename}.npz", "wb") as f:
        np.savez(
            f,
            mu=m.squeeze().cpu().numpy(),
            sigma=s.cpu().numpy(),
            real_sum=fid.real_sum.cpu().numpy(),
            real_cov_sum=fid.real_cov_sum.cpu().numpy(),
            num_real_images=fid.num_real_images.cpu().numpy(),
        )
    logger.info(f"FID statistics saved to {stats_filename}.npz")


if __name__ == "__main__":
    main()
