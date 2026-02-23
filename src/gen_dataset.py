"""Module to generate a dataset from a pre-trained generator."""

import gc
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import ValidationError
from pymdma.image.models.features import ExtractorFactory
from torchvision.transforms.functional import InterpolationMode, resize
from torchvision.utils import save_image
from tqdm import tqdm

from src.metrics.fid.fid import FID
from src.metrics.image_quality import calculate_pymdma_metrics
from src.models import CLDatasetArgs
from src.utils.checkpoint import (
    construct_classifier_from_checkpoint,
    construct_gan_from_checkpoint,
)
from src.utils.logging import configure_logging
from src.utils.utility_functions import gen_seed, setup_reprod

load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:  # pylint: disable=too-many-statements
    """Run process to generate dataset."""
    logger.info("Dataset generation is starting...")

    config = parse_args()

    logger.info(config)

    config.seed = gen_seed() if config.seed is None else config.seed
    setup_reprod(config.seed)

    # Check if companion dataset already exists
    companion_dataset_exists = os.path.isdir(config.out_dir)

    if companion_dataset_exists:
        logger.info(f"✓ Companion dataset exists at {config.out_dir}")
        logger.info("Skipping dataset generation.")
        return

    os.makedirs(config.out_dir, exist_ok=True)

    # load generator
    G, _, _, _ = construct_gan_from_checkpoint(config.gan_path, device=config.device)
    G.eval()
    G.to(config.device)

    if config.fid_stats_path is not None:
        fid = FID(fid_stats_file=config.fid_stats_path, dims=2048, n_images=config.n_samples, device=config.device)
        # for pymdma
        extractor = ExtractorFactory.model_from_name(name="dino_vits8")
        all_synt_features = []

    # Load estimator once if provided
    estimator = None
    if config.estimator_path is not None:
        estimator, _, _, _, _ = construct_classifier_from_checkpoint(config.estimator_path, device=config.device)
        estimator.eval()
        logger.info("Estimator loaded successfully.")
    confusion_distance_sum = 0.0
    confusion_distance_count = 0

    with torch.no_grad():
        for i in tqdm(range(config.n_samples)):
            noise = torch.randn((1, G.params.z_dim), device=config.device)
            gen_image = G(noise)  # .cpu()

            # Calculate confusion distance for this image if estimator is provided
            if estimator is not None:
                prob = estimator(gen_image)[0].item()
                confusion_distance_sum += abs(0.5 - prob)
                confusion_distance_count += 1
            if config.fid_stats_path is not None:
                fid.update(gen_image, (0, 0))

            # move to CPU and normalize to [0, 1] for saving and feature extraction
            gen_image = gen_image.cpu()
            gen_image.clamp_(min=-1.0, max=1.0)
            gen_image.sub_(-1.0).div_(max(1.0 - (-1.0), 1e-5))

            # for pymdma
            if config.fid_stats_path is not None:
                pymdma_images = gen_image
                # Check if the images need to be converted to RGB
                if pymdma_images.shape[1] != 3:
                    # Convert to RGB by repeating across the channel dimension
                    pymdma_images = pymdma_images.repeat(1, 3, 1, 1)
                features = extractor(pymdma_images).detach().cpu().numpy()
                all_synt_features.append(features)
            if config.img_size is not None:
                gen_image = resize(gen_image, config.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
            save_image(gen_image, os.path.join(config.out_dir, f"image_{i:06d}.png"))

    torch.cuda.empty_cache()

    logger.info(f"Generated test noise, stored in {config.out_dir}")
    if config.fid_stats_path is not None and config.calculate_stats:
        dataset_fid = fid.finalize()
        logger.info(f"Dataset FID is: {dataset_fid}")
        # for pymdma
        all_synt_features = np.concatenate(all_synt_features, axis=0)
        all_real_features = fid.data["all_features"]
        pymdma_metrics = calculate_pymdma_metrics(all_real_features, all_synt_features)
        pymdma_metrics = pymdma_metrics.assign(fid=[dataset_fid])

        # Add average confusion distance if it was calculated
        if confusion_distance_count > 0:
            avg_confusion_distance = confusion_distance_sum / confusion_distance_count
            pymdma_metrics = pymdma_metrics.assign(avg_confusion_distance=[avg_confusion_distance])
            logger.info(f"Added average confusion distance to metrics: {avg_confusion_distance:.6f}")

        pymdma_metrics.to_csv(
            path_or_buf=os.path.join(config.out_dir, f"{datetime.now():%Y%m%d_%H%M}_{config.seed}_metrics.csv"),
            index=False,
        )
        logger.info(f"Calculated sythetic images metrics, stored in {config.out_dir}")
        # Clean up large objects to free CUDA memory
        del fid
        del all_synt_features
        del all_real_features
        del extractor

    # Clean up estimator if it was loaded
    if estimator is not None:
        del estimator

    # Delete generator and collect garbage
    del G
    torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> CLDatasetArgs:
    """Parse arguments from cli."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--gan-path", dest="gan_path", required=True, type=str, help="Directory where pre-trained AmbiGAN is located"
    )
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--n-samples", dest="n_samples", required=True, type=int, help="Number of samples to be generated"
    )
    parser.add_argument("--img-size", dest="img_size", type=int, default=None, help="Latent space dimension")
    parser.add_argument("--out-dir", dest="out_dir", required=True, type=str, help="Directory to store the dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, cuda or cpu")
    parser.add_argument(
        "--fid-stats-path", dest="fid_stats_path", type=str, default=None, help="Path to FID statistics file"
    )
    parser.add_argument(
        "--estimator-path",
        dest="estimator_path",
        type=str,
        default=None,
        help="Path to ambiguity estimator checkpoint for computing confusion distance",
    )
    parser.add_argument(
        "--skip-stats", dest="calculate_stats", action="store_false", help="Skip calculating and saving metrics"
    )

    args = parser.parse_args()
    logger.debug(args)

    args_dict = vars(args)

    try:
        config = CLDatasetArgs(**args_dict)
    except ValidationError as e:
        logger.error(f"Argument validation error: {e}")
        raise
    return config


if __name__ == "__main__":
    main()
