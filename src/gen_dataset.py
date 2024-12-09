"""Module to generate a dataset from a pre-trained generator."""

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
from src.models import CLDatasetArgs
from src.utils.checkpoint import construct_gan_from_checkpoint
from src.utils.logging import configure_logging
from src.utils.utility_functions import calculate_pymdma_metrics, gen_seed, set_seed

load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Run process to generate dataset."""
    logger.info("Dataset generation is starting...")

    config = parse_args()

    logger.info(config)

    config.seed = gen_seed() if config.seed is None else config.seed

    set_seed(config.seed)

    G, _, _, _ = construct_gan_from_checkpoint(config.gan_path, device=config.device)
    G.eval()

    if config.fid_stats_path is not None:
        fid = FID(fid_stats_file=config.fid_stats_path, dims=2048, n_images=config.n_samples, device=config.device)
        # for pymdma
        extractor = ExtractorFactory.model_from_name(name="dino_vits8")
        all_synt_features = []

    os.makedirs(config.out_dir, exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(config.n_samples)):
            noise = torch.randn((1, G.params.z_dim), device=config.device)
            gen_image = G(noise).cpu()
            if config.fid_stats_path is not None:
                fid.update(gen_image, (0, 0))
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

    logger.info(f"Generated test noise, stored in {config.out_dir}")
    if config.fid_stats_path is not None:
        dataset_fid = fid.finalize()
        logger.info(f"Dataset FID is: {dataset_fid}")
        # for pymdma
        all_synt_features = np.concatenate(all_synt_features, axis=0)
        all_real_features = fid.data["all_features"]
        pymdma_metrics = calculate_pymdma_metrics(all_real_features, all_synt_features)
        pymdma_metrics = pymdma_metrics.assign(fid=[dataset_fid])
        pymdma_metrics.to_csv(
            path_or_buf=os.path.join(config.out_dir, f"{datetime.now():%Y%m%d_%H%M}_{config.seed}_metrics.csv"),
            index=False,
        )
        logger.info(f"Calculated sythetic images metrics, stored in {config.out_dir}")


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
