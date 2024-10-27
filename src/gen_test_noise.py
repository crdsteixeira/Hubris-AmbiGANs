"""Modure to generate test noise."""

import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from dotenv import load_dotenv
from pydantic import ValidationError

from src.models import CLTestNoiseArgs
from src.utils.logging import configure_logging
from src.utils.utility_functions import create_and_store_z, gen_seed, set_seed

load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Run process to generate test noise."""
    logger.info("Test Noise generation is starting...")

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--nz", dest="nz", required=True, type=int, help="Number of sample to be generated")
    parser.add_argument("--z-dim", dest="z_dim", required=True, type=int, help="Latent space dimension")
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="Directory to store thes test noise",
        default=f"{os.environ['FILESDIR']}/data/z",
    )

    args = parser.parse_args()
    logger.debug(args)

    args_dict = vars(args)

    try:
        config = CLTestNoiseArgs(**args_dict)
    except ValidationError as e:
        logger.error(f"Argument validation error: {e}")
        raise ValidationError(e) from e

    logger.info(config)

    config.seed = gen_seed() if config.seed is None else config.seed

    set_seed(config.seed)

    _, test_noise_path = create_and_store_z(
        config=config,
    )

    logger.info(f"Generated test noise, stored in {test_noise_path}")


if __name__ == "__main__":
    main()
