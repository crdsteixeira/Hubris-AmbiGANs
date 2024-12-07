"""Module to generate FID statistics from dataset."""

import itertools
import logging
import os
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from dotenv import load_dotenv
from pydantic import ValidationError

from src.models import CLFIDStatsArgs
from src.utils.logging import configure_logging

load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataroot",
    type=str,
    dest="dataroot",
    default=f"{os.environ['FILESDIR']}/data",
    help="Directory with dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    dest="dataset",
    default="mnist",
    help="Dataset to use (mnist, fashion-mnist, cifar10 or chest x-ray)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use, cuda or cpu",
)


def main() -> None:
    """Run process to generate FID statistics from given dataset."""
    logger.info("Calculating FID statistics...")
    args = parser.parse_args()
    logger.debug(args)
    args_dict = vars(args)

    try:
        config = CLFIDStatsArgs(**args_dict)
    except ValidationError as e:
        logger.error(f"Argument validation error: {e}")
        raise

    logger.info(config)

    if config.n_classes is not None:
        for neg_class, pos_class in itertools.combinations(range(config.n_classes), 2):
            logger.info(f"{neg_class}vs{pos_class}")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.metrics.fid.fid_cli",
                    "--data",
                    config.dataroot,
                    "--dataset",
                    config.dataset,
                    "--device",
                    config.device,
                    "--pos",
                    str(pos_class),
                    "--neg",
                    str(neg_class),
                ],
                check=False,
                env=os.environ.copy(),
            )
    else:
        raise RuntimeError("Missing number of classes for dataset.")


if __name__ == "__main__":
    main()
