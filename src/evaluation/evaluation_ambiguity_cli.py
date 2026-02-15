"""CLI for ambiguity evaluation."""

import argparse
import logging
import os

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from src.enums import ClassifierType, DatasetNames, DeviceType
from src.utils.logging import configure_logging
from src.utils.utility_functions import setup_reprod

configure_logging()
logger = logging.getLogger(__name__)


class CLAmbiguityArgs(BaseModel):
    """CLI arguments for ambiguity evaluation."""

    dataroot: str = Field(..., description="Directory with dataset")
    out_dir: str = Field(..., description="Output directory for ambiguity evaluation")
    models: list[ClassifierType] = Field(..., description="Models to evaluate")
    datasets: list[DatasetNames] = Field(..., description="Datasets to evaluate")
    device: DeviceType = Field(default=DeviceType.cpu, description="Device to use")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    gan_id: str | None = Field(default=None, description="GAN experiment ID for tracking")


def parse_args() -> CLAmbiguityArgs:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ambiguity evaluation")
    parser.add_argument("--data", dest="dataroot", default=f"{os.environ.get('FILESDIR', '')}/data")
    parser.add_argument("--out-dir", dest="out_dir", required=True)
    parser.add_argument("--models", dest="models", nargs="+", required=True)
    parser.add_argument("--datasets", dest="datasets", nargs="+", required=True)
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gan-id", dest="gan_id", default=None)

    args = parser.parse_args()
    args_dict = vars(args)

    try:
        return CLAmbiguityArgs.model_validate(args_dict)
    except ValidationError as exc:
        logger.error("Argument validation error: %s", exc)
        raise


def main() -> None:
    """Entry point for ambiguity evaluation."""
    logger.info("Ambiguity evaluation is starting...")

    config = parse_args()
    config.seed = np.random.randint(100000) if config.seed is None else config.seed
    setup_reprod(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    logger.info(" > Seed: %s", config.seed)
    logger.info(" > Device: %s", config.device)
    logger.info(" > Models: %s", [model.value for model in config.models])
    logger.info(" > Datasets: %s", [dataset.value for dataset in config.datasets])
    logger.info(" > Output dir: %s", config.out_dir)
    if config.gan_id:
        logger.info(" > GAN ID: %s", config.gan_id)

    logger.warning("Ambiguity evaluation is not implemented yet.")


if __name__ == "__main__":
    main()
