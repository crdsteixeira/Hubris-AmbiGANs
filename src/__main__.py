"""Module to run AmbiGAN process."""

import argparse
import logging
import os
import subprocess

from dotenv import load_dotenv
from pydantic import ValidationError

from src.gan import gan_cli
from src.models import (
    CLAmbigan,
    ClassifierClasses,
    CLFIDStatsArgs,
    CLTestNoiseArgs,
    CLTrainArgs,
    ConfigGAN,
    ConfigMain,
)
from src.utils.logging import configure_logging
from src.utils.read_config import read_main_config

configure_logging()
logger = logging.getLogger(__name__)


def gen_test_noise(config: ConfigMain) -> None:
    """Generate test noise using config parameters."""
    params = CLTestNoiseArgs(
        seed=config.test_noise_seed,
        nz=config.fixed_noise,
        z_dim=config.model.z_dim,
        out_dir=os.path.join(config.out_dir, config.data_dir, "z"),
    )

    args = [
        "python",
        "-m",
        "src.gen_test_noise",
        "--seed",
        str(params.seed),
        "--nz",
        str(params.nz),
        "--z-dim",
        str(params.z_dim),
        "--out-dir",
        str(params.out_dir),
    ]

    subprocess.run(args, check=True)


def gen_pairwise_inception(config: ConfigMain) -> None:
    """Generate pairwise inception using config parameters."""
    params = CLFIDStatsArgs(
        dataroot=os.path.join(config.out_dir, config.data_dir),
        dataset=config.dataset.name,
        device=config.device,
    )

    args = [
        "python",
        "-m",
        "src.metrics.fid.fid_cli",
        "--data",
        params.dataroot,
        "--dataset",
        params.dataset,
        "--device",
        params.device,
        "--pos",
        str(config.dataset.binary.pos),
        "--neg",
        str(config.dataset.binary.neg),
    ]

    subprocess.run(args, check=True)


def gen_classifiers(config: ConfigMain, classifier: ClassifierClasses) -> None:
    """Generate classifier using config parameters and classifier."""
    params = CLTrainArgs(
        dataset_name=config.dataset.name,
        pos_class=config.dataset.binary.pos,
        neg_class=config.dataset.binary.neg,
        data_dir=config.data_dir,
        out_dir=os.path.join(
            config.out_dir,
            "models",
        ),
        name=classifier.name,
        batch_size=classifier.batch_size,
        c_type=classifier.c_type,
        epochs=classifier.epochs,
        early_stop=classifier.early_stop,
        early_acc=classifier.early_acc,
        lr=classifier.lr,
        nf=classifier.nf,
        seed=classifier.seed,
        device=config.device,
        ensemble_type=classifier.ensemble_type,
        ensemble_output_method=classifier.ensemble_output_method,
    )

    args: list[str] = [
        "python",
        "-m",
        "src.classifier.classifier_cli",
        "--dataset_name",
        params.dataset_name,
        "--pos_class",
        str(params.pos_class),
        "--neg_class",
        str(params.neg_class),
        "--data_dir",
        params.data_dir,
        "--out_dir",
        params.out_dir,
        "--batch_size",
        str(classifier.batch_size),
        "--c_type",
        classifier.c_type,
        "--epochs",
        str(classifier.epochs),
        "--lr",
        str(classifier.lr),
        "--nf",
        str(classifier.nf),
        "--seed",
        str(classifier.seed),
        "--device",
        params.device,
    ]

    if classifier.name is not None:
        args.extend(["--name", classifier.name])
    if classifier.early_stop is not None:
        args.extend(["--early_stop", str(classifier.early_stop)])
    if classifier.early_acc is not None:
        args.extend(["--early_acc", str(classifier.early_acc)])
    if classifier.ensemble_type is not None:
        args.extend(["--ensemble_type", classifier.ensemble_type])
    if classifier.ensemble_output_method is not None:
        args.extend(["--ensemble_output_method", classifier.ensemble_output_method])

    subprocess.run(args, check=True)


def gen_gan(config: ConfigMain, fid_stats_path: str, test_noise: str) -> None:
    """Generate GAN using config parameters."""
    params = ConfigGAN(
        project=config.project,
        name=config.name,
        out_dir=config.out_dir,
        data_dir=config.data_dir,
        fid_stats_path=fid_stats_path,
        fixed_noise=config.fixed_noise,
        test_noise=test_noise,
        device=config.device,
        num_workers=config.num_workers,
        num_runs=config.num_runs,
        step_1_seeds=config.step_1_seeds,
        step_2_seeds=config.step_2_seeds,
        dataset=config.dataset,
        model=config.model,
        optimizer=config.optimizer,
        train=config.train,
    )

    gan_cli.main(config=params)


def parse_args() -> CLAmbigan:
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Train AmbiGAN with a config file")
    parser.add_argument("--config", type=str, dest="config_path", required=True, help="Config file")

    # Parse the arguments from command line
    args = parser.parse_args()
    # Convert argparse Namespace to dictionary for validation
    args_dict = vars(args)

    # Validate parsed arguments using Pydantic model
    try:
        validated_args = CLAmbigan.model_validate(args_dict)
        return validated_args
    except ValidationError as e:
        # Print validation error and exit
        logger.error(f"Validation error: {e}")
        raise


def main() -> None:
    """Run process of AmbiGAN training."""
    load_dotenv()
    logger.info("AmbiGANs is starting...")

    args = parse_args()
    config = read_main_config(args.config_path)
    logger.info(f"Loaded experiment configuration from {args.config_path}")

    if config.gen_test_noise:
        gen_test_noise(config)
    if config.gen_pairwise_inception:
        gen_pairwise_inception(config)
    if config.gen_classifiers and config.classifiers is not None:
        for classifier in config.classifiers:
            gen_classifiers(config, classifier)
    if config.gen_gan:
        fid_stats_path = os.path.join(
            config.out_dir,
            config.data_dir,
            "fid-stats",
            f"stats.{config.dataset.name}.{config.dataset.binary.pos}v{config.dataset.binary.neg}.npz",
        )
        test_noise = os.path.join(
            config.out_dir,
            config.data_dir,
            "z",
            f"z_{config.fixed_noise}_{config.model.z_dim}",
        )

        if config.train.step_2.classifier:
            config.train.step_2.classifier = [
                os.path.join(
                    config.out_dir,
                    "models",
                    f"{config.dataset.name}.{config.dataset.binary.pos}v{config.dataset.binary.neg}",
                    c_path,
                )
                for c_path in config.train.step_2.classifier
            ]
        gen_gan(config, fid_stats_path=fid_stats_path, test_noise=test_noise)


if __name__ == "__main__":
    main()
