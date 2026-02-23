"""Module to run AmbiGAN process."""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys

import torch
from dotenv import load_dotenv
from pydantic import ValidationError

from src.gan import gan_cli
from src.models import (
    CLAmbigan,
    ClassifierClasses,
    CLDatasetArgs,
    CLEvaluationArgs,
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


def find_latest_gan_estimator_paths(config: ConfigMain) -> tuple[str, str | None]:
    """Find latest run executed for this specific subset."""
    # For inherently binary datasets (chest-xray), don't append the binary class suffix
    # For multi-class datasets where we select a binary subset (mnist-1v0), do append it
    if config.dataset.name == "chest-xray":
        dataset_dir = config.dataset.name.value
    else:
        dataset_dir = f"{config.dataset.name.value}-{config.dataset.binary.pos}v{config.dataset.binary.neg}"

    gan_root = os.path.join(
        config.out_dir,
        "AmbiGAN",
        dataset_dir,
    )

    subdirs = [os.path.join(gan_root, d) for d in os.listdir(gan_root) if os.path.isdir(os.path.join(gan_root, d))]

    if not subdirs:
        raise ValueError(f"No subdirectories found in {gan_root}")

    estimator_name = None
    try:
        if getattr(config, "train", None) and getattr(config.train, "step_2", None):
            classifier_list = getattr(config.train.step_2, "classifier")
            if classifier_list:
                estimator_name = classifier_list[0]
    except (AttributeError, IndexError, TypeError):
        estimator_name = None

    estimator_path = (
        os.path.join(
            config.out_dir,
            "models",
            f"{config.dataset.name}.{config.dataset.binary.pos}v{config.dataset.binary.neg}",
            estimator_name,
        )
        if estimator_name
        else None
    )

    # select the most recently modified subdirectory
    return max(subdirs, key=os.path.getmtime), estimator_path


def gen_test_noise(config: ConfigMain) -> None:
    """Generate test noise using config parameters."""
    params = CLTestNoiseArgs(
        seed=config.test_noise_seed,
        nz=config.fixed_noise,
        z_dim=config.model.z_dim,
        out_dir=os.path.join(config.out_dir, config.data_dir, "z"),
    )

    args = [
        sys.executable,
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

    subprocess.run(args, check=True, env=os.environ.copy())


def gen_pairwise_inception(config: ConfigMain) -> None:
    """Generate pairwise inception using config parameters."""
    params = CLFIDStatsArgs(
        dataroot=os.path.join(config.out_dir, config.data_dir),
        dataset=config.dataset.name,
        device=config.device,
    )

    args = [
        sys.executable,
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

    subprocess.run(args, check=True, env=os.environ.copy())


def gen_classifiers(config: ConfigMain, classifier: ClassifierClasses) -> None:
    """Generate classifier using config parameters and classifier."""
    params = CLTrainArgs(
        dataset_name=config.dataset.name,
        pos_class=config.dataset.binary.pos,
        neg_class=config.dataset.binary.neg,
        data_dir=os.path.join(config.out_dir, config.data_dir),
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
        sys.executable,
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

    subprocess.run(args, check=True, env=os.environ.copy())


def gen_gan(config: ConfigMain, fid_stats_path: str, test_noise: str) -> None:
    """Generate GAN using config parameters."""
    params = ConfigGAN(
        project=config.project,
        name=config.name,
        out_dir=config.out_dir,
        data_dir=os.path.join(config.out_dir, config.data_dir),
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
    # Clean up CUDA memory to avoid OOM in subsequent steps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def gen_dataset(
    config: ConfigMain, fid_stats_path: str, latest_gan_path: str, estimator_path: str | None = None
) -> None:
    """Generate dataset using config parameters."""
    # search for the directory that matches the first classifier, and select the last epoch
    gan_path = None
    try:
        if config.classifiers and config.classifiers[0].name:
            for entry in os.listdir(latest_gan_path):
                if entry.startswith(config.classifiers[0].name):
                    gan_path = os.path.join(latest_gan_path, entry, str(config.train.step_2.epochs))
                    break
    except (OSError, IndexError) as e:
        logger.warning(f"Could not find classifier directory: {e}")

    if not config.evaluation:
        raise ValueError("evaluation config is required for gen_dataset")

    if not gan_path:
        raise ValueError("Could not determine GAN path from classifier directory")

    params = CLDatasetArgs(
        seed=config.test_noise_seed,
        # img_size=config.img_size, (default)
        n_samples=config.evaluation.companion_n_samples,
        out_dir=os.path.join(latest_gan_path, "companion_dataset", "ambi"),
        gan_path=gan_path,
        device=config.device,
        fid_stats_path=fid_stats_path,
        estimator_path=estimator_path,
    )

    # Skip if companion dataset already exists
    if os.path.exists(params.out_dir) and len(os.listdir(params.out_dir)) > 0:
        logger.info(f"Companion dataset already exists at {params.out_dir}, skipping generation")
        return

    args = [
        sys.executable,
        "-m",
        "src.gen_dataset",
        "--gan-path",
        str(params.gan_path),
        "--seed",
        str(params.seed),
        "--n-samples",
        str(params.n_samples),
        "--device",
        str(params.device),
        "--out-dir",
        str(params.out_dir),
        "--fid-stats-path",
        str(fid_stats_path),
    ]
    if params.estimator_path is not None:
        args.extend(["--estimator-path", str(params.estimator_path)])

    subprocess.run(args, check=True, env=os.environ.copy())
    # Clean up CUDA memory after dataset generation completes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def gen_synthetic_dataset(config: ConfigMain, fid_stats_path: str, latest_gan_path: str) -> None:
    """Generate synthetic dataset from step_1 checkpoint using config parameters."""
    # Use step_1 checkpoint directly from the latest GAN path
    step_1_path = os.path.join(latest_gan_path, "step_1")

    # Read train_state.json to extract the epoch
    train_state_path = os.path.join(step_1_path, "train_state.json")
    with open(train_state_path, encoding="utf-8") as f:
        train_state = json.load(f)
        epoch = train_state.get("epoch")

    gan_path = os.path.join(step_1_path, str(epoch))

    if not config.evaluation:
        raise ValueError("evaluation config is required for gen_synthetic_dataset")

    if not os.path.exists(gan_path):
        raise ValueError(f"GAN checkpoint path does not exist: {gan_path}")

    # For chest-xray, use same number of samples as companion dataset
    # For other datasets, use 200 samples
    n_samples = config.evaluation.companion_n_samples if config.dataset.name == "chest-xray" else 200

    params = CLDatasetArgs(
        seed=config.test_noise_seed,
        # img_size=config.img_size, (default)
        n_samples=n_samples,
        out_dir=os.path.join(latest_gan_path, "synthetic"),
        gan_path=gan_path,
        device=config.device,
        fid_stats_path=fid_stats_path,
        estimator_path=None,
    )

    # Skip if synthetic dataset already exists
    if os.path.exists(params.out_dir) and len(os.listdir(params.out_dir)) > 0:
        logger.info(f"Synthetic dataset already exists at {params.out_dir}, skipping generation")
        return

    args = [
        sys.executable,
        "-m",
        "src.gen_dataset",
        "--gan-path",
        str(params.gan_path),
        "--seed",
        str(params.seed),
        "--n-samples",
        str(params.n_samples),
        "--device",
        str(params.device),
        "--out-dir",
        str(params.out_dir),
        "--fid-stats-path",
        str(fid_stats_path),
        "--skip-stats",
    ]
    if params.estimator_path is not None:
        args.extend(["--estimator-path", str(params.estimator_path)])

    subprocess.run(args, check=True, env=os.environ.copy())
    # Clean up CUDA memory after dataset generation completes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_hubris_evaluation(config: ConfigMain, latest_gan_path: str, estimator_path: str | None = None) -> None:
    """Run evaluation CLI with parameters from `config` and `latest_gan_path`."""
    # Extract GAN ID from path
    gan_id = os.path.basename(latest_gan_path.rstrip("/")).split("_")[-1]

    if config.evaluation is None:
        raise ValueError("evaluation config is required for run_hubris_evaluation")

    config_hubris = config.evaluation.hubris

    if not config_hubris:
        raise ValueError("evaluation config is required for run_hubris_evaluation")

    params = CLEvaluationArgs(
        device=config.device,
        seed=config.test_noise_seed,
        companion_dataroot=os.path.join(latest_gan_path, "companion_dataset"),
        models=config_hubris.models,
        batch_size=config_hubris.batch_size,
        epochs=config_hubris.epochs,
        out_dir=os.path.join(latest_gan_path, "evaluation"),
        estimator_path=estimator_path,
        dataroot=os.path.join(config.out_dir, config.data_dir),
        dataset_name=config.dataset.name,
        pos_class=config.dataset.binary.pos,
        neg_class=config.dataset.binary.neg,
    )

    model_args = [model.value if hasattr(model, "value") else str(model) for model in params.models]
    args: list[str] = [
        sys.executable,
        "-m",
        "src.evaluation.evaluation_hubris_cli",
        "--device",
        str(params.device),
        "--seed",
        str(params.seed),
        "--companion-data",
        str(params.companion_dataroot),
        "--models",
        *model_args,
        "--batch-size",
        str(params.batch_size),
        "--epochs",
        str(params.epochs),
        "--out-dir",
        str(params.out_dir),
        "--estimator-path",
        str(params.estimator_path),
        "--data",
        str(params.dataroot),
        "--dataset",
        str(params.dataset_name),
        "--pos",
        str(params.pos_class),
        "--neg",
        str(params.neg_class),
        "--gan-id",
        gan_id,
    ]

    subprocess.run(args, check=True, env=os.environ.copy())
    # Clean up CUDA memory to avoid OOM in subsequent steps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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
    logger.info("Hubris Benchmarking with AmbiGANs is starting...")

    args = parse_args()
    config = read_main_config(args.config_path)
    logger.info(f"Loaded experiment configuration from {args.config_path}")

    # test noise
    if config.gen_test_noise:
        logger.info("Generating test noise...")
        gen_test_noise(config)

    test_noise = os.path.join(
        config.out_dir,
        config.data_dir,
        "z",
        f"z_{config.fixed_noise}_{config.model.z_dim}",
    )

    # FID stats
    if config.gen_pairwise_inception:
        gen_pairwise_inception(config)
    fid_stats_path = os.path.join(
        config.out_dir,
        config.data_dir,
        "fid-stats",
        f"stats.{config.dataset.name}.{config.dataset.binary.pos}v{config.dataset.binary.neg}.npz",
    )

    # classifiers for step 2
    if config.gen_classifiers and config.classifiers is not None:
        for classifier in config.classifiers:
            gen_classifiers(config, classifier)
    # train ambiGAN
    if config.gen_gan:
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

    # new paths
    gan_path, estimator_path = find_latest_gan_estimator_paths(config)

    if config.gen_dataset:
        # generate companion dataset
        gen_dataset(config, fid_stats_path=fid_stats_path, latest_gan_path=gan_path, estimator_path=estimator_path)
        # Generate synthetic dataset from step_1/latest
        gen_synthetic_dataset(config, fid_stats_path=fid_stats_path, latest_gan_path=gan_path)

    # evaluate binary classifier
    if config.run_hubris_evaluation:
        run_hubris_evaluation(config, latest_gan_path=gan_path, estimator_path=estimator_path)


if __name__ == "__main__":
    main()
