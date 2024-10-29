"""Module to run GAN process."""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import ValidationError

import wandb
from src.classifier.classifier_cache import ClassifierCache
from src.datasets.load import load_dataset
from src.gan.construct_gan import construct_gan, construct_loss
from src.gan.train import train
from src.gan.update_g import UpdateGeneratorGAN
from src.metrics.c_output_hist import OutputsHistogram
from src.metrics.fid.fid import FID
from src.metrics.focd import FOCD
from src.metrics.hubris import Hubris
from src.metrics.loss_term import LossSecondTerm
from src.models import (
    CLAmbigan,
    ConfigGAN,
    FIDMetricsParams,
    GANTrainArgs,
    LoadDatasetParams,
    Step1TrainingArgs,
    Step2TrainingArgs,
)
from src.utils.checkpoint import (
    construct_classifier_from_checkpoint,
    construct_gan_from_checkpoint,
    get_gan_path_at_epoch,
    load_gan_train_state,
)
from src.utils.logging import configure_logging
from src.utils.metrics_logger import MetricsLogger
from src.utils.plot import plot_metrics
from src.utils.read_config import read_config
from src.utils.utility_functions import (
    construct_optimizers,
    construct_weights,
    create_checkpoint_path,
    gen_seed,
    get_epoch_from_state,
    load_z,
    set_seed,
    setup_reprod,
)

configure_logging()
logger = logging.getLogger(__name__)


def parse_args() -> CLAmbigan:
    """Parse arguments from command line for step 2."""
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


def train_modified_gan(
    params: Step2TrainingArgs,
    config: ConfigGAN,
    class_cache: ClassifierCache,
) -> MetricsLogger:
    """Train GAN with extra loss."""
    weight_name, update_g = params.weight
    c_out_hist = OutputsHistogram(class_cache, params.test_noise.size(0))
    run_name = f"{params.c_name}_{weight_name}_{params.s1_epoch}"
    if params.checkpoint_dir is not None:
        checkpoint_dir = os.path.join(params.checkpoint_dir, run_name)
    else:
        checkpoint_dir = os.path.join(f"{os.environ['FILESDIR']}/checkpoint", run_name)

    G, D, _, _ = construct_gan_from_checkpoint(params.gan_path, device=config.device)
    g_optim, d_optim = construct_optimizers(config.optimizer, G, D)
    _, d_crit = construct_loss(config.model.loss, D)

    logger.info(f"Running experiment with classifier {params.c_name} and weight {weight_name} ...")
    if params.seed is not None:
        set_seed(params.seed)
    wandb.init(
        project=config.project,
        group=config.name,
        entity=os.environ["ENTITY"],
        job_type="step-2",
        name=f"{params.run_id}-{run_name}",
        config={
            "id": params.run_id,
            "seed": params.seed,
            "weight": weight_name,
            "train": config.train.step_2.__dict__,
            "step1_epoch": params.s1_epoch,
        },
    )

    params_step_2 = GANTrainArgs(
        dataset=params.dataset,
        device=params.device,
        batch_size=config.train.step_2.batch_size,
        epochs=config.train.step_2.epochs,
        G=G,
        g_opt=g_optim,
        g_updater=update_g,
        D=D,
        d_opt=d_optim,
        d_crit=d_crit,
        test_noise=params.test_noise,
        fid_metrics=params.fid_metrics,
        n_disc_iters=config.train.step_2.disc_iters,
        early_stop=params.early_stop,
        start_early_stop_when=("fid", params.early_stop or 0),
        checkpoint_dir=checkpoint_dir,
        fixed_noise=params.fixed_noise,
        c_out_hist=c_out_hist,
        checkpoint_every=config.train.step_2.checkpoint_every,
        classifier=params.classifier,
        out_dir=config.out_dir,
    )

    _, _, _, eval_metrics = train(
        params=params_step_2,
        config=config,
    )
    wandb.finish()

    return eval_metrics


def train_step2_gan(
    params: Step2TrainingArgs,
    config: ConfigGAN,
    original_fid: FID,
    step_1_train_state: dict,
) -> None:
    """Run GAN training with classifiers."""
    logger.info(" > Start step 2 (gan with modified (loss)")
    step_1_epochs: list[int] | list[str] = ["best"]
    if config.train.step_2.step_1_epochs is not None:
        step_1_epochs = config.train.step_2.step_1_epochs

    ###
    # Train modified GAN
    ###
    step2_metrics: list[pd.DataFrame] = []
    for c_path in config.train.step_2.classifier:
        C_name = os.path.splitext(os.path.basename(c_path))[0]
        C, _, _, _, _ = construct_classifier_from_checkpoint(c_path, device=config.device)

        C.to(config.device)
        C.eval()
        if hasattr(C, "output_feature_maps"):
            C.output_feature_maps = True

        class_cache = ClassifierCache(C)
        fid_metrics = FIDMetricsParams(
            fid=original_fid,
            focd=FOCD(params.test_noise.size(0), config, class_cache, params.dataset),
            conf_dist=LossSecondTerm(class_cache),
            hubris=Hubris(class_cache, params.test_noise.size(0)),
        )

        step2_metrics = []
        for s1_epoch in step_1_epochs:
            epoch = get_epoch_from_state(s1_epoch, step_1_train_state)

            gan_path = get_gan_path_at_epoch(params.gan_path, epoch=epoch)

            if params.g_crit is None:
                logger.error("Failed to construct weights: g_crit must not be None.")
                raise ValueError

            weights = construct_weights(C, config.train.step_2.weight, params.g_crit)
            for weight_name, weight in weights:
                eval_metrics = train_modified_gan(
                    Step2TrainingArgs(
                        dataset=params.dataset,
                        checkpoint_dir=params.checkpoint_dir,
                        gan_path=gan_path,
                        test_noise=params.test_noise,
                        fid_metrics=fid_metrics,
                        classifier=C,
                        c_name=C_name,
                        weight=(weight_name, weight),
                        fixed_noise=params.fixed_noise,
                        device=params.device,
                        seed=params.seed,
                        run_id=params.run_id,
                        step_1_epochs=epoch,
                        epochs=config.train.step_2.epochs,
                        out_dir=config.out_dir,
                        G=params.G,
                        g_opt=params.g_opt,
                        g_updater=params.g_updater,
                        D=params.D,
                        d_opt=params.d_opt,
                        d_crit=params.d_crit,
                        n_disc_iters=config.train.step_2.disc_iters,
                        s1_epoch=s1_epoch,
                    ),
                    config=config,
                    class_cache=class_cache,
                )
                step2_metrics.append(
                    pd.DataFrame(
                        {
                            "fid": eval_metrics.stats["fid"],
                            "conf_dist": eval_metrics.stats["conf_dist"],
                            "hubris": eval_metrics.stats["hubris"],
                            "s1_epochs": [epoch] * len(eval_metrics.stats["fid"]),
                            "weight": [weight] * len(eval_metrics.stats["fid"]),
                            "classifier": [c_path] * len(eval_metrics.stats["fid"]),
                            "epoch": [i + 1 for i in range(len(eval_metrics.stats["fid"]))],
                        }
                    )
                )

    if params.checkpoint_dir is not None:
        checkpoint_dir = os.path.join(params.checkpoint_dir, str(params.run_id))
    else:
        checkpoint_dir = os.path.join(f"{os.environ['FILESDIR']}/checkpoint", str(params.run_id))

    if len(step2_metrics) == 0:
        logger.warning("No step2 metrics were generated; skipping concatenation.")
        return

    step2_metrics = pd.concat(step2_metrics)
    plot_metrics(step2_metrics, checkpoint_dir, f"{C_name}-{params.run_id}")


def train_step1_gan(params: Step1TrainingArgs, config: ConfigGAN) -> tuple[dict, str]:
    """Train GAN step 1."""
    if params.seed is not None:
        setup_reprod(params.seed)
    G, D = construct_gan(config, params.img_size)
    g_optim, d_optim = construct_optimizers(config.optimizer, G, D)
    g_crit, d_crit = construct_loss(config.model.loss, D)
    g_updater = UpdateGeneratorGAN(g_crit)

    logger.info(f"Storing generated artifacts in {params.checkpoint_dir}")
    if params.checkpoint_dir is not None:
        checkpoint_dir = os.path.join(params.checkpoint_dir, str(params.run_id))
    else:
        checkpoint_dir = os.path.join(f"{os.environ['FILESDIR']}/checkpoint", str(params.run_id))

    original_gan_cp_dir = os.path.join(checkpoint_dir, "step_1")

    ###
    # Step 1 (train GAN with normal GAN loss)
    ###
    if not isinstance(config.train.step_1, str):
        early_stop: tuple[str, int] | None = None
        if config.train.step_1.early_stop is not None:
            early_stop = config.train.step_1.early_stop

        wandb.init(
            project=config.project,
            group=config.name,
            entity=os.environ["ENTITY"],
            job_type="step_1",
            name=f"{params.run_id}_step_1",
            config={
                "id": params.run_id,
                "seed": params.seed,
                "gan": config.model,
                "optim": config.optimizer,
                "train": config.train.step_1,
                "dataset": config.dataset,
                "num-workers": config.num_workers,
                "test-noise": params.test_noise_conf,
            },
        )

        params = GANTrainArgs(
            G=G,
            g_opt=g_optim,
            g_updater=g_updater,
            D=D,
            d_opt=d_optim,
            d_crit=d_crit,
            test_noise=params.test_noise,
            fid_metrics=params.fid_metrics,
            n_disc_iters=config.train.step_1.disc_iters,
            early_stop=early_stop,
            checkpoint_dir=original_gan_cp_dir,
            checkpoint_every=config.train.step_1.checkpoint_every,
            fixed_noise=params.fixed_noise,
            dataset=params.dataset,
            epochs=config.train.step_1.epochs,
            out_dir=original_gan_cp_dir,
            batch_size=config.train.step_1.batch_size,
            device=config.device,
        )

        step_1_train_state, _, _, _ = train(params, config)
        wandb.finish()
    else:
        original_gan_cp_dir = config.train.step_1
        step_1_train_state = load_gan_train_state(original_gan_cp_dir)

    return step_1_train_state, original_gan_cp_dir


def main(config: ConfigGAN | None = None) -> None:
    """Run process of GAN training."""
    load_dotenv()
    logger.info("AmbiGANs is starting...")

    if config is None:
        args = parse_args()
        config = read_config(args.config_path)
        logger.info(f"Loaded experiment configuration from {args.config_path}")

    if config.step_1_seeds is None:
        config.step_1_seeds = [gen_seed() for _ in range(config.num_runs)]
    if config.step_2_seeds is None:
        config.step_2_seeds = [gen_seed() for _ in range(config.num_runs)]

    logger.info(f"Using device {config.device}")
    logger.info(f" > Num workers {config.num_workers}")

    ###
    # Setup
    ###
    dataset, _, img_size = load_dataset(
        LoadDatasetParams(
            dataset_name=config.dataset.name,
            dataroot=config.data_dir,
            pos_class=config.dataset.binary.pos,
            neg_class=config.dataset.binary.neg,
            train=True,
        )
    )

    if isinstance(config.fixed_noise, str):
        arr = np.load(config.fixed_noise)
        fixed_noise = torch.Tensor(arr).to(config.device.value)
    else:
        fixed_noise = torch.randn(config.fixed_noise, config.model.z_dim, device=config.device.value)

    test_noise, test_noise_conf = load_z(config.test_noise)
    logger.info(f"Loaded test noise from {config.test_noise}")
    logger.info("\t {test_noise_conf}")

    original_fid = FID(
        fid_stats_file=config.fid_stats_path, dims=2048, n_images=test_noise.size(0), device=config.device
    )

    for i in range(config.num_runs):
        logger.info("##\n")
        logger.info(f"# Starting run {i}\n")
        logger.info("##")

        run_id = wandb.util.generate_id()
        cp_dir = create_checkpoint_path(config, run_id)
        with open(os.path.join(cp_dir, "fixed_noise.npy"), "wb") as f:
            np.save(f, fixed_noise.cpu().numpy())

        fid_metrics = FIDMetricsParams(fid=original_fid)
        step_1_params = Step1TrainingArgs(
            img_size=img_size,
            run_id=run_id,
            test_noise_conf=test_noise_conf,
            fid_metrics=fid_metrics,
            seed=config.step_1_seeds[i],
            checkpoint_dir=cp_dir,
            device=config.device,
            dataset=dataset,
            fixed_noise=fixed_noise,
            test_noise=test_noise,
        )

        step_1_train_state, _ = train_step1_gan(
            params=step_1_params,
            config=config,
        )

        step_2_params = Step2TrainingArgs(
            run_id=run_id,
            seed=config.step_2_seeds[i],
            checkpoint_dir=cp_dir,
            dataset=dataset,
            fixed_noise=fixed_noise,
            test_noise=test_noise,
        )

        train_step2_gan(
            params=step_2_params, config=config, original_fid=original_fid, step_1_train_state=step_1_train_state
        )


if __name__ == "__main__":
    main()
