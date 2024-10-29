"""Module to train GAN."""

import logging
import math

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm

from src.models import (
    CheckpointGAN,
    ConfigGAN,
    GANTrainArgs,
    MetricsParams,
    TrainingState,
)
from src.utils.checkpoint import checkpoint_gan, checkpoint_image
from src.utils.metrics_logger import MetricsLogger
from src.utils.utility_functions import group_images, seed_worker

logger = logging.getLogger(__name__)


def loss_terms_to_str(loss_items: dict[str, float]) -> str:
    """Convert loss terms dictionary to a formatted string."""
    result = ""
    for key, value in loss_items.items():
        result += f"{key}: {value:.4f} "

    return result


def evaluate(
    params: GANTrainArgs,
    stats_logger: MetricsLogger,
) -> None:
    """Evaluate generator performance with metrics and generate plots."""
    training = params.G.training
    params.G.eval()

    start_idx = 0
    num_batches = math.ceil(params.test_noise.size(0) / params.batch_size)

    for _ in tqdm(range(num_batches), desc="Evaluating"):
        real_size = min(params.batch_size, params.test_noise.size(0) - start_idx)

        batch_z = params.test_noise[start_idx : start_idx + real_size]

        with torch.no_grad():
            batch_gen = params.G(batch_z.to(params.device))

        for metric_name, metric in params.fid_metrics.model_dump().items():
            if metric is not None:
                metric.update(batch_gen, (start_idx, real_size))

        if params.c_out_hist is not None:
            params.c_out_hist.update(batch_gen, (start_idx, real_size))

        start_idx += batch_z.size(0)

    for metric_name, metric in params.fid_metrics.model_dump().items():
        if metric is not None:
            result = metric.finalize()
            stats_logger.update_epoch_metric(metric_name, result, prnt=True)
            metric.reset()

    if params.c_out_hist is not None:
        params.c_out_hist.plot()
        params.c_out_hist.reset()
        plt.clf()

    if training:
        params.G.train()


def train_disc(
    params: GANTrainArgs, train_metrics: MetricsLogger, real_data: Tensor, fake_data: torch.tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the discriminator with real and generated data."""
    params.D.zero_grad()

    d_output_real = params.D(real_data)
    d_output_fake = params.D(fake_data)

    # Compute loss, gradients and update net
    d_loss, d_loss_terms = params.d_crit(real_data, fake_data, d_output_real, d_output_fake, params.device)
    d_loss.backward()
    params.d_opt.step()

    # Log metrics
    for loss_term_name, loss_term_value in d_loss_terms.items():
        train_metrics.update_it_metric(loss_term_name, loss_term_value)
    train_metrics.update_it_metric("D_loss", d_loss.item())

    return d_loss, d_loss_terms


def train_gen(
    params: GANTrainArgs, train_metrics: MetricsLogger, noise: torch.tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the generator to improve its output quality."""
    g_loss, g_loss_terms = params.g_updater(params.G, params.D, params.g_opt, noise, params.device)

    # Log metrics
    for loss_term_name, loss_term_value in g_loss_terms.items():
        train_metrics.update_it_metric(loss_term_name, loss_term_value)
    train_metrics.update_it_metric("G_loss", g_loss.item())

    return g_loss, g_loss_terms


def initialize_training_state(params: GANTrainArgs) -> TrainingState:
    """Initialize the training state dictionary."""
    train_state = TrainingState()
    if params.early_stop and params.early_stop[1] is not None:
        if params.start_early_stop_when is not None:
            train_state.pre_early_stop_tracker = 0
            train_state.pre_early_stop_metric = float("inf")
    return train_state


def log_generator_discriminator_metrics(
    train_metrics: MetricsLogger, eval_metrics: MetricsLogger, params: GANTrainArgs
) -> None:
    """Log generator and discriminator metrics."""
    train_metrics.add("G_loss", iteration_metric=True)
    train_metrics.add("D_loss", iteration_metric=True)
    for loss_term in params.g_updater.get_loss_terms():
        train_metrics.add(loss_term, iteration_metric=True)
    for loss_term in params.d_crit.get_loss_terms():
        train_metrics.add(loss_term, iteration_metric=True)
    eval_metrics.add_media_metric("samples")
    eval_metrics.add_media_metric("histogram")
    # Add metrics from fid_metrics to evaluation metrics
    for metric_name, metric in params.fid_metrics.model_dump().items():
        if metric is not None:  # Only add if metric is present
            eval_metrics.add(metric_name)
            logger.debug(f"Added metric: {metric_name} to eval_metrics.")
        else:
            logger.debug(f"Skipping metric: {metric_name} as it is None.")

    # Verify all metrics added
    logger.debug(f"Metrics currently in eval_metrics: {eval_metrics.stats.keys()}")


def evaluate_and_checkpoint(
    params: GANTrainArgs,
    train_state: TrainingState,
    eval_metrics: MetricsLogger,
    train_metrics: MetricsLogger,
    config: CheckpointGAN,
) -> str | None:
    """Evaluate the generator and checkpoint the model."""
    # Evaluate Generator after an epoch and checkpoint the GAN
    with torch.no_grad():
        params.G.eval()
        fake = params.G(params.fixed_noise).detach().cpu()
        params.G.train()

    img = group_images(fake, classifier=params.classifier, device=params.device)
    checkpoint_image(img, train_state.epoch, output_dir=params.checkpoint_dir)
    eval_metrics.log_image("samples", img)

    train_metrics.finalize_epoch()

    # Evaluate GAN
    evaluate(params=params, stats_logger=eval_metrics)
    eval_metrics.finalize_epoch()

    if train_state.epoch == params.epochs or train_state.epoch % params.checkpoint_every == 0:
        return checkpoint_gan(
            params,
            train_state.__dict__,
            {"eval": eval_metrics.stats, "train": train_metrics.stats},
            config,
            epoch=train_state.epoch,
            output_dir=params.checkpoint_dir,
        )

    return None


def train(params: GANTrainArgs, config: ConfigGAN) -> tuple[TrainingState, str | None, MetricsLogger, MetricsLogger]:
    """Run main loop for GAN training."""
    # Initialize training state and dataloader
    checkpoint_data = CheckpointGAN(
        config=config,
        gen_params=params.G.params,
        dis_params=params.D.params,
    )
    fixed_noise = (
        torch.randn(64, params.G.params.z_dim, device=params.device)
        if params.fixed_noise is None
        else params.fixed_noise
    )
    dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        params.dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
    )
    train_metrics: MetricsLogger = MetricsLogger(MetricsParams(prefix="train", log_epoch=True))
    eval_metrics: MetricsLogger = MetricsLogger(MetricsParams(prefix="validation", log_epoch=True))
    train_state: TrainingState = initialize_training_state(params=params)

    log_generator_discriminator_metrics(train_metrics=train_metrics, eval_metrics=eval_metrics, params=params)

    # Initial Evaluation and Checkpoint
    with torch.no_grad():
        params.G.eval()
        fake = params.G(fixed_noise).detach().cpu()
        params.G.train()
    latest_cp: str | None = checkpoint_gan(
        params=params,
        state={},
        stats={},
        config=checkpoint_data,
        epoch=0,
        output_dir=params.checkpoint_dir,
    )
    checkpoint_image(
        group_images(fake, classifier=params.classifier, device=params.device), 0, output_dir=params.checkpoint_dir
    )

    # Begin Training Loop
    logger.info("Training...")
    for epoch in range(1, params.epochs + 1):
        data_iter = iter(dataloader)
        curr_g_iter = 0
        g_iters_per_epoch = int(math.floor(len(dataloader) / params.n_disc_iters))
        iters_per_epoch = g_iters_per_epoch * params.n_disc_iters

        for i in range(1, iters_per_epoch + 1):
            real_data, _ = next(data_iter)
            real_data = real_data.to(params.device)

            noise = torch.randn(params.batch_size, params.G.params.z_dim, device=params.device)
            fake_data = params.G(noise)

            # Update Discriminator
            d_loss, d_loss_terms = train_disc(
                params=params, train_metrics=train_metrics, real_data=real_data, fake_data=fake_data.detach()
            )

            # Update Generator
            if i % params.n_disc_iters == 0:
                curr_g_iter += 1
                noise = torch.randn(params.batch_size, params.G.params.z_dim, device=params.device)
                g_loss, g_loss_terms = train_gen(params=params, train_metrics=train_metrics, noise=noise)

                # Log Statistics
                if curr_g_iter == g_iters_per_epoch:
                    logger.info(
                        "[%d/%d][%d/%d]\tG loss: %.4f %s; D loss: %.4f %s",
                        epoch,
                        params.epochs,
                        curr_g_iter,
                        g_iters_per_epoch,
                        g_loss.item(),
                        loss_terms_to_str(g_loss_terms),
                        d_loss.item(),
                        loss_terms_to_str(d_loss_terms),
                    )

        # Evaluate and Checkpoint After Epoch
        train_state.epoch += 1
        latest_cp = evaluate_and_checkpoint(
            params=params,
            train_state=train_state,
            eval_metrics=eval_metrics,
            train_metrics=train_metrics,
            config=checkpoint_data,
        )

    return train_state, latest_cp, train_metrics, eval_metrics
