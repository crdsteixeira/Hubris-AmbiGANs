"""Module for checkpointing."""

import json
import logging
import os

import torch
import torchvision.utils as vutils
from torch import nn, optim

from src.classifier.construct_classifier import construct_classifier
from src.enums import DeviceType
from src.gan.construct_gan import construct_gan
from src.models import (
    CheckpointGAN,
    CLTrainArgs,
    GANTrainArgs,
    ImageParams,
    TrainClassifierArgs,
    TrainingState,
    TrainingStats,
)

logger = logging.getLogger(__name__)


def checkpoint(
    model: nn.Module,
    model_name: str,
    model_params: TrainClassifierArgs | None,
    train_stats: TrainingStats | None,
    args: CLTrainArgs | None,
    output_dir: str | None = None,
    optimizer: nn.Module | None = None,
) -> str:
    """Save model checkpoint along with training stats and optional optimizer."""
    output_dir = os.path.curdir if output_dir is None else output_dir
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    save_dict = {
        "name": model_name,
        "state": model.state_dict(),
        "stats": train_stats.__dict__ if train_stats else {},
        "params": model_params if model_params else {},
        "args": args.__dict__ if args else {},
    }

    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_stats": train_stats.__dict__ if train_stats else {},
                "model_params": model_params.__dict__ if model_params else {},
                "args": args.__dict__ if args else {},
            },
            f,
            indent=2,
        )

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()

    torch.save(save_dict, os.path.join(output_dir, "classifier.pth"))

    return output_dir


def load_checkpoint(
    path: str, model: nn.Module, device: torch.device | None = None, optimizer: nn.Module | None = None
) -> None:
    """Load a model checkpoint from disk."""
    cp = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(cp["state"])
    model.eval()  # just to be safe

    if optimizer is not None:
        optimizer.load_state_dict(cp["optimizer"])


def construct_classifier_from_checkpoint(
    path: str, device: DeviceType = DeviceType.cpu, optimizer: bool = False
) -> tuple[nn.Module, dict, dict, CLTrainArgs | dict, optim.Optimizer | None]:
    """Construct a classifier model from a saved checkpoint."""
    cp = torch.load(os.path.join(path, "classifier.pth"), map_location=device, weights_only=False)

    logger.info(f" > Loading model from {path} ...")

    model_params = cp["params"]
    logger.info(f"\t. Model: {cp['name']}")
    logger.info(f"\t. Params: {model_params}")

    model = construct_classifier(model_params)
    model.load_state_dict(cp["state"])
    model.eval()

    if optimizer is True:
        opt = optim.Adam(model.parameters())
        opt.load_state_dict(cp["optimizer"].state_dict())
        return model, model_params, cp["stats"], cp["args"], opt

    return model, model_params, cp["stats"], cp["args"], None


def construct_gan_from_checkpoint(
    path: str, device: torch.device | None = None
) -> tuple[nn.Module, nn.Module, optim.Optimizer, optim.Optimizer]:
    """Construct a GAN model from a saved checkpoint."""
    logger.info(f"Loading GAN from {path} ...")
    with open(os.path.join(path, "config.json"), encoding="utf-8") as config_file:
        checkpoint_dict = json.load(config_file)

    checkpoint_data = CheckpointGAN(**checkpoint_dict)

    gen_cp = torch.load(os.path.join(path, "generator.pth"), map_location=device, weights_only=False)
    dis_cp = torch.load(os.path.join(path, "discriminator.pth"), map_location=device, weights_only=False)

    G, D = construct_gan(checkpoint_data.config, ImageParams(image_size=checkpoint_data.gen_params.image_size))

    g_optim = optim.Adam(
        G.parameters(),
        lr=checkpoint_data.config.optimizer.lr,
        betas=(checkpoint_data.config.optimizer.beta1, checkpoint_data.config.optimizer.beta2),
    )
    d_optim = optim.Adam(
        D.parameters(),
        lr=checkpoint_data.config.optimizer.lr,
        betas=(checkpoint_data.config.optimizer.beta1, checkpoint_data.config.optimizer.beta2),
    )

    G.load_state_dict(gen_cp["state"])
    D.load_state_dict(dis_cp["state"])
    g_optim.load_state_dict(gen_cp["optimizer"])
    d_optim.load_state_dict(dis_cp["optimizer"])

    G.eval()
    D.eval()

    return G, D, g_optim, d_optim


def get_gan_path_at_epoch(output_dir: str, epoch: int | str | None = None) -> str:
    """Get the file path for GAN checkpoints at a given epoch."""
    path = output_dir
    if epoch is not None:
        path = os.path.join(path, f"{epoch:02d}")
    return path


def load_gan_train_state(gan_path: str) -> TrainingState:
    """Load GAN training state from a saved file."""
    path = os.path.join(gan_path, "train_state.json")

    with open(path, encoding="utf-8") as in_f:
        train_state = json.load(in_f)
        train_state = TrainingState(**train_state)

    return train_state


def checkpoint_gan(
    params: GANTrainArgs,
    state: dict,
    stats: dict,
    config: CheckpointGAN,
    output_dir: str | None = None,
    epoch: int | None = None,
) -> str:
    """Save GAN checkpoint, including generator, discriminator, and optimizers."""
    rootdir = os.path.curdir if output_dir is None else output_dir

    path = get_gan_path_at_epoch(rootdir, epoch=epoch)

    os.makedirs(path, exist_ok=True)

    torch.save(
        {"state": params.G.state_dict(), "optimizer": params.g_opt.state_dict()},
        os.path.join(path, "generator.pth"),
    )

    torch.save(
        {"state": params.D.state_dict(), "optimizer": params.d_opt.state_dict()},
        os.path.join(path, "discriminator.pth"),
    )
    with open(os.path.join(rootdir, "train_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    with open(os.path.join(rootdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)

    logger.info(f"> Saved checkpoint checkpoint to {path}")

    return path


def checkpoint_image(image: torch.Tensor, epoch: int, output_dir: str | None = None) -> None:
    """Save generated images as checkpoints for visualization."""
    directory: str = os.path.curdir if output_dir is None else output_dir

    directory = os.path.join(directory, "gen_images")
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, f"{epoch:02d}.png")

    vutils.save_image(image, path)
