"""Module for checkpointing"""

import json
import os

import torch
import torchvision.utils as vutils
from torch import nn, optim

from src.classifier.construct_classifier import construct_classifier
from src.gan.construct_gan import construct_gan
from src.models import CLTrainArgs, DeviceType, TrainClassifierArgs, TrainingStats

def checkpoint(
    model: nn.Module,
    model_name: str,
    model_params: TrainClassifierArgs,
    train_stats: TrainingStats,
    args: CLTrainArgs | dict = {},
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
        "stats": train_stats.__dict__,
        "params": model_params,
        "args": args,
    }

    json.dump(
        {
            "train_stats": train_stats.__dict__,
            "model_params": model_params.__dict__,
            "args": args.__dict__ if not type(args) == dict else args,
        },
        open(os.path.join(output_dir, "stats.json"), "w"),
        indent=2,
    )

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()

    torch.save(save_dict, os.path.join(output_dir, "classifier.pth"))

    return output_dir

def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device | None = None,
    optimizer: nn.Module | None = None
) -> None:
    """Load a model checkpoint from disk."""
    cp = torch.load(path, map_location=device)

    model.load_state_dict(cp["state"])
    model.eval()  # just to be safe

    if optimizer is not None:
        optimizer.load_state_dict(cp["optimizer"])

def construct_classifier_from_checkpoint(
    path: str, device: DeviceType = DeviceType.cpu, optimizer: bool = False
) -> tuple[nn.Module, dict, dict, CLTrainArgs | dict, optim.Optimizer | None]:
    """Construct a classifier model from a saved checkpoint."""
    cp = torch.load(os.path.join(path, "classifier.pth"), map_location=device)

    print(f" > Loading model from {path} ...")

    model_params = cp["params"]
    print("\t. Model", cp["name"])
    print("\t. Params: ", model_params)

    model_params["n_classes"] if "n_classes" in model_params else 2

    model = construct_classifier(model_params)
    model.load_state_dict(cp["state"])
    model.eval()

    if optimizer is True:
        opt = optim.Adam(model.parameters())
        opt.load_state_dict(cp["optimizer"].state_dict())
        return model, model_params, cp["stats"], cp["args"], opt
    else:
        return model, model_params, cp["stats"], cp["args"]

def construct_gan_from_checkpoint(
    path: str, device: torch.device | None = None
) -> tuple[nn.Module, nn.Module, optim.Optimizer, optim.Optimizer]:
    """Construct a GAN model from a saved checkpoint."""
    print(f"Loading GAN from {path} ...")
    with open(os.path.join(path, "config.json")) as config_file:
        config = json.load(config_file)

    model_params = config["model"]
    optim_params = config["optimizer"]

    gen_cp = torch.load(os.path.join(path, "generator.pth"), map_location=device)
    dis_cp = torch.load(os.path.join(path, "discriminator.pth"), map_location=device)

    G, D = construct_gan(model_params, tuple(config["model"]["image-size"]))

    g_optim = optim.Adam(
        G.parameters(),
        lr=optim_params["lr"],
        betas=(optim_params["beta1"], optim_params["beta2"]),
    )
    d_optim = optim.Adam(
        D.parameters(),
        lr=optim_params["lr"],
        betas=(optim_params["beta1"], optim_params["beta2"]),
    )

    G.load_state_dict(gen_cp["state"])
    D.load_state_dict(dis_cp["state"])
    g_optim.load_state_dict(gen_cp["optimizer"])
    d_optim.load_state_dict(dis_cp["optimizer"])

    G.eval()
    D.eval()

    return G, D, g_optim, d_optim

def get_gan_path_at_epoch(
    output_dir: str, epoch: int | None = None
) -> str:
    """Get the file path for GAN checkpoints at a given epoch."""
    path = output_dir
    if epoch is not None:
        path = os.path.join(path, f"{epoch:02d}")
    return path

def load_gan_train_state(gan_path: str) -> dict:
    """Load GAN training state from a saved file."""
    path = os.path.join(gan_path, "train_state.json")

    with open(path) as in_f:
        train_state = json.load(in_f)

    return train_state

def checkpoint_gan(
    G: nn.Module,
    D: nn.Module,
    g_opt: optim.Optimizer,
    d_opt: optim.Optimizer,
    state: dict,
    stats: dict,
    config: dict,
    output_dir: str | None = None,
    epoch: int | None = None
) -> str:
    """Save GAN checkpoint, including generator, discriminator, and optimizers."""
    rootdir = os.path.curdir if output_dir is None else output_dir

    path = get_gan_path_at_epoch(rootdir, epoch=epoch)

    os.makedirs(path, exist_ok=True)

    torch.save(
        {"state": G.state_dict(), "optimizer": g_opt.state_dict()},
        os.path.join(path, "generator.pth"),
    )

    torch.save(
        {"state": D.state_dict(), "optimizer": d_opt.state_dict()},
        os.path.join(path, "discriminator.pth"),
    )

    json.dump(state, open(os.path.join(rootdir, "train_state.json"), "w"), indent=2)
    json.dump(stats, open(os.path.join(rootdir, "stats.json"), "w"), indent=2)
    json.dump(config, open(os.path.join(path, "config.json"), "w"), indent=2)

    print(f"> Saved checkpoint checkpoint to {path}")

    return path

def checkpoint_image(
    image: torch.Tensor, epoch: int, output_dir: str | None = None
) -> None:
    """Save generated images as checkpoints for visualization."""
    dir = os.path.curdir if output_dir is None else output_dir

    dir = os.path.join(dir, "gen_images")
    os.makedirs(dir, exist_ok=True)

    path = os.path.join(dir, f"{epoch:02d}.png")

    vutils.save_image(image, path)