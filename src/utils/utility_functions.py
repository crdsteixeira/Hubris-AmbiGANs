"""Module with project utility functions."""

import argparse
import ast
import json
import logging
import math
import os
import random
import subprocess
from datetime import datetime

import numpy as np
import torch
import torchvision.utils as vutils
from torch import nn

from src.enums import WeightType
from src.gan.loss import GeneratorLoss
from src.gan.update_g import (
    UpdateGenerator,
    UpdateGeneratorAmbiGanGaussian,
    UpdateGeneratorAmbiGanGaussianIdentity,
    UpdateGeneratorAmbiGanKLDiv,
    UpdateGeneratorGASTEN,
    UpdateGeneratorGastenMgda,
)
from src.models import CLTestNoiseArgs, CLTrainArgs, ConfigOptimizer, ConfigWeights

logger = logging.getLogger(__name__)


def create_checkpoint_path(config: dict, run_id: str) -> str:
    """Create a path for storing checkpoints."""
    path = os.path.join(
        config["out-dir"],
        config["project"],
        config["name"],
        datetime.now().strftime(f"%b%dT%H-%M_{run_id}"),
    )

    os.makedirs(path, exist_ok=True)

    return path


def create_exp_path(config: dict) -> str:
    """Create an experimental path for storing outputs."""
    path = os.path.join(config["out-dir"], config["name"])

    os.makedirs(path, exist_ok=True)

    return path


def gen_seed(max_val: int = 10000) -> int:
    """Generate a random seed value."""
    return np.random.randint(max_val)


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across multiple libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_reprod(seed: int) -> None:
    """Set deterministic and reproducible behavior for torch."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_seed(seed)


def seed_worker(_: int) -> None:
    """Seed worker process for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_and_store_z(
    config: CLTestNoiseArgs,
    name: str | None = None,
) -> tuple[torch.Tensor, str]:
    """Create a random noise tensor and store it to disk."""
    if name is None:
        name = f"z_{config.nz}_{config.z_dim}"

    noise = torch.randn(config.nz, config.z_dim).numpy()
    out_path = os.path.join(config.out_dir, name)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, "z.npy"), "wb", encoding="utf-8") as f:
        np.savez(f, z=noise)

    if config is not None:
        with open(os.path.join(out_path, "z.json"), "w", encoding="utf-8") as out_json:
            json.dump(config.__dict__, out_json)

    return torch.Tensor(noise), out_path


def load_z(path: str) -> tuple[torch.Tensor, dict]:
    """Load a noise tensor and associated configuration from disk."""
    z_path = os.path.join(path, "z.npy")
    z = np.load(z_path, encoding="utf-8")["z"]

    with open(os.path.join(path, "z.json"), encoding="utf-8") as f:
        conf = json.load(f)

    return torch.Tensor(z), conf


def make_grid(images: torch.Tensor, nrow: int | None = None, total_images: int | None = None) -> torch.Tensor:
    """Create a grid from a batch of images."""
    if nrow is None:
        nrow = int(math.sqrt(images.size(0)))
        if nrow % 1 != 0:
            nrow = 8

    if total_images is not None:
        total_images = math.ceil(total_images / nrow) * nrow

        # Ensure total_images is greater than or equal to images.size(0)
        if total_images > images.size(0):
            blank_images = -torch.ones(
                (
                    total_images - images.size(0),
                    images.size(1),
                    images.size(2),
                    images.size(3),
                )
            )
            images = torch.concat((images, blank_images), 0)

    img = vutils.make_grid(images, padding=2, normalize=True, nrow=int(nrow), value_range=(-1, 1))

    return img


def group_images(images: torch.Tensor, classifier: nn.Module = None, device: torch.device = None) -> torch.Tensor:
    """Group images into different bins and create a grid representation."""
    if classifier is None:
        return make_grid(images)

    y = torch.zeros(images.size(0))
    n_images = images.size(0)

    for i in range(0, n_images, 100):
        i_stop = min(i + 100, n_images)
        y[i:i_stop] = classifier(images[i:i_stop].to(device))

    y, idxs = torch.sort(y)
    images = images[idxs]

    groups = []
    n_divs = 10
    step = 1 / n_divs
    group_start = 0

    largest_group = 0

    for i in range(n_divs):
        up_bound = (i + 1) * step

        group_end = (y > up_bound).nonzero(as_tuple=True)[0]

        if group_end.size()[0] == 0:
            group_end = images.size(0)
        else:
            group_end = group_end[0].item()

        groups.append(images[group_start:group_end])

        largest_group = max(group_end - group_start, largest_group)

        group_start = group_end

    grids = [make_grid(g, nrow=3, total_images=largest_group) for g in groups]
    img = torch.concat(grids, 2)

    return img


def generate_cnn_configs(nf: int | list[int] | list[list[int]] | None) -> list[list[int]]:
    """Generate CNN configurations based on nf parameter."""
    cnn_nfs = []

    if isinstance(nf, int):
        cnns_count = nf
        cnn_nfs = [
            [np.random.randint(1, high=6) for _ in range(np.random.randint(2, high=5))] for _ in range(cnns_count)
        ]
    elif isinstance(nf, list):
        for n in nf:
            if isinstance(n, int):
                cnn = [np.random.randint(1, high=n) for _ in range(np.random.randint(1, high=5))]
            elif isinstance(n, list):
                cnn = [int(c) for c in n]
            else:
                raise ValueError("Invalid type for list element in nf: expected int or list of ints")
            cnn_nfs.append(cnn)
    else:
        raise ValueError("Invalid type for nf: expected int or list of ints")

    return cnn_nfs


def run_training_subprocess(args: CLTrainArgs, cnn_nfs: list[list[int]]) -> None:
    """Run a subprocess to train the classifier."""
    proc = subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "src.classifier.train",
            "--device",
            str(args.device),
            "--data-dir",
            str(args.data_dir),
            "--out-dir",
            str(args.out_dir),
            "--dataset",
            str(args.dataset_name),
            "--pos",
            str(args.pos_class),
            "--neg",
            str(args.neg_class),
            "--classifier-type",
            str(args.c_type),
            "--nf",
            str(cnn_nfs),
            "--name",
            f"{args.c_type}_{args.seed}_{args.epochs}",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--early-acc",
            str(args.early_acc),
        ],
        check=False,
        capture_output=True,
    )
    handle_subprocess_output(proc)


def handle_subprocess_output(proc: subprocess.CompletedProcess) -> None:
    """Handle output from the subprocess."""
    if proc.stdout:
        for line in proc.stdout.split(b"\n"):
            logger.info(line.decode())
    if proc.stderr:
        for line in proc.stderr.split(b"\n"):
            logger.info(line.decode())


def parse_nf(value: str) -> list[int] | int:
    """Parse the value for --nf, which can be either an int or a list of ints."""
    try:
        # Try parsing the value as a list using ast.literal_eval
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, int):
            return [parsed_value]
        if isinstance(parsed_value, list) and all(isinstance(i, int) for i in parsed_value):
            return parsed_value
        raise ValueError
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid value for --nf: '{value}', must be an int or list of ints.") from e


def construct_weights(
    classifier: nn.Module, weight_list: list[ConfigWeights], g_crit: GeneratorLoss
) -> list[tuple[str, UpdateGenerator]]:
    """Construct list of weights from config."""
    weights: list[tuple[str, UpdateGenerator]] = []
    for weight in weight_list:
        if weight is None:
            continue
        for key, value in weight.__dict__.items():
            if value is None:
                continue
            key_enum = WeightType[key]  # Convert the key from the dictionary to a WeightType enum
            match key_enum:
                case WeightType.kldiv:
                    weights.extend(
                        (f"{key}_{alpha}", UpdateGeneratorAmbiGanKLDiv(g_crit, classifier, alpha))
                        for alpha in value.alpha
                    )
                case WeightType.gaussian:
                    weights.extend(
                        (f"{key}_{w.alpha}_{w.var}", UpdateGeneratorAmbiGanGaussian(g_crit, classifier, w.alpha, w.var))
                        for w in value
                    )
                case WeightType.gaussian_v2:
                    weights.extend(
                        (
                            f"{key}_{w.alpha}_{w.var}",
                            UpdateGeneratorAmbiGanGaussianIdentity(g_crit, classifier, w.alpha, w.var),
                        )
                        for w in value
                    )
                case WeightType.cd:
                    weights.extend((f"{key}_{w}", UpdateGeneratorGASTEN(g_crit, classifier, w)) for w in value)
                case WeightType.mgda:
                    weights.extend(
                        (f"{key}_{w}", UpdateGeneratorGastenMgda(g_crit, classifier, normalize=w)) for w in value
                    )
                case _:
                    logger.error(f"Invalid weight specified {key}")
                    raise NotImplementedError

    return weights


def construct_optimizers(
    config: ConfigOptimizer, G: nn.Module, D: nn.Module
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """Cunstruct optimizers for GAN."""
    g_optim = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    d_optim = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    return g_optim, d_optim


def get_epoch_from_state(s1_epoch: int | str, step_1_train_state: dict) -> int | str:
    """Returns the appropriate epoch based on the input value."""
    if s1_epoch == "best":
        return step_1_train_state["best_epoch"]
    elif s1_epoch == "last":
        return step_1_train_state["epoch"]
    else:
        return s1_epoch
