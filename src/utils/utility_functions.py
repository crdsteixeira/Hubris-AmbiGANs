"""Module with project utility functions."""

import itertools
import json
import math
import os
import random
import subprocess
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import torch
import torchvision.utils as vutils
from torch import nn

from src.models import CLTestNoiseArgs, CLTrainArgs


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


def begin_classifier(iterator: Iterable, clf_type: str, l_epochs: list[str], args: CLTrainArgs) -> None:
    """Run a training process for classifiers using subprocess."""
    # Ensure nf is a list, even if a single int is provided
    if isinstance(args.nf, int):
        l_nf = [args.nf]
    elif isinstance(args.nf, list):
        l_nf = args.nf
    else:
        raise ValueError("Invalid type for nf: expected int or list of ints")

    for neg_class, pos_class in iterator:
        print(f"\nGenerating classifiers for {pos_class}v{neg_class} ...")
        for nf, epochs in itertools.product(l_nf, l_epochs):
            print("\n", clf_type, nf, epochs)
            proc = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.classifier.train",
                    "--device",
                    args.device,
                    "--data-dir",
                    args.data_dir,
                    "--out-dir",
                    args.out_dir,
                    "--dataset",
                    args.dataset,
                    "--pos",
                    pos_class,
                    "--neg",
                    neg_class,
                    "--classifier-type",
                    clf_type,
                    "--nf",
                    str(nf),
                    "--epochs",
                    epochs,
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--seed",
                    str(args.seed),
                ],
                check=False,
                capture_output=True,
            )
            for line in proc.stdout.split(b"\n")[-4:-1]:
                print(line.decode())


def begin_ensemble(iterator: Iterable, clf_type: str, l_epochs: list[str], args: CLTrainArgs) -> None:
    """Run an ensemble training process for classifiers using subprocess."""
    # Set the seed if not provided
    initialize_seed(args)

    # Generate CNN configurations
    cnn_nfs = generate_cnn_configs(args.nf)
    print(f"\nFinal CNN list: {cnn_nfs}")

    # Iterate through class pairs and start training subprocess
    for neg_class, pos_class in iterator:
        print(f"\nGenerating classifiers for {pos_class}v{neg_class} ...")
        for epochs in l_epochs:
            print("\n", clf_type, len(cnn_nfs), epochs)
            run_training_subprocess(clf_type, epochs, args, cnn_nfs, pos_class, neg_class)


def initialize_seed(args: CLTrainArgs) -> None:
    """Set random seed if not provided."""
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        args.seed = np.random.randint(100000)


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


def run_training_subprocess(
    clf_type: str, epochs: str, args: CLTrainArgs, cnn_nfs: list[list[int]], pos_class: str, neg_class: str
) -> None:
    """Run a subprocess to train the classifier."""
    proc = subprocess.run(
        [
            "python",
            "-m",
            "src.classifier.train",
            "--device",
            args.device,
            "--data-dir",
            args.data_dir,
            "--out-dir",
            args.out_dir,
            "--dataset",
            args.dataset_name,
            "--pos",
            pos_class,
            "--neg",
            neg_class,
            "--classifier-type",
            clf_type,
            "--nf",
            str(cnn_nfs),
            "--name",
            f"{clf_type.replace(':', '_')}_{args.seed}_{epochs}",
            "--epochs",
            epochs,
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
            print(line.decode())
    if proc.stderr:
        for line in proc.stderr.split(b"\n"):
            print(line.decode())
