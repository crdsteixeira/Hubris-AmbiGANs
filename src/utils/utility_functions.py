"""Module with project utility functions"""

import os
import json
from datetime import datetime

import numpy as np
import torchvision.utils as vutils
import torch
import random
import math
import itertools
import subprocess
from torch import nn


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
    """Setup deterministic and reproducible behavior for torch."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed worker process for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_and_store_z(out_dir: str, n: int, dim: int, name: str = None, config: dict = None) -> tuple[torch.Tensor, str]:
    """Create a random noise tensor and store it to disk."""
    if name is None:
        name = f"z_{n}_{dim}"

    noise = torch.randn(n, dim).numpy()
    out_path = os.path.join(out_dir, name)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, "z.npy"), "wb") as f:
        np.savez(f, z=noise)

    if config is not None:
        with open(os.path.join(out_path, "z.json"), "w") as out_json:
            json.dump(config, out_json)

    return torch.Tensor(noise), out_path


def load_z(path: str) -> tuple[torch.Tensor, dict]:
    """Load a noise tensor and associated configuration from disk."""
    with np.load(os.path.join(path, "z.npy")) as f:
        z = f["z"][:]

    with open(os.path.join(path, "z.json")) as f:
        conf = json.load(f)

    return torch.Tensor(z), conf


def make_grid(images: torch.Tensor, nrow: int = None, total_images: int = None) -> torch.Tensor:
    """Create a grid from a batch of images."""
    if nrow is None:
        nrow = math.sqrt(images.size(0))
        if nrow % 1 != 0:
            nrow = 8
    else:
        if total_images is not None:
            total_images = math.ceil(total_images / nrow) * nrow

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


def begin_classifier(iterator: iter, clf_type: str, l_epochs: list[str], args: object) -> None:
    """Run a training process for classifiers using subprocess."""
    l_nf = list({nf for nf in args.nf.split(",") if nf.isdigit()})
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
                    args.dataroot,
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
                    nf,
                    "--epochs",
                    epochs,
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--seed",
                    str(args.seed),
                ],
                capture_output=True,
            )
            for line in proc.stdout.split(b"\n")[-4:-1]:
                print(line.decode())


def begin_ensemble(iterator: iter, clf_type: str, l_epochs: list[str], args: object) -> None:
    """Run an ensemble training process for classifiers using subprocess."""
    # Set seed, if necessary
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        args.seed = np.random.randint(100000)

    # First get number of CNN's
    cnn_nfs = []
    if "," in args.nf:
        l_nf = list({nf for nf in args.nf.split(",")})
        for n in l_nf:
            if "-" in n:
                cnn = [int(c) for c in n.split("-")]
            else:
                cnn = [np.random.randint(1, high=n) for _ in range(np.random.randint(1, high=4 + 1))]

            cnn_nfs.append(cnn)
    else:
        cnns_count = int(args.nf)
        cnn = [
            [np.random.randint(1, high=5 + 1) for _ in range(np.random.randint(2, high=4 + 1))]
            for _ in range(cnns_count)
        ]
        cnn_nfs.extend(cnn)

    print(f"\nFinal CNN list: {cnn_nfs}")

    for neg_class, pos_class in iterator:
        print(f"\nGenerating classifiers for {pos_class}v{neg_class} ...")
        for (epochs,) in itertools.product(l_epochs):
            print("\n", clf_type, len(cnn_nfs), epochs)
            proc = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.classifier.train",
                    "--device",
                    args.device,
                    "--data-dir",
                    args.dataroot,
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
                    str(cnn_nfs),
                    "--name",
                    str("{}_{}_{}".format(clf_type.replace(":", "_"), args.seed, epochs)),
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
                capture_output=True,
            )
            for line in proc.stdout.split(b"\n"):
                print(line.decode())
            for line in proc.stderr.split(b"\n"):
                print(line.decode())