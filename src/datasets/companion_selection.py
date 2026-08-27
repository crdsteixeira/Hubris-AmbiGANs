"""Select the companion images an evaluation is built from, optionally only the ambiguous ones."""

import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.image_dataset import ImageDataset, get_companion_transform
from src.enums import DeviceType
from src.utils.checkpoint import construct_classifier_from_checkpoint

logger = logging.getLogger(__name__)

# Confusion distance: 0.0 is a perfectly ambiguous sample, 0.5 a perfectly confident one
MAX_CONFUSION_DISTANCE = 0.05

CONFUSION_DISTANCE_CSV = "confusion_distance.csv"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# A complete companion dataset holds 2500 images; runs below this had their generation interrupted
MIN_COMPANION_IMAGES = 2000

# Top-ups live outside `companion_dataset`, which is read wholesale as an ImageFolder and is
# treated as fixed-size elsewhere; `src.gen_dataset` also restarts its numbering on every call.
TOPUP_DIR_NAME = "companion_topup"
TOPUP_ROUND_PREFIX = "round"

# Only images within the threshold count, so each round asks for more than it needs
MAX_TOPUP_ROUNDS = 4
MIN_TOPUP_BATCH = 500
MAX_TOPUP_BATCH = 5000
TOPUP_MARGIN = 1.25


@dataclass(frozen=True)
class RunSpec:
    """Where a companion dataset is, and what guided it."""

    run_dir: Path
    dataset: str
    pos: int
    neg: int
    estimator_path: Path

    @property
    def pair(self) -> str:
        """Return the class pair in `<pos>v<neg>` form."""
        return f"{self.pos}v{self.neg}"


def companion_ambi_dir(run_dir: Path) -> Path:
    """Return the directory holding a run's companion images."""
    return run_dir / "companion_dataset" / "ambi"


def companion_image_count(run_dir: Path) -> int:
    """Count a run's companion images."""
    ambi = companion_ambi_dir(run_dir)
    return sum(1 for _ in ambi.glob("*.png")) if ambi.is_dir() else 0


def companion_images(run_dir: Path) -> list[Path]:
    """List a run's companion images, sorted by file name."""
    return sorted(companion_ambi_dir(run_dir).glob("*.png"))


def find_latest_complete_run(experiment_dir: Path, min_images: int = MIN_COMPANION_IMAGES) -> Path | None:
    """Return the most recent run whose companion dataset finished generating, None if there is none."""
    runs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    # Interrupted runs are the most recent ones precisely because they were restarted
    complete = [d for d in runs if companion_image_count(d) >= min_images]
    if not complete:
        logger.warning("%s: no run has a complete companion dataset", experiment_dir.name)
        return None
    if len(complete) != len(runs):
        logger.info(
            "%s: ignoring %d run(s) with an incomplete companion dataset",
            experiment_dir.name,
            len(runs) - len(complete),
        )
    return max(complete, key=lambda d: d.stat().st_mtime)


def is_canonical_pair(dataset: str, pos: int, neg: int, experiment_dir: Path) -> bool:
    """Report whether an experiment is the one that should contribute images for its class pair."""
    if pos == neg:
        logger.info("Skipping %s: degenerate pair %dv%d", experiment_dir.name, pos, neg)
        return False

    # Testing for the sibling first, rather than requiring pos < neg, keeps datasets whose only
    # pair is recorded as `1v0`
    reverse = experiment_dir.parent / f"{dataset}-{neg}v{pos}"
    if pos > neg and reverse.is_dir() and reverse != experiment_dir:
        logger.info("Skipping %s: %s covers the same pair", experiment_dir.name, reverse.name)
        return False
    return True


def images_in(image_dir: Path) -> list[Path]:
    """List the images of a directory, sorted by file name."""
    if not image_dir.is_dir():
        return []
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _read_run_config(run_dir: Path) -> dict | None:
    """Return the config every GAN checkpoint of a run dumps, so the first one found will do."""
    config_path = min(run_dir.rglob("config.json"), default=None)
    if config_path is None:
        logger.warning("%s: no GAN checkpoint config found", run_dir.name)
        return None

    with open(config_path, encoding="utf-8") as f:
        return json.load(f).get("config", {})


def _default_models_root(run_dir: Path) -> Path:
    """Return the estimator root implied by a run at `<out_dir>/AmbiGAN/<experiment>/<run>`."""
    return run_dir.parents[2] / "models"


def read_run_spec(run_dir: Path, models_root: Path | None = None) -> RunSpec | None:
    """Describe a run from the config its GAN checkpoints carry, None if that config is unusable."""
    config = _read_run_config(run_dir)
    if config is None:
        return None

    binary = config.get("dataset", {}).get("binary", {})
    dataset_name = config.get("dataset", {}).get("name")
    classifiers = config.get("train", {}).get("step_2", {}).get("classifier") or []
    if dataset_name is None or "pos" not in binary or not classifiers:
        logger.warning("%s: incomplete GAN checkpoint config", run_dir.name)
        return None

    estimator = Path(classifiers[0])
    if not estimator.is_dir():
        # The run may have been produced under a different FILESDIR; re-anchor it locally
        root = models_root if models_root is not None else _default_models_root(run_dir)
        relocated = root / estimator.parent.name / estimator.name
        if not relocated.is_dir():
            logger.warning("%s: estimator %s not found on disk", run_dir.name, estimator)
            return None
        logger.info("%s: estimator re-anchored to %s", run_dir.name, relocated)
        estimator = relocated

    return RunSpec(run_dir, dataset_name, int(binary["pos"]), int(binary["neg"]), estimator)


def find_generator_path(run_dir: Path) -> Path | None:
    """Locate the step-2 generator a run's companion images were drawn from."""
    config = _read_run_config(run_dir)
    if config is None:
        return None

    step_2 = config.get("train", {}).get("step_2", {})
    classifiers = step_2.get("classifier") or []
    epochs = step_2.get("epochs")
    if not classifiers or epochs is None:
        logger.warning("%s: config records no step-2 generator", run_dir.name)
        return None

    # The directory is named after the ensemble plus its loss weights, so match by prefix
    prefix = Path(classifiers[0]).name
    for entry in sorted(run_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith(prefix):
            checkpoint_dir = entry / str(epochs)
            if checkpoint_dir.is_dir():
                return checkpoint_dir
            logger.warning("%s: generator epoch %s missing under %s", run_dir.name, epochs, entry.name)

    logger.warning("%s: no step-2 generator directory starting with %s", run_dir.name, prefix)
    return None


def compute_confusion_distance(
    estimator: nn.Module,
    image_paths: list[Path],
    dataset_name: str,
    device: str,
    batch_size: int = 256,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Run the estimator over every image and return its per-image confusion distance."""
    loader = DataLoader(
        ImageDataset([str(p) for p in image_paths], color_mode="RGB", transform=get_companion_transform(dataset_name)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    probs = []
    with torch.inference_mode():
        for images, _ in loader:
            output = estimator(images.to(device))
            if isinstance(output, tuple):
                output = output[0]
            probs.append(output.flatten().cpu())

    full_probs = torch.cat(probs)

    return pd.DataFrame(
        {
            "image": [p.name for p in image_paths],
            "prob": full_probs.numpy(),
            "confusion_distance": (0.50 - full_probs).abs().numpy(),
        }
    )


@dataclass(frozen=True)
class SelectionOptions:
    """How a companion selection is made; the remaining fields matter only under a threshold."""

    max_confusion_distance: float | None = None
    models_root: Path | None = None
    device: DeviceType | None = None
    batch_size: int = 256
    num_workers: int = 4
    allow_generation: bool = True

    @property
    def filters_by_confusion(self) -> bool:
        """Report whether images have to be scored at all."""
        return self.max_confusion_distance is not None

    def resolved_device(self) -> DeviceType:
        """Return the device to score on, preferring CUDA when none was named."""
        return self.device or (DeviceType.cuda_0 if torch.cuda.is_available() else DeviceType.cpu)


def confusion_distances(image_dir: Path, spec: RunSpec, options: SelectionOptions) -> pd.DataFrame:
    """Return the confusion distance of every image in a directory, scoring whatever is not cached."""
    image_paths = images_in(image_dir)
    if not image_paths:
        return pd.DataFrame(columns=["image", "prob", "confusion_distance"])

    csv_path = image_dir / CONFUSION_DISTANCE_CSV
    cached = pd.DataFrame(columns=["image", "prob", "confusion_distance"])
    if csv_path.is_file():
        cached = pd.read_csv(csv_path)

    scored = set(cached["image"])
    missing = [p for p in image_paths if p.name not in scored]

    if missing:
        device = options.resolved_device()
        logger.info("%s: scoring %d image(s) with %s", image_dir, len(missing), spec.estimator_path.name)
        estimator, _, _, _, _ = construct_classifier_from_checkpoint(str(spec.estimator_path), device=device)
        estimator.eval()
        fresh = compute_confusion_distance(
            estimator, missing, spec.dataset, device, options.batch_size, options.num_workers
        )
        del estimator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Concatenating onto an empty frame would only propagate its placeholder dtypes
        cached = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
        cached.to_csv(csv_path, index=False)

    on_disk = {p.name for p in image_paths}
    return cached[cached["image"].isin(on_disk)].reset_index(drop=True)


def _topup_batch_size(needed: int, qualifying: int, scored: int) -> int:
    """Size the next generation round from the yield the run has shown so far."""
    yield_rate = qualifying / scored if scored else 0.0
    if yield_rate <= 0.0:
        return MAX_TOPUP_BATCH
    return int(min(max(math.ceil(needed / yield_rate * TOPUP_MARGIN), MIN_TOPUP_BATCH), MAX_TOPUP_BATCH))


def _generate_images(generator_path: Path, out_dir: Path, n_samples: int, seed: int, device: str) -> list[Path]:
    """Generate a batch of images from a trained AmbiGAN generator into a fresh directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-m",
        "src.gen_dataset",
        "--gan-path",
        str(generator_path),
        "--seed",
        str(seed),
        "--n-samples",
        str(n_samples),
        "--device",
        device,
        "--out-dir",
        str(out_dir),
        "--skip-stats",
    ]
    logger.info("Generating %d image(s) into %s", n_samples, out_dir)
    subprocess.run(args, check=True, env=os.environ.copy())
    return images_in(out_dir)


def _next_topup_round(topup_root: Path) -> int:
    """Return the index of the next round, so rounds never overwrite one another."""
    if not topup_root.is_dir():
        return 0
    existing = [d.name for d in topup_root.iterdir() if d.is_dir() and d.name.startswith(TOPUP_ROUND_PREFIX)]
    return len(existing)


def _topup_root(run_dir: Path) -> Path:
    """Return the directory holding whatever was generated to extend a run's pool."""
    return run_dir / TOPUP_DIR_NAME


def _pool_dirs(run_dir: Path) -> list[Path]:
    """List a run's companion dataset and any top-up rounds."""
    rounds = sorted(d for d in _topup_root(run_dir).glob(f"{TOPUP_ROUND_PREFIX}*") if d.is_dir())
    return [companion_ambi_dir(run_dir), *rounds]


def _full_pool(run_dir: Path) -> list[str]:
    """Return the run's companion dataset as generated, excluding top-ups."""
    # Folding top-ups in would make the unfiltered dataset depend on whether the filtered one was
    # built first; the baseline has to stay reproducible on its own.
    return [str(p) for p in images_in(companion_ambi_dir(run_dir))]


def _ambiguous_pool(run_dir: Path, spec: RunSpec, options: SelectionOptions) -> tuple[list[str], int]:
    """Return a run's images within the confusion threshold, and how many were scored to find them."""
    threshold = options.max_confusion_distance
    qualifying: list[str] = []
    scored = 0
    for image_dir in _pool_dirs(run_dir):
        df = confusion_distances(image_dir, spec, options)
        if df.empty:
            continue
        scored += len(df)
        keep = df.loc[df["confusion_distance"] <= threshold, "image"]
        qualifying.extend(str(image_dir / name) for name in keep)
    return sorted(qualifying), scored


def select_companion_images(
    run_dir: Path,
    n_samples: int | None,
    options: SelectionOptions | None = None,
) -> list[str]:
    """
    Draw `n_samples` images from one run, uniformly over the pool `options` makes eligible.

    Under a confusion threshold a run holding too few eligible images is topped up with generated
    ones, filtered by the same threshold. `n_samples=None` takes the whole pool and never generates.
    Returns fewer than asked only when the pool could not be filled, which is logged.
    """
    options = options or SelectionOptions()

    if not options.filters_by_confusion:
        pool = sorted(_full_pool(run_dir))
        logger.info("%s: sampling %s of %d companion image(s)", run_dir.name, n_samples or "all", len(pool))
        return _draw(pool, n_samples)

    spec = read_run_spec(run_dir, options.models_root)
    if spec is None:
        logger.warning("%s: cannot score companion images without a run spec", run_dir.name)
        return []

    pool, scored = _ambiguous_pool(run_dir, spec, options)
    logger.info(
        "%s: %d/%d companion image(s) within confusion distance %.3f",
        run_dir.name,
        len(pool),
        scored,
        options.max_confusion_distance,
    )

    if n_samples is not None and len(pool) < n_samples and options.allow_generation:
        pool = _extend_pool(pool, scored, run_dir, spec, n_samples, options)

    if n_samples is not None and len(pool) < n_samples:
        logger.warning(
            "%s: only %d image(s) within confusion distance %.3f, using all of them instead of %d",
            run_dir.name,
            len(pool),
            options.max_confusion_distance,
            n_samples,
        )

    return _draw(pool, n_samples)


def _draw(pool: list[str], n_samples: int | None) -> list[str]:
    """Sample `n_samples` paths uniformly from a pool, or return it whole when it is short."""
    if n_samples is None or len(pool) <= n_samples:
        return pool
    chosen = np.random.choice(len(pool), size=n_samples, replace=False)
    return [pool[i] for i in sorted(chosen)]


def _extend_pool(
    pool: list[str],
    scored: int,
    run_dir: Path,
    spec: RunSpec,
    n_samples: int,
    options: SelectionOptions,
) -> list[str]:
    """Generate images from the run's generator until the qualifying pool is large enough."""
    generator_path = find_generator_path(run_dir)
    if generator_path is None:
        logger.warning("%s: cannot generate more images, no step-2 generator found", run_dir.name)
        return pool

    threshold = options.max_confusion_distance
    topup_root = _topup_root(run_dir)
    for attempt in range(MAX_TOPUP_ROUNDS):
        needed = n_samples - len(pool)
        batch = _topup_batch_size(needed, len(pool), scored)
        round_dir = topup_root / f"{TOPUP_ROUND_PREFIX}_{_next_topup_round(topup_root):02d}"
        logger.info(
            "%s: %d image(s) short of %d, generating %d more (round %d/%d)",
            run_dir.name,
            needed,
            n_samples,
            batch,
            attempt + 1,
            MAX_TOPUP_ROUNDS,
        )

        # A fresh seed per round, or every round would regenerate the same images
        seed = int(np.random.randint(0, 2**31 - 1))
        _generate_images(generator_path, round_dir, batch, seed, options.resolved_device())

        df = confusion_distances(round_dir, spec, options)
        scored += len(df)
        keep = df.loc[df["confusion_distance"] <= threshold, "image"]
        pool = sorted([*pool, *(str(round_dir / name) for name in keep)])
        logger.info("%s: pool now holds %d qualifying image(s)", run_dir.name, len(pool))

        if len(pool) >= n_samples:
            break

    return pool
