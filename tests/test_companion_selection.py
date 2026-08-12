"""Module to test how companion images are selected for evaluation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.datasets import companion_selection
from src.datasets.companion_selection import (
    MAX_CONFUSION_DISTANCE,
    SelectionOptions,
    _topup_batch_size,
    companion_ambi_dir,
    find_generator_path,
    find_latest_complete_run,
    is_canonical_pair,
    read_run_spec,
    select_companion_images,
)

ESTIMATOR_NAME = "ensemble_mean_32_mnist_50_3v6_balanced"
GENERATOR_DIR = f"{ESTIMATOR_NAME}_gaussian_0.5_0.1_last"
STEP_2_EPOCHS = 50


def write_images(image_dir: Path, count: int, start: int = 0) -> list[str]:
    """Write `count` tiny greyscale PNGs and return their names."""
    image_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(start, start + count):
        name = f"image_{i:06d}.png"
        Image.new("L", (4, 4), color=i % 256).save(image_dir / name)
        names.append(name)
    return names


def write_confusion_distances(image_dir: Path, names: list[str], distances: list[float]) -> None:
    """Write the per-directory confusion distance cache the selection reads."""
    pd.DataFrame(
        {
            "image": names,
            "prob": [0.5 - d for d in distances],
            "confusion_distance": distances,
        }
    ).to_csv(image_dir / companion_selection.CONFUSION_DISTANCE_CSV, index=False)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Build a run directory shaped like an AmbiGAN run, without any model weights."""
    run = tmp_path / "AmbiGAN" / "mnist-3v6" / "Feb14T07-20_k7n828kk"
    (run / "step_1" / "50").mkdir(parents=True)
    estimator_path = tmp_path / "models" / "mnist.3v6" / ESTIMATOR_NAME
    estimator_path.mkdir(parents=True)
    (run / GENERATOR_DIR / str(STEP_2_EPOCHS)).mkdir(parents=True)

    config = {
        "config": {
            "dataset": {"name": "mnist", "binary": {"pos": 3, "neg": 6}},
            "train": {"step_2": {"epochs": STEP_2_EPOCHS, "classifier": [str(estimator_path)]}},
        }
    }
    with open(run / "step_1" / "50" / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    return run


def test_read_run_spec_describes_the_run(run_dir: Path) -> None:
    """The class pair and guiding estimator come from the GAN checkpoint config."""
    spec = read_run_spec(run_dir)

    assert spec is not None
    assert (spec.dataset, spec.pos, spec.neg, spec.pair) == ("mnist", 3, 6, "3v6")
    assert spec.estimator_path.name == ESTIMATOR_NAME


def test_read_run_spec_reanchors_a_relocated_estimator(run_dir: Path, tmp_path: Path) -> None:
    """A run produced under a different FILESDIR resolves its estimator against the local models root."""
    config_path = run_dir / "step_1" / "50" / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["config"]["train"]["step_2"]["classifier"] = [f"/somewhere/else/models/mnist.3v6/{ESTIMATOR_NAME}"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    spec = read_run_spec(run_dir)

    assert spec is not None
    assert spec.estimator_path == tmp_path / "models" / "mnist.3v6" / ESTIMATOR_NAME


def test_find_latest_complete_run_skips_interrupted_runs(run_dir: Path) -> None:
    """A restarted run holding a handful of images loses to the older run that finished."""
    experiment_dir = run_dir.parent
    write_images(companion_ambi_dir(run_dir), 12)

    interrupted = experiment_dir / "Feb27T18-42_5eivz4dv"
    write_images(companion_ambi_dir(interrupted), 3)
    # Make the interrupted run the most recently modified one
    interrupted.touch()

    assert find_latest_complete_run(experiment_dir, min_images=10) == run_dir


def test_find_latest_complete_run_reports_when_nothing_is_complete(run_dir: Path) -> None:
    """An experiment whose every run was interrupted yields nothing rather than a partial dataset."""
    write_images(companion_ambi_dir(run_dir), 3)

    assert find_latest_complete_run(run_dir.parent, min_images=10) is None


@pytest.mark.parametrize(
    ("pos", "neg", "siblings", "expected"),
    [
        # A pair classified against itself carries no signal
        (8, 8, (), False),
        # Only the pos < neg half of a pair present in both directions is kept
        (9, 8, ("mnist-8v9",), False),
        (8, 9, ("mnist-9v8",), True),
        # An inherently binary dataset records its only pair as 1v0, and must survive
        (1, 0, (), True),
        (3, 6, (), True),
    ],
)
def test_is_canonical_pair(tmp_path: Path, pos: int, neg: int, siblings: tuple[str, ...], expected: bool) -> None:
    """Each class pair contributes exactly once, whichever way round its directory is named."""
    gan_root = tmp_path / "AmbiGAN"
    experiment_dir = gan_root / f"mnist-{pos}v{neg}"
    experiment_dir.mkdir(parents=True)
    for sibling in siblings:
        (gan_root / sibling).mkdir()

    assert is_canonical_pair("mnist", pos, neg, experiment_dir) is expected


def test_find_generator_path_matches_the_step_2_directory(run_dir: Path) -> None:
    """The generator is found by prefix, since its directory also carries the loss weights."""
    assert find_generator_path(run_dir) == run_dir / GENERATOR_DIR / str(STEP_2_EPOCHS)


def test_full_distribution_selection_ignores_confusion_distance(run_dir: Path) -> None:
    """Without a threshold, every companion image is eligible and none is scored."""
    write_images(companion_ambi_dir(run_dir), 20)

    np.random.seed(0)
    selected = select_companion_images(run_dir, 5)

    assert len(selected) == 5
    assert not (companion_ambi_dir(run_dir) / companion_selection.CONFUSION_DISTANCE_CSV).exists()


def test_full_distribution_selection_returns_everything_when_short(run_dir: Path) -> None:
    """Asking for more images than the run holds yields the whole pool rather than failing."""
    write_images(companion_ambi_dir(run_dir), 3)

    assert len(select_companion_images(run_dir, 10)) == 3


def test_ambiguous_selection_keeps_only_images_within_the_threshold(run_dir: Path) -> None:
    """Under a threshold, an image is eligible only if the estimator was undecided about it."""
    ambi = companion_ambi_dir(run_dir)
    names = write_images(ambi, 10)
    # The first four sit inside the threshold, the rest well outside it
    write_confusion_distances(ambi, names, [0.01, 0.02, 0.03, 0.05] + [0.4] * 6)

    np.random.seed(0)
    selected = select_companion_images(run_dir, 3, SelectionOptions(max_confusion_distance=MAX_CONFUSION_DISTANCE))

    assert len(selected) == 3
    assert {Path(p).name for p in selected} <= set(names[:4])


def test_ambiguous_selection_generates_more_when_the_pool_is_short(run_dir: Path, monkeypatch: Any) -> None:
    """A run without enough ambiguous images is topped up, and the extra images face the same threshold."""
    ambi = companion_ambi_dir(run_dir)
    names = write_images(ambi, 10)
    write_confusion_distances(ambi, names, [0.01, 0.02] + [0.4] * 8)

    generated: list[Path] = []

    def fake_generate(_generator_path: Path, out_dir: Path, n_samples: int, _seed: int, _device: str) -> list[Path]:
        fresh = write_images(out_dir, n_samples)
        # Half the generated images land inside the threshold
        write_confusion_distances(out_dir, fresh, [0.01 if i % 2 == 0 else 0.3 for i in range(n_samples)])
        generated.append(out_dir)
        return [out_dir / name for name in fresh]

    monkeypatch.setattr(companion_selection, "_generate_images", fake_generate)
    monkeypatch.setattr(companion_selection, "MIN_TOPUP_BATCH", 10)

    np.random.seed(0)
    selected = select_companion_images(run_dir, 6, SelectionOptions(max_confusion_distance=MAX_CONFUSION_DISTANCE))

    assert len(generated) == 1
    assert len(selected) == 6
    # Every selection is ambiguous, and the two seed images alone could not have filled the quota
    assert any("companion_topup" in path for path in selected)


def test_ambiguous_selection_does_not_generate_when_forbidden(run_dir: Path, monkeypatch: Any) -> None:
    """With generation disabled, a short pool is returned as-is rather than relaxing the threshold."""
    ambi = companion_ambi_dir(run_dir)
    names = write_images(ambi, 5)
    write_confusion_distances(ambi, names, [0.01, 0.02, 0.4, 0.4, 0.4])

    def fail(*_args: Any, **_kwargs: Any) -> list[Path]:
        raise AssertionError("generation should not have been attempted")

    monkeypatch.setattr(companion_selection, "_generate_images", fail)

    options = SelectionOptions(max_confusion_distance=MAX_CONFUSION_DISTANCE, allow_generation=False)
    assert len(select_companion_images(run_dir, 4, options)) == 2


def test_topup_reuses_previously_generated_rounds(run_dir: Path) -> None:
    """Images generated by an earlier top-up stay in the pool instead of being generated again."""
    ambi = companion_ambi_dir(run_dir)
    names = write_images(ambi, 4)
    write_confusion_distances(ambi, names, [0.01, 0.4, 0.4, 0.4])

    round_dir = run_dir / companion_selection.TOPUP_DIR_NAME / "round_00"
    round_names = write_images(round_dir, 4)
    write_confusion_distances(round_dir, round_names, [0.01, 0.02, 0.03, 0.4])

    options = SelectionOptions(max_confusion_distance=MAX_CONFUSION_DISTANCE, allow_generation=False)
    selected = select_companion_images(run_dir, 4, options)

    assert len(selected) == 4
    assert sum("round_00" in path for path in selected) == 3


def test_full_distribution_selection_ignores_topup_rounds(run_dir: Path) -> None:
    """Images generated to satisfy a threshold stay out of the unfiltered dataset, which must not drift."""
    write_images(companion_ambi_dir(run_dir), 4)
    write_images(run_dir / companion_selection.TOPUP_DIR_NAME / "round_00", 8)

    selected = select_companion_images(run_dir, None)

    assert len(selected) == 4
    assert not any("companion_topup" in path for path in selected)


def test_selection_without_a_quota_takes_the_whole_pool(run_dir: Path) -> None:
    """A None quota, as chest-xray uses, returns every eligible image."""
    ambi = companion_ambi_dir(run_dir)
    names = write_images(ambi, 6)
    write_confusion_distances(ambi, names, [0.01, 0.02, 0.03, 0.4, 0.4, 0.4])

    options = SelectionOptions(max_confusion_distance=MAX_CONFUSION_DISTANCE)
    assert len(select_companion_images(run_dir, None, options)) == 3


@pytest.mark.parametrize(
    ("needed", "qualifying", "scored", "expected"),
    [
        # Nothing has qualified yet, so there is no yield to extrapolate from
        (100, 0, 500, companion_selection.MAX_TOPUP_BATCH),
        # A tiny need still generates a worthwhile batch
        (1, 500, 1000, companion_selection.MIN_TOPUP_BATCH),
        # A vanishing yield would ask for more than is worth generating in one round
        (100, 1, 10000, companion_selection.MAX_TOPUP_BATCH),
        # Half the images qualify, so roughly twice the need is asked for
        (400, 500, 1000, 1000),
    ],
)
def test_topup_batch_size_is_bounded(needed: int, qualifying: int, scored: int, expected: int) -> None:
    """The next round is sized from the observed yield, and clamped when that estimate is unusable."""
    assert _topup_batch_size(needed, qualifying, scored) == expected
