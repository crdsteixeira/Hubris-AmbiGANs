"""CL to compute per-image confusion distance of companion datasets against their guiding ensemble."""

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from tqdm import tqdm

from src.datasets.companion_selection import (
    CONFUSION_DISTANCE_CSV,
    MIN_COMPANION_IMAGES,
    companion_ambi_dir,
    companion_image_count,
    companion_images,
    compute_confusion_distance,
    is_canonical_pair,
    read_run_spec,
)
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.logging import configure_logging

load_dotenv()

configure_logging()
logger = logging.getLogger(__name__)

COMBINED_CSV = "confusion_distance_all.csv"

# Experiments whose run is fixed rather than resolved by modification time. `chest-xray` is the only
# experiment with more than one complete run, and the tie-break must not drift between sweeps.
DEFAULT_PINNED_RUNS = {"chest-xray": "Feb23T17-08_7496mtok"}

# Confusion distance is bounded: 0.0 is a perfectly ambiguous sample, 0.5 a perfectly confident one
CD_LIMITS = (0.0, 0.5)
CD_LABEL = "confusion distance"

# Datasets are presented in this order, under these names
DATASET_ORDER = ("mnist", "fashion-mnist", "chest-xray")
DATASET_LABELS = {"mnist": "MNIST", "fashion-mnist": "FashionMNIST", "chest-xray": "Chest-XRay"}

# One hue throughout: the axis already names each dataset, so colour is free to stay constant
# rather than encode an identity the reader can already see.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SERIES_COLOR = "#2a78d6"

# PNG for previewing, PDF for the paper: vector, and what LaTeX includes directly
FIGURE_FORMATS = ("png", "pdf")

HIST_KWARGS = {"x": "confusion_distance", "bins": 50, "binrange": CD_LIMITS, "stat": "density", "color": SERIES_COLOR}
# `cut=0` keeps each violin inside the range the data actually spans: the default KDE tails would run
# past 0.0, implying confusion distances that cannot exist. `density_norm="width"` gives every violin
# the same maximum width; the default ("area") scales a panel by its densest violin, and mnist 0v1 is
# concentrated enough (std 0.0007) to flatten the other 44 pairs into hairlines. Every group holds the
# same 2500 samples, so normalising width rather than area loses nothing.
# Seaborn fixes the dash pattern of the inner lines (dashed median, dotted quartiles) and only the
# weight is ours to set; the two patterns are what tells median and quartiles apart.
VIOLIN_KWARGS = {
    "color": SERIES_COLOR,
    "cut": 0,
    "density_norm": "width",
    "inner": "quart",
    "inner_kws": {"linewidth": 0.7},
}
# Violin bodies are washed out so the quartile lines and outlines read over them
VIOLIN_FILL_ALPHA = 0.5


def find_run_dir(experiment_dir: Path, pinned_run: str | None = None) -> Path | None:
    """
    Select the run of an experiment whose companion dataset should be used.

    A pinned run is honoured verbatim; otherwise the most recent run with a complete companion
    dataset wins. A pin that cannot be honoured is an error rather than a fallback, so that pinning
    an experiment can never silently resolve to a different run.
    """
    if pinned_run is not None:
        run_dir = experiment_dir / pinned_run
        if not run_dir.is_dir():
            logger.error("%s: pinned run %s does not exist", experiment_dir.name, pinned_run)
            return None
        if companion_image_count(run_dir) < MIN_COMPANION_IMAGES:
            logger.error("%s: pinned run %s has an incomplete companion dataset", experiment_dir.name, pinned_run)
            return None
        logger.info("%s: using pinned run %s", experiment_dir.name, pinned_run)
        return run_dir

    runs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    complete = [d for d in runs if companion_image_count(d) >= MIN_COMPANION_IMAGES]
    if not complete:
        return None
    if len(complete) != len(runs):
        logger.info(
            "%s: ignoring %d run(s) with an incomplete companion dataset",
            experiment_dir.name,
            len(runs) - len(complete),
        )
    if len(complete) > 1:
        logger.warning(
            "%s: %d complete runs; falling back to the most recent. Pin one with --pin to make this stable.",
            experiment_dir.name,
            len(complete),
        )

    return max(complete, key=lambda d: d.stat().st_mtime)


def process_experiment(experiment_dir: Path, args: argparse.Namespace) -> pd.DataFrame | None:
    """Compute and persist per-image confusion distances for one experiment's companion dataset."""
    run_dir = find_run_dir(experiment_dir, args.pinned_runs.get(experiment_dir.name))
    if run_dir is None:
        logger.info("Skipping %s: no run with a complete companion dataset", experiment_dir.name)
        return None

    spec = read_run_spec(run_dir, Path(args.models_root))
    if spec is None or not is_canonical_pair(spec.dataset, spec.pos, spec.neg, experiment_dir):
        return None

    out_csv = companion_ambi_dir(run_dir) / CONFUSION_DISTANCE_CSV
    if out_csv.is_file() and not args.overwrite:
        logger.info("%s: reusing %s", experiment_dir.name, out_csv)
        df = pd.read_csv(out_csv)
    else:
        image_paths = companion_images(run_dir)
        estimator, _, _, _, _ = construct_classifier_from_checkpoint(str(spec.estimator_path), device=args.device)
        estimator.eval()
        df = compute_confusion_distance(
            estimator, image_paths, spec.dataset, args.device, args.batch_size, args.num_workers
        )
        del estimator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        df.to_csv(out_csv, index=False)
        logger.info(
            "%s: %d images, mean confusion distance %.6f -> %s",
            experiment_dir.name,
            len(df),
            df["confusion_distance"].mean(),
            out_csv,
        )

    return df.assign(
        dataset=spec.dataset,
        pos_class=spec.pos,
        neg_class=spec.neg,
        pair=spec.pair,
        experiment=experiment_dir.name,
        run_id=run_dir.name,
        estimator=spec.estimator_path.name,
    )


def apply_theme() -> None:
    """Set the recessive chart styling shared by every figure."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Embed TrueType rather than Type 3 fonts: several venues reject Type 3 in submissions
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def dataset_label(dataset_name: str) -> str:
    """Return the display name of a dataset, falling back to its raw name."""
    return DATASET_LABELS.get(dataset_name, dataset_name)


def split_by_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the frame per dataset once, in presentation order, so plotting never re-filters it."""
    groups = {str(name): subset for name, subset in df.groupby("dataset", sort=False)}
    ordered = [name for name in DATASET_ORDER if name in groups]
    # Anything unrecognised still gets plotted rather than silently dropped
    return {name: groups[name] for name in ordered + sorted(set(groups) - set(DATASET_ORDER))}


def soften_violin_fill(ax: Axes) -> None:
    """
    Wash out the violin bodies, leaving their outlines and quartile lines at full strength.

    Seaborn drops the alpha channel from `color`, so the translucent fill has to be applied to the
    drawn bodies rather than requested up front.
    """
    for body in ax.collections:
        body.set_facecolor(to_rgba(SERIES_COLOR, VIOLIN_FILL_ALPHA))


def save_figure(fig: Figure, out_dir: Path, name: str, formats: Sequence[str]) -> None:
    """Write a figure once per requested format, then release it."""
    for suffix in formats:
        fig.savefig(out_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_histograms(groups: dict[str, pd.DataFrame], df: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    """Plot the confusion distance distribution aggregated over all estimators, then split by dataset."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=df, ax=ax, **HIST_KWARGS)
    ax.set(xlim=CD_LIMITS, xlabel=CD_LABEL)
    save_figure(fig, out_dir, "confusion_distance_all", formats)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.5), sharey=True, squeeze=False)
    for ax, (dataset_name, subset) in zip(axes[0], groups.items()):
        sns.histplot(data=subset, ax=ax, **HIST_KWARGS)
        ax.set(xlim=CD_LIMITS, xlabel=CD_LABEL, title=dataset_label(dataset_name))
    save_figure(fig, out_dir, "confusion_distance_by_dataset", formats)

    logger.info("Histograms saved to %s", out_dir)


def plot_violins(groups: dict[str, pd.DataFrame], df: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    """Plot violins of the confusion distance, by dataset and then per class pair."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(
        data=df, x="dataset", y="confusion_distance", order=list(groups), linewidth=0.8, ax=ax, **VIOLIN_KWARGS
    )
    soften_violin_fill(ax)
    ax.set(ylim=CD_LIMITS, ylabel=CD_LABEL, xlabel="")
    ax.set_xticks(range(len(groups)), [dataset_label(name) for name in groups])
    save_figure(fig, out_dir, "confusion_distance_violin_by_dataset", formats)

    # One violin per class pair. Datasets with a single pair carry no spread to show, so they are
    # already fully described by the figure above.
    multi_pair = {name: subset for name, subset in groups.items() if subset["pair"].nunique() > 1}
    if multi_pair:
        tallest = max(subset["pair"].nunique() for subset in multi_pair.values())
        fig, axes = plt.subplots(
            1, len(multi_pair), figsize=(6 * len(multi_pair), max(6.0, 0.24 * tallest)), sharex=True, squeeze=False
        )
        for ax, (dataset_name, subset) in zip(axes[0], multi_pair.items()):
            # Ascending median puts the pairs the GAN found most ambiguous at the top
            order = subset.groupby("pair")["confusion_distance"].median().sort_values().index
            sns.violinplot(
                data=subset, x="confusion_distance", y="pair", order=order, linewidth=0.6, ax=ax, **VIOLIN_KWARGS
            )
            soften_violin_fill(ax)
            ax.set(
                xlim=CD_LIMITS,
                xlabel=CD_LABEL,
                ylabel="class pair" if ax is axes[0][0] else "",
                title=dataset_label(dataset_name),
            )
        save_figure(fig, out_dir, "confusion_distance_violin_by_pair", formats)

    logger.info("Violins saved to %s", out_dir)


def plot_all(df: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    """Draw every figure from one frame."""
    apply_theme()
    groups = split_by_dataset(df)
    plot_histograms(groups, df, out_dir, formats)
    plot_violins(groups, df, out_dir, formats)


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line."""
    files_dir = os.environ.get("FILESDIR", "")
    parser = argparse.ArgumentParser(
        description="Compute per-image confusion distance of every companion dataset against its guiding ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gan-root", default=os.path.join(files_dir, "AmbiGAN"), help="Root with experiment runs")
    parser.add_argument("--models-root", default=os.path.join(files_dir, "models"), help="Root with estimators")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(files_dir, "out", "confusion-distance"),
        help="Directory for the combined CSV and the figures",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for estimator inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker processes for image loading")
    parser.add_argument(
        "--experiments", nargs="+", default=None, help="Restrict to these experiment directories (default: all)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute runs that already have a per-run CSV")
    parser.add_argument("--plots-only", action="store_true", help="Only replot from the existing combined CSV")
    parser.add_argument(
        "--pin",
        nargs="+",
        default=[],
        metavar="EXPERIMENT=RUN_ID",
        help=f"Fix an experiment to a given run, overriding the defaults: {DEFAULT_PINNED_RUNS}",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(FIGURE_FORMATS),
        choices=["png", "pdf", "svg", "eps"],
        help="Image formats to write each figure in",
    )

    args = parser.parse_args()

    args.pinned_runs = dict(DEFAULT_PINNED_RUNS)
    for pin in args.pin:
        experiment, _, run_id = pin.partition("=")
        if not run_id:
            parser.error(f"--pin expects EXPERIMENT=RUN_ID, got {pin!r}")
        args.pinned_runs[experiment] = run_id

    return args


def main() -> None:
    """Compute, persist and plot companion confusion distances across every experiment."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / COMBINED_CSV

    if args.plots_only:
        plot_all(pd.read_csv(combined_csv), out_dir, args.formats)
        return

    experiment_dirs = sorted(d for d in Path(args.gan_root).iterdir() if d.is_dir())
    if args.experiments:
        wanted = set(args.experiments)
        experiment_dirs = [d for d in experiment_dirs if d.name in wanted]

    frames = [process_experiment(d, args) for d in tqdm(experiment_dirs, desc="experiments")]
    frames = [frame for frame in frames if frame is not None]

    if not frames:
        logger.error("No companion dataset was processed; nothing to save")
        return

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(combined_csv, index=False)
    logger.info("Processed %d estimator(s) over %d images -> %s", df["experiment"].nunique(), len(df), combined_csv)

    summary = (
        df.groupby(["dataset", "experiment"])["confusion_distance"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.to_csv(out_dir / "confusion_distance_summary.csv", index=False)

    plot_all(df, out_dir, args.formats)


if __name__ == "__main__":
    main()
