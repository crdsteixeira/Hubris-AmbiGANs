"""
Upload a companion dataset, with its ground truth labels, to the HuggingFace Hub.

    python -m src.utils.huggingface --repo username/repo-name --dataset mnist

Run with --help for the full options. Uploading needs HF_TOKEN set, or `huggingface-cli login`.
Ground truth is the class pair an image was generated for, e.g. [4, 9] for the mnist-4v9 subset.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, Image
from huggingface_hub.utils import HfHubHTTPError

from src.datasets.companion_selection import MAX_CONFUSION_DISTANCE
from src.datasets.datasets import _find_companion_dataset_images

logger = logging.getLogger(__name__)

# CLI names against the directory names the AmbiGAN outputs use
DATASET_DIRS = {"mnist": "mnist", "fashion-mnist": "fashion_mnist", "chest-xray": "chest_xray"}

TEST_MODE_SAMPLES = 5


def load_companion_dataset(dataroot: str, dataset_name: str, max_confusion_distance: float | None = None) -> Dataset:
    """Build a dataset of companion images, each labelled with the class pair it was generated for."""
    image_paths, ground_truths = _find_companion_dataset_images(
        dataroot, dataset_name, max_confusion_distance=max_confusion_distance
    )
    dataset = Dataset.from_dict(
        {
            "image": image_paths,
            # A list is not a column type the Hub round-trips, so labels travel as JSON
            "ground_truth": [json.dumps(gt) for gt in ground_truths],
        }
    )
    logger.info("Loaded %d %s companion images", len(dataset), dataset_name)
    return dataset.cast_column("image", Image())


def upload_to_hub(dataset: Dataset, dataset_name: str, repo_name: str, private: bool = True) -> str:
    """Push a dataset to the Hub as its own config, and return the repository URL."""
    logger.info("Uploading %d %s samples to %s...", len(dataset), dataset_name, repo_name)
    try:
        dataset.push_to_hub(
            repo_id=repo_name,
            config_name=dataset_name,
            split="train",
            private=private,
            commit_message=f"Add {dataset_name} companion dataset",
        )
    except HfHubHTTPError as e:
        if "401" in str(e):
            raise ValueError("Authentication failed. Run `huggingface-cli login` or set HF_TOKEN.") from e
        raise

    return f"https://huggingface.co/datasets/{repo_name}"


def save_metadata(dataset: Dataset, dataset_name: str, output_dir: str) -> Path:
    """Write the ground truth labels and their class-pair distribution to a JSON file."""
    ground_truths = [json.loads(gt) for gt in dataset["ground_truth"]]
    output_path = Path(output_dir) / f"{dataset_name}_metadata.json"

    metadata = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "columns": dataset.column_names,
        "unique_classes": sorted({c for gt in ground_truths for c in gt}),
        "class_pair_distribution": Counter(",".join(map(str, sorted(gt))) for gt in ground_truths),
        "ground_truth_labels": ground_truths,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", output_path)
    return output_path


def resolve_dirs(data_root: str | None) -> tuple[str, str]:
    """Return the data root to read from and the directory metadata is written to."""
    if data_root is not None:
        return data_root, os.path.join(os.path.dirname(data_root), "hf-datasets")

    filesdir = os.environ.get("FILESDIR", ".")
    if filesdir == ".":
        return ".", "hf-datasets"
    return os.path.join(filesdir, "data"), os.path.join(filesdir, "hf-datasets")


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Upload a companion dataset to the HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="Repository, as username/repo-name")
    parser.add_argument("--dataset", choices=sorted(DATASET_DIRS), required=True, help="Dataset to upload")
    parser.add_argument("--data-root", default=None, help="Data root (default: FILESDIR/data)")
    parser.add_argument("--private", action="store_true", help="Upload as private (default: public)")
    parser.add_argument("--test", action="store_true", help=f"Upload only {TEST_MODE_SAMPLES} images")
    parser.add_argument(
        "--ambiguous", action="store_true", help=f"Keep only images with confusion distance <= {MAX_CONFUSION_DISTANCE}"
    )
    return parser.parse_args()


def main() -> None:
    """Load the requested companion dataset, upload it, and save its metadata."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    data_root, metadata_dir = resolve_dirs(args.data_root)
    os.makedirs(metadata_dir, exist_ok=True)
    threshold = MAX_CONFUSION_DISTANCE if args.ambiguous else None
    logger.info(
        "Uploading %s from %s as %s, %s",
        args.dataset,
        data_root,
        "private" if args.private else "public",
        f"confusion distance <= {threshold}" if threshold else "full distribution",
    )

    try:
        dataset = load_companion_dataset(data_root, DATASET_DIRS[args.dataset], threshold)
        if args.test:
            dataset = dataset.select(range(min(TEST_MODE_SAMPLES, len(dataset))))
            logger.info("Test mode: keeping %d image(s)", len(dataset))
        url = upload_to_hub(dataset, args.dataset, args.repo, args.private)
        metadata_path = save_metadata(dataset, args.dataset, metadata_dir)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("✗ %s: %s", args.dataset, e)
        sys.exit(1)

    logger.info("✓ %s: %d samples -> %s (metadata: %s)", args.dataset, len(dataset), url, metadata_path)


if __name__ == "__main__":
    main()
