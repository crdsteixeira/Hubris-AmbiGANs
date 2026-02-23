"""
Module for HuggingFace Hub operations, including dataset uploads.

This module provides utilities to upload companion datasets (with ground truth labels)
to HuggingFace Hub. It supports MNIST, Fashion-MNIST, and Chest X-ray datasets
extracted from AmbiGAN training outputs.

DATASET INFORMATION:
====================
- MNIST and Fashion-MNIST: 9,000 images from 45 binary subsets (200 images each)
- Chest X-ray: All images from single binary directory (no sampling)
- Each image includes ground truth labels indicating valid class pairs

RUNNING THE SCRIPT:
===================

Basic usage - Upload as PUBLIC (default):
    python -m src.utils.huggingface --repo username/repo-name --dataset mnist

Upload as PRIVATE:
    python -m src.utils.huggingface --repo username/repo-name --dataset mnist --private

Upload Chest X-ray:
    python -m src.utils.huggingface --repo username/repo-name --dataset chest-xray

Test mode (5 images only - for verifying upload works):
    python -m src.utils.huggingface --repo username/repo-name --dataset mnist --test

Test Fashion-MNIST as private:
    python -m src.utils.huggingface --repo username/repo-name --dataset fashion-mnist --test --private

With custom data root:
    python -m src.utils.huggingface --repo username/repo-name --dataset mnist --data-root /path/to/data

COMMAND-LINE ARGUMENTS:
=======================
--repo              : HuggingFace repository name (required). Format: username/repo-name
--dataset           : Dataset to upload (required). Choices: mnist, fashion-mnist, chest-xray
--data-root         : Root directory for datasets. Default: FILESDIR/data from env
--private           : Upload as private dataset. Default: public
--test              : Test mode - upload only 5 images to verify upload works

ENVIRONMENT VARIABLES:
======================
FILESDIR            : Root directory for file storage. Metadata saved to FILESDIR/hf-datasets
HF_TOKEN            : HuggingFace API token (required for uploads)

AUTHENTICATION:
===============
Ensure you're authenticated with HuggingFace before running:
    huggingface-cli login

Or set the environment variable:
    export HF_TOKEN=your_hf_token_here

Examples:
=========
# Upload MNIST publicly
python -m src.utils.huggingface --repo user/datasets --dataset mnist

# Test Fashion-MNIST upload
python -m src.utils.huggingface --repo user/datasets --dataset fashion-mnist --test

# Upload Chest X-ray privately
python -m src.utils.huggingface --repo user/datasets --dataset chest-xray --private

# Upload with custom data location
python -m src.utils.huggingface --repo user/datasets --dataset mnist --data-root /media/data

"""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Image
from torchvision.datasets import ImageFolder

from src.datasets.datasets import _find_companion_dataset_images

logger = logging.getLogger(__name__)


def load_companion_dataset_from_directories(
    dataroot: str,
    dataset_name: str,
    n_samples_per_subset: int = 200,
) -> tuple[list[str], list[list[int]]]:
    """
    Load companion dataset images and ground truth labels from directory structure.

    For MNIST and Fashion-MNIST: samples 200 images from each of 45 binary subsets.
    For Chest X-ray: uses all images from the single binary directory (no sampling).

    IMPORTANT: This function delegates to the tested implementation in datasets.py
    to ensure consistent behavior with the evaluation pipeline.

    Args:
        dataroot: Root directory containing the dataset (e.g., FILESDIR)
        dataset_name: Base name of dataset ('mnist', 'fashion_mnist', or 'chest_xray')
        n_samples_per_subset: Number of images to sample per binary subset
                            (ignored for chest_xray)

    Returns:
        Tuple of (image_paths, ground_truth_labels) where ground_truth_labels
        is a list of lists, each containing valid class indices for that image

    Raises:
        ValueError: If no companion dataset images are found

    """
    # Use the tested implementation from datasets.py
    return _find_companion_dataset_images(dataroot, dataset_name, n_samples_per_subset)


def load_companion_dataset_from_imagefolder(
    data_root: str,
    dataset_name: str,
) -> Dataset:
    """
    Load a companion dataset from ImageFolder format.

    Args:
        data_root: Root directory containing the companion dataset
        dataset_name: Name of the dataset (e.g., 'companion-mnist', 'companion-fashion-mnist')

    Returns:
        HuggingFace Dataset object

    Raises:
        FileNotFoundError: If the dataset directory doesn't exist
        ValueError: If the dataset has no valid images

    """
    dataset_path = Path(data_root) / dataset_name
    logger.info(f"Loading {dataset_name} from {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Load using ImageFolder to ensure correct structure
    imagefolder = ImageFolder(root=str(dataset_path))

    if len(imagefolder) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    logger.info(f"Loaded {len(imagefolder)} images from {dataset_name}")

    # Create lists to store image paths and labels
    image_paths = []
    labels = []
    class_names = imagefolder.classes

    for idx in range(len(imagefolder)):
        img_path, label = imagefolder.imgs[idx]
        image_paths.append(img_path)
        labels.append(label)

    # Create dataset dictionary
    data_dict = {
        "image": image_paths,
        "label": labels,
        "class_name": [class_names[label] for label in labels],
    }

    # Create HuggingFace dataset
    hf_dataset = Dataset.from_dict(data_dict)

    # Cast image column to Image feature
    hf_dataset = hf_dataset.cast_column("image", Image())

    logger.info(f"Created HuggingFace dataset with {len(hf_dataset)} samples")

    return hf_dataset


def load_companion_dataset_with_ground_truth(
    dataroot: str,
    dataset_name: str,
    n_samples_per_subset: int = 200,
) -> Dataset:
    """
    Load companion dataset with ground truth labels from AmbiGAN outputs.

    Ground truth labels indicate which binary class pairs each ambiguous image
    belongs to. For example, an image from the 4v9 subset has ground_truth=[4, 9].

    For MNIST and Fashion-MNIST: Samples n_samples_per_subset images from each
    of 45 binary subsets (default 200 per subset = 9,000 total images).

    For Chest X-ray: Uses ALL images from the single binary directory (no sampling)
    with ground_truth=[1, 0].

    Args:
        dataroot: Root directory (typically FILESDIR or points to data directory)
        dataset_name: Base dataset name ('mnist', 'fashion_mnist', or 'chest_xray')
        n_samples_per_subset: Number of samples to take from each binary subset
                            (only applies to MNIST and Fashion-MNIST)

    Returns:
        HuggingFace Dataset with image and ground_truth columns

    Raises:
        FileNotFoundError: If AmbiGAN directory not found
        ValueError: If no companion dataset images found

    """
    # Load image paths and ground truth labels
    image_paths, ground_truths = load_companion_dataset_from_directories(dataroot, dataset_name, n_samples_per_subset)

    logger.info(f"Loading {len(image_paths)} images with ground truth labels")

    # Create dataset dictionary
    data_dict = {
        "image": image_paths,
        "ground_truth": [json.dumps(gt) for gt in ground_truths],  # Store as JSON strings
    }

    # Create HuggingFace dataset
    hf_dataset = Dataset.from_dict(data_dict)

    # Cast image column to Image feature
    hf_dataset = hf_dataset.cast_column("image", Image())

    logger.info(f"Created HuggingFace dataset with {len(hf_dataset)} samples " f"and ground truth labels")

    return hf_dataset


def load_all_companion_datasets(data_root: str) -> DatasetDict:
    """
    Load all companion datasets (MNIST, Fashion-MNIST, and Chest X-ray) with ground truth.

    MNIST and Fashion-MNIST: Sampled from 45 binary subsets (200 images per subset),
    totaling 9k images each.

    Chest X-ray: Uses ALL images from the single binary directory (no sampling).

    Ground truth indicates which binary class pairs each ambiguous image belongs to.

    Args:
        data_root: Root directory (typically FILESDIR)

    Returns:
        DatasetDict with 'mnist', 'fashion-mnist', and 'chest-xray' keys, each containing
        'image' and 'ground_truth' columns

    Raises:
        FileNotFoundError: If required AmbiGAN directory doesn't exist

    """
    datasets = {}

    try:
        # Load MNIST companion dataset
        logger.info("Loading MNIST companion dataset with ground truth...")
        datasets["mnist"] = load_companion_dataset_with_ground_truth(
            data_root,
            "mnist",
            n_samples_per_subset=200,
        )
        logger.info(f"✓ Loaded MNIST: {len(datasets['mnist'])} samples")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load MNIST companion dataset: {e}")

    try:
        # Load Fashion-MNIST companion dataset
        logger.info("Loading Fashion-MNIST companion dataset with ground truth...")
        datasets["fashion-mnist"] = load_companion_dataset_with_ground_truth(
            data_root,
            "fashion_mnist",
            n_samples_per_subset=200,
        )
        logger.info(f"✓ Loaded Fashion-MNIST: {len(datasets['fashion-mnist'])} samples")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load Fashion-MNIST companion dataset: {e}")

    try:
        # Load Chest X-ray companion dataset
        logger.info("Loading Chest X-ray companion dataset with ground truth...")
        datasets["chest-xray"] = load_companion_dataset_with_ground_truth(
            data_root,
            "chest_xray",
            n_samples_per_subset=200,
        )
        logger.info(f"✓ Loaded Chest X-ray: {len(datasets['chest-xray'])} samples")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load Chest X-ray companion dataset: {e}")

    if not datasets:
        raise FileNotFoundError(
            f"No companion datasets found. "
            f"Please ensure AmbiGAN directory exists with companion dataset outputs at "
            f"{os.path.join(os.path.dirname(data_root), 'AmbiGAN')}"
        )

    return DatasetDict(datasets)


def upload_companion_dataset_to_hub(
    dataset: Dataset,
    dataset_name: str,
    repo_name: str,
    private: bool = True,
    commit_message: str = "Upload companion dataset",
) -> str:
    """
    Upload a companion dataset with ground truth to HuggingFace Hub.

    Args:
        dataset: HuggingFace Dataset to upload with 'image' and 'ground_truth' columns
        dataset_name: Name of the dataset split (e.g., 'mnist', 'fashion-mnist')
        repo_name: Repository in HuggingFace Hub format (e.g., 'username/companion-datasets')
        private: Whether the repository should be private
        commit_message: Git commit message for the upload

    Returns:
        URL of the uploaded dataset

    Raises:
        ValueError: If authentication is not set up
        Exception: If upload fails

    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        # Log dataset info
        logger.info(f"Uploading {dataset_name} to {repo_name}...")
        logger.info(f"  - Samples: {len(dataset)}")
        logger.info(f"  - Columns: {dataset.column_names}")

        # Upload to Hub
        dataset.push_to_hub(
            repo_id=repo_name,
            config_name=dataset_name,
            split="train",
            private=private,
            commit_message=commit_message,
        )

        dataset_url = f"https://huggingface.co/datasets/{repo_name}"
        logger.info(f"✓ Successfully uploaded {dataset_name} to {dataset_url}")

        return dataset_url

    except HfHubHTTPError as e:
        if "401" in str(e):
            raise ValueError("Authentication failed.") from e
        raise
    except Exception as e:
        logger.error(f"Upload failed for {dataset_name}: {e}")
        raise


def save_dataset_metadata_locally(
    dataset: Dataset,
    dataset_name: str,
    output_dir: str | None = None,
) -> Path:
    """
    Save dataset metadata (ground truth and image indices) to a JSON file.

    Useful for reproducibility and offline analysis. Saves:
    - Number of samples
    - Ground truth labels for each image
    - Dataset statistics

    Args:
        dataset: HuggingFace Dataset with ground_truth column
        dataset_name: Name of the dataset (e.g., 'mnist', 'fashion-mnist')
        output_dir: Directory to save metadata. If None, uses current directory

    Returns:
        Path to the saved metadata file

    """
    if output_dir is None:
        output_dir = "."

    output_path = Path(output_dir) / f"{dataset_name}_metadata.json"

    # Extract ground truth labels
    ground_truths = []
    if "ground_truth" in dataset.column_names:
        ground_truths = [json.loads(gt) for gt in dataset["ground_truth"]]

    # Compute statistics
    unique_classes = set()
    class_pair_counts = {}
    for gt in ground_truths:
        unique_classes.update(gt)
        class_pair_str = ",".join(map(str, sorted(gt)))
        class_pair_counts[class_pair_str] = class_pair_counts.get(class_pair_str, 0) + 1

    # Create metadata
    metadata = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "columns": dataset.column_names,
        "unique_classes": sorted(list(unique_classes)),
        "num_unique_classes": len(unique_classes),
        "class_pair_distribution": class_pair_counts,
        "ground_truth_labels": ground_truths,
    }

    # Save metadata
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved metadata to {output_path}")

    return output_path


def upload_all_companion_datasets(
    repo_name: str,
    data_root: str | None = None,
    private: bool = True,
    save_metadata: bool = True,
    metadata_dir: str | None = None,
) -> dict:
    """
    Upload all companion datasets to HuggingFace Hub with ground truth labels.

    This function loads companion datasets from AmbiGAN outputs and uploads them to
    the specified HuggingFace repository.

    MNIST and Fashion-MNIST:
    - 45 binary subsets × 200 images per subset = 9,000 images
    - Ground truth labels indicate class pairs (e.g., [4, 9] for mnist-4v9 subset)

    Chest X-ray:
    - Single binary directory with all images (no sampling)
    - Ground truth label: [1, 0] for all images

    Args:
        repo_name: HuggingFace repository name (format: 'username/repo-name')
        data_root: Root directory (typically FILESDIR). If None, uses FILESDIR env var
        private: Whether datasets should be private
        save_metadata: Whether to save ground truth metadata to JSON files
        metadata_dir: Directory to save metadata files. If None, uses current directory

    Returns:
        Dictionary with upload results for each dataset

    Example:
        >>> from src.utils.huggingface import upload_all_companion_datasets
        >>> results = upload_all_companion_datasets(
        ...     repo_name="inesgomes/companion-datasets",
        ...     private=True,
        ...     save_metadata=True
        ... )
        >>> print(results['mnist']['url'])
        https://huggingface.co/datasets/inesgomes/companion-datasets

    """
    # Use FILESDIR environment variable if data_root not provided
    if data_root is None:
        filesdir = os.environ.get("FILESDIR", ".")
        data_root = os.path.join(filesdir, "data") if filesdir != "." else "."

    logger.info(f"Starting companion dataset upload to {repo_name}")
    logger.info(f"Using data root: {data_root}")
    logger.info(f"Private: {private}, Save metadata: {save_metadata}")

    # Load all companion datasets with ground truth
    datasets_dict = load_all_companion_datasets(data_root)

    # Upload each dataset
    results = {}
    for dataset_name, dataset in datasets_dict.items():
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset_name.upper()}")
            logger.info(f"{'='*60}")

            # Upload dataset
            url = upload_companion_dataset_to_hub(
                dataset=dataset,
                dataset_name=dataset_name,
                repo_name=repo_name,
                private=private,
                commit_message=f"Add {dataset_name} companion dataset (9k images with ground truth labels)",
            )

            result = {
                "status": "success",
                "url": url,
                "samples": len(dataset),
            }

            # Save metadata locally if requested
            if save_metadata:
                metadata_path = save_dataset_metadata_locally(
                    dataset,
                    dataset_name,
                    metadata_dir,
                )
                result["metadata"] = str(metadata_path)

            results[dataset_name] = result
            logger.info(f"✓ {dataset_name}: {url}")

        except Exception as e:
            results[dataset_name] = {"status": "failed", "error": str(e)}
            logger.error(f"✗ {dataset_name}: {e}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("UPLOAD SUMMARY")
    logger.info(f"{'='*60}")
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")

    for dataset_name, result in results.items():
        status = result["status"].upper()
        if status == "SUCCESS":
            logger.info(f"  ✓ {dataset_name}: {result['samples']} samples")
        else:
            logger.info(f"  ✗ {dataset_name}: {result['error']}")

    return results


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Upload companion datasets to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload MNIST
  python -m src.utils.huggingface --repo username/companion-datasets --dataset mnist

  # Upload Fashion-MNIST
  python -m src.utils.huggingface --repo username/companion-datasets --dataset fashion-mnist

  # Upload Chest X-ray
  python -m src.utils.huggingface --repo username/companion-datasets --dataset chest-xray

  # Upload with custom data root
  python -m src.utils.huggingface --repo username/companion-datasets --dataset mnist --data-root /path/to/data

  # Test upload with only 5 images
  python -m src.utils.huggingface --repo username/companion-datasets --dataset mnist --test

  # Upload as private
  python -m src.utils.huggingface --repo username/companion-datasets --dataset mnist --private

Note: Metadata is automatically saved to FILESDIR/hf-datasets. Default upload is PUBLIC.
        """,
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repository name (format: username/repo-name)",
    )

    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion-mnist", "chest-xray"],
        required=True,
        help="Dataset to upload: 'mnist', 'fashion-mnist', or 'chest-xray'",
    )

    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory for datasets (default: uses FILESDIR environment variable)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as private dataset (default: public)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: upload only 5 images to verify upload is working",
    )

    args = parser.parse_args()

    # Determine data root and metadata directory
    if args.data_root is None:
        filesdir = os.environ.get("FILESDIR", ".")
        args.data_root = os.path.join(filesdir, "data") if filesdir != "." else "."
        metadata_dir = os.path.join(filesdir, "hf-datasets") if filesdir != "." else "hf-datasets"
    else:
        parent_dir = os.path.dirname(args.data_root)
        metadata_dir = os.path.join(parent_dir, "hf-datasets")

    # Create metadata directory if it doesn't exist
    os.makedirs(metadata_dir, exist_ok=True)

    # Print header
    test_mode_str = " [TEST MODE - 5 IMAGES ONLY]" if args.test else ""
    print("\n" + "=" * 70)
    print(f"COMPANION DATASET UPLOAD TO HUGGINGFACE{test_mode_str}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Repository: {args.repo}")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Visibility: {'PRIVATE' if args.private else 'PUBLIC'}")
    print("  Save Metadata: YES")
    print(f"  Metadata Directory: {metadata_dir}")
    if args.test:
        print("  Images: 5 (test mode)")
    if args.data_root:
        print(f"  Data Root: {args.data_root}")
    print("\nDataset Info:")
    if args.dataset == "mnist":
        print("  - MNIST: 9000 images (45 binary subsets × 200 images)")
    elif args.dataset == "fashion-mnist":
        print("  - Fashion-MNIST: 9000 images (45 binary subsets × 200 images)")
    else:
        print("  - Chest X-ray: 9000 images (45 binary subsets × 200 images)")
    print("\nGround Truth Format:")
    print("  Each image has a 'ground_truth' field containing the class indices")
    print("  for its binary subset (e.g., [4, 9] for mnist-4v9 subset)")
    print("\n" + "=" * 70 + "\n")

    # Load the requested dataset
    try:
        all_datasets = load_all_companion_datasets(args.data_root)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        print(f"\n✗ ERROR: Failed to load datasets: {e}\n")
        exit(1)

    # Get the requested dataset
    if args.dataset not in all_datasets:
        print(f"\n✗ ERROR: Dataset '{args.dataset}' not found. Available: {list(all_datasets.keys())}\n")
        exit(1)

    dataset = all_datasets[args.dataset]

    # Test mode: use only 5 images
    if args.test:
        logger.info(f"\n{'='*60}")
        logger.info("TEST MODE: Limiting to first 5 images")
        logger.info(f"Original dataset size: {len(dataset)}")
        dataset = dataset.select(range(min(5, len(dataset))))
        logger.info(f"Test dataset size: {len(dataset)}")
        logger.info(f"{'='*60}")

    # Upload the dataset
    result = {}
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {args.dataset.upper()}{' [TEST MODE]' if args.test else ''}")
        logger.info(f"{'='*60}")

        # Upload dataset
        url = upload_companion_dataset_to_hub(
            dataset=dataset,
            dataset_name=args.dataset,
            repo_name=args.repo,
            private=args.private,
            commit_message=f"Add {args.dataset} companion dataset (9k images with ground truth labels)",
        )

        result = {
            "status": "success",
            "url": url,
            "samples": len(dataset),
        }

        # Save metadata locally
        metadata_path = save_dataset_metadata_locally(
            dataset,
            args.dataset,
            metadata_dir,
        )
        result["metadata"] = str(metadata_path)

        logger.info(f"✓ {args.dataset}: {url}")

    except Exception as e:
        result = {"status": "failed", "error": str(e)}
        logger.error(f"✗ {args.dataset}: {e}")

    # Print final summary
    print("\n" + "=" * 70)
    print(f"UPLOAD {'COMPLETE' if result['status'] == 'success' else 'FAILED'}{'[TEST MODE]' if args.test else ''}")
    print("=" * 70)

    status = result["status"].upper()
    if status == "SUCCESS":
        print(f"\n✓ {args.dataset:20} → {result['url']}")
        print(f"  Samples: {result['samples']}")
        if args.test:
            print("  Note: Test mode uploaded only 5 images. Full dataset has ~9000 images.")
        if "metadata" in result:
            print(f"  Metadata: {result['metadata']}")
    else:
        print(f"\n✗ {args.dataset:20} → ERROR: {result['error']}")

    print("\n" + "=" * 70 + "\n")

    # Exit with appropriate code
    exit(0 if result["status"] == "success" else 1)
