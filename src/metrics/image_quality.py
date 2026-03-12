"""Image quality metric helpers."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymdma.image.measures.synthesis_val import (
    GIQA,
    Coverage,
    Density,
    ImprovedPrecision,
    ImprovedRecall,
    MultiScaleIntrinsicDistance,
)
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.load import load_dataset
from src.enums import DatasetNames, DeviceType
from src.metrics.fid.fid import FID
from src.models import LoadDatasetParams

logger = logging.getLogger(__name__)


def compute_fid_metric(
    model: torch.nn.Module | None,
    dataloader: DataLoader,
    device: DeviceType | str,
    fid_stats_path: str,
) -> float:
    """
    Compute FID metric from images.

    Args:
        model: Trained classifier model (not used, kept for backward compatibility)
        dataloader: DataLoader with test data
        device: Device to use for computation
        fid_stats_path: Path to FID statistics file

    Returns:
        FID score

    """
    if model is not None:
        model.eval()

    device_obj = DeviceType(device) if isinstance(device, str) else device
    device_str = device_obj.value if isinstance(device_obj, DeviceType) else str(device_obj)
    fid = FID(fid_stats_file=fid_stats_path, dims=2048, n_images=len(dataloader.dataset), device=device_obj)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device_str)
            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)
            fid.update(images, (0, 0))

    return fid.finalize()


def extract_features(
    images: torch.Tensor, device: DeviceType | str, extractor: object | None = None
) -> np.ndarray:  # pylint: disable=redefined-outer-name
    """
    Extract features from images using DINO ViT S/8.

    Args:
        images: Tensor of images with shape (batch_size, channels, height, width)
        device: Device to use for computation
        extractor: Optional cached extractor model. If None, creates a new one.

    Returns:
        Feature array with shape (batch_size, feature_dim)

    """
    try:
        device_str = device.value if isinstance(device, DeviceType) else str(device)

        if extractor is None:
            extractor = ExtractorFactory.model_from_name(name="dino_vits8")
            extractor = extractor.to(device_str)  # type: ignore
            extractor.eval()  # type: ignore
        else:
            extractor = extractor.to(device_str)  # type: ignore

        features_list = []
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            if batch_images.shape[1] != 3:
                batch_images = batch_images.repeat(1, 3, 1, 1)
            batch_images = (batch_images + 1.0) / 2.0
            batch_images = batch_images.clamp(0, 1)

            with torch.no_grad():
                features = extractor(batch_images.to(device_str)).cpu().numpy()  # type: ignore
            features_list.append(features)

        return np.concatenate(features_list, axis=0)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Failed to extract features: %s", e)
        raise


def calculate_pymdma_metrics(real_features: np.ndarray, synt_features: np.ndarray) -> pd.DataFrame:
    """
    Calculate synthetic validation metrics from pymdma library.

    Note: Metric model instances are created once and reused (especially GIQA which is expensive).
    """
    logger.info("Initializing metric models...")
    ip = ImprovedPrecision(k=5)
    ir = ImprovedRecall(k=5)
    giqa = GIQA()
    density = Density()
    coverage = Coverage()
    msid = MultiScaleIntrinsicDistance()

    logger.info("Calculating Improved Precision and Improved Recall")
    ip_result = ip.compute(real_features=real_features, fake_features=synt_features)
    ir_result = ir.compute(real_features=real_features, fake_features=synt_features)
    precision_dataset, _ = ip_result.value
    recall_dataset, _ = ir_result.value

    logger.info("Calculating GIQA QS")
    giqa_qs_result = giqa.compute(real_features=real_features, fake_features=synt_features)
    giqa_qs_dataset, _ = giqa_qs_result.value

    logger.info("Calculating GIQA DS")
    giqa_ds_result = giqa.compute(real_features=synt_features, fake_features=real_features)
    giqa_ds_dataset, _ = giqa_ds_result.value

    logger.info("Calculating Density")
    density_result = density.compute(real_features=real_features, fake_features=synt_features)
    density_dataset, _ = density_result.value

    logger.info("Calculating Coverage")
    coverage_result = coverage.compute(real_features=real_features, fake_features=synt_features)
    coverage_dataset, _ = coverage_result.value

    logger.info("Calculating Multi-Scale Intrinsic Distance")
    msid_result = msid.compute(real_features=real_features, fake_features=synt_features)
    msid_dataset, _ = msid_result.value

    df = pd.DataFrame().assign(
        improved_precision=[precision_dataset],
        improved_recall=[recall_dataset],
        giqa_qs=[giqa_qs_dataset],
        giqa_ds=[giqa_ds_dataset],
        density=[density_dataset],
        coverage=[coverage_dataset],
        msid=[msid_dataset],
    )
    logger.info("Finished PyMDMA metrics calculation")

    return df


def compute_pymdma_metrics_from_images(
    eval_images: torch.Tensor,
    real_features: np.ndarray,
    device: DeviceType | str,
    extractor: object | None = None,
    sample_size: int | None = None,
) -> dict[str, float]:
    """
    Compute pymdma metrics comparing synthetic evaluation dataset against real training features.

    Args:
        eval_images: Synthetic evaluation dataset images
        real_features: Feature array from real training dataset (reference)
        device: Device to use for computation
        extractor: Optional cached extractor model to avoid recreating it
        sample_size: Optional limit on evaluation images for faster metrics (None = use all)

    Returns:
        Dictionary with pymdma metrics

    """
    logger.info("Extracting features from evaluation dataset...")

    if sample_size is not None and len(eval_images) > sample_size:
        indices = np.random.choice(len(eval_images), size=sample_size, replace=False)
        eval_images = eval_images[indices]
        logger.info("Sampled %d evaluation images (full set: %d)", sample_size, len(eval_images))

    synt_features = extract_features(eval_images, device, extractor)

    logger.info("Computing pymdma metrics (synthetic vs real training distribution)...")
    pymdma_df = calculate_pymdma_metrics(real_features, synt_features)

    metrics_dict = {}
    for col in pymdma_df.columns:
        metrics_dict[col] = pymdma_df[col].values[0]

    return metrics_dict


def generate_fid_stats(  # pylint: disable=too-many-statements
    dataroot: str,
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 6,
    device: DeviceType | str = "cpu",
    use_test_set: bool = True,
    n_samples: int = 10000,
) -> str:
    """
    Generate and save FID statistics for a dataset.

    IMPORTANT: This uses InceptionV3-based feature extraction (inception_fid) which is compatible
    with FrechetInceptionDistance. Do NOT use DINO or other extractors for FID statistics.

    Args:
        dataroot: Root directory where datasets are stored
        dataset_name: Name of the dataset (e.g., 'mnist', 'companion-mnist')
        batch_size: Batch size for processing
        num_workers: Number of worker processes for data loading
        device: Device to use ('cpu' or 'cuda:X')
        use_test_set: If True, use test set; if False, use training set
        n_samples: Maximum number of samples to use for statistics (None for all)

    Returns:
        Path to the generated FID statistics file

    """
    stats_dir = Path(dataroot) / "fid-stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    stats_file = stats_dir / f"stats.{dataset_name}.npz"

    # If stats already exist, return the path
    if stats_file.exists():
        logger.info(f"FID statistics already exist at {stats_file}")
        return str(stats_file)

    logger.info(f"Generating FID statistics for {dataset_name}...")

    split = "test" if use_test_set else "train"
    # Load dataset
    try:
        dataset, _, _ = load_dataset(
            LoadDatasetParams(
                dataroot=dataroot,
                dataset_name=DatasetNames(dataset_name),
                pos_class=None,
                neg_class=None,
                split=split,
                pytesting=False,
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

    logger.info(f"Dataset size: {len(dataset)}")

    # Sample down if necessary
    if n_samples is not None and len(dataset) > n_samples:
        logger.info(f"Sampling {n_samples} images from {len(dataset)} total")
        indices = np.random.choice(len(dataset), size=n_samples, replace=False)
        # Convert numpy indices to Python ints (HuggingFace datasets don't accept numpy.int64)
        indices = indices.tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    else:
        if n_samples is not None:
            logger.info(f"Using all {len(dataset)} images (less than requested {n_samples})")

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize FID metric (use repo's FID class to compute reference stats)
    device_obj = DeviceType(device) if isinstance(device, str) else device
    device_str = device_obj.value if isinstance(device_obj, DeviceType) else str(device_obj)
    fid = FID(fid_stats_file=None, dims=2048, n_images=len(dataset), device=device_obj)

    # Initialize feature extractor for FID (use InceptionV3-based FID extractor, not DINO)
    # inception_fid produces 2048-dimensional features compatible with FID
    logger.info("Loading InceptionV3 extractor for FID computation...")

    inception_extractor = ExtractorFactory.model_from_name(name="inception_fid")
    inception_extractor = inception_extractor.to(device_str)
    inception_extractor.eval()

    # Also initialize DINO extractor for pymdma metrics reference
    logger.info("Loading DINO ViT extractor for pymdma metrics...")
    dino_extractor = ExtractorFactory.model_from_name(name="dino_vits8")
    dino_extractor = dino_extractor.to(device_str)
    dino_extractor.eval()
    all_features = []

    # Calculate FID statistics
    logger.info("Computing FID statistics...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing FID stats"):
            images = batch[0]

            if images.ndim < 2:
                raise ValueError(
                    f"Images must have at least two dimensions (batch size and channel), got {images.ndim}D tensor."
                )

            # Convert to RGB by repeating across the channel dimension if needed
            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)

            images = images.to(device_str)

            # NOTE: Directly update fid.fid with is_real=True to accumulate reference statistics
            # We bypass FID.update() because it always uses is_real=False
            # Convert from [-1, 1] to [0, 1] for InceptionV3
            images_normalized = (images + 1.0) / 2.0
            fid.fid.update(images_normalized, is_real=True)

            # Extract DINO features for pymdma metrics reference (on normalized images)
            features = dino_extractor(images_normalized).detach().cpu().numpy()
            all_features.append(features)

    # Extract statistics from FID instance
    # Compute mean and covariance from accumulated statistics
    m = fid.fid.real_sum / fid.fid.num_real_images
    s = fid.fid.real_cov_sum - fid.fid.num_real_images * torch.outer(m, m)

    # Save statistics
    logger.info(f"Saving FID statistics to {stats_file}...")
    with open(f"{stats_file}", "wb") as f:
        np.savez(
            f,
            mu=m.cpu().numpy(),
            sigma=s.cpu().numpy(),
            real_sum=fid.fid.real_sum.cpu().numpy(),
            real_cov_sum=fid.fid.real_cov_sum.cpu().numpy(),
            num_real_images=fid.fid.num_real_images.cpu().numpy(),
            all_features=np.concatenate(all_features, axis=0),
        )

    logger.info(f"FID statistics saved to {stats_file}")
    return str(stats_file)


def find_fid_stats(dataroot: str, dataset_name: str) -> str | None:
    """
    Find FID statistics file for a dataset.

    Args:
        dataroot: Root directory where datasets are stored
        dataset_name: Name of the dataset

    Returns:
        Path to FID stats file if found, None otherwise

    """
    fid_stats_dir = Path(dataroot) / "fid-stats"
    if not fid_stats_dir.exists():
        return None

    # Look for stats file matching the dataset name
    stats_file = fid_stats_dir / f"stats.{dataset_name}.npz"
    if stats_file.exists():
        return str(stats_file)

    return None
