"""Image quality metric helpers."""

import logging

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

from src.enums import DeviceType
from src.metrics.fid.fid import FID

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
