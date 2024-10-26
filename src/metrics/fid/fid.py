"""Module for calculating FID."""

import logging

import numpy as np
import torch
from torch import nn
from torcheval.metrics import FrechetInceptionDistance

from src.enums import DeviceType
from src.metrics.metric import Metric

logger = logging.getLogger(__name__)


class FID(Metric):
    """Class for calculating the Frechet Inception Distance (FID) to measure GAN performance."""

    def __init__(
        self,
        fid_stats_file: str,
        dims: int,
        n_images: int,
        device: DeviceType = DeviceType.cpu,
        feature_map_fn: nn.Module | None = None,
    ) -> None:
        """Initialize the FID metric with necessary parameters for calculation."""
        super().__init__()
        try:
            self.data = np.load(fid_stats_file)
        except FileNotFoundError as e:
            logger.error(f"Failed to load FID stats from {fid_stats_file} with {e}")
            raise FileNotFoundError(e) from e
        self.dims: int = dims
        self.n_images: int = n_images
        self.pred_arr: np.ndarray = np.empty((n_images, dims))
        self.cur_idx: int = 0
        self.device: DeviceType = device
        self.fid = FrechetInceptionDistance(model=feature_map_fn, feature_dim=self.dims, device=self.device)
        self.fid.real_sum = torch.tensor(self.data["real_sum"]).to(self.device).float()
        self.fid.real_cov_sum = torch.tensor(self.data["real_cov_sum"]).to(self.device).float()
        self.fid.num_real_images = torch.tensor(self.data["num_real_images"]).to(self.device).int()

    def update(self, images: torch.Tensor, _: tuple[int, int]) -> None:
        """Update the FID metric with a new batch of generated images."""
        self.fid.update(images=images, is_real=False)

    def finalize(self) -> float:
        """Finalize the FID calculation and return the computed FID value."""
        return self.fid.compute().item()

    def reset(self) -> None:
        """Reset the FID metric to its initial state."""
        self.fid.num_fake_images = torch.tensor(0).to(self.device).int()
        self.fid.fake_sum = torch.zeros(self.dims).to(self.device)
        self.fid.fake_cov_sum = torch.zeros((self.dims, self.dims)).to(self.device)
