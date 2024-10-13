"""Module for calculating FID."""

import numpy as np
import torch
from typing import Callable, Tuple, Union

from src.metrics.metric import Metric
from src.metrics.fid.fid_score import calculate_frechet_distance


class FID(Metric):
    """Class for calculating the Frechet Inception Distance (FID) to measure GAN performance."""

    def __init__(
        self,
        feature_map_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
        dims: int,
        n_images: int,
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        device: str = "cpu",
        eps: float = 1e-6,
    ) -> None:
        """Initialize the FID metric with necessary parameters for calculation."""
        super().__init__()
        self.feature_map_fn = feature_map_fn
        self.dims: int = dims
        self.eps: float = eps
        self.n_images: int = n_images
        self.pred_arr: np.ndarray = np.empty((n_images, dims))
        self.cur_idx: int = 0
        self.mu_real: np.ndarray = mu_real
        self.sigma_real: np.ndarray = sigma_real
        self.device: str = device

    def update(self, images: torch.Tensor, batch: Tuple[int, int]) -> None:
        """Update the FID metric with a new batch of generated images."""
        start_idx, batch_size = batch

        with torch.no_grad():
            pred = self.feature_map_fn(images, start_idx, batch_size)

        pred_np = pred.cpu().numpy()
        self.pred_arr[self.cur_idx : self.cur_idx + pred_np.shape[0]] = pred_np
        self.cur_idx += pred_np.shape[0]

    def finalize(self) -> float:
        """Finalize the FID calculation and return the computed FID value."""
        act = self.pred_arr
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        self.result = calculate_frechet_distance(mu, sigma, self.mu_real, self.sigma_real, eps=self.eps)
        return self.result

    def reset(self) -> None:
        """Reset the FID metric to its initial state."""
        self.pred_arr = np.empty((self.n_images, self.dims))
        self.cur_idx = 0
        self.result = float("inf")
