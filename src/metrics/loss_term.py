# pylint: skip-file

import torch
from torch import Tensor

from src.metrics.metric import Metric
from src.classifier.classifier_cache import ClassifierCache


class LossSecondTerm(Metric):
    """Metric to calculate the confusion distance term of loss for GAN evaluation."""

    def __init__(self, C: ClassifierCache) -> None:
        """Initialize LossSecondTerm metric."""
        super().__init__()
        self.C = C
        self.count = 0
        self.acc = 0
        self.result = float("inf")

    def update(self, images: Tensor, batch: tuple[int, int]) -> None:
        """Update the metric values using the given batch of generated images."""
        start_idx, batch_size = batch

        with torch.no_grad():
            c_output = self.C.get(images, start_idx, batch_size)

        term_2 = (torch.tensor(0.5) - c_output).abs().sum().item()

        self.acc += term_2
        self.count += images.size(0)

    def finalize(self) -> float:
        """Finalize the metric computation and return the result."""
        self.result = self.acc / self.count

        return self.result

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        self.count = 0
        self.acc = 0
        self.result = float("inf")
