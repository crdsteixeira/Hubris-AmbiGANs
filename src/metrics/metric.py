"""Module for base metric class."""

from typing import Any


class Metric:
    """Base class for computing and managing metrics."""

    def __init__(self) -> None:
        """Initialize the Metric with an initial result value of infinity."""
        self.result: float = float("inf")

    def update(self, images: Any, batch: tuple[int, int]) -> None:
        """Update the metric calculation with a new batch of images."""
        raise NotImplementedError

    def finalize(self) -> float:
        """Finalize and return the computed metric value."""
        raise NotImplementedError

    def get_result(self) -> float:
        """Get the current result value of the metric."""
        return self.result

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        raise NotImplementedError
