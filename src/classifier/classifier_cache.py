"""Classifier Cache Module."""

from torch import Tensor, nn


class ClassifierCache:
    """Cache classifier outputs to avoid redundant computations on the same batch."""

    def __init__(self, C: nn.Module) -> None:
        """
        Initialize the ClassifierCache with a classifier object.

        Args:
            C (nn.Module): The classifier model to be used. This classifier is assumed
            to return a list where the last element is the final output, and the second-to-last
            element is the feature maps (if requested).

        Attributes:
            C (nn.Module): The classifier object.
            last_output (tensor): Stores the output of the classifier from the last batch.
            last_output_feature_maps (tensor or None): Stores the feature maps of the last batch.
            last_batch_idx (int or None): Stores the index of the last processed batch.
            last_batch_size (int or None): Stores the size of the last processed batch.

        """
        super().__init__()
        self.C = C
        self.last_output: Tensor | None = None
        self.last_output_feature_maps: Tensor | None = None
        self.last_batch_idx: int | None = None
        self.last_batch_size: int | None = None

    def get(
        self, x: Tensor, batch_idx: int, batch_size: int, output_feature_maps: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Retrieve the cached classifier output or recompute it if the batch has changed.

        Caching Logic
            - If the batch index or size changes, it recomputes the classifier output and updates the cache.
            - If `output_feature_maps` is True and feature maps exist, it returns both the final output and
            the feature maps.
            - If `output_feature_maps` is False, it returns only the final output.
            - If the classifier does not return feature maps, the cache stores `None` for feature maps.

        Raises
        ------
            IndexError: If the classifier output structure does not match the expected format.

        """
        if batch_idx != self.last_batch_idx or batch_size != self.last_batch_size:
            output = self.C(x, output_feature_maps=True)

            # Check if the classifier returns both feature maps and output
            if isinstance(output, tuple) and len(output) > 1:
                self.last_output_feature_maps = output[-1]
                self.last_output = output[-2]
            else:
                self.last_output = output
                self.last_output_feature_maps = None  # No feature maps returned

            self.last_batch_idx = batch_idx
            self.last_batch_size = batch_size

        if output_feature_maps and self.last_output_feature_maps is not None:
            return self.last_output, self.last_output_feature_maps
        return self.last_output
