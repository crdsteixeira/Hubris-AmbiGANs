"""Module to test classifier cache."""

from unittest.mock import MagicMock

import torch
from torch import nn

from src.classifier.classifier_cache import ClassifierCache


# Mock classifier for testing purposes
class MockClassifier(nn.Module):
    """Mock Classifier class."""

    def forward(self, _: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Mock forward function."""
        if output_feature_maps:
            return [
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 10),
            ]  # Mock feature maps and final output
        return [torch.randn(1, 10)]  # Only final output


def test_cache_basic_usage() -> None:
    """Test if cached output is reused on repeated calls with same batch_idx and batch_size."""
    mock_classifier = MockClassifier()
    mock_classifier.forward = MagicMock(return_value=[torch.randn(1, 3, 64, 64), torch.randn(1, 10)])

    cache = ClassifierCache(mock_classifier)

    x = torch.randn(1, 3, 64, 64)
    output_1 = cache.get(x, batch_idx=1, batch_size=1, output_feature_maps=True)
    output_2 = cache.get(x, batch_idx=1, batch_size=1, output_feature_maps=True)

    assert mock_classifier.forward.call_count == 1, "Classifier should only be called once for the same batch"
    assert output_1 == output_2, "Cached result should be the same when the same batch is used"


def test_cache_invalidation() -> None:
    """Test if cache refreshes when a new batch_idx or batch_size is passed."""
    mock_classifier = MockClassifier()
    mock_classifier.forward = MagicMock(return_value=[torch.randn(1, 3, 64, 64), torch.randn(1, 10)])

    cache = ClassifierCache(mock_classifier)

    x = torch.randn(1, 3, 64, 64)
    # First call, cache should store result
    cache.get(x, batch_idx=1, batch_size=1, output_feature_maps=True)
    # Second call with different batch index, should trigger recalculation
    cache.get(x, batch_idx=2, batch_size=1, output_feature_maps=True)

    assert mock_classifier.forward.call_count == 2, "Classifier should be called again for different batch"


def test_feature_maps_returned() -> None:
    """Test if feature maps and final output are returned correctly when requested."""
    mock_classifier = MockClassifier()
    mock_classifier.forward = MagicMock(return_value=[torch.randn(1, 3, 64, 64), torch.randn(1, 10)])

    cache = ClassifierCache(mock_classifier)

    x = torch.randn(1, 3, 64, 64)
    output, feature_maps = cache.get(x, batch_idx=1, batch_size=1, output_feature_maps=True)

    assert output is not None, "Output should not be None"
    assert feature_maps is not None, "Feature maps should not be None"
    assert len(feature_maps.size()) == 4, "Feature maps should be a 4D tensor"


def test_no_feature_maps_returned() -> None:
    """Test if only the final output is returned when feature maps are not requested."""
    mock_classifier = MockClassifier()
    mock_classifier.forward = MagicMock(return_value=[torch.randn(1, 10)])

    cache = ClassifierCache(mock_classifier)

    x = torch.randn(1, 3, 64, 64)
    output = cache.get(x, batch_idx=1, batch_size=1, output_feature_maps=False)

    assert output is not None, "Output should not be None when feature maps are not requested"
    assert mock_classifier.forward.call_count == 1, "Classifier should be called once"
