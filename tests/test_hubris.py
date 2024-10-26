"""Module to test Hubris metric."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from src.metrics.hubris import Hubris
from src.classifier.classifier_cache import ClassifierCache


@pytest.fixture
def mock_classifier_cache() -> ClassifierCache:
    """Fixture to provide a mock classifier cache."""
    mock_cache = MagicMock(spec=ClassifierCache)
    
    # Create a mock for the C attribute and models
    mock_C = MagicMock()
    mock_C.models = [MagicMock(), MagicMock()]  # Assume there are two models
    mock_cache.configure_mock(C=mock_C)
    
    return mock_cache


@pytest.fixture
def hubris_metric(mock_classifier_cache: ClassifierCache) -> Hubris:
    """Fixture to provide an instance of Hubris."""
    dataset_size = 100
    return Hubris(mock_classifier_cache, dataset_size)


def test_initialization(hubris_metric: Hubris, mock_classifier_cache: ClassifierCache) -> None:
    """Test Hubris initialization."""
    assert hubris_metric.C == mock_classifier_cache
    assert hubris_metric.dataset_size == 100
    assert hubris_metric.preds.shape == (100,)
    assert hubris_metric.output_clfs == len(mock_classifier_cache.C.models)
    if hubris_metric.output_clfs > 0:
        assert hubris_metric.clf_preds.shape == (hubris_metric.output_clfs, 100)


def test_update(hubris_metric: Hubris) -> None:
    """Test Hubris update method."""
    # Mock data
    images = torch.randn((10, 3, 64, 64))  # Example batch of 10 images
    batch = (0, 10)  # Starting index and batch size

    # Mock behavior for C.get()
    mock_y_hat = torch.randn(10).float()
    mock_y_preds = [torch.randn(10).float(), torch.randn(10).float()]
    hubris_metric.C.get = MagicMock(return_value=(mock_y_hat, [mock_y_preds]))

    # Call the update function
    hubris_metric.update(images, batch)

    # Check if values were updated correctly
    assert torch.equal(hubris_metric.preds[0:10], mock_y_hat)
    for i in range(hubris_metric.output_clfs):
        assert torch.equal(hubris_metric.clf_preds[i, 0:10], mock_y_preds[i])


def test_compute_no_reference(hubris_metric: Hubris) -> None:
    """Test Hubris compute method without reference predictions."""
    preds = torch.rand(10)
    score = hubris_metric.compute(preds)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0  # Hubris score should be in range [0, 1]


def test_compute_with_reference(hubris_metric: Hubris) -> None:
    """Test Hubris compute method with reference predictions."""
    preds = torch.rand(10)
    ref_preds = torch.rand(10)
    score = hubris_metric.compute(preds, ref_preds)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0  # Hubris score should be in range [0, 1]


def test_finalize(hubris_metric: Hubris) -> None:
    """Test Hubris finalize method."""
    hubris_metric.preds = torch.rand(100)
    result = hubris_metric.finalize()

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_get_clfs(hubris_metric: Hubris) -> None:
    """Test Hubris get_clfs method."""
    hubris_metric.clf_preds = torch.rand((hubris_metric.output_clfs, hubris_metric.dataset_size))
    results = hubris_metric.get_clfs()

    assert len(results) == hubris_metric.output_clfs
    for score in results:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_reset(hubris_metric: Hubris) -> None:
    """Test Hubris reset method."""
    hubris_metric.preds = torch.rand(100)
    hubris_metric.result = torch.tensor([0.5])

    hubris_metric.reset()

    assert torch.equal(hubris_metric.preds, torch.zeros(100, dtype=float))
    assert torch.equal(hubris_metric.result, torch.tensor([1.0]))


