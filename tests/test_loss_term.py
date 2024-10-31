"""Module for testing the Confusion Distance metric."""

from unittest.mock import MagicMock

import pytest
import torch

from src.classifier.classifier_cache import ClassifierCache
from src.metrics.loss_term import LossSecondTerm


@pytest.fixture
def mock_classifier_cache() -> ClassifierCache:
    """Fixture to provide a mock classifier cache."""
    mock_cache = MagicMock(spec=ClassifierCache)
    return mock_cache


@pytest.fixture
def loss_second_term(mock_classifier_cache: ClassifierCache) -> LossSecondTerm:
    """Fixture to provide an instance of LossSecondTerm."""
    return LossSecondTerm(mock_classifier_cache)


def test_initialization(loss_second_term: LossSecondTerm, mock_classifier_cache: ClassifierCache) -> None:
    """Test LossSecondTerm initialization."""
    assert loss_second_term.C == mock_classifier_cache
    assert loss_second_term.count == 0
    assert loss_second_term.acc == 0
    assert loss_second_term.result == float("inf")


def test_update(loss_second_term: LossSecondTerm) -> None:
    """Test LossSecondTerm update method."""
    # Mock data
    images = torch.randn((10, 3, 64, 64))  # Example batch of 10 images
    batch = (0, 10)  # Starting index and batch size

    # Mock classifier output
    mock_output = torch.rand(10)
    loss_second_term.C.get = MagicMock(return_value=mock_output)

    # Call the update method
    loss_second_term.update(images, batch)

    # Calculate expected term_2 value
    expected_term_2 = (torch.tensor(0.5) - mock_output).abs().sum().item()

    # Check accumulated value and count
    assert loss_second_term.acc == expected_term_2
    assert loss_second_term.count == images.size(0)


def test_finalize(loss_second_term: LossSecondTerm) -> None:
    """Test LossSecondTerm finalize method."""
    # Mock data
    loss_second_term.acc = 25.0
    loss_second_term.count = 5

    # Call finalize and verify the result
    result = loss_second_term.finalize()
    assert result == 5.0  # 25.0 / 5
    assert loss_second_term.result == 5.0


def test_finalize_with_zero_count(loss_second_term: LossSecondTerm) -> None:
    """Test finalize method when count is zero."""
    # Ensure that finalize returns float('inf') when count is zero
    result = loss_second_term.finalize()
    assert result == float("inf")
    assert loss_second_term.result == float("inf")


def test_reset(loss_second_term: LossSecondTerm) -> None:
    """Test LossSecondTerm reset method."""
    # Set non-default values to ensure reset works
    loss_second_term.count = 5
    loss_second_term.acc = 10.0
    loss_second_term.result = 3.0

    # Call reset
    loss_second_term.reset()

    # Check that values have been reset
    assert loss_second_term.count == 0
    assert loss_second_term.acc == 0
    assert loss_second_term.result == float("inf")
