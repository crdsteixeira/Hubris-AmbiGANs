"""Test for accuracy module."""

import torch

from src.metrics.accuracy import binary_accuracy, multiclass_accuracy


def test_binary_accuracy_avg() -> None:
    """Test binary accuracy with averaging enabled."""
    y_pred = torch.tensor([0.8, 0.3, 0.7, 0.2])
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    accuracy = binary_accuracy(y_pred, y_true, avg=True)
    expected_accuracy = torch.tensor(1.0)  # All predictions are correct
    assert torch.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"


def test_binary_accuracy_no_avg() -> None:
    """Test binary accuracy without averaging (sum of correct predictions)."""
    y_pred = torch.tensor([0.8, 0.3, 0.7, 0.2])
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    accuracy = binary_accuracy(y_pred, y_true, avg=False)
    expected_accuracy = torch.tensor(4)  # All predictions are correct
    assert accuracy == expected_accuracy, f"Expected {expected_accuracy}, but got {accuracy}"


def test_binary_accuracy_threshold() -> None:
    """Test binary accuracy with a custom threshold."""
    y_pred = torch.tensor([0.6, 0.4, 0.7, 0.3])
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    accuracy = binary_accuracy(y_pred, y_true, avg=True, threshold=0.5)
    expected_accuracy = torch.tensor(1.0)  # All predictions are correct with threshold 0.5
    assert torch.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"


def test_multiclass_accuracy_avg() -> None:
    """Test multiclass accuracy with averaging enabled."""
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    y_true = torch.tensor([1, 2, 0])
    accuracy = multiclass_accuracy(y_pred, y_true, avg=True)
    expected_accuracy = torch.tensor(1.0)  # All predictions are correct
    assert torch.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"


def test_multiclass_accuracy_no_avg() -> None:
    """Test multiclass accuracy without averaging (sum of correct predictions)."""
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    y_true = torch.tensor([1, 2, 0])
    accuracy = multiclass_accuracy(y_pred, y_true, avg=False)
    expected_accuracy = torch.tensor(3)  # All predictions are correct
    assert accuracy == expected_accuracy, f"Expected {expected_accuracy}, but got {accuracy}"


def test_multiclass_accuracy_incorrect() -> None:
    """Test multiclass accuracy with incorrect predictions."""
    y_pred = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.6, 0.3], [0.1, 0.1, 0.8]])
    y_true = torch.tensor([1, 2, 0])
    accuracy = multiclass_accuracy(y_pred, y_true, avg=True)
    expected_accuracy = torch.tensor(0.0)  # None of the predictions are correct
    assert torch.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"


def test_binary_accuracy_partial_correct() -> None:
    """Test binary accuracy where some predictions are correct and others are incorrect."""
    y_pred = torch.tensor([0.8, 0.3, 0.2, 0.7])
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    accuracy = binary_accuracy(y_pred, y_true, avg=True)
    expected_accuracy = torch.tensor(0.5)  # Two predictions are correct out of four
    assert torch.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"


def test_multiclass_accuracy_partial_correct() -> None:
    """Test multiclass accuracy with some correct predictions."""
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    y_true = torch.tensor([1, 0, 1])
    accuracy = multiclass_accuracy(y_pred, y_true, avg=True)
    expected_accuracy = torch.tensor(2 / 3)  # Two out of three predictions are correct
    assert torch.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"
