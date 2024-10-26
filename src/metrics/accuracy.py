"""Module to calculate custom accuracy."""

import torch


def binary_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, avg: bool = True, threshold: float = 0.5
) -> torch.Tensor:
    """Calculate binary classification accuracy."""
    correct = (y_pred > threshold) == y_true

    return correct.sum() if avg is False else correct.type(torch.float32).mean()


def multiclass_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, avg: bool = True, _: None = None) -> torch.Tensor:
    """Calculate multiclass classification accuracy."""
    pred = y_pred.max(1, keepdim=True)[1]

    correct = pred.eq(y_true.view_as(pred))

    return correct.sum() if avg is False else correct.type(torch.float32).mean()
