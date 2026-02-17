"""Module to calculate custom accuracy."""

import torch


def binary_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, avg: bool = True, threshold: float | None = 0.5
) -> torch.Tensor:
    """Calculate binary classification accuracy."""
    correct = (y_pred > threshold) == y_true

    return correct.sum() if avg is False else correct.type(torch.float32).mean()


def binary_precision_recall_f1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 for binary classification."""
    preds = (y_pred > threshold).view(-1)
    labels = y_true.view(-1)

    preds = preds.to(torch.int64)
    labels = labels.to(torch.int64)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def multiclass_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    avg: bool = True,
) -> torch.Tensor:
    """Calculate multiclass classification accuracy."""
    pred = y_pred.max(1, keepdim=True)[1]

    correct = pred.eq(y_true.view_as(pred))

    return correct.sum() if avg is False else correct.type(torch.float32).mean()


def top_n_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, n: int = 2) -> float:
    """Calculate top-n accuracy for multiclass classification."""
    _, top_n_preds = torch.topk(y_pred, n, dim=1)
    y_true_expanded = y_true.unsqueeze(1).expand_as(top_n_preds)
    correct = top_n_preds.eq(y_true_expanded).any(dim=1)

    return correct.sum().item() / len(y_true)
