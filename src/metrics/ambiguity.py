"""Metrics helpers for ambiguity evaluation."""

import logging

import torch

logger = logging.getLogger(__name__)


def compute_entropy(predictions: torch.Tensor) -> float:
    """
    Compute entropy from classifier predictions (softmax probabilities).

    Args:
        predictions: Tensor of shape (n_samples, n_classes) with probabilities

    Returns:
        Mean entropy across samples (in nats, using natural logarithm)

    """
    epsilon = 1e-10
    # Clamp predictions to valid probability range [epsilon, 1-epsilon]
    predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
    # Ensure probabilities sum to 1 for each sample (renormalize after clamping)
    predictions = predictions / predictions.sum(dim=1, keepdim=True)
    # Compute entropy: H(X) = -Σ p(x) * log(p(x))
    entropy = -(predictions * torch.log(predictions)).sum(dim=1)

    return entropy.mean().item()


def compute_top_pairs(predictions: torch.Tensor, ground_truth: list[list[int]]) -> float:
    """
    Compute top pairs metric for ambiguous datasets.

    For each sample, checks if the two most likely predicted classes equal
    the two true classes (i.e., the set of top 2 predictions matches the set
    of ground truth classes). Returns the percentage of inputs where this is true.

    Args:
        predictions: Tensor of shape (n_samples, n_classes) with softmax probabilities
        ground_truth: List of lists where each element is a list of ground truth class indices

    Returns:
        Percentage of inputs where top 2 predictions equal the ground truth classes

    """
    if len(predictions) != len(ground_truth):
        logger.warning("Mismatch: got %d predictions but %d ground truth labels", len(predictions), len(ground_truth))
        return 0.0

    _, top_2_indices = torch.topk(predictions, k=2, dim=1)
    top_2_indices = top_2_indices.cpu().numpy()

    matches = 0
    for top_2, gt in zip(top_2_indices, ground_truth):
        if not gt:
            continue
        if set(top_2) == set(gt):
            matches += 1

    return float(matches / len(ground_truth) * 100) if ground_truth else 0.0
