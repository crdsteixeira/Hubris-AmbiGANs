"""Module for a Fully Connected Neural Network."""

import torch
from torch import nn

from src.models import ClassifierParams


class Classifier(nn.Module):
    """Fully Connected classifier with multiple dense layers, dropout, and flexible configuration."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the Fully Connected classifier using ClassifierParams."""
        super().__init__()
        num_channels, height, width = params.img_size
        input_size = num_channels * height * width
        num_classes = params.n_classes

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1 if num_classes == 2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Fully Connected classifier."""
        return self.model(x)
