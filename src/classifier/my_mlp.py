"""Module for creating MLP."""

import torch
from torch import nn

from src.models import ClassifierParams  # Import the data model


class Classifier(nn.Module):
    """Multi-layer Perceptron (MLP) classifier for image data based on given parameters."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize MLP classifier with the given parameters."""
        super().__init__()

        # Extract img_size, num_classes, and nf from params
        num_channels, height, width = params.img_size
        input_size = num_channels * height * width
        num_classes = params.n_classes
        nf = params.nf

        # Block 1: Flatten the input and apply the first Linear layer
        self.blocks = nn.ModuleList()
        block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, nf),
        )
        self.blocks.append(block_1)

        # Block 2: Output layer
        predictor = nn.Sequential(
            nn.Linear(nf, 1 if num_classes == 2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1),
        )
        self.blocks.append(predictor)

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Perform a forward pass through the MLP."""
        intermediate_outputs = []

        # Forward pass through each block
        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        # If binary classification, flatten the output to remove the singleton dimension
        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        # Return feature maps if requested, otherwise only the final output
        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
