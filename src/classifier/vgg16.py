"""Module for VGG-16 classifier."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from src.models import ClassifierParams


class Classifier(nn.Module):
    """VGG-16 classifier for image classification with customizable output features."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the VGG-16 classifier."""
        super().__init__()
        num_classes = params.n_classes
        self.device = params.device
        self.num_classes = num_classes

        # Load pre-trained VGG-16
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head
        in_features = self.model.classifier[6].in_features
        # For binary classification, output 1 channel with sigmoid; for multiclass, output num_classes with no activation
        out_features = 1 if num_classes == 2 else num_classes
        self.model.classifier[6] = nn.Linear(in_features, out_features)

        # Add activation function
        if num_classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()  # No activation for multiclass (CrossEntropyLoss expects logits)

        # Unfreeze only the classifier head (not conv layers)
        # Unfreezing conv blocks causes excessive gradient memory usage
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Forward pass for VGG-16 classifier."""
        del output_feature_maps
        # Handle grayscale images (convert to RGB if needed)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            # For other channel sizes, try to adapt
            x = x[:, :3] if x.shape[1] > 3 else F.pad(x, (0, 0, 0, 0, 0, 3 - x.shape[1]))

        # Ensure minimum input size (32x32) for VGG16
        # VGG16 is designed for larger images, so upscale small images
        if x.shape[2] < 32 or x.shape[3] < 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        # Get model output and apply activation
        output = self.model(x)
        output = self.activation(output)

        # Squeeze binary output from [batch_size, 1] to [batch_size] for BCELoss compatibility
        if self.num_classes == 2:
            output = output.squeeze(-1)

        return output
