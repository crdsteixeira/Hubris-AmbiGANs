"""Module for VGG-16 classifier."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

from src.models import ClassifierParams


class Classifier(nn.Module):
    """VGG-16 classifier for image classification with customizable output features."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the VGG-16 classifier."""
        super().__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.out_features = out_features

        # Load pre-trained VGG-16
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, out_features)

        # Unfreeze the new classifier layer
        self.model.classifier[6].requires_grad = True

        # Preprocessing transform to resize and normalize
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Forward pass for VGG-16 classifier."""
        del output_feature_maps
        # Handle grayscale images (convert to RGB)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            # For other channel sizes, try to adapt
            x = x[:, :3] if x.shape[1] > 3 else F.pad(x, (0, 0, 0, 0, 0, 3 - x.shape[1]))

        # Resize if needed
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Apply preprocessing
        x = self.transform(x)

        return self.model(x)
