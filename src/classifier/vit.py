"""Module for Vision Transformer (ViT) classifier."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.models import vision_transformer

from src.models import ClassifierParams


class Classifier(nn.Module):
    """Vision Transformer (ViT) classifier for image classification with customizable output features."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the ViT classifier."""
        super().__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.out_features = out_features

        # Load pre-trained ViT-B/16
        self.model = vision_transformer.vit_b_16(weights=vision_transformer.ViT_B_16_Weights.DEFAULT)

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, out_features)

        # Unfreeze the new classifier layer
        self.model.heads.head.requires_grad = True

        # Preprocessing transform to resize and normalize
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Forward pass for ViT classifier."""
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
