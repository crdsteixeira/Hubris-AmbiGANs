"""Module for a DenseNet-based classifier."""

import torch
from torch import nn
from torchvision.models import densenet121

from src.models import ClassifierParams


class Classifier(nn.Module):
    """DenseNet121-based classifier for image classification with custom FC layers and dropout."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the DenseNet121 classifier using ClassifierParams."""
        super().__init__()
        num_channels, height, width = params.img_size
        num_classes = params.n_classes

        self.blocks = nn.ModuleList()

        # DenseNet expects 3-channel 224x224 images
        # For MNIST/FashionMNIST (28x28, 1-channel), we need to reshape
        if height != 224 or width != 224 or num_channels != 3:
            # Create a reshape block that converts input to (224, 224, 3)
            self.blocks.append(
                nn.Sequential(
                    DenseNetReshape(num_channels),
                )
            )
        else:
            # If already 224x224x3, just add identity
            self.blocks.append(nn.Identity())

        # Load pretrained DenseNet121 and remove the classification head
        densenet = densenet121(weights="DEFAULT")

        # Remove the classifier layer
        # Keep everything up to and including the final relu and adaptive average pooling
        self.feature_extractor = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
        )

        # Get the number of output features from DenseNet121 (last conv layer has 1024 channels)
        num_features = densenet.classifier.in_features

        # Add the feature extractor as a block (MUST come before pooling)
        self.blocks.append(self.feature_extractor)

        # Add global average pooling (after feature extraction)
        self.blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Custom FC layers with lower dropout (0.1 as in the Keras code, though using 0.2 for consistency with other models)
        fc_block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.blocks.append(fc_block_1)

        fc_block_2 = nn.Sequential(
            nn.Linear(256, 125),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.blocks.append(fc_block_2)

        # Output layer
        output_block = nn.Sequential(
            nn.Linear(125, 1 if num_classes == 2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1),
        )
        self.blocks.append(output_block)

        # Freeze DenseNet121 backbone - only train FC layers
        # Unfreezing dense blocks causes excessive gradient memory usage
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DenseNet121 classifier."""
        # Apply reshape if needed
        x = self.blocks[0](x)

        # Apply feature extractor
        x = self.blocks[1](x)  # DenseNet121 feature extractor
        x = self.blocks[2](x)  # AdaptiveAvgPool2d
        x = x.view(x.size(0), -1)  # Flatten

        # Apply FC layers and output
        for i in range(3, len(self.blocks)):
            x = self.blocks[i](x)

        # If binary classification, flatten the output to remove the singleton dimension
        if x.shape[1] == 1:
            x = x.flatten()

        return x


class DenseNetReshape(nn.Module):
    """Custom layer to reshape MNIST/FashionMNIST images to DenseNet input format."""

    def __init__(self, num_channels: int) -> None:
        """Initialize the reshape layer."""
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input from (batch, channels, h, w) to (batch, 3, 224, 224)."""
        if self.num_channels == 1:
            # Convert single channel to 3 channels by repeating
            x = x.repeat(1, 3, 1, 1)

        # Resize to 224x224 using interpolation
        if x.size(2) != 224 or x.size(3) != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        return x
