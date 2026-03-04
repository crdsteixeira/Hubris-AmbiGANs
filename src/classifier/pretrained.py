"""Module for creating pretrained models."""

import torch
from torch import nn
from torchvision import transforms
from transformers import AutoModel, AutoModelForImageClassification
from transformers.modeling_utils import PreTrainedModel

from src.models import ClassifierParams


class ClassifierVIT(nn.Module):
    """Vision Transformer (VIT) classifier for image classification with customizable output features."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the VIT-based classifier."""
        super().__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        self.model.num_labels = out_features
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def forward(self, x: torch.Tensor, _: bool = False) -> torch.Tensor:
        """Forward pass for VIT classifier."""
        images = torch.stack([self.transforms(i).to(self.device) for i in x])
        return self.model(images)["logits"]


class ClassifierResnet(nn.Module):
    """ResNet-based classifier with a customizable output layer for image classification tasks."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the ResNet-based classifier."""
        super().__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-mnist")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.model.config.hidden_sizes[-1],
                out_features=out_features,
            ),
        )
        self.model.num_labels = out_features

    def forward(self, x: torch.Tensor, _: bool = False) -> torch.Tensor:
        """Forward pass for ResNet classifier."""
        return self.model(x)["logits"]


class ClassifierMLP(nn.Module):
    """Multi-layer Perceptron (MLP) classifier with customizable output layer and image preprocessing."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the MLP-based classifier."""
        super().__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device

        original_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized

        # Create a patched version that handles missing attribute
        def patched_mark_tied(self: PreTrainedModel) -> None:
            if not hasattr(self, "all_tied_weights_keys"):
                # For custom models without this attribute, use _tied_weights_keys if available
                if hasattr(self, "_tied_weights_keys"):
                    self.all_tied_weights_keys = {}  # Empty dict is safe for models without tied weights
                else:
                    return None  # Skip if neither attribute exists
            return original_mark_tied(self)

        # Apply monkey patch temporarily
        PreTrainedModel.mark_tied_weights_as_initialized = patched_mark_tied

        try:
            self.model = AutoModel.from_pretrained("dacorvo/mnist-mlp", trust_remote_code=True)
        finally:
            # Restore original method
            PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.model.config.hidden_size, out_features=out_features),
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda pil_img: pil_img.convert("L")),  # Grayscale for MLP
                # Ensure the input size matches what MLP expects
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081)),
                transforms.Lambda(torch.flatten),
            ]
        )

    def forward(self, x: torch.Tensor, _: bool = False) -> torch.Tensor:
        """Forward pass for MLP classifier."""
        images = torch.stack([self.transforms(i).to(self.device) for i in x])
        output = self.model(images)
        if output.shape[-1] == 1:  # Apply squeeze if binary classification (output is 1D)
            output = output.squeeze(-1)
        return output
