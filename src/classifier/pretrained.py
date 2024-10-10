import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel, AutoModelForImageClassification

from src.models import ClassifierParams


class ClassifierVIT(nn.Module):
    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the VIT-based classifier."""
        super(ClassifierVIT, self).__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.model = AutoModelForImageClassification.from_pretrained(
            "farleyknight-org-username/vit-base-mnist"
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Linear(
            in_features=self.model.classifier.in_features, out_features=out_features
        )
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

    def forward(
        self, x: torch.Tensor, output_feature_maps: bool = False
    ) -> torch.Tensor:
        """Forward pass for VIT classifier."""
        images = torch.stack([self.transforms(i).to(self.device) for i in x])
        return self.model(images)["logits"]


class ClassifierResnet(nn.Module):
    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the ResNet-based classifier."""
        super(ClassifierResnet, self).__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.model = AutoModelForImageClassification.from_pretrained(
            "fxmarty/resnet-tiny-mnist"
        )

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

    def forward(
        self, x: torch.Tensor, output_feature_maps: bool = False
    ) -> torch.Tensor:
        """Forward pass for ResNet classifier."""
        return self.model(x)["logits"]


class ClassifierMLP(nn.Module):
    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the MLP-based classifier."""
        super(ClassifierMLP, self).__init__()
        out_features = 1 if params.n_classes == 2 else params.n_classes
        self.device = params.device
        self.model = AutoModel.from_pretrained(
            "dacorvo/mnist-mlp", trust_remote_code=True
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.model.config.hidden_size, out_features=out_features
            ),
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(
                    lambda pil_img: pil_img.convert("L")
                ),  # Grayscale for MLP
                # Ensure the input size matches what MLP expects
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081)),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )

    def forward(
        self, x: torch.Tensor, output_feature_maps: bool = False
    ) -> torch.Tensor:
        """Forward pass for MLP classifier."""
        images = torch.stack([self.transforms(i).to(self.device) for i in x])
        output = self.model(images)
        if (
            output.shape[-1] == 1
        ):  # Apply squeeze if binary classification (output is 1D)
            output = output.squeeze(-1)
        return output
