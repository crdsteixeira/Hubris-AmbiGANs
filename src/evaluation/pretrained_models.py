"""Module for pre-trained evaluation."""

import torch
from torch import nn
from torch.nn import BCELoss
from torch.nn.functional import sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextConfig,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    ViTConfig,
)

from src.enums import DeviceType


class HuggingFaceModel(nn.Module):
    """HuggingFace model class."""

    def retrain(self, dataloader: DataLoader, epochs: int = 10, device: DeviceType = DeviceType.cpu) -> None:
        """Retrain pre-trained model."""
        self.model.to(device)
        optimizer = Adam(self.model.parameters())
        criterion = BCELoss()
        pbar = tqdm(range(epochs))
        for _ in pbar:
            self.model.train()
            batch_loss = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                preds = self.forward(images)
                loss = criterion(preds, labels.float())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            pbar.set_postfix(BatchLoss=batch_loss)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward method for ConvNext wrapper."""
        images = (images + 1.0) / 2.0
        if images.shape[1] != 3:
            # Convert to RGB by repeating across the channel dimension
            images = images.repeat(1, 3, 1, 1)

        inputs = self.processor(images, return_tensors="pt", do_rescale=False).to(images.device)
        logits = self.model(**inputs).logits
        return sigmoid(logits).squeeze()


class ConvNext(HuggingFaceModel):
    """ConvNext wrapper class."""

    def __init__(self) -> None:
        """Convnext wrapper initialization."""
        super().__init__()
        self.processor, self.model = self._load_convnext()

    def _load_convnext(self) -> tuple[ConvNextImageProcessor, ConvNextForImageClassification]:
        """Load ConvNext model from pretrained HuggingFace location."""
        config = ConvNextConfig.from_pretrained("facebook/convnext-tiny-224")
        config.num_labels = 1
        processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        model = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-tiny-224", config=config, ignore_mismatched_sizes=True
        )
        return processor, model


class ViT(HuggingFaceModel):
    """Vit wrapper class."""

    def __init__(self) -> None:
        """Vit wrapper initialization."""
        super().__init__()
        self.processor, self.model = self._load_vit()

    def _load_vit(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """Vit model from pretrained HuggingFace location."""
        config = ViTConfig.from_pretrained("NeuronZero/CXR-Classifier")
        config.num_labels = 1
        processor = AutoImageProcessor.from_pretrained("NeuronZero/CXR-Classifier")
        model = AutoModelForImageClassification.from_pretrained(
            "NeuronZero/CXR-Classifier", config=config, ignore_mismatched_sizes=True
        )
        return processor, model
