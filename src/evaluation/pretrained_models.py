"""Module for pre-trained evaluation."""

import logging

import torch
import wandb
from torch import nn
from torch.nn import BCELoss
from torch.nn.functional import sigmoid
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextConfig,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    EfficientNetConfig,
    ViTConfig,
)

from src.enums import DeviceType

logger = logging.getLogger(__name__)


class HuggingFaceModel(nn.Module):
    """
    Base class for HuggingFace pre-trained models.

    Provides common functionality for loading and retraining HuggingFace image classification models.
    """

    def retrain(self, dataloader: DataLoader, epochs: int = 10, device: DeviceType = DeviceType.cpu) -> None:
        """
        Retrain pre-trained model with optional wandb logging.

        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train
            device: Device to use for training

        """
        self.model.to(device)
        optimizer = Adam(self.model.parameters())
        criterion = BCELoss()
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            num_batches = 0
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

                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                # Log batch-level metrics if wandb is enabled
                wandb.log(
                    {
                        "batch_loss": batch_loss,
                        "epoch": epoch,
                    }
                )

            # Compute average loss for epoch
            avg_epoch_loss = epoch_loss / num_batches
            pbar.set_postfix(Loss=avg_epoch_loss)

            # Log epoch-level metrics if wandb is enabled
            wandb.log(
                {
                    "epoch_loss": avg_epoch_loss,
                    "epoch": epoch,
                }
            )
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f}")

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
    """Vision Transformer (ViT) wrapper class for CXR classification."""

    def __init__(self) -> None:
        """Initialize ViT model for chest X-ray classification."""
        super().__init__()
        self.processor, self.model = self._load_vit()

    def _load_vit(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """
        Load ViT model from HuggingFace Hub for CXR classification.

        Returns:
            Tuple containing the image processor and pre-trained model.

        """
        config = ViTConfig.from_pretrained("NeuronZero/CXR-Classifier")
        config.num_labels = 1
        processor = AutoImageProcessor.from_pretrained("NeuronZero/CXR-Classifier")
        model = AutoModelForImageClassification.from_pretrained(
            "NeuronZero/CXR-Classifier", config=config, ignore_mismatched_sizes=True
        )
        return processor, model

    def retrain(self, dataloader: DataLoader, epochs: int = 10, device: DeviceType = DeviceType.cpu) -> None:
        """
        Retrain ViT model with AdamW optimizer.

        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train
            device: Device to use for training

        """
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.05)
        criterion = BCELoss()  # keep same output (probabilities)

        for _ in range(epochs):
            self.model.train()
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                preds = self.forward(images)  # still sigmoid probs
                loss = criterion(preds, labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class EfficientNetV2(HuggingFaceModel):
    """EfficientNetV2 wrapper class."""

    def __init__(self) -> None:
        """EfficientNetV2 wrapper initialization."""
        super().__init__()
        self.processor, self.model = self._load_efficientnetv2()

    def _load_efficientnetv2(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """Load EfficientNetV2 model from HuggingFace Hub."""
        model_id = "google/efficientnet-b2"
        config = EfficientNetConfig.from_pretrained(model_id)
        config.num_labels = 1
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        return processor, model


class Swin(HuggingFaceModel):
    """Swin Transformer wrapper class for image classification."""

    def __init__(self) -> None:
        """Initialize Swin Transformer model."""
        super().__init__()
        self.processor, self.model = self._load_swin()

    def _load_swin(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """
        Load Swin Transformer model from HuggingFace Hub.

        Returns:
            Tuple containing the image processor and pre-trained model.

        """
        model_id = "microsoft/swinv2-tiny-patch4-window8-256"
        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = 1
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        return processor, model

    def retrain(self, dataloader: DataLoader, epochs: int = 10, device: DeviceType = DeviceType.cpu) -> None:
        """
        Retrain Swin model with AdamW optimizer.

        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train
            device: Device to use for training

        """
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.05)  # Swin-specific
        criterion = BCELoss()  # keep identical outputs to other models

        pbar = tqdm(range(epochs))
        for _ in pbar:
            self.model.train()
            epoch_loss, num_batches = 0.0, 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                preds = self.forward(images)  # still sigmoid probs
                loss = criterion(preds, labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg = epoch_loss / max(1, num_batches)
            pbar.set_postfix(Loss=avg)
