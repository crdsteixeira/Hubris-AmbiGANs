"""Module for pre-trained evaluation."""

import gc

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextConfig,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    EfficientNetConfig,
    ViTConfig,
)

from src.enums import DeviceType


class HuggingFaceModel(nn.Module):
    """HuggingFace model class."""

    def _freeze_backbone(self, freeze_ratio: float = 0.8) -> None:
        """
        Freeze backbone layers for efficient fine-tuning.

        Args:
            freeze_ratio: Fraction of model parameters to freeze (0-1)

        """
        params = list(self.model.parameters())
        num_to_freeze = int(len(params) * freeze_ratio)
        for param in params[:num_to_freeze]:
            param.requires_grad = False

    def retrain(self, dataloader: DataLoader, epochs: int = 10, device: DeviceType = DeviceType.cpu) -> None:
        """
        Retrain pre-trained model with optimized training strategy.

        Uses mixed precision training, learning rate scheduling, and selective parameter
        freezing for faster convergence.
        """
        self.model.to(device)

        # Freeze 80% of backbone - only fine-tune top layers
        self._freeze_backbone(freeze_ratio=0.8)

        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_params, lr=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = BCEWithLogitsLoss()

        pbar = tqdm(range(epochs))

        for _ in pbar:
            self.model.train()
            batch_loss = 0
            num_batches = 0

            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Use autocast for mixed precision training
                with torch.autocast(device_type=str(device).split(":", maxsplit=1)[0], dtype=torch.float16):
                    # Forward pass
                    preds = self.forward(images)
                    loss = criterion(preds.squeeze(), labels.float())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                num_batches += 1

            # Step scheduler after each epoch
            scheduler.step()

            # Cleanup memory only once per epoch (not per batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            pbar.set_postfix(BatchLoss=batch_loss / max(num_batches, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward method for ConvNext wrapper."""
        images = (images + 1.0) / 2.0
        if images.shape[1] != 3:
            # Convert to RGB by repeating across the channel dimension
            images = images.repeat(1, 3, 1, 1)

        inputs = self.processor(images, return_tensors="pt", do_rescale=False).to(images.device)
        logits = self.model(**inputs).logits

        # Ensure output is always [batch_size, num_classes]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        return logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Get probability predictions for inference (applies sigmoid to logits)."""
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits)
            # Return 1D tensor for compatibility with evaluation code
            return probs.squeeze(-1)


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
    """Swin Transformer wrapper class."""

    def __init__(self) -> None:
        """Swin Transformer wrapper initialization."""
        super().__init__()
        self.processor, self.model = self._load_swin()

    def _load_swin(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """Load Swin Transformer model from HuggingFace Hub."""
        model_id = "microsoft/swin-tiny-patch4-window7-224"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id, ignore_mismatched_sizes=True)

        # Explicitly replace the classifier head for binary classification
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 1)
        model.config.num_labels = 1

        return processor, model
