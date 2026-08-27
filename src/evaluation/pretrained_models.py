"""Module for pre-trained evaluation."""

import logging
import math

import torch
import wandb
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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

CONVNEXT_MODEL_ID = "facebook/convnext-tiny-224"
EFFICIENTNETV2_MODEL_ID = "google/efficientnet-b2"
SWIN_MODEL_ID = "microsoft/swinv2-tiny-patch4-window8-256"
VIT_MODEL_ID = "google/vit-base-patch16-224"
# ViT backbone used for every dataset before VIT_MODEL_ID; kept so that checkpoints
# finetuned from it are recognised as stale instead of silently reloaded.
VIT_LEGACY_CXR_MODEL_ID = "NeuronZero/CXR-Classifier"

# Identifies the fine-tuning recipe below. Bump it whenever the recipe changes so that
# checkpoints produced by an older one are retrained instead of silently reused.
TRAINING_RECIPE_ID = "adamw-lr5e-5-wd0.05-warmup0.1-cosine-clip1.0-bcelogits-v1"


class HuggingFaceModel(nn.Module):
    """
    Base class for HuggingFace pre-trained models.

    Provides common functionality for loading and retraining HuggingFace image classification models.
    """

    #: Shared fine-tuning hyperparameters. Every model uses the same recipe so that the
    #: hubris comparison reflects the architecture rather than the optimizer settings.
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.05
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    def _param_groups(self) -> list[dict]:
        """Split parameters so that weight decay skips biases and normalisation weights."""
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            target = no_decay if param.ndim <= 1 or name.endswith(".bias") else decay
            target.append(param)
        return [
            {"params": decay, "weight_decay": self.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @staticmethod
    def _lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
        """Linear warmup followed by cosine decay, as a multiplier on the base learning rate."""
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    @staticmethod
    def _log(metrics: dict) -> None:
        """Log metrics to wandb when a run is active."""
        if wandb.run is not None:
            wandb.log(metrics)

    def _validation_loss(self, dataloader: DataLoader, criterion: nn.Module, device: DeviceType) -> float:
        """Compute the mean validation loss."""
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                logits = self.forward_logits(images.to(device))
                total_loss += criterion(logits, labels.to(device).float()).item()
                num_batches += 1
        return total_loss / max(1, num_batches)

    def retrain(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        device: DeviceType = DeviceType.cpu,
        val_dataloader: DataLoader | None = None,
        patience: int = 3,
    ) -> None:
        """
        Fine-tune the pre-trained model.

        Uses AdamW with linear warmup and cosine decay, gradient clipping, and a
        logit-space BCE loss. When a validation loader is given, the epoch with the
        lowest validation loss is restored at the end and training stops early after
        `patience` epochs without improvement.

        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train
            device: Device to use for training
            val_dataloader: Optional DataLoader used for model selection
            patience: Epochs without validation improvement before stopping early

        """
        self.model.to(device)
        optimizer = AdamW(self._param_groups(), lr=self.LEARNING_RATE)
        criterion = BCEWithLogitsLoss()

        total_steps = max(1, epochs * len(dataloader))
        warmup_steps = int(self.WARMUP_RATIO * total_steps)
        scheduler = LambdaLR(optimizer, lambda step: self._lr_scale(step, warmup_steps, total_steps))
        logger.info(
            "Fine-tuning %s: lr=%g wd=%g warmup=%d/%d steps",
            self.model_id,
            self.LEARNING_RATE,
            self.WEIGHT_DECAY,
            warmup_steps,
            total_steps,
        )

        best_val_loss, best_state, stale_epochs = math.inf, None, 0
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.train()
            epoch_loss, num_batches = 0.0, 0
            for images, labels in dataloader:
                logits = self.forward_logits(images.to(device))
                loss = criterion(logits, labels.to(device).float())

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                self._log({"batch_loss": loss.item(), "lr": scheduler.get_last_lr()[0], "epoch": epoch})

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            metrics = {"epoch_loss": avg_epoch_loss, "epoch": epoch}
            postfix = {"loss": avg_epoch_loss}

            if val_dataloader is not None:
                val_loss = self._validation_loss(val_dataloader, criterion, device)
                metrics["val_loss"] = val_loss
                postfix["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1

            pbar.set_postfix(**postfix)
            self._log(metrics)
            logger.info("Epoch %d/%d - %s", epoch + 1, epochs, metrics)

            if val_dataloader is not None and stale_epochs >= patience:
                logger.info("No validation improvement for %d epochs, stopping early.", patience)
                break

        if best_state is not None:
            logger.info("Restoring best epoch (val_loss %.6f)", best_val_loss)
            self.model.load_state_dict(best_state)
            self.model.to(device)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits. Training uses these directly for numerical stability."""
        images = (images + 1.0) / 2.0
        if images.shape[1] != 3:
            # Convert to RGB by repeating across the channel dimension
            images = images.repeat(1, 3, 1, 1)

        inputs = self.processor(images, return_tensors="pt", do_rescale=False).to(images.device)
        # squeeze(-1) rather than squeeze(): a trailing batch of size 1 must stay 1-D.
        return self.model(**inputs).logits.squeeze(-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return sigmoid(self.forward_logits(images))


class ConvNext(HuggingFaceModel):
    """ConvNext wrapper class."""

    def __init__(self) -> None:
        """Convnext wrapper initialization."""
        super().__init__()
        self.model_id = CONVNEXT_MODEL_ID
        self.processor, self.model = self._load_convnext()

    def _load_convnext(self) -> tuple[ConvNextImageProcessor, ConvNextForImageClassification]:
        """Load ConvNext model from pretrained HuggingFace location."""
        config = ConvNextConfig.from_pretrained(self.model_id)
        config.num_labels = 1
        processor = ConvNextImageProcessor.from_pretrained(self.model_id)
        model = ConvNextForImageClassification.from_pretrained(
            self.model_id, config=config, ignore_mismatched_sizes=True
        )
        return processor, model


class ViT(HuggingFaceModel):
    """Vision Transformer (ViT) wrapper class."""

    def __init__(self, model_id: str = VIT_MODEL_ID) -> None:
        """
        Initialize ViT model.

        Args:
            model_id: HuggingFace model id to load the backbone from.

        """
        super().__init__()
        self.model_id = model_id
        self.processor, self.model = self._load_vit()

    def _load_vit(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """
        Load ViT model from HuggingFace Hub.

        Returns:
            Tuple containing the image processor and pre-trained model.

        """
        logger.info("Loading ViT backbone from: %s", self.model_id)
        config = ViTConfig.from_pretrained(self.model_id)
        config.num_labels = 1
        processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageClassification.from_pretrained(
            self.model_id, config=config, ignore_mismatched_sizes=True
        )
        return processor, model


class EfficientNetV2(HuggingFaceModel):
    """EfficientNetV2 wrapper class."""

    def __init__(self) -> None:
        """EfficientNetV2 wrapper initialization."""
        super().__init__()
        self.model_id = EFFICIENTNETV2_MODEL_ID
        self.processor, self.model = self._load_efficientnetv2()

    def _load_efficientnetv2(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """Load EfficientNetV2 model from HuggingFace Hub."""
        model_id = self.model_id
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
        self.model_id = SWIN_MODEL_ID
        self.processor, self.model = self._load_swin()

    def _load_swin(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        """
        Load Swin Transformer model from HuggingFace Hub.

        Returns:
            Tuple containing the image processor and pre-trained model.

        """
        model_id = self.model_id
        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = 1
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        return processor, model
