"""Configuration management for multiclass classifier training."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """Configuration for datasets to train on."""

    datasets: list[str] = Field(
        default=["mnist", "fashion-mnist", "chest-xray"],
        description="List of dataset names to train on",
    )


class ClassifierConfig(BaseModel):
    """Configuration for classifiers to train."""

    classifiers: list[str] = Field(
        default=["cnn", "vgg16", "densenet", "resnet50", "vit"],
        description="List of classifier types to train",
    )


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    batch_size: int = Field(default=64, description="Batch size for training")
    epochs: int = Field(default=50, description="Number of training epochs")
    lr: float = Field(default=5e-4, description="Learning rate")
    optimizer: str = Field(default="adam", description="Optimizer type (adam, sgd)")
    weight_decay: float = Field(default=1e-5, description="L2 regularization weight decay")
    momentum: float = Field(default=0.9, description="Momentum for SGD optimizer")
    lr_schedule: str = Field(default="none", description="Learning rate schedule (none, cosine, step)")
    warmup_epochs: int = Field(default=0, description="Number of warmup epochs")
    dropout: float = Field(default=0.5, description="Dropout rate")


class MulticlassTrainingConfig(BaseModel):
    """Complete configuration for multiclass training."""

    dataset: str = Field(
        default="mnist",
        description="Dataset name to train on",
    )
    classifiers: list[str] = Field(
        default=["cnn", "vgg16", "densenet", "resnet50", "vit"],
        description="List of classifier types to train",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Default training parameters",
    )
    per_classifier_training: dict[str, dict] = Field(
        default_factory=dict,
        description="Classifier-specific training parameter overrides",
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_training_config(self, classifier_name: str | None = None) -> TrainingConfig:
        """
        Get training configuration for a classifier.

        Returns the classifier-specific config if available, otherwise the default.

        Args:
            classifier_name: Name of the classifier (optional)

        Returns:
            TrainingConfig with classifier-specific overrides applied

        """
        # Start with default config
        config_dict = self.training.model_dump()

        # Apply classifier-specific overrides (if classifier provided)
        if classifier_name and classifier_name in self.per_classifier_training:
            config_dict.update(self.per_classifier_training[classifier_name])

        return TrainingConfig(**config_dict)


def load_config(config_path: str | Path) -> MulticlassTrainingConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        MulticlassTrainingConfig instance

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config is invalid

    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        # Validate and create config object
        config = MulticlassTrainingConfig(**config_dict)

        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  - Dataset: {config.dataset}")
        logger.info(f"  - Classifiers: {config.classifiers}")
        logger.info("  - Default Training Params:")
        logger.info(f"      • Epochs: {config.training.epochs}")
        logger.info(f"      • Learning Rate: {config.training.lr}")
        logger.info(f"      • Batch Size: {config.training.batch_size}")
        logger.info(f"      • Optimizer: {config.training.optimizer}")
        logger.info(f"      • Weight Decay: {config.training.weight_decay}")
        if config.per_classifier_training:
            logger.info("  - Classifier-Specific Overrides:")
            for classifier_name, overrides in config.per_classifier_training.items():
                logger.info(f"      • {classifier_name}: {overrides}")

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}") from e


def get_default_config_path() -> Path:
    """
    Get the path to the default configuration file.

    Returns:
        Path to default config file (experiments/train-classifiers/multiclass_training_config.yaml)

    """
    # Get the project root (assuming we're in src/classifier/)
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "experiments" / "train-classifiers" / "multiclass_training_config.yaml"
    return config_path


def load_config_with_defaults(config_path: str | Path | None = None) -> MulticlassTrainingConfig:
    """
    Load configuration from file, or use defaults if file not found.

    Args:
        config_path: Path to configuration file (optional). If None, looks for default location

    Returns:
        MulticlassTrainingConfig instance

    """
    if config_path is None:
        config_path = get_default_config_path()

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Configuration file not found at {config_path}. " f"Using default configuration.")
        return MulticlassTrainingConfig()

    return load_config(config_path)
