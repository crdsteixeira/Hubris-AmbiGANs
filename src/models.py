"""Pydantic moels validation and settings management."""

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn, optim
from torch.utils.data import Dataset

from src.enums import (
    ArchitectureType,
    ChestXrayClasses,
    Cifar10Classes,
    ClassifierType,
    DatasetNames,
    DeviceType,
    EnsembleType,
    LossType,
    MnistClasses,
    OutputMethod,
    TrainingStage,
)
from src.gan.loss import DiscriminatorLoss, GeneratorLoss
from src.gan.update_g import UpdateGenerator
from src.metrics.fid.fid import FID
from src.metrics.hubris import Hubris
from src.metrics.loss_term import LossSecondTerm

load_dotenv()


class ClassifierParams(BaseModel):
    """Specifys parameters of different classifier types."""

    type: ClassifierType  # Only 'cnn', 'mlp', or 'ensemble'
    img_size: tuple[int, int, int] = Field(..., description="Tuple of three integers for image size")
    n_classes: int = Field(..., description="Number of classes (should be >= 2)")
    nf: int | list[int] | list[list[int]] | None = Field(
        None,
        description="Number of filters for cnn/mlp. Can be None for pretrained ensembles.",
    )
    ensemble_type: EnsembleType | None = Field(None, description="Type of ensemble when applicable")
    output_method: OutputMethod | None = Field(None, description="Output method for ensemble when applicable")
    device: DeviceType = Field(
        DeviceType.cpu, description="Device for computation ('cpu' or 'cuda')"
    )  # Using Enum for device validation

    @model_validator(mode="after")
    def validate_n_classes(self) -> "ClassifierParams":
        """Validate that n_classes is at least 2."""
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        return self

    @model_validator(mode="before")
    @classmethod
    def parse_ensemble_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate the ensemble type and output method from the classifier type string."""
        type_value = values.get("type")

        if isinstance(type_value, str) and type_value.startswith("ensemble:"):
            # Split and validate
            parts = type_value.split(":")
            if len(parts) != 3:
                raise ValueError("Invalid ensemble format. Must be 'ensemble:<ensemble_type>:<output_method>'")

            # Validate ensemble_type and output_method
            ensemble_type_str, output_method_str = parts[1], parts[2]
            try:
                values["ensemble_type"] = EnsembleType(ensemble_type_str)
                values["output_method"] = OutputMethod(output_method_str)
            except ValueError as e:
                raise ValueError(f"Invalid values for ensemble_type or output_method: {str(e)}") from e

            # Set type to 'ensemble'
            values["type"] = ClassifierType.ensemble

        return values

    @model_validator(mode="after")
    def check_ensemble_fields(self) -> "ClassifierParams":
        """Validate that both ensemble_type and output_method are provided for ensemble type classifiers."""
        if self.type == ClassifierType.ensemble:
            if not self.ensemble_type or not self.output_method:
                raise ValueError("ensemble_type and output_method must be provided for ensemble type classifiers")
        return self

    @model_validator(mode="after")
    def check_img_size(self) -> "ClassifierParams":
        """Validate img_size based on the classifier type and ensemble type."""
        img_size = self.img_size
        if self.type == ClassifierType.ensemble and self.ensemble_type == EnsembleType.pretrained:
            # Pretrained models accept 1-channel images
            if img_size[0] != 1:
                raise ValueError("Pretrained models in ensemble expect 1-channel images (got {img_size[0]} channels)")
        else:
            # Non-pretrained models expect 3-channel images
            if img_size[0] > 3:
                raise ValueError(
                    "img_size must be a tuple of three integers with max 3 channels for non-pretrained models"
                )

        if len(img_size) != 3:
            raise ValueError("img_size must be a tuple of three integers")
        return self

    @model_validator(mode="after")
    def check_nf(self) -> "ClassifierParams":
        """Validate nf based on classifier type and ensemble type."""
        if self.type == ClassifierType.cnn:
            if not isinstance(self.nf, list) or any(not isinstance(n, int) for n in self.nf):
                raise ValueError("For cnn type, nf must be a list of integers.")
        elif self.type == ClassifierType.mlp:
            if not isinstance(self.nf, int):
                raise ValueError("For mlp type, nf must be a single integer.")
        elif self.type == ClassifierType.ensemble:
            if self.ensemble_type == EnsembleType.cnn:
                if not isinstance(self.nf, list) or any(not isinstance(n, list) for n in self.nf):
                    raise ValueError("For ensemble with cnn type, nf must be a list of lists of integers.")
            elif self.ensemble_type == EnsembleType.pretrained:
                if self.nf is not None:
                    raise ValueError("For ensemble with pretrained type, nf should be None.")
        return self


class SpatialParams(BaseModel):
    """Pydantic model to validate parameters for the pooling operation."""

    in_size: int | tuple[int, int] = Field(..., description="Input size (height or width) for spatial operation.")
    out_size: int | tuple[int, int] | None = Field(
        None, description="Output size (height or width) for spatial operation."
    )
    kernel: int = Field(..., description="Kernel size for pooling or convolution.")
    dilation: int | None = Field(1, description="Dilation factor for pooling or convolution.")
    padding: int | None = Field(0, description="Padding applied to the input.")
    stride: int | None = Field(
        None,
        description="Stride for pooling or convolution. Defaults to kernel size if not provided.",
    )
    in_channels: int | None = Field(None, description="Input channels for spatial operation.")
    out_channels: int | None = Field(None, description="Output channels for spatial operation.")
    bias: bool | None = Field(None, description="Use bias during spatial operation")

    @model_validator(mode="after")
    def check_valid_values(self) -> "SpatialParams":
        """Validate that the input size, kernel, dilation, padding, and stride are all non-negative."""
        if self.kernel < 0:
            raise ValueError("Kernel size must be non-negative.")
        if self.stride is not None and self.stride < 0:
            raise ValueError("Stride must be non-negative.")
        if self.padding is not None and self.padding < 0:
            raise ValueError("Padding must be non-negative.")
        if self.dilation is not None and self.dilation < 0:
            raise ValueError("Dilation must be non-negative.")
        if self.in_channels is not None and self.in_channels < 0:
            raise ValueError("Number input channels must be non-negative.")
        if self.out_channels is not None and self.out_channels < 0:
            raise ValueError("Number output channels must be non-negative.")
        return self


class PoolParams(SpatialParams):
    """Arguments for Pooling."""

    in_size: int = Field(..., description="Input size (height or width) for pooling operation.")
    padding: int = Field(0, description="Padding applied to the input.")
    dilation: int = Field(1, description="Dilation factor for pooling or convolution.")

    @model_validator(mode="after")
    def validate_in_size(self) -> "PoolParams":
        """Validate inut image size."""
        if self.in_size < 0:
            raise ValueError("Input image size must be non-negative.")
        return self


class PadParams(SpatialParams):
    """Arguments for Padding."""

    in_size: int = Field(..., description="Input size (height or width) for padding operation.")
    out_size: int = Field(..., description="Output size (height or width) for padding operation.")
    stride: int = Field(..., description="Stride for pooling or convolution. Defaults to kernel size if not provided.")

    @model_validator(mode="after")
    def validate_in_size(self) -> "PadParams":
        """Validate input image size."""
        if self.in_size < 0:
            raise ValueError("Input image size must be non-negative.")
        return self


class ConvParams(SpatialParams):
    """Arguments for Convolution."""

    in_channels: int = Field(..., description="Number of input channels for convolutional operation.")
    out_channels: int = Field(..., description="Number of output channels for convolutional operation.")
    in_size: tuple[int, int] = Field(..., description="Input size (height or width) for padding operation.")
    out_size: tuple[int, int] = Field(..., description="Output size (height or width) for padding operation.")
    stride: int = Field(..., description="Stride for convolutional operation.")
    bias: bool = Field(True, description="Use bias during convolutional operation")

    @model_validator(mode="after")
    def validate_in_size(self) -> "PadParams":
        """Validate input image size."""
        if self.in_size[0] < 0 or self.in_size[0] < 0:
            raise ValueError("Input image size must be non-negative.")
        return self


class EvaluateParams(BaseModel):
    """Pydantic model for `evaluate` function parameters."""

    device: DeviceType = Field(..., description="Device for computation ('cpu', 'cuda').")
    verbose: bool = Field(True, description="Whether to show progress bar or not.")
    desc: str = Field("Validate", description="Description for the progress bar.")
    header: str | None = Field(None, description="Optional header for logging.")


class TrainArgsBase(BaseModel):
    """Pydantic base model for training arguments."""

    epochs: int = Field(..., description="Number of training epochs")
    early_acc: float = Field(1.0, description="Early accuracy threshold")
    out_dir: str = Field(..., description="Output directory for saving checkpoints")
    batch_size: int = Field(64, description="Batch size for training")
    lr: float = Field(default=5e-4, description="Learning rate for the optimizer")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    device: DeviceType = Field(..., description="Device for computation ('cpu', 'cuda')")


class TrainArgs(TrainArgsBase):
    """Pydantic model for training arguments."""

    early_stop: int | None = Field(None, description="Early stopping criterion")


class TrainingStats(BaseModel):
    """Pydantic model for tracking training statistics."""

    best_loss: float = Field(
        float("inf"),
        description="The lowest validation loss encountered during training.",
    )
    best_epoch: int = Field(0, description="Epoch with the lowest validation loss.")
    early_stop_tracker: int = Field(0, description="Counter for early stopping criteria.")
    cur_epoch: int = Field(0, description="Current epoch during training.")
    train_acc: list[float] = Field(default_factory=list, description="List of training accuracies for each epoch.")
    train_loss: list[float] = Field(default_factory=list, description="List of training losses for each epoch.")
    val_acc: list[float] = Field(
        default_factory=list,
        description="List of validation accuracies for each epoch.",
    )
    val_loss: list[float] = Field(default_factory=list, description="List of validation losses for each epoch.")
    test_acc: float = Field(0.0, description="List of test accuracies for each epoch.")
    test_loss: float = Field(0.0, description="List of test losses for each epoch.")


class DatasetClasses(BaseModel):
    """Arguments for Dataset Classes."""

    dataset_name: DatasetNames = Field(..., description="Name of the dataset.")
    pos_class: int = Field(..., description="Positive class for binary classification.")
    neg_class: int = Field(..., description="Negative class for binary classification.")

    @model_validator(mode="after")
    def validate_classes_for_binary_datasets(self) -> "DatasetClasses":
        """Validate that pos_class and neg_class are provided and valid for the dataset."""
        # If positive or negative class is specified, ensure both are given
        if self.pos_class is None or self.neg_class is None:
            raise ValueError(
                f"""Both pos_class and neg_class must be provided for"""
                f"""binary classification with dataset {self.dataset_name}."""
            )

        # Define valid classes for each dataset
        dataset_class_mapping = {
            DatasetNames.mnist: MnistClasses,
            DatasetNames.fashion_mnist: MnistClasses,
            DatasetNames.cifar10: Cifar10Classes,
            DatasetNames.chest_xray: ChestXrayClasses,
        }

        # Get the enum class based on dataset_name
        valid_classes_enum = dataset_class_mapping.get(self.dataset_name)

        # Check if pos_class and neg_class are valid members of the enum
        if valid_classes_enum:
            if self.pos_class not in [item.value for item in valid_classes_enum]:
                raise ValueError(
                    f"Invalid pos_class '{self.pos_class}' for dataset '{self.dataset_name}'. "
                    f"Valid classes are {[item.value for item in valid_classes_enum]}."
                )
            if self.neg_class not in [item.value for item in valid_classes_enum]:
                raise ValueError(
                    f"Invalid neg_class '{self.neg_class}' for dataset '{self.dataset_name}'. "
                    f"Valid classes are {[item.value for item in valid_classes_enum]}."
                )

        return self


class CLTrainArgs(DatasetClasses):
    """Pydantic model for parsing and validating command-line training arguments."""

    data_dir: str = Field(default=f"{os.environ['FILESDIR']}/data", description="Path to dataset")
    out_dir: str = Field(
        default=f"{os.environ['FILESDIR']}/models",
        description="Path to generated files",
    )
    name: str | None = Field(default=None, description="Name of the classifier for output files")
    batch_size: int = Field(default=64, description="Batch size for training")
    c_type: ClassifierType = Field(
        default=ClassifierType.cnn,
        description="Classifier type ('cnn', 'mlp', or 'ensemble')",
    )
    epochs: int = Field(default=2, description="Number of epochs to train for")
    early_stop: int | None = Field(default=None, description="Early stopping criteria (optional)")
    early_acc: float = Field(default=1.0, description="Early accuracy threshold for backpropagation")
    lr: float = Field(default=5e-4, description="Learning rate for the optimizer")
    nf: int | list[int] = Field(default=2, description="Number of filters or features in the model")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    device: DeviceType = Field(DeviceType.cpu, description="Device for computation ('cpu' or 'cuda')")
    n_classes: int = Field(None, description="Number of classes in the dataset")
    ensemble_type: EnsembleType | None = Field(None, description="Type of ensemble when applicable")
    ensemble_output_method: OutputMethod | None = Field(None, description="Output method for ensemble when applicable")

    @model_validator(mode="after")
    def set_n_classes_based_on_dataset(self) -> "CLTrainArgs":
        """Set the number of classes based on the dataset."""
        dataset_class_mapping = {
            DatasetNames.mnist: len(MnistClasses),
            DatasetNames.fashion_mnist: len(MnistClasses),
            DatasetNames.cifar10: len(Cifar10Classes),
            DatasetNames.chest_xray: len(ChestXrayClasses)
        }

        # Set the number of classes based on the dataset provided
        if self.dataset_name in dataset_class_mapping:
            self.n_classes = dataset_class_mapping[self.dataset_name]
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        
        return self



# Mix-in model that combines both training and classifier arguments
class TrainClassifierArgs(TrainArgs, ClassifierParams):
    """Combined model for training arguments and classifier parameters."""


class DatasetParams(BaseModel):
    """Parameters for configuring dataset loading."""

    dataroot: str = Field(..., description="Directory where the dataset is stored.")
    train: bool = Field(True, description="Indicates whether to load the training set or the test set.")
    pytesting: bool = Field(
        False, description="Indicates whether to load only a fraction of the dataset for pytesting purpose."
    )


class LoadDatasetParams(DatasetParams, DatasetClasses):
    """Pydantic model for parameters to load a dataset."""


class ImageParams(BaseModel):
    """Base model for validating image dimensions."""

    image_size: tuple[int, int, int] = Field(..., description="Dimensions of the image (channels, height, width).")

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Ensure that image dimensions have valid positive integers."""
        if len(v) != 3:
            raise ValueError("Image size must be a tuple of three integers (channels, height, width).")
        if any(d <= 0 for d in v):
            raise ValueError("All image dimensions (channels, height, width) must be positive integers.")
        return v


class GenParams(ImageParams):
    """Pydantic model for parameters to configure the Generator."""

    z_dim: int = Field(100, gt=0, description="Dimension of the latent code. Must be greater than zero.")
    n_blocks: int = Field(3, gt=0, description="Number of transposed convolutional blocks. Must be greater than zero.")
    filter_dim: int = Field(
        64,
        gt=0,
        description="Base number of filters in the first transposed convolutional layer. Must be greater than zero.",
    )

    @field_validator("n_blocks")
    @classmethod
    def validate_n_blocks(cls, v: int) -> int:
        """Ensure that the number of blocks is reasonable for model complexity."""
        if v < 1:
            raise ValueError("Number of blocks must be at least 1.")
        return v


class DisParams(ImageParams):
    """Pydantic model for parameters to configure the Discriminator."""

    filter_dim: int = Field(
        64, gt=0, description="Base dimension of filters in convolutional layers. Must be greater than zero."
    )
    n_blocks: int = Field(
        2, gt=0, description="Number of convolutional blocks in the discriminator. Must be greater than zero."
    )
    use_batch_norm: bool = Field(
        True, description="Indicates whether to use batch normalization in intermediate layers."
    )
    is_critic: bool = Field(
        False, description="Indicates if the discriminator is a critic (without sigmoid at the output)."
    )

    @field_validator("n_blocks")
    @classmethod
    def validate_n_blocks(cls, v: int) -> int:
        """Ensure the number of blocks is reasonable for model complexity."""
        if v < 1:
            raise ValueError("Number of blocks must be at least 1.")
        return v


class ConfigBinary(BaseModel):
    """Configuration for binary classes."""

    pos: int = Field(..., description="Binary classification with pos class.")
    neg: int = Field(..., description="Binary classification with neg class.")


class ConfigDatasetParams(BaseModel):
    """Params from config for the dataset."""

    name: DatasetNames = Field(..., description="Name of the dataset.")
    binary: ConfigBinary = Field(..., description="Binary classification with pos and neg classes.")


class ConfigArchitecture(BaseModel):
    """Configuration for model architecture."""

    name: ArchitectureType = Field(
        ..., description="Architecture type, such as dcgan, dcgan_deprecated, or resnet_deprecated."
    )
    g_filter_dim: int = Field(..., description="Filter dimension for the generator.")
    d_filter_dim: int = Field(..., description="Filter dimension for the discriminator.")
    g_num_blocks: int = Field(..., description="Number of generator blocks (for DCGAN).")
    d_num_blocks: int = Field(..., description="Number of discriminator blocks (for DCGAN).")


class ConfigLoss(BaseModel):
    """Configuration for loss."""

    name: LossType = Field(..., description="Loss function name, e.g., wgan-gp or ns.")


class ConfigLossWG(ConfigLoss):
    """Configuration for WGAN-GP loss."""

    args: float = Field(..., description="Arguments for the loss function for WGAN-GP.")


class ConfigModel(BaseModel):
    """Configuration for model."""

    z_dim: int = Field(..., description="Dimension of the latent code.")
    architecture: ConfigArchitecture = Field(..., description="Architecture configuration.")
    loss: ConfigLoss | ConfigLossWG = Field(..., description="Loss function configuration.")


class ConfigOptimizer(BaseModel):
    """Configuration for the optimizer."""

    lr: float = Field(..., description="Learning rate.")
    beta1: float | int = Field(..., description="Beta1 value for the optimizer.")
    beta2: float | int = Field(..., description="Beta2 value for the optimizer.")


class ConfigStep1(BaseModel):
    """Configuration for each training step 1."""

    epochs: int | None = Field(None, description="Number of epochs for training.")
    checkpoint_every: int | None = Field(None, description="Frequency of checkpointing during training.")
    batch_size: int | None = Field(None, description="Batch size for training.")
    disc_iters: int | None = Field(None, description="Number of discriminator iterations per generator iteration.")
    early_stop: tuple[str, int] | None = Field(None, description="Early stop criteria for GAN training")


class ConfigKLDiv(BaseModel):
    """Configuration for KL-Div."""

    alpha: list[float] = Field(..., description="Alpha config for KL-Div.")


class ConfigGaussian(BaseModel):
    """Configuration for Gaussian."""

    alpha: float = Field(..., description="Alpha config for Gaussian.")
    var: float = Field(..., description="Varience config for Gaussian.")


class ConfigCD(BaseModel):
    """Configuration for ConfusionDistance."""

    alpha: list[float] = Field(..., description="Alpha config for CD loss.")


class ConfigMGDA(BaseModel):
    """Configuration for ConfusionDistance."""

    norm: list[bool] | None = Field(False, description="Enable normalization for MGDA loss.")


class ConfigWeights(BaseModel):
    """Configuration for weigths."""

    kldiv: ConfigKLDiv | None = Field(None, description="Config for KL-Div.")
    gaussian: list[ConfigGaussian] | None = Field(None, description="Config for Gaussian.")
    gaussian_v2: list[ConfigGaussian] | None = Field(None, description="Config for Gaussian-identity (no-output).")
    cd: ConfigCD | None = Field(None, description="Config for CD loss.")
    mgda: ConfigMGDA | None = Field(None, description="Config for MGDA loss.")


class ConfigStep2(BaseModel):
    """Configuration for training step 2."""

    epochs: int | None = Field(None, description="Number of epochs for training.")
    checkpoint_every: int | None = Field(None, description="Frequency of checkpointing during training.")
    batch_size: int | None = Field(None, description="Batch size for training.")
    disc_iters: int | None = Field(None, description="Number of discriminator iterations per generator iteration.")
    classifier: list[str] = Field(..., description="Paths to classifier checkpoints.")
    weight: list[ConfigWeights] = Field(..., description="Weights for step-2 training.")
    step_1_epochs: list[int] | None = Field(None, description="GAN's to use")

    @field_validator("classifier", mode="before")
    @classmethod
    def validate_classifier_paths(cls, value: str) -> str:
        """Validate classifier path exists."""
        if isinstance(value, str):
            full_path = os.path.join(os.environ["FILESDIR"], value)
            if not os.path.exists(full_path):
                raise ValueError(f"Classifier path does not exist: {full_path}")
            return full_path
        return value


class ConfigTrain(BaseModel):
    """Configuration for each step."""

    step_1: ConfigStep1 | str = Field(..., description="Configuration for step 1 training.")
    step_2: ConfigStep2 = Field(..., description="Configuration for step 2 training.")


class ConfigGAN(BaseModel):
    """Full configuration for GAN training."""

    project: str = Field(..., description="Project name.")
    name: str = Field(..., description="Run name.")
    out_dir: str = Field(..., description="Path to output directory.")
    data_dir: str = Field(..., description="Path to data directory.")
    fid_stats_path: str = Field(..., description="Path to FID statistics file.")
    fixed_noise: str | int = Field(..., description="Path to fixed noise or number of fixed samples.")
    test_noise: str = Field(..., description="Path to test noise file.")
    compute_fid: bool | None = Field(None, description="Flag to compute FID score.")
    device: DeviceType = Field(DeviceType.cuda, description="Device to use, e.g., cpu or cuda.")
    num_workers: int = Field(0, description="Number of workers for data loading.")
    num_runs: int = Field(1, description="Number of runs.")
    step_1_seeds: list[int] = Field(None, description="Random seeds for step 1.")
    step_2_seeds: list[int] | None = Field(None, description="Random seeds for step 2.")
    dataset: ConfigDatasetParams = Field(..., description="Dataset parameters.")
    model: ConfigModel = Field(..., description="Model parameters.")
    optimizer: ConfigOptimizer = Field(..., description="Optimizer parameters.")
    train: ConfigTrain = Field(..., description="Training configuration.")

    @model_validator(mode="after")
    def validate_seeds(self) -> dict:
        """Validate seed configuration for consistency with the number of runs."""
        num_runs = self.num_runs
        if self.step_1_seeds and len(self.step_1_seeds) != num_runs:
            raise ValueError("Number of seeds must match the number of runs for step_1_seeds.")
        if self.train and self.step_2_seeds and len(self.step_2_seeds) != num_runs:
            raise ValueError("Number of seeds must match the number of runs for step_2_seeds.")
        return self


class CLAmbigan(BaseModel):
    """Command-line arguments for AmbiGAN."""

    config_path: str = Field(..., description="Config file")


class FIDMetricsParams(BaseModel):
    """Metrics for calculating FID."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fid: FID | None = Field(None, description="List of FID values during training stage")
    focd: FID | None = Field(None, description="List of FOCD values during training stage")
    conf_dist: LossSecondTerm | None = Field(None, description="List of CD values during training stage")
    hubris: Hubris | None = Field(None, description="List of Hubris values during training stage")


class MetricsParams(BaseModel):
    """Pydantic model for MetricsLogger initialization parameters."""

    prefix: TrainingStage | None = Field(None, description="Prefix indicating training stage in metric")
    log_epoch: bool = Field(True, description="Flag indicating if epoc should be logged")
    g_loss: list[float] | None = Field(None, description="List of generator losses")
    d_loss: list[float] | None = Field(None, description="List of discriminator losses")
    fid: list[FIDMetricsParams] | None = Field(None, description="List of FID metrics")


class GANTrainArgs(TrainArgsBase):
    """Arguments for GAN training, extending base training arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    G: nn.Module
    g_opt: optim.Optimizer
    g_crit: GeneratorLoss | None = None
    g_updater: UpdateGenerator
    D: nn.Module
    d_opt: optim.Optimizer
    d_crit: DiscriminatorLoss
    test_noise: Tensor
    fid_metrics: FIDMetricsParams
    n_disc_iters: int
    early_stop: tuple[str, int] | None = None
    start_early_stop_when: tuple[str, int] | None = None
    checkpoint_dir: str | None = None
    checkpoint_every: int = 10
    fixed_noise: Tensor | None = None
    c_out_hist: Any | None = None
    classifier: nn.Module | None = None
    dataset: Dataset


class Step1TrainingArgs(GANTrainArgs):
    """Extra arguments for step1 of GAN Training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    img_size: ImageParams
    run_id: int
    test_noise_conf: dict

class Step2TrainingArgs(GANTrainArgs):
    """Extra arguments for step2 of GAN Training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    s1_epoch: str
    c_name: str
    gan_path: str
    weight: tuple[str, UpdateGenerator]
    run_id: int



class TrainingState(BaseModel):
    """Training state model to track training progress."""

    epoch: int = 0
    early_stop_tracker: int = 0
    best_epoch: int = 0
    best_epoch_metric: float = float("inf")
    pre_early_stop_tracker: int | None = None
    pre_early_stop_metric: float | None = None


class CheckpointGAN(BaseModel):
    """Constructing GAN from checkpoint params."""

    config: ConfigGAN
    gen_params: GenParams
    dis_params: DisParams


class CLFIDArgs(DatasetClasses):
    """Pydantic model for validating FID CLI arguments."""

    model_config = ConfigDict(protected_namespaces=())
    dataroot: str = Field(default=f"{os.environ['FILESDIR']}/data", description="Directory with dataset")
    batch_size: int = Field(default=64, description="Batch size to use")
    model_path: str | None = Field(default=None, description="Path to classifier. If None, uses InceptionV3")
    num_workers: int = Field(default=6, description="Number of worker processes for data loading")
    device: DeviceType = Field(default=DeviceType.cpu, description="Device to use (cuda, cuda:0, or cpu)")
    name: str | None = Field(default=None, description="Name of generated .npz file")

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str | None) -> str | None:
        """Ensure model path exists if provided."""
        if value is not None and not os.path.exists(value):
            raise ValueError(f"Model path '{value}' does not exist.")
        return value

class CLTestNoiseArgs(BaseModel):
    seed: int | None =  Field(None, description="Random seed for reproducibility")
    nz: int = Field(..., description="Number of sample to be generated")
    z_dim: int = Field(..., description="Latent space dimension")
    out_dir: str = Field(f"{os.environ['FILESDIR']}/data/z", description="Directory to store thes test noise")


class CLFIDStatsArgs(BaseModel):
    dataroot: str = Field(default=f"{os.environ.get('FILESDIR', '')}/data", description="Directory with dataset",)
    dataset: DatasetNames = Field(default=DatasetNames.mnist, description="Dataset to use (mnist, fashion-mnist, cifar10 or chest x-ray)")
    device: DeviceType = Field(default=DeviceType.cpu, description="Device to use, cuda or cpu")
    n_classes: int = Field(None, description="Number of classes in the dataset")

    @model_validator(mode="after")
    def set_n_classes_based_on_dataset(self) -> "CLFIDStatsArgs":
        """Set the number of classes based on the dataset."""
        dataset_class_mapping = {
            DatasetNames.mnist: len(MnistClasses),
            DatasetNames.fashion_mnist: len(MnistClasses),
            DatasetNames.cifar10: len(Cifar10Classes),
            DatasetNames.chest_xray: len(ChestXrayClasses)
        }

        # Set the number of classes based on the dataset provided
        if self.dataset in dataset_class_mapping:
            self.n_classes = dataset_class_mapping[self.dataset]
        else:
            raise ValueError(f"Dataset '{self.dataset}' is not supported.")
        
        return self

