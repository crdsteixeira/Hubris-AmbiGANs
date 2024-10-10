from pydantic import BaseModel, Field, model_validator
from typing import Tuple, List, Union, Optional
from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

class ClassifierType(str, Enum):
    cnn = 'cnn'
    mlp = 'mlp'
    ensemble = 'ensemble'


class EnsembleType(str, Enum):
    pretrained = 'pretrained'
    cnn = 'cnn'


class OutputMethod(str, Enum):
    meta_learner = 'meta-learner'
    mean = 'mean'
    linear = 'linear'
    identity = 'identity'


class DeviceType(str, Enum):
    cpu = 'cpu'
    cuda = 'cuda'

class TrainingStage(str, Enum):
    train = 'train'
    optimize = 'optimize'
    test = 'test'


class ClassifierParams(BaseModel):
    type: ClassifierType  # Only 'cnn', 'mlp', or 'ensemble'
    img_size: Tuple[int, int, int] = Field(..., description="Tuple of three integers for image size")
    n_classes: int = Field(..., description="Number of classes (should be >= 2)")
    nf: Optional[Union[int, List[int], List[List[int]]]] = Field(None, description="Number of filters for cnn/mlp. Can be None for pretrained ensembles.")
    ensemble_type: Optional[EnsembleType] = Field(None, description="Type of ensemble when applicable")
    output_method: Optional[OutputMethod] = Field(None, description="Output method for ensemble when applicable")
    device: DeviceType = Field(DeviceType.cpu, description="Device for computation ('cpu' or 'cuda')")  # Using Enum for device validation

    @model_validator(mode="after")
    def validate_n_classes(self):
        """Validates that n_classes is at least 2."""
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        return self

    @model_validator(mode="before")
    @classmethod
    def parse_ensemble_type(cls, values):
        """Parses and validates the ensemble type and output method from the classifier type string."""
        type_value = values.get('type')
        
        if isinstance(type_value, str) and type_value.startswith('ensemble:'):
            # Split and validate
            parts = type_value.split(':')
            if len(parts) != 3:
                raise ValueError("Invalid ensemble format. Must be 'ensemble:<ensemble_type>:<output_method>'")

            # Validate ensemble_type and output_method
            ensemble_type_str, output_method_str = parts[1], parts[2]
            try:
                values['ensemble_type'] = EnsembleType(ensemble_type_str)
                values['output_method'] = OutputMethod(output_method_str)
            except ValueError as e:
                raise ValueError(f"Invalid values for ensemble_type or output_method: {str(e)}")

            # Set type to 'ensemble'
            values['type'] = ClassifierType.ensemble

        return values

    @model_validator(mode="after")
    def check_ensemble_fields(self):
        """Validates that both ensemble_type and output_method are provided for ensemble type classifiers."""
        if self.type == ClassifierType.ensemble:
            if not self.ensemble_type or not self.output_method:
                raise ValueError("ensemble_type and output_method must be provided for ensemble type classifiers")
        return self

    @model_validator(mode="after")
    def check_img_size(self):
        """Validates img_size based on the classifier type and ensemble type."""
        img_size = self.img_size
        if self.type == ClassifierType.ensemble and self.ensemble_type == EnsembleType.pretrained:
            # Pretrained models accept 1-channel images
            if img_size[0] != 1:
                raise ValueError("Pretrained models in ensemble expect 1-channel images (got {img_size[0]} channels)")
        else:
            # Non-pretrained models expect 3-channel images
            if img_size[0] > 3:
                raise ValueError('img_size must be a tuple of three integers with max 3 channels for non-pretrained models')
        
        if len(img_size) != 3:
            raise ValueError('img_size must be a tuple of three integers')
        return self
    
    @model_validator(mode="after")
    def check_nf(self):
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
    

class PoolingParams(BaseModel):
    """
    Pydantic model to validate parameters for the pooling operation.
    """
    in_size: int = Field(..., description="Input size (height or width) for pooling operation.")
    kernel: int = Field(..., description="Kernel size for pooling or convolution.")
    dilation: int = Field(1, description="Dilation factor for pooling or convolution.")
    padding: int = Field(0, description="Padding applied to the input.")
    stride: Optional[int] = Field(None, description="Stride for pooling or convolution. Defaults to kernel size if not provided.")

    @model_validator(mode="after")
    def check_valid_values(self):
        """
        Validates that the input size, kernel, dilation, padding, and stride are all non-negative.
        """
        if self.in_size < 0 or self.kernel < 0 or self.dilation < 0 or self.padding < 0:
            raise ValueError("Input size, kernel, dilation, and padding must be non-negative.")
        if self.stride is not None and self.stride < 0:
            raise ValueError("Stride must be non-negative.")
        return self

class EvaluateParams(BaseModel):
    """Pydantic model for `evaluate` function parameters."""
    device: DeviceType = Field(..., description="Device for computation ('cpu', 'cuda').")
    verbose: bool = Field(True, description="Whether to show progress bar or not.")
    desc: str = Field('Validate', description="Description for the progress bar.")
    header: Optional[str] = Field(None, description="Optional header for logging.")

class DefaultTrainParams(BaseModel):
    """Pydantic model for default_train_fn parameters."""
    early_acc: float = Field(1.0, description="Early accuracy threshold for backpropagation.")
    device: DeviceType = Field(DeviceType.cuda, description="Device for computation ('cpu' or 'cuda').")

class TrainArgs(BaseModel):
    """Pydantic model for training arguments."""
    epochs: int = Field(..., description="Number of training epochs")
    early_stop: Optional[int] = Field(None, description="Early stopping criterion")
    early_acc: float = Field(1.0, description="Early accuracy threshold")
    out_dir: str = Field(..., description="Output directory for saving checkpoints")
    batch_size: int = Field(64, description="Batch size for training")

class TrainingStats(BaseModel):
    """Pydantic model for tracking training statistics."""
    best_loss: float = Field(float('inf'), description="The lowest validation loss encountered during training.")
    best_epoch: int = Field(0, description="Epoch with the lowest validation loss.")
    early_stop_tracker: int = Field(0, description="Counter for early stopping criteria.")
    cur_epoch: int = Field(0, description="Current epoch during training.")
    train_acc: List[float] = Field(default_factory=list, description="List of training accuracies for each epoch.")
    train_loss: List[float] = Field(default_factory=list, description="List of training losses for each epoch.")
    val_acc: List[float] = Field(default_factory=list, description="List of validation accuracies for each epoch.")
    val_loss: List[float] = Field(default_factory=list, description="List of validation losses for each epoch.")
    test_acc: List[float] = Field(default_factory=list, description="List of test accuracies for each epoch.")
    test_loss: List[float] = Field(default_factory=list, description="List of test losses for each epoch.")

class CLTrainArgs(BaseModel):
    """Pydantic model for parsing and validating command-line training arguments."""
    data_dir: str = Field(default=f"{os.environ['FILESDIR']}/data", description="Path to dataset")
    out_dir: str = Field(default=f"{os.environ['FILESDIR']}/models", description="Path to generated files")
    name: Optional[str] = Field(default=None, description="Name of the classifier for output files")
    dataset_name: str = Field(default='mnist', description="Dataset (mnist or fashion-mnist)")
    pos_class: int = Field(default=7, description="Positive class for binary classification")
    neg_class: int = Field(default=1, description="Negative class for binary classification")
    batch_size: int = Field(default=64, description="Batch size for training")
    c_type: str = Field(default='mlp', description="Classifier type ('cnn', 'mlp', or 'ensemble')")
    epochs: int = Field(default=2, description="Number of epochs to train for")
    early_stop: Optional[int] = Field(default=None, description="Early stopping criteria (optional)")
    early_acc: float = Field(default=1.0, description="Early accuracy threshold for backpropagation")
    lr: float = Field(default=5e-4, description="Learning rate for the optimizer")
    nf: str = Field(default="2", description="Number of filters or features in the model")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    device: DeviceType = Field(DeviceType.cuda, description="Device for computation ('cpu' or 'cuda')") 
