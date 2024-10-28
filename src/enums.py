"""Module for Enums."""

from enum import Enum


class ClassifierType(str, Enum):
    """Enumerate different types of classifiers."""

    cnn = "cnn"
    mlp = "mlp"
    ensemble = "ensemble"


class EnsembleType(str, Enum):
    """Enumerate different types of ensembles of classifiers."""

    pretrained = "pretrained"
    cnn = "cnn"


class OutputMethod(str, Enum):
    """Enumerate different types of output methods from classifiers."""

    meta_learner = "meta-learner"
    mean = "mean"
    linear = "linear"
    identity = "identity"


class DeviceType(str, Enum):
    """Enumerate different types of devices."""

    cpu = "cpu"
    cuda = "cuda"


class TrainingStage(str, Enum):
    """Enumerate different types of training stages."""

    train = "train"
    optimize = "optimize"
    test = "test"
    validation = "validation"


class DatasetNames(str, Enum):
    """Enumeration of supported datasets."""

    mnist = "mnist"
    fashion_mnist = "fashion-mnist"
    cifar10 = "cifar10"
    chest_xray = "chest-xray"

    @classmethod
    def valid_dataset(cls, name: str) -> bool:
        """Validate if the dataset name is supported and return True if valid, else raise ValueError."""
        if name not in cls._value2member_map_:
            raise ValueError(
                f"Dataset '{name}' is not supported. Available datasets are: {list(cls._value2member_map_.keys())}"
            )
        return True


class MnistClasses(int, Enum):
    """Valid classes for MNIST and Fashion-MNIST datasets."""

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


class Cifar10Classes(int, Enum):
    """Valid classes for CIFAR-10 dataset."""

    AIRPLANE = 0
    AUTOMOBILE = 1
    BIRD = 2
    CAT = 3
    DEER = 4
    DOG = 5
    FROG = 6
    HORSE = 7
    SHIP = 8
    TRUCK = 9


class ChestXrayClasses(int, Enum):
    """Valid classes for Chest-Xray dataset."""

    PNEUMONIA = 0
    NORMAL = 1


class ArchitectureType(str, Enum):
    """Valid names for Architecture types."""

    dcgan = "dcgan"
    resnet_deprecated = "resnet_deprecated"
    dcgan_deprecated = "dcgan_deprecated"


class LossType(str, Enum):
    """Valid names for Loss types."""

    ns = "ns"
    wgan = "wgan-gp"

class WeightType(str, Enum):
    """Valid weight names"""

    kldiv = "kldiv"
    gaussian = "gaussian"
    gaussian_v2 = "gaussian_v2"
    cd = "cd"
    mgda = "mgda"