"""Test load dataset module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.load import get_function, load_dataset
from src.datasets.utils import BinaryDataset
from src.enums import DatasetNames
from src.models import DatasetParams, ImageParams, LoadDatasetParams


class MockDataset:
    """Mock dataset for test."""

    def __init__(self, size: int = 100, channels: int = 1, height: int = 28, width: int = 28) -> None:
        """Init function for mock dataset."""
        self.size = size
        # Create mock data with the appropriate shape
        self.data = torch.rand(
            (size, channels, height, width)
        )  # Creating a tensor of shape [size, channels, height, width]
        self.targets = torch.randint(0, 2, (size,))  # Random binary labels for demonstration purposes

    def __len__(self) -> int:
        """Return len of mock dataset."""
        return self.size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a tuple (tensor, label) to simulate a dataset item."""
        return self.data[idx], self.targets[idx]


@pytest.fixture
def dataset_params(tmp_path: str) -> LoadDatasetParams:
    """Fixture to provide dataset parameters for different dataset names."""
    return LoadDatasetParams(
        dataroot=str(tmp_path), train=True, dataset_name=DatasetNames.mnist, pos_class=7, neg_class=1, pytesting=True
    )


# Tests for get_function


@pytest.mark.parametrize(
    "dataset_name, expected_function_name",
    [
        (DatasetNames.mnist, "get_mnist"),
        (DatasetNames.fashion_mnist, "get_fashion_mnist"),
        (DatasetNames.cifar10, "get_cifar10"),
        (DatasetNames.chest_xray, "get_chest_xray"),
    ],
)
def test_get_function(dataset_name: DatasetNames, expected_function_name: str) -> None:
    """Test the retrieval of dataset loading function for different datasets."""
    func = get_function(dataset_name)
    assert func.__name__ == expected_function_name, f"Expected {expected_function_name}, but got {func.__name__}"


# Tests for load_dataset


@pytest.mark.parametrize(
    "dataset_name",
    [
        DatasetNames.mnist,
        DatasetNames.fashion_mnist,
        DatasetNames.cifar10,
    ],
)
def test_load_standard_dataset(dataset_params: LoadDatasetParams, dataset_name: DatasetNames) -> None:
    """Test loading standard datasets like MNIST, Fashion MNIST, and CIFAR-10."""
    dataset_params.dataset_name = dataset_name
    dataset, num_classes, image_size = load_dataset(dataset_params)

    assert isinstance(dataset, Dataset), "Returned dataset should be a Dataset instance"
    assert len(dataset) > 0, "Dataset should not be empty"
    assert num_classes > 0, "Number of classes should be greater than zero"
    assert (
        isinstance(image_size, ImageParams) and len(image_size.image_size) == 3
    ), "Image size should be a ImageParams, tuple of length 3"


@pytest.mark.parametrize(
    "dataset_name",
    [
        DatasetNames.mnist,
        DatasetNames.fashion_mnist,
        DatasetNames.cifar10,
    ],
)
def test_load_binary_dataset(dataset_params: LoadDatasetParams, dataset_name: DatasetNames) -> None:
    """Test loading and converting a dataset for binary classification."""
    dataset_params.dataset_name = dataset_name
    dataset_params.pos_class = 0  # Assuming class 0 is valid
    dataset_params.neg_class = 1  # Assuming class 1 is valid

    dataset, num_classes, image_size = load_dataset(dataset_params)

    assert isinstance(dataset, Dataset), "Returned dataset should be a Dataset instance"
    assert len(dataset) > 0, "Dataset should not be empty"
    assert num_classes == 2, "Number of classes should be 2 for binary classification"
    assert (
        isinstance(image_size, ImageParams) and len(image_size.image_size) == 3
    ), "Image size should be a ImageParams, tuple of length 3"


def test_load_invalid_dataset(dataset_params: LoadDatasetParams) -> None:
    """Test that an invalid dataset name raises a ValueError."""
    dataset_params.dataset_name = "invalid_dataset_name"

    with pytest.raises(ValueError, match="Dataset 'invalid_dataset_name' is not supported."):
        load_dataset(dataset_params)


@patch("src.datasets.load.get_function")
def test_load_dataset_function_call(mock_get_function: MagicMock, dataset_params: LoadDatasetParams) -> None:
    """Test that the correct loading function is called when loading a dataset."""
    # Set up the mock to return a fake dataset
    mock_dataset = MockDataset(size=10, channels=1, height=28, width=28)  # Fake dataset
    mock_get_function.return_value = MagicMock(return_value=mock_dataset)

    dataset_params.dataset_name = DatasetNames.mnist

    # Call load_dataset
    dataset, num_classes, _ = load_dataset(dataset_params)

    # Verify the expected call
    expected_params = DatasetParams(
        dataroot=dataset_params.dataroot, train=dataset_params.train, pytesting=dataset_params.pytesting
    )
    mock_get_function.assert_called_once_with(dataset_params.dataset_name)
    download_function = mock_get_function.return_value
    download_function.assert_called_once_with(expected_params)

    # Ensure the mock dataset function returned was called with the provided dataset_params
    assert isinstance(dataset, BinaryDataset)  # Mock is used as a dummy for Dataset type
    assert num_classes == 2  # Since we use a BinaryDataset, num_classes should be set to 2


def test_load_chest_xray_dataset(dataset_params: LoadDatasetParams) -> None:
    """Test loading the Chest X-ray dataset with 10% fraction for pytest."""
    dataset_params.dataset_name = DatasetNames.chest_xray
    dataset_params.pytesting = True
    dataset_params.pos_class = 1
    dataset_params.neg_class = 0

    dataset, num_classes, image_size = load_dataset(dataset_params)

    assert isinstance(dataset, Dataset), "Returned dataset should be a Dataset instance"
    assert len(dataset) > 0, "Dataset should not be empty"
    assert num_classes > 0, "Number of classes should be greater than zero"
    assert (
        isinstance(image_size, ImageParams) and len(image_size.image_size) == 3
    ), "Image size should be a ImageParams, tuple of length 3"


@pytest.mark.parametrize(
    "dataset_name",
    [
        DatasetNames.mnist,
        DatasetNames.fashion_mnist,
        DatasetNames.cifar10,
        DatasetNames.chest_xray,
    ],
)
def test_load_dataset_classes_range(dataset_params: LoadDatasetParams, dataset_name: DatasetNames) -> None:
    """Test that the pos_class and neg_class values are within the valid range for each dataset."""
    with pytest.raises(ValueError, match=r".*Invalid pos_class.*|.*Invalid neg_class.*"):
        load_dataset(
            LoadDatasetParams(
                dataset_name=dataset_name,
                pos_class=-1,  # Invalid class index
                neg_class=-2,  # Invalid class index
                pytesting=dataset_params.pytesting,
                dataroot=dataset_params.dataroot,
                train=dataset_params.train,
            )
        )


# Test loading MNIST with incorrect positive and negative class inputs
@pytest.mark.parametrize(
    "dataset_name",
    [
        DatasetNames.mnist,
        DatasetNames.fashion_mnist,
        DatasetNames.cifar10,
    ],
)
def test_invalid_pos_neg_classes(dataset_params: LoadDatasetParams, dataset_name: DatasetNames) -> None:
    """Test if loading the dataset with invalid positive and negative class indices raises an error."""
    with pytest.raises(ValueError, match=r".*Invalid pos_class.*|.*Invalid neg_class.*"):
        load_dataset(
            LoadDatasetParams(
                dataset_name=dataset_name,
                pos_class=1000,  # Invalid class index
                neg_class=2000,  # Invalid class index
                pytesting=dataset_params.pytesting,
                dataroot=dataset_params.dataroot,
                train=dataset_params.train,
            )
        )
