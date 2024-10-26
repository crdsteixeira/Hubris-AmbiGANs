import pytest
import torch
from torch.utils.data import Dataset
from unittest.mock import MagicMock
from src.datasets.utils import BinaryDataset
from src.enums import DatasetNames
from src.models import LoadDatasetParams

class MockOriginalDataset(Dataset):
    """Mock dataset for testing purposes."""

    def __init__(self):
        # Create a simple dataset: ten examples of each class (0 and 1)
        self.data = [torch.tensor([[i]], dtype=torch.float32) for i in range(20)]
        self.targets = [1] * 10 + [7] * 10

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], self.targets[index]


@pytest.fixture
def mock_original_dataset() -> MockOriginalDataset:
    """Fixture for a mock original dataset for testing."""
    return MockOriginalDataset()


@pytest.fixture
def valid_dataset_params(tmp_path) -> LoadDatasetParams:
    """Fixture to provide valid dataset parameters for loading a binary dataset."""
    return LoadDatasetParams(
        dataroot=str(tmp_path),
        train=True,
        pytesting=True,
        dataset_name=DatasetNames.mnist,
        pos_class=7,
        neg_class=1,
    )


@pytest.fixture
def invalid_dataset_params(tmp_path) -> LoadDatasetParams:
    """Fixture to provide invalid dataset parameters that should result in an empty binary dataset."""
    return LoadDatasetParams(
        dataroot=str(tmp_path),
        train=True,
        pytesting=True,
        dataset_name=DatasetNames.mnist,
        pos_class=7,  
        neg_class=1,  
    )


def test_binary_dataset_initialization(mock_original_dataset: MockOriginalDataset, valid_dataset_params: LoadDatasetParams) -> None:
    """Test that the BinaryDataset initializes correctly when given valid classes."""
    binary_dataset = BinaryDataset(mock_original_dataset, valid_dataset_params)
    assert len(binary_dataset) == 20, "BinaryDataset length should match the filtered examples in the mock dataset."
    assert torch.equal(binary_dataset.targets, torch.hstack([torch.zeros(10), torch.ones(10)])), "BinaryDataset targets should match filtered targets."


def test_binary_dataset_initialization_empty(mock_original_dataset: MockOriginalDataset, invalid_dataset_params: LoadDatasetParams) -> None:
    """Test that initializing the BinaryDataset with unavailable classes raises a ValueError."""
    invalid_dataset_params.pos_class = 10 # Overwirte with nvalid class index not in the dataset
    invalid_dataset_params.neg_class = 11 # Overwirte with nvalid class index not in the dataset
    with pytest.raises(ValueError, match=r"The dataset .* is empty.*"):
        BinaryDataset(mock_original_dataset, invalid_dataset_params)


def test_binary_dataset_length(mock_original_dataset: MockOriginalDataset, valid_dataset_params: LoadDatasetParams, tmp_path) -> None:
    """Test the length of BinaryDataset."""
    binary_dataset = BinaryDataset(mock_original_dataset, valid_dataset_params)
    assert len(binary_dataset) == 20, "The length of BinaryDataset should match the number of valid samples."


def test_binary_dataset_get_item(mock_original_dataset: MockOriginalDataset, valid_dataset_params: LoadDatasetParams) -> None:
    """Test that the __getitem__ method returns the correct data and label."""
    binary_dataset = BinaryDataset(mock_original_dataset, valid_dataset_params)
    sample, label = binary_dataset[0]
    
    expected_sample = mock_original_dataset.data[0]
    assert torch.equal(sample, expected_sample), f"The data returned should match the corresponding original data. Expected {expected_sample}, but got {sample}"

    expected_label = torch.tensor(0.0)  # Since class 0 was set as the neg_class in the params
    assert torch.equal(label, expected_label), f"Expected label {expected_label}, but got {label}"