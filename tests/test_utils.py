"""Test for utils methods module."""

import pytest
import torch
from torch.utils.data import Dataset

from src.datasets.utils import BinaryDataset
from src.enums import DatasetNames
from src.models import LoadDatasetParams


class MockOriginalDataset(Dataset):
    """Mock dataset for testing purposes."""

    def __init__(self) -> None:
        """Create a simple dataset: ten examples of each class (0 and 1)."""
        self.data = [torch.tensor([[i]], dtype=torch.float32) for i in range(20)]
        self.targets = [1] * 10 + [7] * 10

    def __len__(self) -> int:
        """Get number of samples in mock dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Get a specific item from mock dataset."""
        return self.data[index], self.targets[index]


@pytest.fixture
def mock_original_dataset() -> MockOriginalDataset:
    """Fixture for a mock original dataset for testing."""
    return MockOriginalDataset()


@pytest.fixture
def valid_dataset_params(tmp_path: str) -> LoadDatasetParams:
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
def invalid_dataset_params(tmp_path: str) -> LoadDatasetParams:
    """Fixture to provide invalid dataset parameters that should result in an empty binary dataset."""
    return LoadDatasetParams(
        dataroot=str(tmp_path),
        train=True,
        pytesting=True,
        dataset_name=DatasetNames.mnist,
        pos_class=7,
        neg_class=1,
    )


def test_binary_dataset_initialization(
    mock_original_dataset: MockOriginalDataset, valid_dataset_params: LoadDatasetParams
) -> None:
    """Test that the BinaryDataset initializes correctly when given valid classes."""
    binary_dataset = BinaryDataset(mock_original_dataset, valid_dataset_params)
    assert len(binary_dataset) == 20, "BinaryDataset length should match the filtered examples in the mock dataset."
    assert torch.equal(
        binary_dataset.targets, torch.hstack([torch.zeros(10), torch.ones(10)])
    ), "BinaryDataset targets should match filtered targets."


def test_binary_dataset_initialization_empty(
    mock_original_dataset: MockOriginalDataset, invalid_dataset_params: LoadDatasetParams
) -> None:
    """Test that initializing the BinaryDataset with unavailable classes raises a ValueError."""
    invalid_dataset_params.pos_class = 10  # Overwirte with nvalid class index not in the dataset
    invalid_dataset_params.neg_class = 11  # Overwirte with nvalid class index not in the dataset
    with pytest.raises(ValueError, match=r"The dataset .* is empty.*"):
        BinaryDataset(mock_original_dataset, invalid_dataset_params)


def test_binary_dataset_length(
    mock_original_dataset: MockOriginalDataset, valid_dataset_params: LoadDatasetParams, tmp_path: str
) -> None:
    """Test the length of BinaryDataset."""
    binary_dataset = BinaryDataset(mock_original_dataset, valid_dataset_params)
    assert len(binary_dataset) == 20, "The length of BinaryDataset should match the number of valid samples."


def test_binary_dataset_get_item(
    mock_original_dataset: MockOriginalDataset, valid_dataset_params: LoadDatasetParams
) -> None:
    """Test that the __getitem__ method returns the correct data and label."""
    binary_dataset = BinaryDataset(mock_original_dataset, valid_dataset_params)
    sample, label = binary_dataset[0]

    expected_sample = mock_original_dataset.data[0]
    assert torch.equal(
        sample, expected_sample
    ), f"The data returned should match the corresponding original data. Expected {expected_sample}, but got {sample}"

    expected_label = torch.tensor(0.0)  # Since class 0 was set as the neg_class in the params
    assert torch.equal(label, expected_label), f"Expected label {expected_label}, but got {label}"

def test_binary_dataset_balancing() -> None:
    """Test that the balance_dataset method correctly balances the dataset."""

    # Create an unbalanced mock dataset
    class UnbalancedMockDataset(Dataset):
        """Mock dataset with unbalanced classes for testing."""
        

        def __init__(self) -> None:
            """Initialize with 8 negatives (class 1) and 2 positives (class 7)."""
            self.data = [torch.tensor([[i]], dtype=torch.float32) for i in range(10)]
            self.targets = [1] * 8 + [7] * 2  # Majority class is 1, minority is 7

        def __len__(self) -> int:
            """Return the number of samples in the mock dataset."""
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            """Get a specific item from the mock dataset."""
            return self.data[index], self.targets[index]

    unbalanced_dataset = UnbalancedMockDataset()

    # Count positives and negatives before balancing
    original_targets = torch.tensor(
        [1.0 if t == 7 else 0.0 for t in unbalanced_dataset.targets]
    )
    num_pos_before = (original_targets == 1).sum().item()
    num_neg_before = (original_targets == 0).sum().item()

    # Define dataset parameters
    dataset_params = LoadDatasetParams(
        dataroot="",
        train=True,
        pytesting=True,
        dataset_name=DatasetNames.mnist,
        pos_class=7,
        neg_class=1,
    )

    # Initialize BinaryDataset (which balances the dataset)
    binary_dataset = BinaryDataset(unbalanced_dataset, dataset_params)

    # Count positives and negatives after balancing
    num_pos = (binary_dataset.targets == 1).sum().item()
    num_neg = (binary_dataset.targets == 0).sum().item()

    # Check that the dataset is now balanced
    assert num_pos == num_neg, f"Dataset is not balanced: {num_pos} positives, {num_neg} negatives."

    # Calculate expected length after balancing
    num_to_add = abs(num_pos_before - num_neg_before)
    expected_length = len(unbalanced_dataset) + num_to_add
    assert len(binary_dataset) == expected_length, f"Expected length {expected_length}, got {len(binary_dataset)}."

    # Verify that oversampled data are duplicates of minority class samples
    original_length = len(unbalanced_dataset)
    oversampled_data = binary_dataset.data[original_length:]
    original_minority_data = binary_dataset.data[(binary_dataset.targets == 1).nonzero(as_tuple=True)[0][:num_pos_before]]

    for sample in oversampled_data:
        assert any(torch.equal(sample, orig_sample) for orig_sample in original_minority_data), \
            "Oversampled data contains unexpected samples."
