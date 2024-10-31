"""Module to test dataset retrieval functions."""

import pytest
import torch
from torch.utils.data import Dataset

from src.datasets.datasets import (
    get_chest_xray,
    get_cifar10,
    get_fashion_mnist,
    get_mnist,
)
from src.models import DatasetParams


@pytest.fixture
def dataset_params(tmp_path: str) -> DatasetParams:
    """Fixture to provide default dataset parameters using pytest tmp_path."""
    return DatasetParams(dataroot=str(tmp_path), train=True, pytesting=True)


def test_get_mnist(dataset_params: DatasetParams) -> None:
    """Test retrieval of the MNIST dataset."""
    dataset = get_mnist(dataset_params)
    assert isinstance(dataset, Dataset), "Returned object should be a Dataset"
    assert len(dataset) > 0, "Dataset should not be empty"


def test_get_fashion_mnist(dataset_params: DatasetParams) -> None:
    """Test retrieval of the Fashion-MNIST dataset."""
    dataset = get_fashion_mnist(dataset_params)
    assert isinstance(dataset, Dataset), "Returned object should be a Dataset"
    assert len(dataset) > 0, "Dataset should not be empty"


def test_get_cifar10(dataset_params: DatasetParams) -> None:
    """Test retrieval of the CIFAR-10 dataset."""
    dataset = get_cifar10(dataset_params)
    assert isinstance(dataset, Dataset), "Returned object should be a Dataset"
    assert len(dataset) > 0, "Dataset should not be empty"


def test_get_chest_xray(dataset_params: DatasetParams) -> None:
    """Test retrieval of the Chest X-ray dataset."""
    dataset = get_chest_xray(dataset_params)
    assert isinstance(dataset, Dataset), "Returned object should be a Dataset"
    assert len(dataset) > 0, "Dataset should not be empty"
    # Test if data and targets properties are accessible
    assert isinstance(dataset.data, torch.Tensor), "Dataset data should be a tensor"
    assert isinstance(dataset.targets, torch.Tensor), "Dataset targets should be a tensor"


def test_chest_xray_dataset_data_and_targets(dataset_params: DatasetParams) -> None:
    """Test data and targets properties of Chest X-ray dataset."""
    dataset = get_chest_xray(dataset_params)
    data = dataset.data
    targets = dataset.targets
    assert isinstance(data, torch.Tensor), "Data property should return a torch tensor"
    assert isinstance(targets, torch.Tensor), "Targets property should return a torch tensor"
    assert len(data) == len(targets), "Data and targets should have the same length"
