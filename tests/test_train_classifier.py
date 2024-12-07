""""Test Train classifier."""

import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.train_classifier import default_train_fn, evaluate, save_predictions
from src.enums import ClassifierType, DeviceType, TrainingStage, DatasetNames
from src.metrics.accuracy import binary_accuracy
from src.models import ClassifierParams, EvaluateParams, TrainClassifierArgs


# Mock Dataset
@pytest.fixture
def mock_dataloader() -> DataLoader:
    """Fixture to create a mock DataLoader for testing."""
    X: torch.Tensor = torch.randn(10, 3, 28, 28)
    y: torch.Tensor = torch.randint(0, 2, (10, 1), dtype=torch.float)
    dataset: TensorDataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=5)


# Mock Classifier
class MockClassifier(nn.Module):
    """Mock classifier module."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize a mock classifier model."""
        super().__init__()
        self.params = params
        self.linear: nn.Linear = nn.Linear(28 * 28 * 3, 1)
        self.train_models: bool = False
        self.optimize: bool = False

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Perform forward pass through the mock classifier."""
        x = x.view(x.size(0), -1)
        output: torch.Tensor = torch.sigmoid(self.linear(x)).view(-1, 1)
        if output_feature_maps:
            return output, [output]
        return output

    @property
    def m_val(self) -> bool:
        """Return whether multiple models are available."""
        return False


# Fixture for MockClassifier with specific model parameters
@pytest.fixture
def mock_classifier() -> MockClassifier:
    """Fixture to create a mock classifier with predefined model parameters."""
    model_params: ClassifierParams = ClassifierParams(
        type=ClassifierType.mlp,
        img_size=(3, 28, 28),
        nf=64,
        n_classes=2,
        device=DeviceType.cpu,
    )
    return MockClassifier(model_params)


# Test for the `evaluate` function
def test_evaluate(mock_dataloader: DataLoader, mock_classifier: MockClassifier) -> None:
    """Test evaluate function returns accuracy and loss as floats."""
    criterion: nn.Module = nn.BCELoss()
    acc_fun = binary_accuracy
    params: EvaluateParams = EvaluateParams(device=DeviceType.cpu, verbose=False, desc="Testing")

    acc, loss = evaluate(mock_classifier, mock_dataloader, criterion, acc_fun, params)
    assert isinstance(acc, float), "Accuracy should be a float"
    assert isinstance(loss, float), "Loss should be a float"


# Test for the `default_train_fn`
def test_default_train_fn(mock_dataloader: DataLoader, mock_classifier: MockClassifier, tmp_path: str) -> None:
    """Test default_train_fn to ensure it computes gradients and accuracy."""
    criterion: nn.Module = nn.BCELoss()
    acc_fun = binary_accuracy
    args_train_classifier = TrainClassifierArgs(
        type=ClassifierType.mlp,
        img_size=(3, 28, 28),
        n_classes=2,
        nf=64,
        epochs=1,
        early_acc=0.9,
        device=DeviceType.cpu,
        out_dir=str(tmp_path),  # Example output directory
        dataset_name=DatasetNames.mnist,
        pos_class=1,
        neg_class=7,
    )

    for X, Y in mock_dataloader:
        loss, acc = default_train_fn(mock_classifier, X, Y, criterion, acc_fun, 1.0, args_train_classifier)
        assert loss.requires_grad, "Loss should require gradients"
        assert isinstance(acc, torch.Tensor), "Accuracy should be a Tensor (torch.float)"


# Test for `save_predictions`
def test_save_predictions(mock_dataloader: DataLoader, mock_classifier: MockClassifier, tmp_path: str) -> None:
    """Test save_predictions function to ensure predictions are saved properly."""
    args_train_classifier = TrainClassifierArgs(
        type=ClassifierType.mlp,
        img_size=(3, 28, 28),
        n_classes=2,
        nf=64,
        epochs=1,
        early_acc=1.0,
        device=DeviceType.cpu,
        out_dir=str(tmp_path),
        dataset_name=DatasetNames.mnist,
        pos_class=1,
        neg_class=7,
    )
    dataset_name: TrainingStage = TrainingStage.test
    cp_path: str = str(tmp_path)

    save_predictions(mock_classifier, mock_dataloader, args_train_classifier, dataset_name, cp_path)

    assert os.path.exists(os.path.join(cp_path, "test_y_hat.npy")), "Numpy file should be saved"
    assert os.path.exists(os.path.join(cp_path, "test_y_hat.svg")), "SVG file should be saved"
