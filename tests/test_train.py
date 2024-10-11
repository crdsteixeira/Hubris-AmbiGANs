import os
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.train import (default_train_fn, evaluate, main,
                                  save_predictions, train)
from src.metrics.accuracy import binary_accuracy
from src.models import (ClassifierParams, ClassifierType, CLTrainArgs,
                        DefaultTrainParams, DeviceType, EvaluateParams,
                        TrainArgs, TrainingStage)


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
    def __init__(self) -> None:
        """Initialize a mock classifier model."""
        super(MockClassifier, self).__init__()
        self.linear: nn.Linear = nn.Linear(28 * 28 * 3, 1)
        self.train_models: bool = False
        self.optimize: bool = False

    def forward(
        self, x: torch.Tensor, output_feature_maps: bool = False
    ) -> torch.Tensor:
        """Perform forward pass through the mock classifier."""
        x: torch.Tensor = x.view(x.size(0), -1)
        print(f"Input to linear layer shape: {x.shape}")
        output: torch.Tensor = torch.sigmoid(self.linear(x)).view(-1, 1)
        if output_feature_maps:
            return [output, output]
        return output

    @property
    def m_val(self) -> bool:
        """Return whether multiple models are available."""
        return False


# Test for the `evaluate` function
def test_evaluate(mock_dataloader: DataLoader) -> None:
    """Test evaluate function returns accuracy and loss as floats."""
    model: MockClassifier = MockClassifier()
    criterion: nn.Module = nn.BCELoss()
    acc_fun = binary_accuracy
    params: EvaluateParams = EvaluateParams(
        device=DeviceType.cpu, verbose=False, desc="Testing"
    )

    acc, loss = evaluate(model, mock_dataloader, criterion, acc_fun, params)
    assert isinstance(acc, float), "Accuracy should be a float"
    assert isinstance(loss, float), "Loss should be a float"


# Test for the `default_train_fn`
def test_default_train_fn(mock_dataloader: DataLoader) -> None:
    """Test default_train_fn to ensure it computes gradients and accuracy."""
    model: MockClassifier = MockClassifier()
    criterion: nn.Module = nn.BCELoss()
    acc_fun = binary_accuracy
    params: DefaultTrainParams = DefaultTrainParams(
        early_acc=0.9, device=DeviceType.cpu
    )

    for X, Y in mock_dataloader:
        X, Y = X.to(DeviceType.cpu.value), Y.to(DeviceType.cpu.value)
        loss, acc = default_train_fn(model, X, Y, criterion, acc_fun, params)
        assert loss.requires_grad, "Loss should require gradients"
        assert isinstance(
            acc, torch.Tensor
        ), "Accuracy should be a Tensor (torch.float)"


# Test for the `train` function
def test_train(mock_dataloader: DataLoader, tmp_path: str) -> None:
    """Test train function to ensure the model trains correctly."""
    model: MockClassifier = MockClassifier()
    criterion: nn.Module = nn.BCELoss()
    acc_fun = binary_accuracy
    opt: optim.Optimizer = optim.Adam(model.parameters(), lr=0.001)
    args: TrainArgs = TrainArgs(epochs=1, early_acc=1.0, out_dir=str(tmp_path))
    model_params: ClassifierParams = ClassifierParams(
        type=ClassifierType.mlp,
        img_size=(3, 28, 28),
        nf=64,
        n_classes=2,
        device=DeviceType.cpu,
    )

    stats, cp_path = train(
        model,
        opt,
        criterion,
        mock_dataloader,
        mock_dataloader,
        mock_dataloader,
        acc_fun,
        args,
        name="mock_model",
        model_params=model_params,
        device=DeviceType.cpu,
    )

    assert isinstance(stats.train_acc, list), "Train accuracy should be a list"
    assert isinstance(cp_path, str), "Checkpoint path should be a string"


# Test for `save_predictions`
def test_save_predictions(mock_dataloader: DataLoader, tmp_path: str) -> None:
    """Test save_predictions function to ensure predictions are saved properly."""
    model: MockClassifier = MockClassifier()
    device: DeviceType = DeviceType.cpu
    dataset_name: TrainingStage = TrainingStage.test
    cp_path: str = str(tmp_path)

    dataloader: DataLoader = mock_dataloader
    save_predictions(model, dataloader, device, dataset_name, cp_path)

    assert os.path.exists(
        os.path.join(cp_path, "test_y_hat.npy")
    ), "Numpy file should be saved"
    assert os.path.exists(
        os.path.join(cp_path, "test_y_hat.svg")
    ), "SVG file should be saved"


@pytest.mark.parametrize("device", [DeviceType.cpu, DeviceType.cuda])
def test_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: str,
    device: DeviceType,
    mock_dataloader: DataLoader,
) -> None:
    """Test main function to ensure all training steps execute correctly."""

    # Skip the test if CUDA is not available and the device is set to CUDA
    if device == DeviceType.cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping CUDA tests.")

    def mock_parse_args() -> CLTrainArgs:
        return CLTrainArgs(
            data_dir=str(tmp_path),
            out_dir=str(tmp_path),
            dataset_name="mnist",
            device=device,
        )

    monkeypatch.setattr("src.classifier.train.parse_args", mock_parse_args)

    def mock_load_dataset(
        *args, **kwargs
    ) -> Tuple[TensorDataset, int, Tuple[int, int, int]]:
        dataset = mock_dataloader.dataset
        return dataset, 2, (3, 28, 28)

    monkeypatch.setattr("src.classifier.train.load_dataset", mock_load_dataset)
    monkeypatch.setattr("src.classifier.train.setup_reprod", lambda x: None)
    monkeypatch.setattr(
        "src.classifier.train.construct_classifier_from_checkpoint",
        lambda *args, **kwargs: [MockClassifier()],
    )
    monkeypatch.setattr(
        "src.classifier.train.construct_classifier",
        lambda *args, **kwargs: MockClassifier(),
    )
    monkeypatch.setattr(
        "src.classifier.train.DataLoader", lambda *args, **kwargs: mock_dataloader
    )
    monkeypatch.setattr(
        "src.utils.checkpoint.checkpoint", lambda *args, **kwargs: str(
            tmp_path)
    )

    main()

    assert os.path.exists(tmp_path), "The output directory should be created"
