import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.classifier_cli import parse_args
from src.classifier.train_classifier import main, train
from src.enums import ClassifierType, DeviceType
from src.metrics.accuracy import binary_accuracy
from src.models import ClassifierParams, CLTrainArgs, TrainClassifierArgs
from tests.test_train_classifier import MockClassifier


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
            return [output, output]
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


def test_train(mock_dataloader: DataLoader, mock_classifier: MockClassifier, tmp_path: str) -> None:
    """Test train function to ensure the model trains correctly."""

    def mock_parse_args() -> CLTrainArgs:
        return CLTrainArgs(
            data_dir=str(tmp_path),
            out_dir=str(tmp_path),
            dataset_name="mnist",
            device=DeviceType.cpu,
            pos_class=7,
            neg_class=1,
        )

    criterion: nn.Module = nn.BCELoss()
    acc_fun = binary_accuracy
    args_train_classifier = TrainClassifierArgs(
        type=ClassifierType.mlp,
        img_size=(3, 28, 28),
        n_classes=2,
        nf=64,
        epochs=1,
        early_acc=1.0,
        device=DeviceType.cpu,
        out_dir=str(tmp_path),
    )

    stats, cp_path = train(
        mock_classifier,
        criterion,
        mock_dataloader,
        mock_dataloader,
        acc_fun,
        train_classifier_args=args_train_classifier,
        cl_args=mock_parse_args(),
    )

    assert isinstance(stats.train_acc, list), "Train accuracy should be a list"
    assert isinstance(cp_path, str), "Checkpoint path should be a string"


@pytest.mark.parametrize("device", [DeviceType.cpu, DeviceType.cuda])
def test_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: str,
    device: DeviceType,
    mock_dataloader: DataLoader,
    mock_classifier: MockClassifier,
) -> None:
    """Test main function to ensure all training steps execute correctly."""
    # Skip the test if CUDA is not available and the device is set to CUDA
    if device == DeviceType.cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping CUDA tests.")

    def mock_parse_args() -> CLTrainArgs:
        return CLTrainArgs(
            data_dir=str(tmp_path), out_dir=str(tmp_path), dataset_name="mnist", device=device, pos_class=7, neg_class=1
        )

    monkeypatch.setattr("src.classifier.classifier_cli.parse_args", mock_parse_args)

    def mock_load_dataset(*args, **kwargs) -> tuple[TensorDataset, int, tuple[int, int, int]]:
        dataset = mock_dataloader.dataset
        return dataset, 2, (3, 28, 28)

    monkeypatch.setattr("src.classifier.classifier_cli.load_dataset", mock_load_dataset)
    monkeypatch.setattr("src.classifier.classifier_cli.setup_reprod", lambda x: None)
    monkeypatch.setattr(
        "src.classifier.classifier_cli.construct_classifier_from_checkpoint",
        lambda *args, **kwargs: [mock_classifier],
    )
    monkeypatch.setattr(
        "src.classifier.classifier_cli.construct_classifier",
        lambda *args, **kwargs: mock_classifier,
    )
    monkeypatch.setattr("src.classifier.classifier_cli.DataLoader", lambda *args, **kwargs: mock_dataloader)
    monkeypatch.setattr("src.utils.checkpoint.checkpoint", lambda *args, **kwargs: str(tmp_path))

    main()

    assert os.path.exists(tmp_path), "The output directory should be created"


def test_parse_args_valid():
    """Test parse_args function with valid command-line arguments."""
    test_args = [
        "train_classifier.py",
        "--data_dir",
        "/some/data/dir",
        "--out_dir",
        "/some/output/dir",
        "--dataset_name",
        "mnist",
        "--pos_class",
        "0",
        "--neg_class",
        "1",
        "--batch_size",
        "32",
        "--epochs",
        "10",
        "--c_type",
        "cnn",
        "--device",
        "cuda",
    ]

    with patch("sys.argv", test_args):
        args = parse_args()
        assert args.data_dir == "/some/data/dir"
        assert args.out_dir == "/some/output/dir"
        assert args.dataset_name == "mnist"
        assert args.pos_class == 0
        assert args.neg_class == 1
        assert args.batch_size == 32
        assert args.epochs == 10
        assert args.c_type == "cnn"
        assert args.device == "cuda"


def test_parse_args_missing_required():
    """Test parse_args function to ensure it raises ValidationError on missing required arguments."""
    test_args = ["train_classifier.py", "--data_dir", "/some/data/dir"]  # Missing several required arguments

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            parse_args()


def test_parse_args_invalid_type():
    """Test parse_args function to ensure it raises ValidationError on invalid argument type."""
    test_args = [
        "train_classifier.py",
        "--data_dir",
        "/some/data/dir",
        "--out_dir",
        "/some/output/dir",
        "--dataset_name",
        "mnist",
        "--pos_class",
        "0",
        "--neg_class",
        "1",
        "--batch_size",
        "invalid_batch_size",  # This should be an integer
        "--epochs",
        "10",
        "--c_type",
        "cnn",
        "--device",
        "cuda",
    ]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            parse_args()
