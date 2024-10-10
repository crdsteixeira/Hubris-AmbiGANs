import pytest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from src.classifier.ensemble import Ensemble
from src.models import ClassifierParams, DeviceType, OutputMethod, EnsembleType
from typing import Tuple, Union

# Mock classifiers for testing purposes
class MockCNN(nn.Module):
    def __init__(self, n_classes: int, *args, **kwargs) -> None:
        super(MockCNN, self).__init__()
        self.n_classes = 1 if n_classes == 2 else n_classes

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if output_feature_maps:
            return torch.randn(x.shape[0], 64, 64, 64, requires_grad=True), torch.randn(x.shape[0], self.n_classes, requires_grad=True)
        return torch.randn(x.shape[0], self.n_classes, requires_grad=True)

class MockPretrained(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MockPretrained, self).__init__()

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        return torch.randn(x.shape[0], 10)

# Test: Ensemble Initialization
def test_ensemble_cnn_initialization() -> None:
    """Test if CNN ensemble initializes correctly with valid params."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.mean,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    assert len(ensemble.models) == len(params.nf), "CNN ensemble should have the same number of models as nf length."

def test_ensemble_pretrained_initialization() -> None:
    """Test if pretrained ensemble initializes correctly."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(1, 64, 64),
        n_classes=10,
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    assert len(ensemble.models) == 2, "Pretrained ensemble should initialize with 2 models."
    assert not ensemble.train_models, "Pretrained models should not be set to train."

def test_ensemble_missing_ensemble_type_and_output_method():
    """Test that a ValueError is raised if ensemble_type or output_method are missing for ensemble classifiers."""
    
    # Case 1: Missing both ensemble_type and output_method
    with pytest.raises(ValueError, match="ensemble_type and output_method must be provided for ensemble type classifiers"):
        params = ClassifierParams(
            type='ensemble',
            img_size=(3, 64, 64),
            n_classes=10,
            nf=None,
            device=DeviceType.cpu
        )
    
    # Case 2: Missing only ensemble_type
    with pytest.raises(ValueError, match="ensemble_type and output_method must be provided for ensemble type classifiers"):
        params = ClassifierParams(
            type='ensemble',
            img_size=(3, 64, 64),
            n_classes=10,
            nf=None,
            output_method=OutputMethod.mean,  # output_method provided, but ensemble_type is missing
            device=DeviceType.cpu
        )

    # Case 3: Missing only output_method
    with pytest.raises(ValueError, match="output_method must be provided for ensemble type classifiers"):
        params = ClassifierParams(
            type='ensemble',
            img_size=(3, 64, 64),
            n_classes=10,
            nf=None,
            ensemble_type=EnsembleType.cnn,  # ensemble_type provided, but output_method is missing
            device=DeviceType.cpu
        )

# Test: Forward Pass
def test_ensemble_forward_mean() -> None:
    """Test forward pass with 'mean' output method."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.mean,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    x = torch.randn(1, 3, 64, 64)

    output = ensemble(x, output_feature_maps=False)
    assert output is not None, "Output should not be None"
    assert output.shape == (1, 10), "Output shape should be correct for classification."

def test_ensemble_forward_with_feature_maps() -> None:
    """Test forward pass with 'output_feature_maps' set to True."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.mean,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    x = torch.randn(1, 3, 64, 64)

    output, feature_maps = ensemble(x, output_feature_maps=True)
    assert output is not None, "Output should not be None"
    assert len(feature_maps) == len(params.nf), f"Expected {len(params.nf)} feature maps, but got {len(feature_maps)}"
    for fmap in feature_maps:
        assert fmap.shape == (1, 64, 64, 64), f"Feature map shape mismatch: {fmap.shape}"

# Test: Output Method - Meta Learner
def test_ensemble_meta_learner() -> None:
    """Test ensemble with meta-learner as output method."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.meta_learner,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    x = torch.randn(1, 3, 64, 64)

    output = ensemble(x, output_feature_maps=False)
    assert output.shape == (1, 10), f"Output shape should be (1, 10), but got {output.shape}"

def test_ensemble_mean_output_binary() -> None:
    """Test ensemble with mean output method for binary classification produces a scalar."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=2,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.mean,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    x = torch.randn(1, 3, 64, 64)

    output = ensemble(x, output_feature_maps=False)
    assert output.shape == (1,), f"Expected output shape (1,) for binary classification, but got {output.shape}"

def test_ensemble_linear_output_binary() -> None:
    """Test ensemble with linear output method for binary classification produces a scalar."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=2,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.linear,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    x = torch.randn(1, 3, 64, 64)

    output = ensemble(x, output_feature_maps=False)
    assert output.shape == (1,), f"Expected output shape (1,) for binary classification, but got {output.shape}"

# Test: Train Helper
def test_train_helper() -> None:
    """Test train_helper function."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.mean,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    X = torch.randn(10, 3, 64, 64)
    Y = torch.randint(0, 10, (10,))
    crit = nn.CrossEntropyLoss()
    acc_fun = MagicMock(return_value=0.9)

    loss, acc = ensemble.train_helper(None, X, Y, crit, acc_fun)
    assert loss is not None, "Loss should not be None"
    assert acc > 0, "Accuracy should be calculated"

# Test: Optimize Helper
def test_optimize_helper() -> None:
    """Test optimize_helper function."""
    params = ClassifierParams(
        type='ensemble',
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[[32], [64]],
        ensemble_type=EnsembleType.cnn,
        output_method=OutputMethod.linear,
        device=DeviceType.cpu
    )
    ensemble = Ensemble(params)
    ensemble.models = nn.ModuleList([MockCNN(ensemble.n_classes), MockCNN(ensemble.n_classes)])  # Mock the CNN models
    X = torch.randn(10, 3, 64, 64)
    Y = torch.randint(0, 10, (10,))
    crit = nn.CrossEntropyLoss()
    acc_fun = MagicMock(return_value=0.9)

    loss, acc = ensemble.optimize_helper(None, X, Y, crit, acc_fun)
    assert loss is not None, "Loss should not be None"
    assert acc > 0, "Accuracy should be calculated"
