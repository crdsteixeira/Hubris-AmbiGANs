import pytest
import torch
from src.classifier.pretrained import ClassifierVIT, ClassifierResnet, ClassifierMLP
from src.models import ClassifierParams, DeviceType, EnsembleType, OutputMethod

# Mock input data for testing
@pytest.fixture
def mock_pretrained_data():
    return torch.randn(1, 1, 224, 224)  # Mock a batch of 224x224 single-channel images

@pytest.fixture
def mock_grayscale_data():
    return torch.randn(1, 1, 28, 28)  # Mock a batch of 28x28 single-channel images for MLP


# Test initialization for each model

def test_vit_initialization():
    """Test that ViT classifier initializes correctly with single-channel image for pretrained ensemble."""
    params = ClassifierParams(
        type="ensemble",
        img_size=(1, 224, 224),  # Single-channel for pretrained
        n_classes=10,
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,  # Added output_method
        device=DeviceType.cpu
    )
    vit_model = ClassifierVIT(params)
    assert vit_model is not None, "ViT model should be initialized successfully."


def test_resnet_initialization():
    """Test that ResNet classifier initializes correctly with single-channel image for pretrained ensemble."""
    params = ClassifierParams(
        type="ensemble",
        img_size=(1, 224, 224),  # Single-channel for pretrained
        n_classes=10,
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,  # Added output_method
        device=DeviceType.cpu
    )
    resnet_model = ClassifierResnet(params)
    assert resnet_model is not None, "ResNet model should be initialized successfully."


def test_mlp_initialization():
    """Test that MLP classifier initializes correctly with single-channel image."""
    params = ClassifierParams(
        type="mlp",
        img_size=(1, 28, 28),  # Single-channel for MLP
        n_classes=10,
        nf=64,  # Set nf to a valid integer for MLP
        device=DeviceType.cpu
    )
    mlp_model = ClassifierMLP(params)
    assert mlp_model is not None, "MLP model should be initialized successfully."


# Test forward pass for each model

def test_vit_forward_pass(mock_pretrained_data):
    """Test forward pass of the ViT classifier with 1-channel pretrained data."""
    params = ClassifierParams(
        type="ensemble",
        img_size=(1, 224, 224),  # Single-channel for pretrained
        n_classes=10,
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,  # Added output_method
        device=DeviceType.cpu
    )
    vit_model = ClassifierVIT(params)
    output = vit_model(mock_pretrained_data)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"


def test_resnet_forward_pass(mock_pretrained_data):
    """Test forward pass of the ResNet classifier with 1-channel pretrained data."""
    params = ClassifierParams(
        type="ensemble",
        img_size=(1, 224, 224),  # Single-channel for pretrained
        n_classes=10,
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,  # Added output_method
        device=DeviceType.cpu
    )
    resnet_model = ClassifierResnet(params)
    output = resnet_model(mock_pretrained_data)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"


def test_mlp_forward_pass(mock_grayscale_data):
    """Test forward pass of the MLP classifier with 1-channel grayscale data."""
    params = ClassifierParams(
        type="mlp",
        img_size=(1, 28, 28),  # Single-channel for MLP
        n_classes=10,
        nf=64,  # Set nf to a valid integer for MLP
        device=DeviceType.cpu
    )
    mlp_model = ClassifierMLP(params)
    output = mlp_model(mock_grayscale_data)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"


# Test binary classification output (n_classes=2)

def test_vit_binary_classification(mock_pretrained_data):
    """Test binary classification output (1D) from ViT classifier."""
    params = ClassifierParams(
        type="ensemble",
        img_size=(1, 224, 224),  # Single-channel for pretrained
        n_classes=2,  # Binary classification
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,  # Added output_method
        device=DeviceType.cpu
    )
    vit_model = ClassifierVIT(params)
    output = vit_model(mock_pretrained_data)
    assert output.shape == (1, 1), f"Expected output shape (1, 1) for binary classification, but got {output.shape}"


def test_resnet_binary_classification(mock_pretrained_data):
    """Test binary classification output (1D) from ResNet classifier."""
    params = ClassifierParams(
        type="ensemble",
        img_size=(1, 224, 224),  # Single-channel for pretrained
        n_classes=2,  # Binary classification
        nf=None,
        ensemble_type=EnsembleType.pretrained,
        output_method=OutputMethod.mean,  # Added output_method
        device=DeviceType.cpu
    )
    resnet_model = ClassifierResnet(params)
    output = resnet_model(mock_pretrained_data)
    assert output.shape == (1, 1), f"Expected output shape (1, 1) for binary classification, but got {output.shape}"


def test_mlp_binary_classification(mock_grayscale_data):
    """Test binary classification output (1D) from MLP classifier."""
    params = ClassifierParams(
        type="mlp",
        img_size=(1, 28, 28),  # Single-channel for MLP
        n_classes=2,  # Binary classification
        nf=64,  # Set nf to a valid integer for MLP
        device=DeviceType.cpu
    )
    mlp_model = ClassifierMLP(params)
    output = mlp_model(mock_grayscale_data)
    assert output.shape == (1,), f"Expected output shape (1,) for binary classification, but got {output.shape}"
