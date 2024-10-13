from typing import List

import pytest
import torch
from torch import Tensor

from src.classifier.simple_cnn import Classifier, pool_out
from src.models import ClassifierParams, ClassifierType, DeviceType, PoolParams


# Mock input data for the tests
@pytest.fixture
def mock_data() -> Tensor:
    """Fixture to provide mock input data (batch_size=1, channels=3, height=64, width=64)."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def binary_mock_data() -> Tensor:
    """Fixture to provide mock input data for binary classification (batch_size=1, channels=3, height=64, width=64)."""
    return torch.randn(1, 3, 64, 64)


# Test pool_out function
def test_pool_out() -> None:
    """Test the pool_out function with different pooling parameters."""
    # Test case 1: Simple pooling with no padding, stride or dilation
    params: PoolParams = PoolParams(in_size=32, kernel=2)
    assert pool_out(params) == 16, "Expected output size for 32x32 input with 2x2 pooling should be 16x16"

    # Test case 2: With padding
    params = PoolParams(in_size=32, kernel=2, padding=1)
    assert pool_out(params) == 17, "Expected output size for 32x32 input with 2x2 pooling and padding should be 17x17"

    # Test case 3: With stride
    params = PoolParams(in_size=32, kernel=2, stride=1)
    assert pool_out(params) == 31, "Expected output size for 32x32 input with 2x2 pooling and stride 1 should be 31x31"

    # Test case 4: With dilation
    params = PoolParams(in_size=32, kernel=2, dilation=2)
    assert (
        pool_out(params) == 15
    ), "Expected output size for 32x32 input with 2x2 pooling and dilation 2 should be 15x15"


# Test Classifier initialization
def test_classifier_initialization() -> None:
    """Test the initialization of the CNN classifier using ClassifierParams."""
    params: ClassifierParams = ClassifierParams(
        type=ClassifierType.cnn,
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[16, 32],
        device=DeviceType.cpu,
    )
    model: Classifier = Classifier(params)

    assert isinstance(model, Classifier), "Model should be an instance of the Classifier class."
    assert len(model.blocks) == 3, "Classifier should have two convolutional blocks and one fully connected block."


# Test Classifier forward pass
def test_classifier_forward_pass(mock_data: Tensor) -> None:
    """Test the forward pass of the CNN classifier using ClassifierParams."""
    params: ClassifierParams = ClassifierParams(
        type=ClassifierType.cnn,
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[16, 32],
        device=DeviceType.cpu,
    )
    model: Classifier = Classifier(params)

    # Perform forward pass
    output: Tensor = model(mock_data)

    assert output.shape == (
        1,
        10,
    ), f"Expected output shape to be (1, 10), but got {output.shape}"


# Test Classifier with binary classification
def test_classifier_binary_classification(binary_mock_data: Tensor) -> None:
    """Test the classifier with binary classification (output should be flattened)."""
    params: ClassifierParams = ClassifierParams(
        type=ClassifierType.cnn,
        img_size=(3, 64, 64),
        n_classes=2,  # Binary classification
        nf=[16, 32],
        device=DeviceType.cpu,
    )
    model: Classifier = Classifier(params)

    # Perform forward pass
    output: Tensor = model(binary_mock_data)

    assert output.shape == (1,), f"Expected output shape for binary classification to be (1,), but got {output.shape}"


# Test Classifier with feature maps output
def test_classifier_with_feature_maps(mock_data: Tensor) -> None:
    """Test the classifier with feature map output enabled."""
    params: ClassifierParams = ClassifierParams(
        type=ClassifierType.cnn,
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[16, 32],
        device=DeviceType.cpu,
    )
    model: Classifier = Classifier(params)

    # Perform forward pass with output_feature_maps=True
    feature_maps: list[Tensor] = model(mock_data, output_feature_maps=True)

    assert len(feature_maps) == 3, f"Expected 3 feature maps, but got {len(feature_maps)}"
    assert feature_maps[-1].shape == (
        1,
        10,
    ), f"Expected last feature map to have shape (1, 10), but got {feature_maps[-1].shape}"
