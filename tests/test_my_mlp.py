import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from src.classifier.my_mlp import Classifier
from src.enums import DeviceType
from src.models import ClassifierParams


def test_valid_mlp_params():
    """Test valid MLP classifier params."""
    params = ClassifierParams(
        type="mlp",
        img_size=(3, 28, 28),
        n_classes=10,
        nf=64,  # Valid single integer for MLP
        device=DeviceType.cpu,
    )
    assert params.nf == 64


def test_invalid_mlp_nf():
    """Test invalid 'nf' value for MLP (should raise validation error)."""
    with pytest.raises(ValidationError):
        ClassifierParams(
            type="mlp",
            img_size=(3, 28, 28),
            n_classes=10,
            nf=[64, 128],  # Invalid: nf should be a single integer for MLP
            device=DeviceType.cpu,
        )


def test_invalid_img_size():
    """Test invalid img_size (should raise validation error)."""
    with pytest.raises(ValidationError):
        ClassifierParams(
            type="mlp",
            img_size=(28, 28),  # Invalid: must be a tuple of three integers
            n_classes=10,
            nf=64,
            device=DeviceType.cpu,
        )


def test_invalid_number_of_classes():
    """Test invalid number of classes (should raise validation error)."""
    with pytest.raises(ValidationError):
        ClassifierParams(
            type="mlp",
            img_size=(3, 28, 28),
            n_classes=1,  # Invalid: number of classes should be at least 2
            nf=64,
            device=DeviceType.cpu,
        )


def test_mlp_binary_classification_output():
    """Test forward pass for binary classification produces scalar output."""
    params = ClassifierParams(
        type="mlp",
        img_size=(1, 28, 28),  # Grayscale image
        n_classes=2,  # Binary classification
        nf=64,
        device=DeviceType.cpu,
    )
    model = Classifier(params)

    x = torch.randn(1, 1, 28, 28)  # Batch of 1 image
    output = model(x)

    # Check that output is a scalar for binary classification
    assert output.shape == (1,), f"Expected shape (1,), but got {output.shape}"


def test_mlp_multiclass_classification_output():
    """Test forward pass for multi-class classification produces correct output."""
    params = ClassifierParams(
        type="mlp",
        img_size=(3, 28, 28),  # RGB image
        n_classes=10,  # Multi-class classification
        nf=64,
        device=DeviceType.cpu,
    )
    model = Classifier(params)

    x = torch.randn(1, 3, 28, 28)  # Batch of 1 image
    output = model(x)

    # Check that output matches the number of classes for multi-class classification
    assert output.shape == (1, 10), f"Expected shape (1, 10), but got {output.shape}"


def test_mlp_output_with_feature_maps():
    """Test forward pass with feature maps output enabled."""
    params = ClassifierParams(
        type="mlp",
        img_size=(1, 28, 28),  # Grayscale image
        n_classes=2,  # Binary classification
        nf=64,
        device=DeviceType.cpu,
    )
    model = Classifier(params)

    x = torch.randn(1, 1, 28, 28)  # Batch of 1 image
    outputs = model(x, output_feature_maps=True)

    # Check that feature maps are returned as a list of intermediate outputs
    assert isinstance(outputs, list), "Expected a list of intermediate outputs"
    assert len(outputs) == 2, "Expected 2 sets of outputs (one for each block)"

    # Final output should still be scalar for binary classification
    assert outputs[-1].shape == (1,), f"Expected final output shape (1,), but got {outputs[-1].shape}"


def test_invalid_input_shape():
    """Test that an invalid input shape raises an error."""
    params = ClassifierParams(type="mlp", img_size=(3, 28, 28), n_classes=10, nf=64, device=DeviceType.cpu)
    model = Classifier(params)

    # Input with incorrect shape
    x = torch.randn(1, 3, 32, 32)  # 32x32 instead of 28x28
    with pytest.raises(RuntimeError):
        model(x)


def test_mlp_device_compatibility():
    """Test that the MLP model works on both CPU and GPU (if available)."""
    params = ClassifierParams(type="mlp", img_size=(3, 28, 28), n_classes=10, nf=64, device=DeviceType.cpu)
    model = Classifier(params)

    x = torch.randn(1, 3, 28, 28)

    # Test on CPU
    output_cpu = model(x.to(DeviceType.cpu.value))
    assert output_cpu is not None, "Output on CPU should not be None"

    # Test on GPU (if available)
    if torch.cuda.is_available():
        model.cuda()
        output_gpu = model(x.to(DeviceType.cuda.value))
        assert output_gpu is not None, "Output on GPU should not be None"


def test_mlp_initialization_properties():
    """Test if MLP initializes with correct properties."""
    params = ClassifierParams(type="mlp", img_size=(1, 28, 28), n_classes=2, nf=64, device=DeviceType.cpu)
    model = Classifier(params)

    assert len(model.blocks) == 2, "There should be two blocks in the model"
    assert isinstance(model.blocks[0][1], nn.Linear), "First block should have a Linear layer"
    assert isinstance(model.blocks[1][0], nn.Linear), "Second block should have the final Linear layer"
