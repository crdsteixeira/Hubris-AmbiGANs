"""Module to test construct classifier."""

import pytest
from torch import nn

from src.classifier.construct_classifier import construct_classifier
from src.classifier.my_mlp import Classifier as MyMLP
from src.classifier.simple_cnn import Classifier as SimpleCNN
from src.enums import ClassifierType, DeviceType, EnsembleType, DatasetNames
from src.models import TrainClassifierArgs


@pytest.fixture
def cnn_params() -> TrainClassifierArgs:
    """Fixture to provide default parameters for a CNN classifier."""
    return TrainClassifierArgs(
        type=ClassifierType.cnn,
        img_size=(3, 28, 28),
        n_classes=10,
        nf=[32],
        device=DeviceType.cpu,
        epochs=10,  # Add required field 'epochs'
        out_dir="./models",  # Add required field 'out_dir'
        early_stop=None,  # Optional field
        early_acc=0.9,  # Optional field
        lr=0.001,  # Add learning rate
        batch_size=32,  # Batch size for training
        seed=42,  # Optional seed for reproducibility
        dataset_name=DatasetNames.mnist,
        pos_class=1,
        neg_class=7,
    )


@pytest.fixture
def mlp_params() -> TrainClassifierArgs:
    """Fixture to provide default parameters for an MLP classifier."""
    return TrainClassifierArgs(
        type=ClassifierType.mlp,
        img_size=(3, 28, 28),
        n_classes=10,
        nf=128,
        device=DeviceType.cpu,
        epochs=10,  # Add required field 'epochs'
        out_dir="./models",  # Add required field 'out_dir'
        early_stop=None,  # Optional field
        early_acc=0.9,  # Optional field
        lr=0.001,  # Add learning rate
        batch_size=32,  # Batch size for training
        seed=42,  # Optional seed for reproducibility
        dataset_name=DatasetNames.mnist,
        pos_class=1,
        neg_class=7,
    )


@pytest.fixture
def ensemble_params() -> TrainClassifierArgs:
    """Fixture to provide default parameters for an ensemble classifier."""
    return TrainClassifierArgs(
        type=ClassifierType.ensemble,
        ensemble_type=EnsembleType.cnn,
        output_method="mean",
        img_size=(3, 28, 28),
        n_classes=10,
        nf=[[32], [64]],  # Matches the number of models in the ensemble
        device=DeviceType.cpu,
        epochs=10,  # Add required field 'epochs'
        out_dir="./models",  # Add required field 'out_dir'
        early_stop=None,  # Optional field
        early_acc=0.9,  # Optional field
        lr=0.001,  # Add learning rate
        batch_size=32,  # Batch size for training
        seed=42,  # Optional seed for reproducibility
        dataset_name=DatasetNames.mnist,
        pos_class=1,
        neg_class=7,
    )


def test_construct_cnn(cnn_params: TrainClassifierArgs) -> None:
    """Test construction of a CNN classifier."""
    model = construct_classifier(cnn_params)
    assert isinstance(model, nn.Module), "Model should be a PyTorch Module"
    # Verify model is an instance of SimpleCNN
    assert isinstance(model, SimpleCNN), "Model should be an instance of SimpleCNN"
    # Check if it is from the correct module
    assert model.__module__ == "src.classifier.simple_cnn", "Model should come from 'simple_cnn' module"

    # Check for presence of convolutional layers inside 'blocks'
    assert hasattr(model, "blocks"), "'SimpleCNN' should have a 'blocks' attribute for layers"
    assert isinstance(model.blocks, nn.ModuleList), "'blocks' should be an instance of 'nn.ModuleList'"

    # Verify that the layers inside blocks contain Conv2d and MaxPool2d
    found_conv_layer = False
    found_pool_layer = False
    for block in model.blocks:
        if isinstance(block, nn.Sequential):
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    found_conv_layer = True
                elif isinstance(layer, nn.MaxPool2d):
                    found_pool_layer = True

    assert found_conv_layer, "SimpleCNN should have at least one Conv2d layer"
    assert found_pool_layer, "SimpleCNN should have at least one MaxPool2d layer"


def test_construct_mlp(mlp_params: TrainClassifierArgs) -> None:
    """Test construction of an MLP classifier."""
    model = construct_classifier(mlp_params)
    assert isinstance(model, nn.Module), "Model should be a PyTorch Module"
    # Verify model is an instance of MyMLP
    assert isinstance(model, MyMLP), "Model should be an instance of MyMLP"
    # Check if it is from the correct module
    assert model.__module__ == "src.classifier.my_mlp", "Model should come from 'my_mlp' module"

    # Verify specific layers in the MLP classifier
    assert hasattr(model, "blocks"), "MLP Classifier should have 'blocks' attribute"
    assert isinstance(model.blocks, nn.ModuleList), "'blocks' should be an instance of nn.ModuleList"
    assert len(model.blocks) == 2, "MLP should contain exactly two blocks"

    # Verify Block 1: Flatten and Linear Layer
    block_1 = model.blocks[0]
    assert isinstance(block_1, nn.Sequential), "Block 1 should be a Sequential model"
    assert isinstance(block_1[0], nn.Flatten), "Block 1 should contain a Flatten layer as the first layer"
    assert isinstance(block_1[1], nn.Linear), "Block 1 should contain a Linear layer"

    # Verify Block 2: Output Predictor Layer
    block_2 = model.blocks[1]
    assert isinstance(block_2, nn.Sequential), "Block 2 should be a Sequential model"
    assert isinstance(block_2[0], nn.Linear), "Block 2 should contain a Linear layer"
    assert isinstance(
        block_2[1], nn.Sigmoid | nn.Softmax
    ), "Block 2 should contain either Sigmoid or Softmax for output"


def test_construct_ensemble(ensemble_params: TrainClassifierArgs) -> None:
    """Test construction of an ensemble classifier."""
    model = construct_classifier(ensemble_params)
    assert isinstance(model, nn.Module), "Model should be a PyTorch Module"
    assert model.__class__.__name__ == "Ensemble", "Model should be an instance of Ensemble"


def test_invalid_classifier_type() -> None:
    """Test construction with an invalid classifier type."""
    with pytest.raises(ValueError, match="Input should be 'cnn', 'mlp' or 'ensemble'"):
        construct_classifier(
            TrainClassifierArgs(
                type="invalid_type",  # Invalid classifier type
                img_size=(3, 28, 28),
                n_classes=10,
                nf=32,
                device=DeviceType.cpu,
                epochs=10,  # Add required field 'epochs'
                out_dir="./models",  # Add required field 'out_dir'
                lr=0.001,  # Add learning rate
                batch_size=32,  # Batch size for training
                dataset_name=DatasetNames.mnist,
                pos_class=1,
                neg_class=7,
            )
        )


def test_ensemble_nf_mismatch() -> None:
    """Test construction of an ensemble with nf parameter mismatch."""
    invalid_ensemble_params = TrainClassifierArgs(
        type=ClassifierType.ensemble,
        ensemble_type=EnsembleType.cnn,
        output_method="mean",
        img_size=(3, 28, 28),
        n_classes=10,
        nf=[[32]],
        device=DeviceType.cpu,
        epochs=10,  # Add required field 'epochs'
        out_dir="./models",  # Add required field 'out_dir'
        lr=0.001,  # Add learning rate
        batch_size=32,  # Batch size for training
        dataset_name=DatasetNames.mnist,
        pos_class=1,
        neg_class=7,
    )
    with pytest.raises(ValueError, match="Ensemble must have more than one model, but got*"):
        construct_classifier(invalid_ensemble_params)
