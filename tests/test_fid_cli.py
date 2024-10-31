"""Test FID CLI arguments."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from src.metrics.fid.fid_cli import get_feature_map_function, main
from src.models import CLFIDArgs


@pytest.fixture
def mock_args() -> dict:
    """Fixture to provide mocked CLI arguments for FID calculation."""
    return {
        "dataroot": "./mock_data",
        "dataset_name": "fashion-mnist",
        "pos_class": 3,
        "neg_class": 0,
        "batch_size": 64,
        "model_path": None,
        "num_workers": 2,
        "device": "cpu",
        "name": None,
    }


@pytest.fixture
def mock_dataset() -> list[torch.Tensor]:
    """Fixture to provide a mocked dataset."""
    return [torch.randn((3, 32, 32)) for _ in range(100)]


@patch("src.metrics.fid.fid_cli.load_dataset")
@patch("src.metrics.fid.fid_cli.FrechetInceptionDistance")
@patch("src.metrics.fid.fid_cli.construct_classifier_from_checkpoint")
def test_get_feature_map_function(
    mock_construct_classifier: MagicMock, mock_fid: MagicMock, mock_load_dataset: MagicMock, mock_args: MagicMock
) -> None:
    """Test get_feature_map_function."""
    # Case 1: Model path is provided and valid model returned
    mock_model = MagicMock()
    mock_construct_classifier.return_value = [mock_model]

    # Use the actual dictionary from mock_args fixture
    config = CLFIDArgs(**mock_args)

    # Scenario where model_path is provided
    config.model_path = "mock_path"

    feature_map_fn = get_feature_map_function(config)

    mock_construct_classifier.assert_called_once_with("mock_path", "cpu")
    assert feature_map_fn == mock_model

    # Case 2: Model path is None
    config.model_path = None
    feature_map_fn = get_feature_map_function(config)
    assert feature_map_fn is None


@patch("src.metrics.fid.fid_cli.load_dataset")
@patch("src.metrics.fid.fid_cli.FrechetInceptionDistance")
@patch("numpy.savez")
@patch("src.metrics.fid.fid_cli.torch.utils.data.DataLoader")
def test_main(
    mock_dataloader: MagicMock,
    mock_savez: MagicMock,
    mock_fid: MagicMock,
    mock_load_dataset: MagicMock,
    mock_args: MagicMock,
) -> None:
    """Test the main function in fid_cli.py."""
    mock_load_dataset.return_value = ("mocked_data", None, None)

    # Mocking FID instance properly with Tensors
    mock_fid_instance = mock_fid.return_value
    mock_fid_instance.real_sum = torch.tensor([1.0])
    mock_fid_instance.num_real_images = torch.tensor(1.0)
    mock_fid_instance.real_cov_sum = torch.eye(2048)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        mock_args["dataroot"] = tmpdirname

        # Mock arguments namespace
        with patch("argparse.ArgumentParser.parse_args", return_value=Namespace(**mock_args)):
            with patch("src.models.CLFIDArgs", wraps=CLFIDArgs) as _:
                # Simulate passing arguments through parser and running the main function
                main()

                # Assertions
                mock_load_dataset.assert_called_once()

                mock_dataloader.assert_called_once()
                mock_fid.assert_called_once()

                # Check if numpy.savez was called correctly
                mock_savez.assert_called_once()
                _, kwargs = mock_savez.call_args
                assert "mu" in kwargs
                assert "sigma" in kwargs


def test_argument_validation() -> None:
    """Test CLFIDArgs pydantic model validation for different scenarios."""
    # Valid arguments
    valid_args = {
        "dataroot": "./mock_data",
        "dataset_name": "fashion-mnist",
        "pos_class": 3,
        "neg_class": 0,
        "batch_size": 64,
        "model_path": None,
        "num_workers": 2,
        "device": "cpu",
        "name": None,
    }
    try:
        CLFIDArgs(**valid_args)
    except ValidationError:
        pytest.fail("Validation failed for valid arguments")

    # Invalid dataset
    invalid_args = valid_args.copy()
    invalid_args["dataset_name"] = "invalid-dataset"
    with pytest.raises(ValidationError):
        CLFIDArgs(**invalid_args)

    # Invalid pos_class
    invalid_args = valid_args.copy()
    invalid_args["pos_class"] = "invalid"
    with pytest.raises(ValidationError):
        CLFIDArgs(**invalid_args)


@patch("numpy.savez")
@patch("torch.utils.data.DataLoader")
@patch("src.metrics.fid.fid_cli.load_dataset")
@patch("src.metrics.fid.fid_cli.FrechetInceptionDistance")
def test_fid_statistics_calculation(
    mock_fid: MagicMock,
    mock_load_dataset: MagicMock,
    mock_dataloader: MagicMock,
    mock_savez: MagicMock,
    mock_args: MagicMock,
) -> None:
    """Test if FID statistics are being calculated and saved properly."""
    # Mock dataset loading and DataLoader
    mock_load_dataset.return_value = ("mocked_data", None, None)
    mock_dataloader.return_value = [torch.tensor([[[[0.0]]]])] * 5

    # Properly mock FID instance attributes as Tensors
    mock_fid_instance = mock_fid.return_value
    mock_fid_instance.real_sum = torch.tensor([1.0])
    mock_fid_instance.num_real_images = torch.tensor(1.0)
    mock_fid_instance.real_cov_sum = torch.eye(2048)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        mock_args["dataroot"] = tmpdirname

        with patch("argparse.ArgumentParser.parse_args", return_value=Namespace(**mock_args)):
            main()

            # Check if numpy.savez was called correctly
            mock_savez.assert_called_once()
            _, kwargs = mock_savez.call_args
            assert "mu" in kwargs
            assert "sigma" in kwargs
