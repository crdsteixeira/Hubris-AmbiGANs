"""Module to test test noise generation."""

import os
import random
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from src.models import CLTestNoiseArgs
from src.utils.utility_functions import create_and_store_z, gen_seed, set_seed


@pytest.fixture
def mock_args() -> CLTestNoiseArgs:
    """Fixture to provide mock command line arguments."""
    return CLTestNoiseArgs(seed=42, nz=5, z_dim=100, out_dir="mock_out")


@patch("os.makedirs")
@patch("numpy.savez")
@patch("builtins.open", new_callable=mock_open)
def test_create_and_store_z(
    mock_file: MagicMock, mock_savez: MagicMock, mock_makedirs: MagicMock, mock_args: MagicMock
) -> None:
    """Test that noise tensor z is created and stored correctly."""
    z, path = create_and_store_z(config=mock_args)

    # Construct the expected directory path that would be created
    expected_out_path = os.path.join(mock_args.out_dir, f"z_{mock_args.nz}_{mock_args.z_dim}")

    # Check if directories are created (with the correct path)
    mock_makedirs.assert_called_once_with(expected_out_path, exist_ok=True)

    # Check if the file was opened correctly to store the noise
    expected_z_path = os.path.join(expected_out_path, "z.npy")
    expected_json_path = os.path.join(expected_out_path, "z.json")

    # Since `open` is called twice, we need to check both calls
    mock_file.assert_has_calls(
        [
            call(expected_z_path, "wb"),
            call(expected_json_path, "w", encoding="utf-8"),
        ],
        any_order=True,
    )

    # Check that numpy.savez() was called to save the noise
    mock_savez.assert_called_once()

    # Ensure the returned tensor has the expected shape
    assert z.shape == (mock_args.nz, mock_args.z_dim)

    # Ensure the path is correct
    assert path == expected_out_path


def test_gen_seed() -> None:
    """Test generating a random seed."""
    seed = gen_seed()
    assert isinstance(seed, int)
    assert 0 <= seed < 10000


def test_set_seed() -> None:
    """Test setting the random seed."""
    seed = 42
    # Set the seed using the utility function
    set_seed(seed)

    # Generate random numbers from numpy, torch, and random
    np_random_value_1 = np.random.rand()
    torch_random_value_1 = torch.rand(1).item()
    random_value_1 = random.random()

    # Set the seed again to ensure reproducibility
    set_seed(seed)

    # Generate random numbers again
    np_random_value_2 = np.random.rand()
    torch_random_value_2 = torch.rand(1).item()
    random_value_2 = random.random()

    # Check if the generated values are the same, ensuring reproducibility
    assert np_random_value_1 == np_random_value_2, "Numpy values do not match for the same seed"
    assert torch_random_value_1 == torch_random_value_2, "Torch values do not match for the same seed"
    assert random_value_1 == random_value_2, "Random values do not match for the same seed"


@patch("argparse.ArgumentParser.parse_args")
@patch("src.utils.utility_functions.create_and_store_z")
@patch("src.utils.logging.configure_logging")
@patch("dotenv.load_dotenv")
def test_main(
    mock_load_dotenv: MagicMock,
    mock_configure_logging: MagicMock,
    mock_create_and_store_z: MagicMock,
    mock_parse_args: MagicMock,
) -> None:
    """Test the main function end-to-end."""
    mock_args = MagicMock(seed=42, nz=5, z_dim=100, out_dir="mock_out")
    mock_parse_args.return_value = mock_args
    mock_create_and_store_z.return_value = (0, "test")

    # Import the main function after mocking dependencies
    from src.gen_test_noise import main

    # Call the main function
    main()

    # Assertions to verify behavior
    mock_load_dotenv.assert_called_once()
    mock_configure_logging.assert_called_once()
    mock_parse_args.assert_called_once()  # Make sure parse_args was called

    # Test if the validation passed without raising a ValidationError
    try:
        config = CLTestNoiseArgs(seed=42, nz=5, z_dim=100, out_dir="mock_out")
    except ValidationError:
        pytest.fail("Validation failed with valid arguments.")

    if config.seed:
        set_seed(config.seed)
    mock_create_and_store_z.assert_called_once_with(config=config)
