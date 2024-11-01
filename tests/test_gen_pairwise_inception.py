"""Test Module for generating FID statistics from dataset."""

from collections.abc import Generator
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import ValidationError

from src.gen_pairwise_inception import main
from src.models import CLFIDStatsArgs


# Fixture for setting up valid arguments
@pytest.fixture
def valid_args() -> dict[str, str]:
    """Provide valid arguments for FID statistics."""
    return {
        "dataroot": "mock_data",
        "dataset": "mnist",
        "device": "cpu",
    }


# Mock load_dotenv, configure_logging and subprocess in a setup fixture
@pytest.fixture(autouse=True)
@patch("dotenv.load_dotenv")
@patch("src.utils.logging.configure_logging")
def setup_logging_and_env(
    mock_configure_logging: MagicMock, mock_load_dotenv: MagicMock
) -> Generator[None, None, None]:
    """Fixture to mock environment setup and logging configuration."""
    mock_load_dotenv.return_value = None
    mock_configure_logging.return_value = None
    yield


# Optimized Test with Mocking Pydantic Validation and Subprocess Call
@patch("argparse.ArgumentParser.parse_args")
@patch("subprocess.run")
@patch("itertools.combinations")
def test_main_subprocess_calls(
    mock_combinations: MagicMock, mock_subprocess_run: MagicMock, mock_parse_args: MagicMock, valid_args: dict[str, str]
) -> None:
    """Test the main function end-to-end with subprocess mocking."""
    # Mock the command line arguments
    mock_parse_args.return_value = MagicMock(**valid_args)
    # Mock class combinations to reduce the scope of the test
    mock_combinations.return_value = [(0, 1)]  # Limit to only one combination
    # Mock subprocess call to simulate successful execution
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # Call the main function
    main()

    # Assertions to ensure subprocess was called and with the correct parameters
    assert mock_subprocess_run.call_count == 1
    mock_subprocess_run.assert_called_with(
        [
            "python",
            "-m",
            "src.metrics.fid.fid_cli",
            "--data",
            valid_args["dataroot"],
            "--dataset",
            valid_args["dataset"],
            "--device",
            valid_args["device"],
            "--pos",
            str(1),
            "--neg",
            str(0),
        ],
        check=False,
    )


@patch("argparse.ArgumentParser.parse_args")
def test_pydantic_validation_failure(mock_parse_args: MagicMock) -> None:
    """Test if ValidationError is raised for invalid arguments."""
    # Provide invalid arguments
    mock_parse_args.return_value = MagicMock(
        dataroot="mock_data",
        dataset="invalid_dataset",
        device="cpu",
    )

    # Expect ValidationError due to invalid dataset value
    with pytest.raises(ValidationError):
        main()


@patch("argparse.ArgumentParser.parse_args")
def test_pydantic_validation_success(mock_parse_args: MagicMock, valid_args: dict[str, str]) -> None:
    """Test if the Pydantic validation succeeds for valid arguments."""
    mock_parse_args.return_value = MagicMock(**valid_args)

    # Test the validation
    try:
        config = CLFIDStatsArgs(**valid_args)
    except ValidationError:
        pytest.fail("Validation failed with valid arguments.")

    assert config.dataset == valid_args["dataset"]
    assert config.dataroot == valid_args["dataroot"]
    assert config.device == valid_args["device"]


@patch("argparse.ArgumentParser.parse_args")
@patch("src.models.CLFIDStatsArgs")
@patch("itertools.combinations")
@patch("src.gen_pairwise_inception.logger")
def test_main_logging_and_validation(
    mock_logger: MagicMock,
    mock_combinations: MagicMock,
    mock_CLFIDStatsArgs: MagicMock,
    mock_parse_args: MagicMock,
    valid_args: dict[str, str],
) -> None:
    """Test logging and validation integration."""
    # Mock arguments and validation
    mock_parse_args.return_value = MagicMock(**valid_args)
    mock_CLFIDStatsArgs.return_value = CLFIDStatsArgs(**valid_args)

    # Mock combinations to limit the output
    mock_combinations.return_value = [(0, 1)]  # Only return one combination

    # Run the main function
    main()

    # Check logger calls for the process
    mock_logger.info.assert_any_call("Calculating FID statistics...")
    mock_logger.info.assert_any_call(mock_CLFIDStatsArgs.return_value)
    mock_logger.info.assert_any_call("0vs1")


@patch("argparse.ArgumentParser.parse_args")
@patch("subprocess.run")
@patch("itertools.combinations")
def test_multiple_class_combinations(
    mock_combinations: MagicMock, mock_subprocess_run: MagicMock, mock_parse_args: MagicMock, valid_args: dict[str, str]
) -> None:
    """Test subprocess is called for multiple class combinations."""
    # Mock the command line arguments
    mock_parse_args.return_value = MagicMock(**valid_args)
    # Provide more class combinations to check multiple subprocess executions
    mock_combinations.return_value = [(0, 1), (1, 2), (2, 3)]  # Simulating three class pairs
    # Mock subprocess call to simulate successful execution
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # Run the main function
    main()

    # Ensure subprocess was called the correct number of times
    assert mock_subprocess_run.call_count == 3
    # Create expected calls list
    expected_calls = [
        call(
            [
                "python",
                "-m",
                "src.metrics.fid.fid_cli",
                "--data",
                valid_args["dataroot"],
                "--dataset",
                valid_args["dataset"],
                "--device",
                valid_args["device"],
                "--pos",
                str(pos),
                "--neg",
                str(neg),
            ],
            check=False,
        )
        for neg, pos in mock_combinations.return_value
    ]

    # Assert all calls were made as expected
    mock_subprocess_run.assert_has_calls(expected_calls, any_order=True)
