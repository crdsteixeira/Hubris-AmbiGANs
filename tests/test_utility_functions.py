"""Module for testing utility functions."""

import json
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch

from src.models import CLTrainArgs, CLTestNoiseArgs
from src.utils.utility_functions import (
    create_and_store_z,
    create_checkpoint_path,
    create_exp_path,
    gen_seed,
    generate_cnn_configs,
    group_images,
    handle_subprocess_output,
    load_z,
    make_grid,
    run_training_subprocess,
    seed_worker,
    set_seed,
    setup_reprod,
)


@pytest.fixture
def mock_config():
    """Fixture to provide a mock configuration dictionary."""
    return {"out-dir": "mock_out", "project": "test_project", "name": "test_run"}


@pytest.fixture
def mock_args() -> None:
    """Fixture to provide mock CLTrainArgs for classifier training."""
    return CLTrainArgs(
        device="cpu",
        data_dir="data_dir",
        out_dir="output_dir",
        dataset_name="mnist",
        batch_size=32,
        lr=0.001,
        seed=42,
        nf=[2, 3],
        early_acc=0.95,
        pos_class=1,
        neg_class=7,
    )


def test_create_checkpoint_path(mock_config) -> None:
    """Test creating a checkpoint path."""
    with patch("os.makedirs") as mock_makedirs:
        path = create_checkpoint_path(mock_config, "001")
        assert "mock_out" in path
        assert "test_project" in path
        assert "test_run" in path
        mock_makedirs.assert_called_once_with(path, exist_ok=True)


def test_create_exp_path(mock_config) -> None:
    """Test creating an experimental path."""
    with patch("os.makedirs") as mock_makedirs:
        path = create_exp_path(mock_config)
        assert path == os.path.join("mock_out", "test_run")
        mock_makedirs.assert_called_once_with(path, exist_ok=True)


def test_gen_seed() -> None:
    """Test generating a random seed."""
    seed = gen_seed()
    assert isinstance(seed, int)
    assert 0 <= seed < 10000


def test_set_seed() -> None:
    """Test setting a seed for reproducibility."""
    set_seed(42)
    assert np.random.randint(100) == np.random.RandomState(42).randint(100)


def test_setup_reprod() -> None:
    """Test setting up reproducibility settings."""
    with patch("torch.backends.cudnn") as mock_cudnn, patch("src.utils.utility_functions.set_seed") as mock_set_seed:
        setup_reprod(42)
        mock_cudnn.deterministic = True
        mock_cudnn.benchmark = False
        mock_set_seed.assert_called_once_with(42)


def test_seed_worker() -> None:
    """Test seeding worker processes for reproducibility."""
    with (
        patch("torch.initial_seed", return_value=42),
        patch("random.seed") as mock_random_seed,
        patch("numpy.random.seed") as mock_np_seed,
    ):
        seed_worker(42)
        worker_seed = 42 % 2**32
        mock_np_seed.assert_called_once_with(worker_seed)
        mock_random_seed.assert_called_once_with(worker_seed)


@patch("os.makedirs")
@patch("numpy.savez")
def test_create_and_store_z(mock_savez, mock_makedirs):
    """Test creating and storing noise tensor z."""
    out_dir = "mock_out"
    n, dim = 5, 100
    with patch("builtins.open", mock_open()) as mock_file:
        z, path = create_and_store_z(CLTestNoiseArgs(
            out_dir=out_dir,
            nz=n,
            z_dim=dim
        ))

        assert isinstance(z, torch.Tensor)
        assert z.shape == (n, dim)
        mock_makedirs.assert_called_once_with(path, exist_ok=True)
        mock_file.assert_any_call(os.path.join(path, "z.npy"), "wb", encoding="utf-8")
        mock_savez.assert_called_once()


def test_load_z() -> None:
    """Test loading noise tensor z from disk."""
    mock_z_data = np.random.rand(5, 100)
    mock_conf_data = {"key": "value"}

    # Mock np.load to return our mock tensor
    with (
        patch("numpy.load") as mock_np_load,
        patch("builtins.open", mock_open(read_data=json.dumps(mock_conf_data))) as mock_file,
    ):
        mock_np_load.return_value = {"z": mock_z_data}

        # Call the load_z function
        z, conf = load_z("mock_path")

        # Assertions to ensure the correct behavior
        assert isinstance(z, torch.Tensor)
        assert z.shape == (5, 100)
        assert conf == mock_conf_data

        # Ensure files were opened correctly
        mock_file.assert_called_once_with(os.path.join("mock_path", "z.json"), encoding="utf-8")
        mock_np_load.assert_called_once_with(os.path.join("mock_path", "z.npy"), encoding="utf-8")


def test_make_grid() -> None:
    """Test making a grid of images."""
    images = torch.randn(16, 3, 64, 64)
    grid = make_grid(images, nrow=4)
    assert isinstance(grid, torch.Tensor)
    assert grid.shape[0] == 3  # RGB channels


def test_group_images() -> None:
    """Test grouping images with classifier."""
    images = torch.randn(100, 3, 64, 64)
    classifier = MagicMock(return_value=torch.randint(0, 10, (100,)))
    grouped_images = group_images(images, classifier=classifier, device=torch.device("cpu"))
    assert isinstance(grouped_images, torch.Tensor)
    classifier.assert_called()



def test_generate_cnn_configs() -> None:
    """Test generating CNN configurations."""
    configs = generate_cnn_configs(3)
    assert len(configs) == 3
    assert all(isinstance(cfg, list) for cfg in configs)


@patch("subprocess.run")
def test_run_training_subprocess(mock_run, mock_args) -> None:
    """Test running a classifier training subprocess."""
    run_training_subprocess(
        CLTrainArgs(
            dataset_name="mnist",
            pos_class=0,
            neg_class=1,
            c_type="cnn",
            epochs="10",
            args=mock_args,
        ), cnn_nfs=[[2, 3], [3, 4]],
    )
    mock_run.assert_called_once()


@patch("src.utils.utility_functions.logger")
def test_handle_subprocess_output(mock_logger) -> None:
    """Test handling output from subprocess."""
    # Mock subprocess result with stdout and stderr
    proc_mock = MagicMock()
    proc_mock.stdout = b"mock_stdout\nline2\n"
    proc_mock.stderr = b"mock_stderr\nline2\n"

    # Capture printed output
    captured_output = StringIO()
    sys.stdout = captured_output  # Redirect stdout

    # Call the function with the mocked CompletedProcess object
    handle_subprocess_output(proc_mock)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Assertions for checking stdout and stderr are printed correctly
    mock_logger.info.assert_any_call("mock_stdout")
    mock_logger.info.assert_any_call("line2")
    mock_logger.info.assert_any_call("mock_stderr")
