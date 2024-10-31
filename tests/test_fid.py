"""Test FID module."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.enums import DeviceType
from src.metrics.fid.fid import FID


@pytest.fixture
def fid_stats_file(tmp_path: str) -> str:
    """Fixture to create a temporary FID stats file."""
    stats_file = os.path.join(tmp_path, "fid_stats.npz")
    np.savez(
        stats_file,
        real_sum=np.random.rand(2048),
        real_cov_sum=np.random.rand(2048, 2048),
        num_real_images=np.array(1000),
    )
    return str(stats_file)


@pytest.fixture
def fid_instance(fid_stats_file: str) -> FID:
    """Fixture to provide an instance of the FID class."""
    return FID(fid_stats_file=fid_stats_file, dims=2048, n_images=100, device=DeviceType.cpu)


def test_initialization_success(fid_stats_file: str) -> None:
    """Test successful initialization of the FID class."""
    fid = FID(fid_stats_file=fid_stats_file, dims=2048, n_images=100, device=DeviceType.cpu)
    assert fid.dims == 2048
    assert fid.n_images == 100
    assert isinstance(fid.pred_arr, np.ndarray)
    assert fid.cur_idx == 0


def test_initialization_file_not_found() -> None:
    """Test initialization with a non-existent FID stats file."""
    with pytest.raises(FileNotFoundError):
        FID(fid_stats_file="non_existent_file.npz", dims=2048, n_images=100, device=DeviceType.cpu)


def test_update(fid_instance: FID) -> None:
    """Test updating the FID metric with a batch of generated images."""
    images = torch.rand((10, 3, 32, 32))  # Example batch of 10 images
    fid_instance.update(images, (0, 0))

    # Check if the `update` function of `FrechetInceptionDistance` was called properly
    assert fid_instance.fid.num_fake_images > 0


def test_finalize(fid_instance: FID) -> None:
    """Test finalizing the FID metric calculation."""
    images = torch.rand((10, 3, 32, 32))  # Example batch of 10 images
    fid_instance.update(images, (0, 0))
    fid_value = fid_instance.finalize()

    assert isinstance(fid_value, float)  # Finalize should return a float FID score


def test_reset(fid_instance: FID) -> None:
    """Test resetting the FID metric to its initial state."""
    # Update the metric first
    images = torch.rand((10, 3, 32, 32))
    fid_instance.update(images, (0, 0))

    # Reset the metric
    fid_instance.reset()

    # Verify that internal attributes are reset correctly
    assert fid_instance.fid.num_fake_images == 0
    assert torch.equal(fid_instance.fid.fake_sum, torch.zeros(fid_instance.dims).to(fid_instance.device))
    assert torch.equal(
        fid_instance.fid.fake_cov_sum, torch.zeros((fid_instance.dims, fid_instance.dims)).to(fid_instance.device)
    )


def test_update_on_different_devices(fid_stats_file: str) -> None:
    """Test the FID metric update method on different devices."""
    # Test on CPU
    fid_cpu = FID(fid_stats_file=fid_stats_file, dims=2048, n_images=100, device=DeviceType.cpu)
    images = torch.rand((10, 3, 32, 32))
    fid_cpu.update(images, (0, 0))
    assert fid_cpu.fid.num_fake_images > 0

    # Test on CUDA (if available)
    if torch.cuda.is_available():
        fid_cuda = FID(fid_stats_file=fid_stats_file, dims=2048, n_images=100, device=DeviceType.cuda)
        images_cuda = torch.rand((10, 3, 32, 32)).to(DeviceType.cuda)
        fid_cuda.update(images_cuda, (0, 0))
        assert fid_cuda.fid.num_fake_images > 0


def test_finalize_with_mock(fid_instance: MagicMock) -> None:
    """Test finalizing the FID calculation with mocked values."""
    with patch.object(fid_instance.fid, "compute", return_value=torch.tensor(50.0)) as mock_compute:
        fid_value = fid_instance.finalize()
        assert fid_value == 50.0
        mock_compute.assert_called_once()


def test_invalid_images_update(fid_instance: MagicMock) -> None:
    """Test updating FID metric with invalid images."""
    invalid_images = torch.rand((10,))  # Malformed input
    with pytest.raises(ValueError):
        fid_instance.update(invalid_images, (0, 0))
