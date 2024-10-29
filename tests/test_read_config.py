"""Test for read config module."""

import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError

from src.models import ConfigWeights
from src.utils.read_config import read_config


@pytest.fixture
def sample_config_data() -> dict:
    """Fixture to provide sample GAN configuration data as a dictionary."""
    return {
        "project": "gan_project",
        "name": "gan_experiment",
        "out_dir": "output_directory",
        "data_dir": "data_directory",
        "fid_stats_path": "fid_stats.yaml",
        "fixed_noise": 100,
        "test_noise": "test_noise.yaml",
        "compute_fid": True,
        "device": "cuda",
        "num_workers": 4,
        "num_runs": 2,
        "step_1_seeds": [42, 43],
        "step_2_seeds": [44, 45],
        "dataset": {
            "name": "cifar10",
            "binary": {"pos": 0, "neg": 5},
        },
        "model": {
            "z_dim": 100,
            "architecture": {
                "name": "dcgan",
                "g_filter_dim": 64,
                "d_filter_dim": 64,
                "g_num_blocks": 3,
                "d_num_blocks": 3,
            },
            "loss": {
                "name": "wgan-gp",
                "args": 10,
            },
        },
        "optimizer": {
            "lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
        },
        "train": {
            "step_1": "step_1_model.yaml",
            "step_2": {
                "epochs": 50,
                "checkpoint_every": 10,
                "batch_size": 32,
                "disc_iters": 5,
                "classifier": ["classifier_1.yaml"],
                "weight": [
                    {"gaussian": [{"alpha": 0.5, "var": 0.1}]},
                    {"cd": {"alpha": [1, 2.5]}},
                ],
            },
        },
    }


def write_yaml_file(data: dict) -> str:
    """Write YAML data to a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        with open(temp_file.name, "w", encoding="utf-8") as file:
            yaml.dump(data, file)
    return temp_file.name


def test_read_config_valid(sample_config_data: dict) -> None:
    """Test reading a valid configuration."""
    config_file_path = write_yaml_file(sample_config_data)
    try:
        config = read_config(config_file_path)
        assert config.name == sample_config_data["name"]

        # Adjusting the assertion to access weight properly
        weight = config.train.step_2.weight[0]
        if isinstance(weight, ConfigWeights) and weight.gaussian:
            assert weight.gaussian[0].alpha == 0.5
        else:
            raise AssertionError("Expected a ConfigWeights object with 'gaussian' containing 'alpha'.")
    finally:
        os.remove(config_file_path)


def test_read_config_missing_field(sample_config_data: dict) -> None:
    """Test reading a configuration with a missing required field."""
    del sample_config_data["name"]  # Remove required field
    config_file_path = write_yaml_file(sample_config_data)
    try:
        with pytest.raises(ValidationError, match="missing"):
            read_config(config_file_path)
    finally:
        os.remove(config_file_path)


def test_read_config_invalid_field_type(sample_config_data: dict) -> None:
    """Test reading a configuration with an invalid field type."""
    sample_config_data["num_runs"] = "invalid_integer"  # Invalid type for num_runs
    config_file_path = write_yaml_file(sample_config_data)
    try:
        with pytest.raises(ValidationError, match="int_parsing"):
            read_config(config_file_path)
    finally:
        os.remove(config_file_path)


def test_read_config_add_paths(sample_config_data: dict) -> None:
    """Test reading a configuration and validating that paths are updated properly on any OS."""
    config_file_path = write_yaml_file(sample_config_data)
    os.environ["FILESDIR"] = os.path.join(os.sep, "base", "path")  # Cross-platform

    try:
        config = read_config(config_file_path)
        assert config.out_dir == os.path.join(os.environ["FILESDIR"], "output_directory")
        assert config.data_dir == os.path.join(os.environ["FILESDIR"], "data_directory")
        assert config.train.step_2.classifier[0] == os.path.join(os.environ["FILESDIR"], "classifier_1.yaml")
    finally:
        os.remove(config_file_path)
