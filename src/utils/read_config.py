"""Module for reading config from YAML."""

import os

import yaml

from src.models import ConfigGAN, ConfigMain


def read_config(path: str) -> ConfigGAN:
    """Read and parse GAN configuration from a YAML file."""
    with open(path, encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    # Add base paths using FILESDIR
    files_dir = os.environ.get("FILESDIR", "")
    for rel_path_key in ["out_dir", "data_dir", "fid_stats_path", "test_noise"]:
        if rel_path_key in config_data and isinstance(config_data[rel_path_key], str):
            config_data[rel_path_key] = os.path.join(files_dir, config_data[rel_path_key])

    # Update classifier paths
    if "train" in config_data and "step_2" in config_data["train"]:
        classifiers = config_data["train"]["step_2"].get("classifier", [])
        if isinstance(classifiers, list):
            config_data["train"]["step_2"]["classifier"] = [
                os.path.join(files_dir, rel_path) for rel_path in classifiers
            ]

    return ConfigGAN(**config_data)


def read_main_config(path: str) -> ConfigMain:
    """Read and parse AmbiGAN configuration from a YAML file."""
    with open(path, encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    # Add base paths using FILESDIR
    files_dir = os.environ.get("FILESDIR", "")
    for rel_path_key in ["out_dir", "data_dir", "fid_stats_path", "test_noise"]:
        if rel_path_key in config_data and isinstance(config_data[rel_path_key], str):
            config_data[rel_path_key] = os.path.join(files_dir, config_data[rel_path_key])

    return ConfigMain(**config_data)
