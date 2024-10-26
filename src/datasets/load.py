"""Module for loading the datasets."""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

from src.datasets.datasets import (
    get_chest_xray,
    get_cifar10,
    get_fashion_mnist,
    get_mnist,
)
from src.datasets.utils import BinaryDataset
from src.enums import DatasetNames
from src.models import DatasetParams, LoadDatasetParams, ImageParams

logger = logging.getLogger(__name__)


def get_function(dataset_name: DatasetNames) -> Callable[[DatasetParams], Any]:
    """Retrieve the function to load the dataset."""
    mapping: dict[DatasetNames, Callable[[DatasetParams], Any]] = {
        DatasetNames.mnist: get_mnist,
        DatasetNames.fashion_mnist: get_fashion_mnist,
        DatasetNames.cifar10: get_cifar10,
        DatasetNames.chest_xray: get_chest_xray,
    }
    return mapping[dataset_name]


def load_dataset(params: LoadDatasetParams) -> tuple[Dataset, int, ImageParams]:
    """
    Load dataset, optionally modify it for binary classification.
    Return  with class and image size information.
    """
    try:
        DatasetNames.valid_dataset(params.dataset_name)
        logger.info(f"{params.dataset_name.value} is a valid dataset.")
    except ValueError as e:
        logger.error(e)
        raise e

    download_function = get_function(params.dataset_name)
    download_params = DatasetParams(dataroot=params.dataroot, train=params.train, pytesting=params.pytesting)
    dataset = download_function(download_params)

    # Check if the dataset is empty and log an error
    if len(dataset) == 0:
        error = f"The dataset '{params.dataset_name}' is empty. Please verify the data availability."
        logger.error(error)
        raise ValueError(error)

    image_size = tuple(dataset.data.shape[1:])
    if len(image_size) == 2:
        image_size = (1, *image_size)

    # TODO: check is this is the correct way to change from HWC to CHW
    elif len(image_size) == 3 and image_size[2] == 3:
        image_size = (image_size[2], image_size[0], image_size[1])

    targets = dataset.targets if torch.is_tensor(dataset.targets) else torch.tensor(dataset.targets)
    num_classes = targets.unique().size()

    if params.pos_class is not None and params.neg_class is not None:
        num_classes = 2
        dataset = BinaryDataset(dataset, params)

    return dataset, num_classes, ImageParams(image_size=image_size)
