"""Utility module for datasets."""

import logging

import torch
from torch.utils.data import Dataset

from src.models import LoadDatasetParams

logger = logging.getLogger(__name__)


class BinaryDataset(Dataset):
    """Dataset for binary classification from a given dataset with specified positive and negative classes."""

    def __init__(self, original_dataset: Dataset, params: LoadDatasetParams) -> None:
        """Initialize the BinaryDataset by filtering the original dataset for specified classes."""
        data = []
        targets = []

        for X, y in original_dataset:
            if y in (params.pos_class, params.neg_class):
                data.append(torch.unsqueeze(X, dim=0))
                targets.append(
                    torch.ones((1,), dtype=torch.float32)
                    if y == params.pos_class
                    else torch.zeros((1,), dtype=torch.float32)
                )

        if len(data) == 0:
            error = f"""The dataset '{params.dataset_name}' is empty. \n
              The classes you provided were {params.pos_class} and {params.neg_class}. \n
              They are not available in the dataset provided."""
            logger.error(error)
            raise ValueError(error)

        self.data: torch.Tensor = torch.vstack(data)
        self.targets: torch.Tensor = torch.hstack(targets)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the data and label at the specified index."""
        return self.data[index], self.targets[index]
