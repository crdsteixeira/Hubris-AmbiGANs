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
        self.transform = original_dataset.transform

        self.balance_dataset()

    def balance_dataset(self) -> None:
        """Oversample the minority class to balance the dataset."""
        pos_indices = (self.targets == 1).nonzero(as_tuple=True)[0]
        neg_indices = (self.targets == 0).nonzero(as_tuple=True)[0]

        # Determine which class is the minority
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        if num_pos == num_neg:
            return  # Dataset is already balanced

        # Oversample the minority class
        if num_pos < num_neg:
            minority_indices = pos_indices
            num_to_add = num_neg - num_pos
            minority_target = 1

        else:
            minority_indices = neg_indices
            num_to_add = num_pos - num_neg
            minority_target = 0

        # Randomly duplicate samples from the minority class
        oversampled_data = self.data[minority_indices][
            torch.randint(len(minority_indices), (num_to_add,), generator=torch.manual_seed(42))
        ]
        oversampled_targets = torch.full((num_to_add,), fill_value=minority_target)

        # Append the oversampled data to the existing dataset
        self.data = torch.vstack([self.data, oversampled_data])
        self.targets = torch.hstack([self.targets, oversampled_targets])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the data and label at the specified index."""
        return self.data[index], self.targets[index]
