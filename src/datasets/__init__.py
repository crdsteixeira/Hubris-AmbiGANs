import torch

from src.constants import DatasetManager

from .utils import BinaryDataset


def load_dataset(name, data_dir, pos_class=None, neg_class=None, train=True):
    if not DatasetManager.valid_dataset(name):
        print(f"{name} dataset not supported")
        exit(-1)

    dataset = DatasetManager.get_dataset(name, data_dir, train=train)

    image_size = tuple(dataset.data.shape[1:])
    if len(image_size) == 2:
        image_size = (1, *image_size)
    elif len(image_size) == 3 and image_size[2] == 3:
        image_size = (image_size[2], image_size[0], image_size[1])

    targets = (
        dataset.targets
        if torch.is_tensor(dataset.targets)
        else torch.tensor(dataset.targets)
    )
    num_classes = targets.unique().size()

    if pos_class is not None and neg_class is not None:
        num_classes = 2
        dataset = BinaryDataset(dataset, pos_class, neg_class)

    return dataset, num_classes, image_size
