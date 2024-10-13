# pylint: skip-file

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from dotenv import load_dotenv

from src.constants import DatasetManager  # Updated to use DatasetManager
from src.datasets.utils import BinaryDataset
from src.metrics import fid
from src.metrics.fid import get_inception_feature_map_fn
from src.utils.checkpoint import construct_classifier_from_checkpoint

# Load environment variables
load_dotenv()

# Argument parser setup
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data",
    dest="dataroot",
    default=f"{os.environ['FILESDIR']}/data",
    help="Directory with dataset",
)
parser.add_argument(
    "--dataset",
    dest="dataset",
    default="fashion-mnist",
    help="Dataset (mnist, fashion-mnist, etc.)",
)
parser.add_argument(
    "--pos",
    dest="pos_class",
    default=3,
    type=int,
    help="Positive class for binary classification",
)
parser.add_argument(
    "--neg",
    dest="neg_class",
    default=0,
    type=int,
    help="Negative class for binary classification",
)
parser.add_argument("--batch-size", type=int, default=64, help="Batch size to use")
parser.add_argument(
    "--model-path",
    dest="model_path",
    default=None,
    type=str,
    help="Path to classifier. If none, uses InceptionV3",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=6,
    help="Number of worker processes for data loading",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda, cuda:0, or cpu)")
parser.add_argument("--name", dest="name", default=None, help="Name of generated .npz file")


def load_dataset(args):
    """Load the specified dataset."""
    if DatasetManager.valid_dataset(args.dataset):
        return DatasetManager.get_dataset(args.dataset, args.dataroot)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")


def get_feature_map_function(args, device, dset):
    """Get the feature map function based on model path."""
    if args.model_path is None:
        return get_inception_feature_map_fn(device), None
    else:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError("Model Path doesn't exist")

        model = construct_classifier_from_checkpoint(args.model_path)[0]
        model.to(device)
        model.eval()
        model.output_feature_maps = True

        def feature_map_fn(images, batch):
            return model(images, batch)[1]

        dims = feature_map_fn(dset.data[0:1], (0, 1)).size(1)
        return feature_map_fn, dims


def main():
    args = parser.parse_args()
    print(args)

    # Set device
    device = torch.device("cpu" if args.device is None else args.device)
    print("Using device:", device)

    # Load dataset
    try:
        dset = load_dataset(args)
    except ValueError as e:
        print(e)
        exit(-1)

    # Handle binary classification
    binary_classification = args.pos_class is not None and args.neg_class is not None
    if binary_classification:
        dset = BinaryDataset(dset, args.pos_class, args.neg_class)

    print("Dataset size:", len(dset))

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Get feature map function and dimensions
    get_feature_map_fn, dims = get_feature_map_function(args, device, dset)

    # Calculate FID statistics
    m, s = fid.calculate_activation_statistics_dataloader(dataloader, get_feature_map_fn, dims=dims, device=device)

    # Save FID statistics
    os.makedirs(os.path.join(args.dataroot, "fid-stats"), exist_ok=True)
    stats_filename = os.path.join(args.dataroot, "fid-stats", f'{args.name or f"stats.{args.dataset}"}')
    with open(f"{stats_filename}.npz", "wb") as f:
        np.savez(f, mu=m, sigma=s)
    print(f"FID statistics saved to {stats_filename}.npz")


if __name__ == "__main__":
    main()
