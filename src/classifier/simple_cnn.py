"""Module for a Simple CNN."""

import math
import logging

import torch
from torch import nn

from src.models import ClassifierParams, PoolParams

logger = logging.getLogger(__name__)


def pool_out(params: PoolParams) -> int:
    """Calculate the output size after a pooling operation using PoolingParams."""
    stride = params.kernel if params.stride is None else params.stride
    out_size = (params.in_size + 2 * params.padding - params.dilation * (params.kernel - 1) - 1) / stride + 1
    return int(math.floor(out_size))


class Classifier(nn.Module):
    """CNN classifier that processes images with convolutional layers and combines outputs for classification."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize the CNN classifier using ClassifierParams."""
        super().__init__()
        nc: int
        nh: int
        nw: int
        nc, nh, nw = params.img_size
        self.blocks = nn.ModuleList()

        n_in = nc
        if isinstance(params.nf, list):
            for nf in params.nf:
                # Ensure nf is an int for type safety.
                assert isinstance(nf, int), f"Expected nf to be an int, got {type(nf)}"
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(n_in, nf, 3, padding="same"),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                )
                pool_params = PoolParams(in_size=nh, kernel=2)
                nh = pool_out(pool_params)
                nw = pool_out(pool_params)
                n_in = nf

        else:
            error = f"Expected params.nf to be a list, got type: {type(params.nf)}"
            logger.error(error)
            raise TypeError(error)

        self.blocks.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(nh * nw * n_in, 1 if params.n_classes == 2 else params.n_classes),
                nn.Sigmoid() if params.n_classes == 2 else nn.Softmax(dim=1),
            )
        )

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False) -> torch.Tensor:
        """Forward pass through the CNN classifier."""
        intermediate_outputs = []
        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
