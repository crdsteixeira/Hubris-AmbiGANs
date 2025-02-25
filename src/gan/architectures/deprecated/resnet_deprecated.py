import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, List


class ConvScale(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        scale: Union[str, None] = None,
    ) -> None:
        """Convolutional layer with optional upsampling or downsampling."""
        super().__init__()
        self.block = nn.Sequential()

        if scale == "up":
            self.block.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding="same"))

        if scale == "down":
            self.block.append(nn.AvgPool2d(2))

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass through ConvScale block."""
        return self.block(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Union[str, None] = None,
        batch_norm: bool = True,
        size: Tuple[int, int] = (-1, -1),
    ) -> None:
        """Residual block with optional upsampling or downsampling."""
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        nh, nw = size

        if in_channels == out_channels and scale is None:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvScale(in_channels, out_channels, scale=scale)

        if scale == "down":
            conv1 = ConvScale(in_channels, in_channels, scale=None)
            conv2 = ConvScale(in_channels, out_channels, scale=scale)
        elif scale == "up":
            conv1 = ConvScale(in_channels, out_channels, scale=scale)
            conv2 = ConvScale(out_channels, out_channels, scale=None)
        elif scale is None:
            conv1 = ConvScale(in_channels, out_channels)
            conv2 = ConvScale(out_channels, out_channels)
        else:
            raise Exception("invalid scale option")

        norm1 = nn.BatchNorm2d(in_channels) if batch_norm else nn.Identity()
        nc = in_channels if scale == "down" else out_channels
        if scale == "up":
            nh *= 2
            nw *= 2

        norm2 = nn.BatchNorm2d(nc) if batch_norm else nn.Identity()

        self.conv1 = nn.Sequential(
            norm1,
            nn.ReLU(),
            conv1,
        )
        self.conv2 = nn.Sequential(
            norm2,
            nn.ReLU(),
            conv2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass through Residual block."""
        shortcut = self.shortcut(x)
        output = x
        output = self.conv1(output)
        output = self.conv2(output)
        output += shortcut
        return output


class Generator(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int, int],
        z_dim: int,
        gf_dim: int,
    ) -> None:
        """Generator model to generate images from latent vectors."""
        super().__init__()
        self.image_size = image_size
        self.z_dim = z_dim
        self.gf_dim = gf_dim

        n_channels = image_size[0]

        self.project = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * gf_dim, bias=False),
        )
        self.blocks = nn.Sequential(
            ResNetBlock(gf_dim, gf_dim, scale="up"),
            ResNetBlock(gf_dim, gf_dim, scale="up"),
            ResNetBlock(gf_dim, gf_dim, scale="up"),
        )
        self.gen = nn.Sequential(nn.BatchNorm2d(gf_dim), ConvScale(gf_dim, n_channels), nn.Tanh())

    def forward(self, z: Tensor) -> Tensor:
        """Generate an image from input latent vector z."""
        output = self.project(z)
        output = self.blocks(output.view(-1, self.gf_dim, 4, 4))
        output = self.gen(output)
        return output


class ReduceMean(nn.Module):
    def __init__(self, dim: Union[int, List[int]]) -> None:
        """Layer to reduce mean over a specified dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Perform mean reduction over specified dimension."""
        return x.mean(dim=self.dim)


class Discriminator(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int, int],
        df_dim: int = 128,
        use_batch_norm: bool = True,
        is_critic: bool = False,
    ) -> None:
        """Discriminator model to classify real and generated images."""
        super().__init__()
        self.image_size = image_size
        self.df_dim = df_dim

        n_channels = image_size[0]

        self.blocks = nn.Sequential(
            ResNetBlock(
                n_channels,
                df_dim,
                scale="down",
                batch_norm=use_batch_norm,
                size=(32, 32),
            ),
            ResNetBlock(df_dim, df_dim, scale="down", batch_norm=use_batch_norm, size=(16, 16)),
            ResNetBlock(df_dim, df_dim, scale=None, batch_norm=use_batch_norm, size=(8, 8)),
            ResNetBlock(df_dim, df_dim, scale=None, batch_norm=use_batch_norm, size=(8, 8)),
        )

        self.predict = nn.Sequential(nn.ReLU(), ReduceMean(dim=[2, 3]), nn.Linear(df_dim, 1))

        if not is_critic:
            self.predict.append(nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """Classify input image tensor x as real or fake."""
        output = x
        output = self.blocks(output)
        output = self.predict(output).squeeze()
        return output