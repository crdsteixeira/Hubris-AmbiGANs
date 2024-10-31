"""Module for DCGAN."""

import numpy as np
from torch import Tensor, nn

from src.models import ConvParams, DisParams, GenParams, PadParams


def weights_init(m: nn.Module) -> None:
    """Initialize weights for the given neural network layer based on its class type."""
    classname = m.__class__.__name__
    if classname.find("Same") != -1:
        nn.init.normal_(m.conv_transpose_2d.weight.data, 0.0, 0.02)
        if m.conv_transpose_2d.bias is not None:
            nn.init.constant_(m.conv_transpose_2d.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def conv_out_size_same(size: int, stride: int) -> int:
    """Calculate the output size of a convolution operation with 'same' padding."""
    return int(np.ceil(float(size) / float(stride)))


def compute_padding_same(params: PadParams) -> tuple[int, int]:
    """Compute the padding and output padding needed to maintain the same output size as the input size."""
    res = (params.in_size - 1) * params.stride - params.out_size + params.kernel

    out_padding = 0 if (res % 2 == 0) else 1
    padding = (res + out_padding) / 2

    return int(padding), int(out_padding)


class ConvTranspose2dSame(nn.Module):
    """ConvTranspose2d layer with padding calculation to maintain the same input and output size."""

    def __init__(self, params: ConvParams) -> None:
        """Initialize ConvTranspose2d with parameters to keep the same input and output dimensions."""
        super().__init__()

        in_height, in_width = params.in_size
        out_height, out_width = params.out_size

        pad_height, out_pad_height = compute_padding_same(
            PadParams(in_size=in_height, out_size=out_height, kernel=params.kernel, stride=params.stride)
        )

        pad_width, out_pad_width = compute_padding_same(
            PadParams(in_size=in_width, out_size=out_width, kernel=params.kernel, stride=params.stride)
        )

        pad = (pad_height, pad_width)
        out_pad = (out_pad_height, out_pad_width)

        self.conv_transpose_2d = nn.ConvTranspose2d(
            params.in_channels, params.out_channels, params.kernel, params.stride, pad, out_pad, bias=params.bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ConvTranspose2d layer."""
        return self.conv_transpose_2d(x)


class Generator(nn.Module):
    """Generative model that uses transposed convolution layers to produce images from latent vectors."""

    def __init__(self, params: GenParams) -> None:
        """
        Initialize the Generator with specified architecture parameters.

        Args:
            params: instance of GenParams model.
            params.image_size (Tuple[int, int, int]): Dimensions of the output image (channels, height, width).
            params.z_dim (int): Dimension of the latent code. Default is 100.
            params.n_blocks (int): Number of transposed convolutional blocks. Default is 3.
            params.filter_dim (int): Base number of filters in the first transposed convolutional layer. Default is 64.

        """
        super().__init__()
        self.params = params

        n_channels, current_size_height, current_size_width = self.params.image_size
        conv_blocks_rev = nn.ModuleList()

        for i in range(self.params.n_blocks):
            current_size_height_smaller = conv_out_size_same(current_size_height, 2)
            current_size_width_smaller = conv_out_size_same(current_size_width, 2)

            if i == 0:
                block = nn.Sequential(
                    ConvTranspose2dSame(
                        ConvParams(
                            in_channels=self.params.filter_dim,
                            out_channels=n_channels,
                            in_size=(current_size_height_smaller, current_size_width_smaller),
                            out_size=(current_size_height, current_size_width),
                            kernel=4,
                            stride=2,
                            bias=True,
                        )
                    ),
                    nn.Tanh(),
                )
            else:
                in_channels = self.params.filter_dim * 2**i
                out_channels = self.params.filter_dim * 2 ** (i - 1)

                block = nn.Sequential(
                    ConvTranspose2dSame(
                        ConvParams(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            in_size=(current_size_height_smaller, current_size_width_smaller),
                            out_size=(current_size_height, current_size_width),
                            kernel=4,
                            stride=2,
                            bias=False,
                        )
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )

            current_size_height, current_size_width = current_size_height_smaller, current_size_width_smaller

            conv_blocks_rev.append(block)

        self.project_out_reshape_dim = (
            self.params.filter_dim * (2 ** (self.params.n_blocks - 1)),
            current_size_height,
            current_size_width,
        )

        self.project = nn.Sequential(
            nn.ConvTranspose2d(
                self.params.z_dim,
                self.project_out_reshape_dim[0],
                (current_size_height, current_size_width),
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(self.project_out_reshape_dim[0]),
            nn.ReLU(True),
        )

        self.conv_blocks = nn.Sequential()
        for i in reversed(conv_blocks_rev):
            self.conv_blocks.append(i)

        self.apply(weights_init)

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass through the generator."""
        z = self.project(z.view(-1, self.params.z_dim, 1, 1))
        z = self.conv_blocks(z.view(-1, *self.project_out_reshape_dim))
        return z


class Discriminator(nn.Module):
    """Discriminator model used to classify generated and real images for GAN training."""

    def __init__(self, params: DisParams) -> None:
        """
        Initialize the Discriminator with specified architecture parameters.

        Args:
            params: instance of DisParams model.
            params.image_size (Tuple[int, int, int]): Input image dimensions (channels, height, width).
            params.n_blocks (int): Number of convolutional blocks in the discriminator. Default is 2.
            params.filter_dim (int): Base dimension of filters in convolutional layers. Default is 64.
            params.use_batch_norm (bool): Whether to use batch normalization in intermediate layers. Default is True.
            params.is_critic (bool=False): Indicates whether the discriminator is a critic (without sigmoid).

        """
        super().__init__()
        self.params = params

        n_channels, current_size_height, current_size_width = self.params.image_size

        self.conv_blocks = nn.Sequential()

        for i in range(self.params.n_blocks):
            current_size_height = conv_out_size_same(current_size_height, 2)
            current_size_width = conv_out_size_same(current_size_width, 2)

            out_channels = self.params.filter_dim * 2**i
            if i == 0:
                in_channels = n_channels
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                in_channels = self.params.filter_dim * 2 ** (i - 1)
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    (
                        nn.BatchNorm2d(out_channels)
                        if self.params.use_batch_norm
                        else nn.LayerNorm(normalized_shape=[out_channels, current_size_height, current_size_width])
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )

            self.conv_blocks.append(block)

        self.predict = nn.Sequential(
            nn.Conv2d(
                in_channels=self.params.filter_dim * 2 ** (self.params.n_blocks - 1),
                out_channels=1,
                kernel_size=(current_size_height, current_size_width),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Flatten(),
        )

        if not self.params.is_critic:
            self.predict.append(nn.Sigmoid())

        self.apply(weights_init)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the discriminator."""
        for block in self.conv_blocks:
            x = block(x)
        return self.predict(x).squeeze()
