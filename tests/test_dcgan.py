import pytest
import torch
from torch import nn, Tensor
from src.gan.architectures.dcgan import (
    Generator,
    Discriminator,
    ConvTranspose2dSame,
    weights_init,
    conv_out_size_same,
    compute_padding_same,
)
from src.models import ConvParams, DisParams, GenParams, PadParams

@pytest.fixture
def gen_params():
    """Fixture to provide valid parameters for the Generator."""
    return GenParams(
        image_size=(1, 28, 28),  # grayscale image, 28x28
        z_dim=100,
        n_blocks=3,
        filter_dim=64
    )

@pytest.fixture
def dis_params():
    """Fixture to provide valid parameters for the Discriminator."""
    return DisParams(
        image_size=(1, 28, 28),  # grayscale image, 28x28
        n_blocks=2,
        filter_dim=64,
        use_batch_norm=True,
        is_critic=False
    )

@pytest.fixture
def conv_params():
    """Fixture to provide parameters for ConvTranspose2dSame."""
    return ConvParams(
        in_channels=64,
        out_channels=1,
        in_size=(14, 14),
        out_size=(28, 28),
        kernel=4,
        stride=2,
        bias=True
    )

@pytest.fixture
def pad_params():
    """Fixture to provide parameters for padding calculation."""
    return PadParams(
        in_size=14,
        out_size=28,
        kernel=4,
        stride=2
    )


def test_weights_init() -> None:
    """Test weight initialization for a Conv layer."""
    layer = nn.Conv2d(3, 6, kernel_size=3)
    weights_init(layer)
    assert layer.weight.mean().item() == pytest.approx(0.0, abs=0.1), "Weights should be initialized around 0."
    assert layer.bias is not None and layer.bias.mean().item() == pytest.approx(0.0, abs=0.1), "Biases should be initialized around 0."


def test_conv_out_size_same() -> None:
    """Test computation of output size for convolution with 'same' padding."""
    size = 28
    stride = 2
    expected_output = 14
    assert conv_out_size_same(size, stride) == expected_output, f"Expected output size {expected_output}, got {conv_out_size_same(size, stride)}"


def test_compute_padding_same(pad_params: PadParams) -> None:
    """Test computation of padding for ConvTranspose2d layer to maintain same output size."""
    padding, out_padding = compute_padding_same(pad_params)
    assert padding == 1, "Expected padding of 1"
    assert out_padding == 0, "Expected output padding of 0"


def test_conv_transpose_2d_same(conv_params: ConvParams) -> None:
    """Test forward pass through ConvTranspose2dSame layer."""
    layer = ConvTranspose2dSame(conv_params)
    input_tensor = torch.randn(1, 64, 14, 14)  # batch size 1, 64 channels, 14x14
    output = layer(input_tensor)
    assert output.shape == (1, 1, 28, 28), f"Expected output shape (1, 1, 28, 28), got {output.shape}"


def test_generator_forward(gen_params: GenParams) -> None:
    """Test forward pass through the Generator."""
    generator = Generator(gen_params)
    z = torch.randn(1, gen_params.z_dim)  # batch size 1, latent vector of length z_dim
    output = generator(z)
    assert output.shape == (1, *gen_params.image_size), f"Expected output shape {(1, *gen_params.image_size)}, got {output.shape}"


def test_discriminator_forward(dis_params: DisParams) -> None:
    """Test forward pass through the Discriminator."""
    discriminator = Discriminator(dis_params)
    x = torch.randn(1, *dis_params.image_size)  # batch size 1, image size (channels, height, width)
    output = discriminator(x)
    assert isinstance(output.item(), float), "Expected output to be a scalar float value"
