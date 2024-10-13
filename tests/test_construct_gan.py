"""Module for testing of construct gan."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from torch import nn

from src.gan.construct_gan import construct_gan, construct_loss
from src.gan.loss import DiscriminatorLoss, GeneratorLoss
from src.models import (
    ArchitectureType,
    ConfigArchitecture,
    ConfigBinary,
    ConfigCD,
    ConfigDatasetParams,
    ConfigGAN,
    ConfigLoss,
    ConfigLossWG,
    ConfigModel,
    ConfigOptimizer,
    ConfigStep2,
    ConfigTrain,
    ConfigWeights,
    DeviceType,
    ImageParams,
    LossType,
)


@pytest.fixture
def gan_config_dcgan() -> ConfigGAN:
    """Fixture for DCGAN configuration."""
    return ConfigGAN(
        project="test_project",
        name="test_gan",
        out_dir="./out",
        data_dir="./data",
        fid_stats_path="./fid_stats.npz",
        fixed_noise="./fixed_noise.pt",
        test_noise="./test_noise.pt",
        compute_fid=False,
        device=DeviceType.cpu,
        num_workers=2,
        num_runs=1,
        step_1_seeds=[42],
        step_2_seeds=None,
        dataset=ConfigDatasetParams(
            name="mnist",
            binary=ConfigBinary(
                pos=9,
                neg=8,
            ),
        ),
        model=ConfigModel(
            z_dim=100,
            architecture=ConfigArchitecture(
                name=ArchitectureType.dcgan,
                g_filter_dim=64,
                d_filter_dim=64,
                g_num_blocks=3,
                d_num_blocks=3,
            ),
            loss=ConfigLoss(name=LossType.ns),
        ),
        optimizer=ConfigOptimizer(lr=0.0002, beta1=0.5, beta2=0.999),
        train=ConfigTrain(
            step_1="/my_path/to_model.pth",
            step_2=ConfigStep2(
                epochs=10,
                checkpoint_every=1,
                batch_size=32,
                disc_iters=1,
                classifier=["my_classifier.pth"],
                weight=[ConfigWeights(cd=ConfigCD(alpha=[2.1]))],
            ),
        ),
    )


@pytest.fixture
def image_params() -> ImageParams:
    """Fixture for image parameters."""
    return ImageParams(image_size=(3, 64, 64))


def test_construct_gan_dcgan(gan_config_dcgan: ConfigGAN, image_params: ImageParams) -> None:
    """Test constructing DCGAN Generator and Discriminator."""
    G, D = construct_gan(gan_config_dcgan, image_params.image_size)
    assert isinstance(G, nn.Module), "Generator is not an instance of nn.Module"
    assert isinstance(D, nn.Module), "Discriminator is not an instance of nn.Module"


def test_construct_gan_deprecated_architecture(gan_config_dcgan: ConfigGAN, image_params: ImageParams) -> None:
    """Test constructing deprecated DCGAN architecture."""
    gan_config_dcgan.model.architecture.name = ArchitectureType.dcgan_deprecated
    G, D = construct_gan(gan_config_dcgan, image_params.image_size)
    assert isinstance(G, nn.Module), "Generator is not an instance of nn.Module for deprecated DCGAN"
    assert isinstance(D, nn.Module), "Discriminator is not an instance of nn.Module for deprecated DCGAN"


def test_construct_gan_resnet_deprecated(gan_config_dcgan: ConfigGAN, image_params: ImageParams) -> None:
    """Test constructing deprecated ResNet architecture."""
    gan_config_dcgan.model.architecture.name = ArchitectureType.resnet_deprecated
    G, D = construct_gan(gan_config_dcgan, image_params.image_size)
    assert isinstance(G, nn.Module), "Generator is not an instance of nn.Module for ResNet"
    assert isinstance(D, nn.Module), "Discriminator is not an instance of nn.Module for ResNet"


@pytest.fixture
def ns_loss_config() -> ConfigLoss:
    """Fixture for non-saturating loss configuration."""
    return ConfigLoss(name=LossType.ns)


@pytest.fixture
def wgan_loss_config() -> ConfigLossWG:
    """Fixture for WGAN loss configuration."""
    return ConfigLossWG(name=LossType.wgan, args=10.0)


def test_construct_loss_ns(ns_loss_config: ConfigLoss) -> None:
    """Test constructing non-saturating loss functions."""
    D = MagicMock(spec=nn.Module)
    g_loss, d_loss = construct_loss(ns_loss_config, D)
    assert isinstance(g_loss, GeneratorLoss), "Generator loss is not an instance of GeneratorLoss for NS loss"
    assert isinstance(
        d_loss, DiscriminatorLoss
    ), "Discriminator loss is not an instance of DiscriminatorLoss for NS loss"


def test_construct_loss_wgan(wgan_loss_config: ConfigLossWG) -> None:
    """Test constructing WGAN loss functions."""
    D = MagicMock(spec=nn.Module)
    g_loss, d_loss = construct_loss(wgan_loss_config, D)
    assert isinstance(g_loss, GeneratorLoss), "Generator loss is not an instance of GeneratorLoss for WGAN loss"
    assert isinstance(
        d_loss, DiscriminatorLoss
    ), "Discriminator loss is not an instance of DiscriminatorLoss for WGAN loss"


def test_construct_loss_invalid() -> None:
    """Test constructing loss with an invalid loss type."""
    D = MagicMock(spec=nn.Module)
    with pytest.raises(ValidationError, match=r".*validation error for ConfigLoss.*"):
        construct_loss(ConfigLoss(name="invalid_loss"), D)


def test_construct_gan_invalid_architecture(gan_config_dcgan: ConfigGAN, image_params: ImageParams) -> None:
    """Test constructing GAN with an invalid architecture type."""
    gan_config_dcgan.model.architecture.name = "invalid_architecture"
    with pytest.raises(ValueError, match="Invalid architecture type.*"):
        construct_gan(gan_config_dcgan, image_params.image_size)
