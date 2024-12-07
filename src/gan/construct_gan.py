"""Module for constructing GAN."""

import logging

from torch import nn

from src.enums import ArchitectureType, LossType
from src.gan.architectures.dcgan import Discriminator, Generator
from src.gan.architectures.deprecated.dcgan_deprecated import (
    Discriminator as DiscriminatorDeprecated,
)
from src.gan.architectures.deprecated.dcgan_deprecated import (
    Generator as GeneratorDeprecated,
)
from src.gan.architectures.deprecated.resnet_deprecated import (
    Discriminator as ResnetDiscriminatorDeprecated,
)
from src.gan.architectures.deprecated.resnet_deprecated import (
    Generator as ResnetGeneratorDeprecated,
)
from src.gan.loss import (
    DiscriminatorLoss,
    GeneratorLoss,
    NS_DiscriminatorLoss,
    NS_GeneratorLoss,
    W_GeneratorLoss,
    WGP_DiscriminatorLoss,
)
from src.models import (
    ConfigGAN,
    ConfigLoss,
    ConfigLossWG,
    DisParams,
    GenParams,
    ImageParams,
)

logger = logging.getLogger(__name__)


def construct_gan(config: ConfigGAN, img_size: ImageParams) -> tuple[nn.Module, nn.Module]:
    """Construct generator and discriminator models based on the given configuration."""
    if config.model.loss.name != LossType.wgan:
        use_batch_norm = True
        is_critic = False
    else:
        use_batch_norm = False
        is_critic = True

    if config.model.architecture.name == ArchitectureType.dcgan:
        G = Generator(
            GenParams(
                image_size=img_size.image_size,
                z_dim=config.model.z_dim,
                n_blocks=config.model.architecture.g_num_blocks,
                filter_dim=config.model.architecture.g_filter_dim,
            )
        ).to(config.device.value)

        D = Discriminator(
            DisParams(
                image_size=img_size.image_size,
                filter_dim=config.model.architecture.d_filter_dim,
                n_blocks=config.model.architecture.d_num_blocks,
                use_batch_norm=use_batch_norm,
                is_critic=is_critic,
            )
        ).to(config.device.value)

    # These architectures are deprecated and were kept for compatibility
    # Won't be maintained

    elif config.model.architecture.name == ArchitectureType.dcgan_deprecated:
        G = GeneratorDeprecated(
            img_size.image_size,
            z_dim=config.model.z_dim,
            filter_dim=config.model.architecture.g_filter_dim,
            n_blocks=config.model.architecture.g_num_blocks,
        ).to(config.device.value)

        D = DiscriminatorDeprecated(
            img_size.image_size,
            filter_dim=config.model.architecture.d_filter_dim,
            n_blocks=config.model.architecture.d_num_blocks,
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(config.device.value)

    elif config.model.architecture.name == ArchitectureType.resnet_deprecated:
        G = ResnetGeneratorDeprecated(
            img_size.image_size, z_dim=config.model.z_dim, gf_dim=config.model.architecture.g_filter_dim
        ).to(config.device.value)
        D = ResnetDiscriminatorDeprecated(
            img_size.image_size,
            df_dim=config.model.architecture.d_filter_dim,
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(config.device.value)

    else:
        valid_architecture_types = [loss.value for loss in ArchitectureType]  # Get the list of valid loss types
        error = (
            f"""Invalid architecture type '{config.model.architecture.name}'."""
            f"""Supported types are: {', '.join(valid_architecture_types)}."""
        )
        logger.error(error)
        raise ValueError(error)

    return G, D


def construct_generator_loss(config: ConfigLoss | ConfigLossWG) -> GeneratorLoss:
    """Construct loss functions for generator and discriminator based on the given configuration."""
    if config.name == LossType.ns:
        return NS_GeneratorLoss()
    if config.name == LossType.wgan:
        return W_GeneratorLoss()

    valid_loss_types = [loss.value for loss in LossType]  # Get the list of valid loss types
    error = f"Invalid loss type '{config.name}'. Supported types are: {', '.join(valid_loss_types)}."
    logger.error(error)
    raise ValueError(error)


def construct_discriminator_loss(config: ConfigLoss | ConfigLossWG, D: nn.Module) -> DiscriminatorLoss:
    """Construct loss functions for discriminator based on the given configuration."""
    if config.name == LossType.ns:
        return NS_DiscriminatorLoss()
    if config.name == LossType.wgan and isinstance(config, ConfigLossWG):
        return WGP_DiscriminatorLoss(D, config.args)

    valid_loss_types = [loss.value for loss in LossType]  # Get the list of valid loss types
    error = f"Invalid loss type '{config.name}'. Supported types are: {', '.join(valid_loss_types)}."
    logger.error(error)
    raise ValueError(error)
