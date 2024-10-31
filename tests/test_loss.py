"""Module for testing of loss function classes."""

import pytest
import torch
from torch import nn

from src.gan.loss import (
    DiscriminatorLoss,
    GeneratorLoss,
    NS_DiscriminatorLoss,
    NS_GeneratorLoss,
    W_DiscrimatorLoss,
    W_GeneratorLoss,
    WGP_DiscriminatorLoss,
)


@pytest.fixture
def real_fake_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixture to provide real and fake data tensors."""
    real_data = torch.randn(16, 3, 64, 64)
    fake_data = torch.randn(16, 3, 64, 64)
    return real_data, fake_data


@pytest.fixture
def discriminator_outputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixture to provide discriminator outputs for real and fake data."""
    real_output = torch.sigmoid(torch.randn(16, 1))
    fake_output = torch.sigmoid(torch.randn(16, 1))
    return real_output, fake_output


@pytest.fixture
def device() -> torch.device:
    """Fixture to provide device for computation."""
    return torch.device("cpu")


# Test NS_DiscriminatorLoss


def test_ns_discriminator_loss(
    real_fake_data: tuple[torch.Tensor, torch.Tensor],
    discriminator_outputs: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    """Test the non-saturating loss for the discriminator."""
    real_data, fake_data = real_fake_data
    real_output, fake_output = discriminator_outputs

    ns_loss = NS_DiscriminatorLoss()
    loss, _ = ns_loss(real_data, fake_data, real_output, fake_output, device)

    assert isinstance(loss, torch.Tensor), "NS_DiscriminatorLoss should return a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"


# Test W_DiscrimatorLoss


def test_w_discriminator_loss(
    real_fake_data: tuple[torch.Tensor, torch.Tensor],
    discriminator_outputs: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    """Test the Wasserstein loss for the discriminator."""
    real_data, fake_data = real_fake_data
    real_output, fake_output = discriminator_outputs

    w_loss = W_DiscrimatorLoss()
    loss, _ = w_loss(real_data, fake_data, real_output, fake_output, device)

    assert isinstance(loss, torch.Tensor), "W_DiscrimatorLoss should return a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"


# Test WGP_DiscriminatorLoss


def test_wgp_discriminator_loss(
    real_fake_data: tuple[torch.Tensor, torch.Tensor],
    discriminator_outputs: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    """Test the Wasserstein loss with gradient penalty for the discriminator."""
    real_data, fake_data = real_fake_data
    real_output, fake_output = discriminator_outputs

    D = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),
    )  # Simple discriminator for testing
    wgp_loss = WGP_DiscriminatorLoss(D, lmbda=10.0)
    loss, terms = wgp_loss(real_data, fake_data, real_output, fake_output, device)

    assert isinstance(loss, torch.Tensor), "WGP_DiscriminatorLoss should return a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "W_distance" in terms, "Terms should contain 'W_distance'"
    assert "D_loss" in terms, "Terms should contain 'D_loss'"
    assert "GP" in terms, "Terms should contain 'GP'"


# Test NS_GeneratorLoss


def test_ns_generator_loss(device: torch.device) -> None:
    """Test the non-saturating loss for the generator."""
    output = torch.sigmoid(torch.randn(16, 1))
    ns_g_loss = NS_GeneratorLoss()
    loss = ns_g_loss(device, output)

    assert isinstance(loss, torch.Tensor), "NS_GeneratorLoss should return a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"


# Test W_GeneratorLoss


def test_w_generator_loss(device: torch.device) -> None:
    """Test the Wasserstein loss for the generator."""
    output = torch.randn(16, 1)
    w_g_loss = W_GeneratorLoss()
    loss = w_g_loss(device, output)

    assert isinstance(loss, torch.Tensor), "W_GeneratorLoss should return a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"


# Test base classes (DiscriminatorLoss and GeneratorLoss)


def test_discriminator_loss_base() -> None:
    """Test the base DiscriminatorLoss class for NotImplementedError."""
    loss = DiscriminatorLoss([])
    with pytest.raises(NotImplementedError):
        loss(None, None, None, None, None)


def test_generator_loss_base() -> None:
    """Test the base GeneratorLoss class for NotImplementedError."""
    loss = GeneratorLoss([])
    with pytest.raises(NotImplementedError):
        loss(None, None)
