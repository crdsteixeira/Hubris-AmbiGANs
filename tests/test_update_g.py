import pytest
from unittest.mock import MagicMock
import torch
from torch import nn, optim
from typing import Callable
from src.gan.update_g import (
    UpdateGeneratorGAN,
    UpdateGeneratorGASTEN,
    UpdateGeneratorGastenMgda,
    UpdateGeneratorAmbiGanGaussian,
    UpdateGeneratorAmbiGanKLDiv,
    UpdateGeneratorAmbiGanGaussianIdentity,
)
from src.models import DeviceType

@pytest.fixture
def mock_models():
    # Mock Generator and Discriminator
    G = MagicMock(spec=nn.Module)
    D = MagicMock(spec=nn.Module)
    C = MagicMock(spec=nn.Module)
    C_identity = MagicMock(spec=nn.Module)

    # Set up a mock for optimizer
    optimizer = MagicMock(spec=optim.Optimizer)

    # Mock noise and device
    noise = torch.randn((16, 100), requires_grad=True)  # Example shape with requires_grad=True
    device = torch.device(DeviceType.cpu)

    # Mocking G, D, and C outputs to have requires_grad=True
    fake_data = torch.randn((16, 3, 64, 64), requires_grad=True)  # Example shape with requires_grad=True
    G.return_value = fake_data
    D.return_value = torch.rand((16, 1), requires_grad=True)  # Output from discriminator also requires grad
    C.return_value = torch.rand((16, 1), requires_grad=True)  # Classifier output also requires grad

    # Let's simulate two different tensors for different purposes (e.g., class predictions, auxiliary features)
    C_identity.return_value = (
        torch.sigmoid(torch.randn((16, 10))),  # Prediction from the several classifiers 
        torch.sigmoid(torch.randn((16, 1)))    # Final predictions from ambiguity estimator 
    )

    return G, D, C, C_identity, optimizer, noise, device

@pytest.fixture
def mock_crit():
    # Mock criterion that returns a scalar tensor requiring gradient
    return MagicMock(return_value=torch.tensor(1.0, requires_grad=True))


def test_update_generator_gan(mock_models: tuple, mock_crit: Callable) -> None:
    """Test UpdateGeneratorGAN for correct loss and terms output."""
    G, D, _, _, optimizer, noise, device = mock_models

    updater = UpdateGeneratorGAN(crit=mock_crit)
    loss, terms = updater(G, D, optimizer, noise, device)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "original_g_loss" not in terms, "There shouldn't be any loss terms for UpdateGeneratorGAN"


def test_update_generator_gasten(mock_models: tuple, mock_crit: Callable) -> None:
    """Test UpdateGeneratorGASTEN for correct loss and terms output."""
    G, D, C, _, optimizer, noise, device = mock_models
    alpha = 0.5

    updater = UpdateGeneratorGASTEN(crit=mock_crit, C=C, alpha=alpha)
    loss, terms = updater(G, D, optimizer, noise, device)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "original_g_loss" in terms, "Terms should contain 'original_g_loss'"
    assert "conf_dist_loss" in terms, "Terms should contain 'conf_dist_loss'"


def test_update_generator_gasten_mgda(mock_models: tuple, mock_crit: Callable) -> None:
    """Test UpdateGeneratorGastenMgda for correct loss and terms output."""
    G, D, C, _, optimizer, noise, device = mock_models
    alpha = 1.0
    normalize = True

    updater = UpdateGeneratorGastenMgda(crit=mock_crit, C=C, alpha=alpha, normalize=normalize)
    loss, terms = updater(G, D, optimizer, noise, device)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "original_g_loss" in terms, "Terms should contain 'original_g_loss'"
    assert "conf_dist_loss" in terms, "Terms should contain 'conf_dist_loss'"
    assert "scale1" in terms, "Terms should contain 'scale1'"
    assert "scale2" in terms, "Terms should contain 'scale2'"


def test_update_generator_ambigan_gaussian(mock_models: tuple, mock_crit: Callable) -> None:
    """Test UpdateGeneratorAmbiGanGaussian for correct loss and terms output."""
    G, D, C, _, optimizer, noise, device = mock_models
    alpha = 0.5
    var = 1.0

    updater = UpdateGeneratorAmbiGanGaussian(crit=mock_crit, C=C, alpha=alpha, var=var)
    loss, terms = updater(G, D, optimizer, noise, device)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "original_g_loss" in terms, "Terms should contain 'original_g_loss'"
    assert "conf_dist_loss" in terms, "Terms should contain 'conf_dist_loss'"


def test_update_generator_ambigan_kldiv(mock_models: tuple, mock_crit: Callable) -> None:
    """Test UpdateGeneratorAmbiGanKLDiv for correct loss and terms output."""
    G, D, _, C_identity, optimizer, noise, device = mock_models
    alpha = 0.5

    updater = UpdateGeneratorAmbiGanKLDiv(crit=mock_crit, C=C_identity, alpha=alpha)
    loss, terms = updater(G, D, optimizer, noise, device)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "original_g_loss" in terms, "Terms should contain 'original_g_loss'"
    assert "conf_dist_loss" in terms, "Terms should contain 'conf_dist_loss'"


def test_update_generator_ambigan_gaussian_identity(mock_models: tuple, mock_crit: Callable) -> None:
    """Test UpdateGeneratorAmbiGanGaussianIdentity for correct loss and terms output."""
    G, D, _, C_identity, optimizer, noise, device = mock_models
    alpha = 0.5
    var = 1.0

    updater = UpdateGeneratorAmbiGanGaussianIdentity(crit=mock_crit, C=C_identity, alpha=alpha, var=var)
    loss, terms = updater(G, D, optimizer, noise, device)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(terms, dict), "Terms should be a dictionary"
    assert "original_g_loss" in terms, "Terms should contain 'original_g_loss'"
    assert "conf_dist_loss" in terms, "Terms should contain 'conf_dist_loss'"
