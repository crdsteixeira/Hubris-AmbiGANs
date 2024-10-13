"""Module for GAN loss functions."""

import torch
import torch.nn.functional as F
from torch import autograd, nn


class DiscriminatorLoss:
    """Base class for discriminator loss with loss term tracking."""

    def __init__(self, terms: list[str]) -> None:
        """Initialize DiscriminatorLoss with loss terms."""
        self.terms = terms

    def __call__(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the discriminator loss."""
        raise NotImplementedError

    def get_loss_terms(self) -> list[str]:
        """Get the list of loss terms."""
        return self.terms


class NS_DiscriminatorLoss(DiscriminatorLoss):
    """Non-saturating loss for the discriminator."""

    def __init__(self) -> None:
        """Initialize NS_DiscriminatorLoss."""
        super().__init__([])

    def __call__(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        """Compute non-saturating loss for the discriminator."""
        ones = torch.ones_like(real_output, dtype=torch.float, device=device)
        zeros = torch.zeros_like(fake_output, dtype=torch.float, device=device)

        return F.binary_cross_entropy(real_output, ones) + F.binary_cross_entropy(fake_output, zeros), {}


class W_DiscrimatorLoss(DiscriminatorLoss):
    """Wasserstein loss for the discriminator."""

    def __init__(self) -> None:
        """Initialize W_DiscriminatorLoss."""
        super().__init__([])

    def __call__(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        """Compute Wasserstein loss for the discriminator."""
        d_loss_real = -real_output.mean()
        d_loss_fake = fake_output.mean()

        return d_loss_real + d_loss_fake, {}


class WGP_DiscriminatorLoss(DiscriminatorLoss):
    """Wasserstein loss with gradient penalty for the discriminator."""

    def __init__(self, D: nn.Module, lmbda: float) -> None:
        """Initialize WGP_DiscriminatorLoss."""
        super().__init__(["W_distance", "D_loss", "GP"])
        self.D = D
        self.lmbda = lmbda

    def calc_gradient_penalty(
        self, real_data: torch.Tensor, fake_data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Calculate gradient penalty for the Wasserstein loss."""
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_data)

        interpolates = real_data + alpha * (fake_data - real_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size(), device=device)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=[1, 2, 3]))
        gradient_penalty = ((gradients_norm - 1.0) ** 2).mean()

        return gradient_penalty

    def __call__(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        """Compute Wasserstein loss with gradient penalty for the discriminator."""
        d_loss_real = -real_output.mean()
        d_loss_fake = fake_output.mean()

        d_loss = d_loss_real + d_loss_fake
        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data, device)

        w_distance = -d_loss_real - d_loss_fake

        return d_loss + self.lmbda * gradient_penalty, {
            "W_distance": w_distance.item(),
            "D_loss": d_loss.item(),
            "GP": gradient_penalty.item(),
        }


class GeneratorLoss:
    """Base class for generator loss with loss term tracking."""

    def __init__(self, terms: list[str]) -> None:
        """Initialize GeneratorLoss with loss terms."""
        self.terms = terms

    def __call__(self, device: torch.device, output: torch.Tensor) -> torch.Tensor:
        """Compute the generator loss."""
        raise NotImplementedError

    def get_loss_terms(self) -> list[str]:
        """Get the list of loss terms."""
        return self.terms


class NS_GeneratorLoss(GeneratorLoss):
    """Non-saturating loss for the generator."""

    def __init__(self) -> None:
        """Initialize NS_GeneratorLoss."""
        super().__init__([])

    def __call__(self, device: torch.device, output: torch.Tensor) -> torch.Tensor:
        """Compute non-saturating loss for the generator."""
        ones = torch.ones_like(output, dtype=torch.float, device=device)

        return F.binary_cross_entropy(output, ones)


class W_GeneratorLoss(GeneratorLoss):
    """Wasserstein loss for the generator."""

    def __init__(self) -> None:
        """Initialize W_GeneratorLoss."""
        super().__init__([])

    def __call__(self, device: torch.device, output: torch.Tensor) -> torch.Tensor:
        """Compute Wasserstein loss for the generator."""
        d_loss_fake = output.mean()

        return -d_loss_fake
