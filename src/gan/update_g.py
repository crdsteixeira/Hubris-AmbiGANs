"""Module for updating the generator."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor, full_like, hstack, log, nn, optim
from torch.nn import BCELoss, GaussianNLLLoss, KLDivLoss
from torch.nn.utils import clip_grad_norm_

from src.utils.min_norm_solvers import MinNormSolver


class UpdateGenerator:
    """Base class for updating a generator in GAN training."""

    def __init__(self, crit: Callable) -> None:
        """Initialize UpdateGenerator with a criterion function."""
        self.crit = crit

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Call function to update the generator."""
        raise NotImplementedError

    def get_loss_terms(self) -> list[str]:
        """Get the list of loss terms."""
        raise NotImplementedError


class UpdateGeneratorGAN(UpdateGenerator):
    """Update the generator using the classic GAN loss."""

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Update the generator using GAN loss."""
        G.zero_grad()

        fake_data = G(noise)
        output = D(fake_data)

        loss = self.crit(device, output)

        loss.backward()
        clip_grad_norm_(G.parameters(), 1.00)
        optimizer.step()

        return loss, {}

    def get_loss_terms(self) -> list[str]:
        """Return the list of loss terms."""
        return []


class UpdateGeneratorGASTEN(UpdateGenerator):
    """Update the generator using GASTEN loss, incorporating a classifier with a confidence distance term."""

    def __init__(self, crit: Callable, C: nn.Module, alpha: float) -> None:
        """Initialize UpdateGeneratorGASTEN with a criterion, classifier, and alpha value."""
        super().__init__(crit)
        self.C = C
        self.alpha = alpha

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Update the generator using GASTEN loss."""
        G.zero_grad()

        fake_data = G(noise)

        output = D(fake_data)
        term_1 = self.crit(device, output)

        clf_output = self.C(fake_data)
        term_2 = (0.5 - clf_output).abs().mean()

        loss = term_1 + self.alpha * term_2

        loss.backward()
        optimizer.step()

        return loss, {"original_g_loss": term_1.item(), "conf_dist_loss": term_2.item()}

    def get_loss_terms(self) -> list[str]:
        """Return the list of loss terms."""
        return ["original_g_loss", "conf_dist_loss"]


class UpdateGeneratorGastenMgda(UpdateGenerator):
    """
    Update the generator using GASTEN loss with MGDA .
    (Multiple Gradient Descent Algorithm) for multi-objective optimization.
    """

    def __init__(self, crit: Callable, C: nn.Module, alpha: float = 1, normalize: bool = False) -> None:
        """Initialize UpdateGeneratorGastenMgda with a criterion, classifier, alpha, and normalization flag."""
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.normalize = normalize

    def gradient_normalizers(self, grads: list[torch.Tensor], loss: torch.Tensor) -> float:
        """Compute gradient normalizers."""
        return loss.item() * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads]))

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: torch.Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Update the generator using GASTEN with MGDA loss."""
        # Compute gradients of each loss function wrt parameters

        # Term 1
        G.zero_grad()

        fake_data = G(noise)

        output = D(fake_data)
        term_1 = self.crit(device, output)

        term_1.backward()
        term_1_grads = [param.grad.clone().detach() for param in G.parameters() if param.grad is not None]

        # Term 2
        G.zero_grad()

        fake_data = G(noise)

        c_output = self.C(fake_data)
        term_2 = (0.5 - c_output).abs().mean()

        term_2.backward()
        term_2_grads = [param.grad.clone().detach() for param in G.parameters() if param.grad is not None]

        if self.normalize:
            gn1 = self.gradient_normalizers(term_1_grads, term_1)
            gn2 = self.gradient_normalizers(term_2_grads, term_2)

            for gr_i, element in enumerate(term_1_grads):
                term_1_grads[gr_i] = element / gn1
            for gr_i, element in enumerate(term_2_grads):
                term_2_grads[gr_i] = element / gn2

        scale, _ = MinNormSolver.find_min_norm_element([term_1_grads, term_2_grads])

        assert scale is not None

        # Scaled back-propagation
        G.zero_grad()

        fake_data = G(noise)

        output = D(fake_data)
        term_1 = self.crit(device, output)

        clf_output = self.C(fake_data)
        term_2 = (0.5 - clf_output).abs().mean()

        loss = scale[0] * term_1 + scale[1] * term_2

        loss.backward()
        optimizer.step()

        return loss, {
            "original_g_loss": term_1.item(),
            "conf_dist_loss": term_2.item(),
            "scale1": scale[0],
            "scale2": scale[1],
        }

    def get_loss_terms(self) -> list[str]:
        """Return the list of loss terms."""
        return ["original_g_loss", "conf_dist_loss", "scale1", "scale2"]


class UpdateGeneratorAmbiGanGaussian(UpdateGenerator):
    """Update the generator using AmbiGAN with Gaussian loss for ambiguity reduction."""

    def __init__(self, crit: Callable, C: nn.Module, alpha: float, var: float) -> None:
        """Initialize UpdateGeneratorAmbiGanGaussian with a criterion, classifier, alpha, and variance."""
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss()
        self.target = 0.5

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Update the generator using AmbiGAN with Gaussian loss."""
        G.zero_grad()

        optimizer.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data)
        # update from ensemble
        target = full_like(input=clf_output, fill_value=self.target, device=device)
        var = full_like(input=clf_output, fill_value=self.var, device=device)
        loss_1 = self.c_loss(clf_output, target, var)
        loss_1.backward()
        clip_grad_norm_(G.parameters(), 1.00 * self.alpha)
        optimizer.step()
        # update from discriminator
        optimizer.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        loss_2 = self.crit(device, output)
        loss_2.backward()
        clip_grad_norm_(G.parameters(), 1.00)
        optimizer.step()

        loss = loss_1 + loss_2

        return loss, {"original_g_loss": loss_2.item(), "conf_dist_loss": loss_1.item()}

    def get_loss_terms(self) -> list[str]:
        """Return the list of loss terms."""
        return ["original_g_loss", "conf_dist_loss"]


class UpdateGeneratorAmbiGanKLDiv(UpdateGenerator):
    """Update the generator using AmbiGAN with KL divergence loss for ambiguity reduction."""

    def __init__(self, crit: Callable, C: nn.Module, alpha: float) -> None:
        """Initialize UpdateGeneratorAmbiGanKLDiv with a criterion, classifier, and alpha."""
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.c_loss = KLDivLoss(reduction="none")
        self.target = 0.5
        self.eps = 1e-9
        self.crit = BCELoss(reduction="none")

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Update the generator using GASTEN with KL divergence loss."""
        G.zero_grad()

        optimizer.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data, output_feature_maps=True)
        # update from ensemble
        loss_1 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
        for c_pred in clf_output[0].mT:

            class_prob = hstack((c_pred.unsqueeze(-1), 1.0 - c_pred.unsqueeze(-1)))

            # Ensure class_prob has a minimum value (self.eps) and maximum value (1.0) for stability
            class_prob = class_prob.clamp(min=self.eps, max=1.0)
            # Compute the target and ensure it has the same shape as class_prob
            target = full_like(input=class_prob, fill_value=self.target, device=device)
            # Compute the KL Divergence loss, accumulate it
            loss_1 = loss_1 + self.c_loss(log(class_prob), target).sum()
            # loss_1 += self.c_loss(log(class_prob.clip(self.eps, 1.0)), target)

        # update from discriminator
        output = D(fake_data)
        target = full_like(input=output, fill_value=1.0, device=device)
        loss_2 = self.crit(output, target).sum()

        # Combine the losses
        loss = (self.alpha * loss_1 + loss_2).sum()

        # loss = hstack((self.alpha * loss_1, loss_2.unsqueeze(-1))).sum()

        loss.backward()
        optimizer.step()

        return loss, {
            "original_g_loss": loss_2.sum().item(),
            "conf_dist_loss": loss_1.sum().item(),
        }

    def get_loss_terms(self) -> list[str]:
        """Return the list of loss terms."""
        return ["original_g_loss", "conf_dist_loss"]


class UpdateGeneratorAmbiGanGaussianIdentity(UpdateGenerator):
    """
    Update the generator using AmbiGAN with Gaussian loss for ambiguity reduction.
    Using 'Identity' version (no combination of ouputs).
    """

    def __init__(self, crit: Callable, C: nn.Module, alpha: float, var: float) -> None:
        """Initialize UpdateGeneratorGASTEN_gaussianV2 with a criterion, classifier, alpha, and variance."""
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss(reduction="none")
        self.target = 0.5
        self.crit = BCELoss(reduction="none")

    def __call__(
        self, G: nn.Module, D: nn.Module, optimizer: optim.Optimizer, noise: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, Any]]:
        """Update the generator using GASTEN with Gaussian V2 loss."""
        G.zero_grad()

        optimizer.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data, output_feature_maps=True)
        # update from ensemble
        loss_1 = torch.tensor(0.0, dtype=torch.float32)
        for c_pred in clf_output[0].mT:

            # Compute the target and ensure it has the same shape as c_pred
            target = full_like(input=c_pred, fill_value=self.target, device=device)
            # Compute variance and ensure it has the same shape as c_pred
            var = full_like(input=c_pred, fill_value=self.var, device=device)

            loss_1 = loss_1 + self.c_loss(c_pred, target, var).sum()

        # update from discriminator
        output = D(fake_data)
        target = full_like(input=output, fill_value=1.0, device=device)
        loss_2 = self.crit(output, target)

        # Combine the losses
        loss = (self.alpha * loss_1 + loss_2).sum()

        # loss = hstack((self.alpha * loss_1, loss_2)).sum()

        loss.backward()
        optimizer.step()

        return loss, {
            "original_g_loss": loss_2.sum().item(),
            "conf_dist_loss": loss_1.sum().item(),
        }

    def get_loss_terms(self) -> list[str]:
        """Return the list of loss terms."""
        return ["original_g_loss", "conf_dist_loss"]
