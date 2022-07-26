import torch
import torch.autograd as autograd
import torch.nn.functional as F


def valid_loss(config):
    valid_names = {'wgan-gp', 'ns'}
    if config["name"].lower() not in valid_names:
        return False

    if config["name"].lower() == "wgan-gp":
        return "args" in config and "lambda" in config["args"]

    return True


class DiscriminatorLoss:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        raise NotImplementedError

    def get_loss_terms(self):
        return self.terms


class NS_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self):
        super().__init__([])

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        ones = torch.ones_like(real_output, dtype=torch.float, device=device)
        zeros = torch.zeros_like(fake_output, dtype=torch.float, device=device)

        return F.binary_cross_entropy(real_output, ones) + F.binary_cross_entropy(fake_output, zeros), {}


class W_DiscrimatorLoss(DiscriminatorLoss):
    def __init__(self):
        super().__init__([])

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        d_loss_real = -real_output.mean()
        d_loss_fake = fake_output.mean()

        return d_loss_real + d_loss_fake, {}


class WGP_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, D, lmbda):
        super().__init__(['W_distance', 'D_loss', 'GP'])
        self.D = D
        self.lmbda = lmbda

    def calc_gradient_penalty(self, real_data, fake_data, device):
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = real_data + alpha * (fake_data - real_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size(), device=device),
                                  create_graph=True, retain_graph=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        d_loss_real = -real_output.mean()
        d_loss_fake = fake_output.mean()

        d_loss = d_loss_real + d_loss_fake
        gradient_penalty = self.calc_gradient_penalty(
            real_data, fake_data, device)

        w_distance = - d_loss_real - d_loss_fake

        return d_loss + self.lmbda * gradient_penalty, {'W_distance': w_distance.item(), 'D_loss': d_loss.item(), 'GP': gradient_penalty.item()}


class GeneratorLoss:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, device, output, fake_data):
        raise NotImplementedError

    def get_loss_terms(self):
        return self.terms


class NS_GeneratorLoss(GeneratorLoss):
    def __init__(self):
        super().__init__([])

    def __call__(self, device, output, fake_data):
        ones = torch.ones_like(output, dtype=torch.float, device=device)

        return F.binary_cross_entropy(output, ones), {}


class W_GeneratorLoss(GeneratorLoss):
    def __init__(self):
        super().__init__([])

    def __call__(self, device, output, fake_data):
        d_loss_fake = output.mean()

        return - d_loss_fake, {}


class ThresholdDistBinary_GeneratorLoss(GeneratorLoss):
    def __init__(self, g_loss, classifier, alpha=0.5):
        super().__init__(['original_g_loss', 'conf_dist_loss'])
        self.g_loss = g_loss
        self.classifier = classifier
        self.alpha = alpha

    def __call__(self, device, output, fake_data):
        term_1 = self.g_loss(device, output, fake_data)

        c_output = self.classifier(fake_data)
        term_2 = 2 * (0.5 - c_output).abs().mean()

        loss = (1 - self.alpha) * term_1 + self.alpha * term_2

        return loss, {'original_g_loss': term_1.item(), 'conf_dist_loss': term_2.item()}
