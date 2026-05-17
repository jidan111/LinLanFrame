from .structs import *
from .functions import *


def l1(fake_sample, true_sample):
    return torch.abs(fake_sample - true_sample)


def l2(fake_sample, true_sample):
    return torch.square(true_sample - fake_sample)


class VQAutoEncoderLoss(nn.Module):
    def __init__(self, book_weight=1., have_perception: bool = False, perception_weight=1.,
                 perception_net="alex"):
        super(VQAutoEncoderLoss, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec_loss = l1
        self.book_weight = book_weight
        self.have_perception = have_perception
        self.perception_weight = perception_weight
        if self.have_perception:
            self.perception_loss = LPIPS(net=perception_net).to(device).eval()
            for param in self.perception_loss.parameters():
                param.requires_grad = False

    def forward(self, true_sample, fake_sample, book_loss, log_var: torch.Tensor = None):
        rec_loss = self.rec_loss(fake_sample, true_sample)
        if self.have_perception:
            p_loss = self.perception_loss(fake_sample, true_sample)
            rec_loss = rec_loss + self.perception_weight * p_loss
        if log_var:
            rec_loss = rec_loss / torch.exp(log_var) + log_var
        rec_loss = rec_loss.mean()
        loss = rec_loss + self.book_weight * book_loss
        return loss


class AutoEncoderKLLoss(nn.Module):
    def __init__(self, have_perception: bool = False, perception_weight=1., perception_net="alex", kl_weight=1e-6):
        super(AutoEncoderKLLoss, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec_loss = l1
        self.kl_weight = kl_weight
        self.have_perception = have_perception
        self.perception_weight = perception_weight
        if self.have_perception:
            self.perception_loss = LPIPS(net=perception_net).to(device).eval()
            for param in self.perception_loss.parameters():
                param.requires_grad = False

    def forward(self, true_sample: torch.Tensor, fake_sample: torch.Tensor, latent: DiagonalGaussianDistribution,
                log_var: torch.Tensor = None):
        batch_size = true_sample.shape[0]
        rec_loss = self.rec_loss(fake_sample, true_sample)
        kl = latent.kl().sum() / batch_size
        if self.have_perception:
            p_loss = self.perception_loss(fake_sample, true_sample)
            rec_loss = rec_loss + self.perception_weight * p_loss
        if log_var:
            rec_loss = rec_loss / torch.exp(log_var) + log_var
        rec_loss = rec_loss.sum() / batch_size
        return rec_loss + kl * self.kl_weight


class GAN_HingeLoss(nn.Module):
    def __init__(self):
        super(GAN_HingeLoss, self).__init__()
        self._ = ""

    def forward(self, fake_sample, true_sample):
        loss_real = torch.mean(F.relu(1. - true_sample))
        loss_fake = torch.mean(F.relu(1. + fake_sample))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss


class WGAN_GP_Loss(nn.Module):
    def __init__(self, lambda_gp=10):
        super(WGAN_GP_Loss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, model, fake_sample, true_sample, create_graph=True, retain_graph=True):
        alpha = torch.rand(size=(true_sample.shape[0], 1, 1, 1)).to(true_sample.device)
        interpolates = (alpha * true_sample + ((1 - alpha) * fake_sample)).requires_grad_(True).to(
            true_sample.device)
        d_interpolates = model(interpolates)
        fake = torch.ones(size=(true_sample.shape[0], 1), requires_grad=False).to(true_sample.device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        true_score = model(true_sample)
        fake_score = model(fake_sample.detach())
        return -true_score.mean() + fake_score.mean() + self.lambda_gp * gradient_penalty
