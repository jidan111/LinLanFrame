from ..import_package import *


def l1(fake_sample, true_sample):
    return torch.abs(true_sample - fake_sample)


def l2(fake_sample, true_sample):
    return torch.square(true_sample - fake_sample)


class ESRLoss(nn.Module):
    def __init__(self, have_perception: bool = False, perception_weight=1., perception_net="vgg"):
        super(ESRLoss, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rec_loss = l1
        self.have_perception = have_perception
        if self.have_perception:
            self.p_loss = LPIPS(net=perception_net).to(device).eval()
            self.p_weight = perception_weight
            for param in self.p_loss.parameters():
                param.requires_grad = False

    def forward(self, fake_sample, true_sample):
        batch_size = fake_sample.shape[0]
        rec_loss = self.rec_loss(fake_sample=fake_sample, true_sample=true_sample)
        if self.have_perception:
            p_loss = self.p_loss(fake_sample, true_sample)
            rec_loss = rec_loss + self.p_weight * p_loss
        rec_loss = rec_loss.sum() / batch_size
        return rec_loss


class GAN_HingeLoss(nn.Module):
    def __init__(self):
        super(GAN_HingeLoss, self).__init__()
        self._ = ""

    def forward(self, fake_sample, true_sample):
        loss_real = torch.mean(F.relu(1. - true_sample))
        loss_fake = torch.mean(F.relu(1. + fake_sample))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss
