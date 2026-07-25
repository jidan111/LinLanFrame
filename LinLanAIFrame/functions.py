from .import_package import *


def global_latents_scaling_factor(global_latents):
    """
    :param global_latents: [batch, latent_dim, H, W]
    :return:
    """
    global_latents = global_latents.flatten(1)
    std_global = global_latents.std().item()
    scaling_factor = 1 / std_global
    return scaling_factor


def get_latent_mean_std(latent: torch.Tensor):
    assert latent.dim() == 4, f"输入应为4D张量 [B,C,H,W]，但得到 {latent.shape}"
    mean = latent.mean(dim=(0, 2, 3))  # [latent_dim]
    std = latent.std(dim=(0, 2, 3))  # [latent_dim]
    std = std + 1e-8
    return mean, std
