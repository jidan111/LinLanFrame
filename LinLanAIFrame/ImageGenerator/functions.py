from ..import_package import *


def get_depth(h, limit=4):
    assert h % 2 == 0, f"图片无法进行半采样{h}%2!=0"
    cnt = 0
    while h > limit:
        if h % 2 != 0:
            break
        h //= 2
        cnt += 1
    return cnt, h


def l1(fake_sample, true_sample):
    return torch.abs(true_sample - fake_sample)


def l2(fake_sample, true_sample):
    return torch.square(true_sample - fake_sample)


def channels_get_norms(channels):
    split = [32, 16, 8, 4]
    if channels <= 4:
        return nn.BatchNorm2d(channels)  # 用BatchNorm
    for i in split:
        if channels % i == 0:
            return nn.GroupNorm(num_channels=channels, num_groups=channels // i)  # 用GroupNorm
    return nn.BatchNorm2d(channels)  # 用BatchNorm


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# 对模型参数普归一化, model.apply(add_sn)
def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        parametrizations.spectral_norm(m)


def KLDivergence_Q2P(q_sample, p_sample, min_eps=1e-10, max_eps=1., dim=None):
    """
    描述用q代替p时, 会损失多少信息, 输入前需要softmax
    :param q_sample: 未知分布(概率分布, [0,1])
    :param p_sample: 目标分布(概率分布, [0,1])
    :param min_eps:
    :param max_eps:
    :param dim:
    :return:
    """
    if dim is None:
        dim = list(range(1, len(q_sample.shape)))
    q_sample = q_sample.clamp(min=min_eps, max=max_eps)
    p_sample = p_sample.clamp(min=min_eps, max=max_eps)
    z = torch.sum(q_sample * torch.log2(q_sample / p_sample), dim=dim)
    return z


def compute_gradient_penalty(D, real_samples, fake_samples, create_graph=True, retain_graph=True):
    alpha = torch.rand(size=(real_samples.shape[0], 1, 1, 1)).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(real_samples.device)
    d_interpolates = D(interpolates)
    fake = torch.ones(size=(real_samples.shape[0], 1), requires_grad=False).to(real_samples.device)
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
    return gradient_penalty


def get_load_state_dict_from_compile(file, device="cuda"):
    new_dict = OrderedDict()
    for k, v in torch.load(file, map_location=device).items():
        key = k.replace("_orig_mod.", "")
        new_dict[key] = v
    return new_dict
