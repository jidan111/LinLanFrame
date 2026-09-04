from .import_package import *


class ConfigModule(nn.Module):
    """
    自动捕获子类初始化参数，无需手动传递！
    所有子类只需要写：super().__init__() 即可
    加载子类只需要 Object(**config)即可加载子类，方便复现
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        subclass = self.__class__
        sig = inspect.signature(subclass.__init__)
        frame = inspect.currentframe().f_back  # 获取调用栈（子类 __init__）
        support_types = (str, int, float, list, dict, tuple, bool)
        local_vars = frame.f_locals
        params = list(sig.parameters.keys())[1:]
        config = {
            k: local_vars[k] for k in params if k in local_vars and isinstance(local_vars[k], support_types)
        }
        self.config = {self.__class__.__name__: config}

    def save_config(self, file_name: str = None):
        with open(file_name, "w") as f:
            json.dump(self.config, f)


class EMA:
    def __init__(
            self,
            model: nn.Module,
            decay: float = 0.9999,
    ):
        self.decay = decay
        self.step = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = 1 - (1 - self.decay) * (1 - 1 / max(self.step, 1))
        for name, param in model.named_parameters():
            self.shadow[name].mul_(decay).add_(
                param.data, alpha=1 - decay
            )

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        backup = {}
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup):
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])

    def state_dict(self):
        return {
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "step": self.step,
        }

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.step = state["step"]


class DiagonalGaussianDistribution(object):
    def __init__(self, tensor, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__()
        assert tensor.shape[1] % 2 == 0, f"输入的潜在向量无法划分为均值和方差, {tensor.shape[1]}%2 != 0"
        self.dim = list(range(1, len(tensor.shape)))
        self.params = tensor
        self.mean, self.log_var = tensor.chunk(2, dim=1)
        self.log_var = self.log_var.clamp(-30.0, 20.0)
        self.deterministic = deterministic
        if deterministic:
            self.var, self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.log_var)
            self.var = self.log_var.exp()

    def sample(self):
        out = self.mean + self.std * torch.randn_like(self.mean).to(self.params.device)
        return out

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1. - self.log_var, dim=self.dim)
            else:
                return 0.5 * torch.sum((self.mean - other.mean).pow(
                    2) / other.var + self.var / other.var - 1. - self.log_var + other.logvar, dim=self.dim)

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logwopi = np.log(2. * np.pi)
        return 0.5 * torch.sum(logwopi + self.log_var + (sample - self.mean).pow(2) / self.var, dim=self.dim)
