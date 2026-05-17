from .structs import *


class Diffusion(ConfigModule):
    def __init__(self, model=lambda x_, t_, o=None: x_, image_shape=(3, 28, 28), step_nums=100, step_dim=128,
                 schedule_name="linear", betas=(1e-4, 0.02)):
        super(Diffusion, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_shape = image_shape
        self.step_nums = step_nums
        self.step_embedding = nn.Embedding(step_nums, step_dim).to(device)
        self.model = model.to(device)
        self.device = device
        self.loss = nn.MSELoss().to(device=device)
        for k, v in self.set_params(schedule_name=schedule_name, betas=betas).items():
            self.register_buffer(k, v.to(device))

    def get_alpha_beta(self, schedule_name="linear", betas=(0.04, 0.02), s=0.008):
        if schedule_name == "linear":
            beta = torch.linspace(start=betas[0], end=betas[1], steps=self.step_nums).view(self.step_nums, 1, 1, 1)
            alpha = 1 - beta
            return alpha, beta
        elif schedule_name == "cosine":
            steps = self.step_nums + 1
            x = torch.linspace(0, self.step_nums, steps)
            alpha_bar = torch.cos((x / self.step_nums + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # 归一化
            alpha_bar = alpha_bar[1:]  # 对齐步长
            alpha = alpha_bar[1:] / alpha_bar[:-1]
            alpha = torch.cat([alpha_bar[[0]], alpha])
            beta = 1 - alpha
            beta = beta.view(self.step_nums, 1, 1, 1)
            alpha = alpha.view(self.step_nums, 1, 1, 1)
            return alpha, beta
        else:
            raise NotImplementedError(
                f"调度算法 {schedule_name} 未预设")

    def set_params(self, schedule_name="linear", betas=(0.04, 0.02), s=0.008):
        alpha, beta = self.get_alpha_beta(schedule_name=schedule_name, betas=betas, s=s)
        alpha_bar = torch.cumprod(alpha, dim=0)
        sqrt_beta = torch.sqrt(beta)  # 前后都用
        sqrt_alpha = torch.sqrt(alpha) + 1e-8  # 前后都用
        sqrt_alpha_bar = torch.sqrt(alpha_bar) + 1e-8  #
        sqrt_one_sub_alpha_bar = torch.sqrt(1 - alpha_bar) + 1e-8
        return {"sqrt_alpha_bar": sqrt_alpha_bar, "sqrt_one_sub_alpha_bar": sqrt_one_sub_alpha_bar, "beta": beta,
                "sqrt_alpha": sqrt_alpha, "sqrt_beta": sqrt_beta}

    def add_noise(self, x0, noise, t):
        xt = self.sqrt_alpha_bar[t] * x0 + self.sqrt_one_sub_alpha_bar[t] * noise
        return xt

    def forward(self, x, txt=None):
        batch_size, *_ = x.shape
        t = torch.randint(low=0, high=self.step_nums, size=(batch_size,), dtype=torch.long, device=self.device)
        t_embed = self.step_embedding(t)
        noise = torch.randn_like(x, device=x.device)
        xt = self.add_noise(x0=x, noise=noise, t=t)
        pre_noise = self.model(xt, t_embed) if txt is None else self.model(xt, t_embed, txt)
        loss = self.loss(pre_noise, noise)
        return loss

    @torch.no_grad()
    def __clean_noise_p(self, xt, t, txt=None):
        index = t
        batch_size = xt.shape[0]
        t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        t_embed = self.step_embedding(t)
        pre_noise = self.model(xt, t_embed) if txt is None else self.model(xt, t_embed, txt)
        z = torch.randn_like(pre_noise, device=self.device)
        x_t_prev_mean = (xt - (self.beta[t] / self.sqrt_one_sub_alpha_bar[t]) * pre_noise) / self.sqrt_alpha[t]
        if index > 0:
            return x_t_prev_mean + self.sqrt_beta[t] * z
        return x_t_prev_mean

    @torch.no_grad()
    def __sample_ddpm(self, batch_size=4, txt=None):
        x = torch.randn(size=(batch_size, *self.image_shape), device=self.device)
        for i in tqdm(range(self.step_nums - 1, -1, -1), desc="DDPM Sampling"):
            x = self.__clean_noise_p(xt=x, t=i, txt=txt)
        return x

    @torch.no_grad()
    def __clean_noise_i(self, xt, t1, t2, txt=None, sigma=0.):
        """t1->t2"""
        index = t2
        batch_size = xt.shape[0]
        t1 = torch.full((batch_size,), t1, dtype=torch.long, device=self.device)
        t2 = torch.full((batch_size,), t2, dtype=torch.long, device=self.device)
        t_embed = self.step_embedding(t1)
        t1_pre_noise = self.model(xt, t_embed) if txt is None else self.model(xt, t_embed, txt)
        x0 = (xt - self.sqrt_one_sub_alpha_bar[t1] * t1_pre_noise) / self.sqrt_alpha_bar[t1]
        if index == 0:
            return x0
        noise = torch.randn_like(x0)
        x_mean = self.sqrt_alpha_bar[t2] * x0
        t2_pre_noise = self.sqrt_one_sub_alpha_bar[t2] * t1_pre_noise
        return x_mean + t2_pre_noise + sigma * noise

    @torch.no_grad()
    def __sample_ddim(self, batch_size=4, x=None, txt=None, step=2, sigma=0.):
        steps_arr = list(range(self.step_nums - 1, -1, -step))
        if x is None:
            x = torch.randn(size=(batch_size, *self.image_shape), device=self.device)
        for i in tqdm(steps_arr, desc="DDIM Sampling"):
            next_t = max(i - step, 0)
            x = self.__clean_noise_i(xt=x, t1=i, t2=next_t, txt=txt, sigma=sigma)
            if next_t == 0:
                break
        return x

    @torch.no_grad()
    def __clean_noise_dpm_2m(self, xt, t0_pre_noise=None, t1=0, t2=0, txt=None):
        """t1->t2"""
        batch_size = xt.shape[0]
        t_tensor = torch.full((batch_size,), t1, dtype=torch.long, device=self.device)
        t_embed = self.step_embedding(t_tensor)
        t1_pre_noise = self.model(xt, t_embed) if txt is None else self.model(xt, t_embed, txt)
        if t0_pre_noise is not None:
            t1_pre_noise = (3 * t1_pre_noise - t0_pre_noise) / 2
        x0 = (xt - self.sqrt_one_sub_alpha_bar[t1] * t1_pre_noise) / self.sqrt_alpha_bar[t1]
        if t2 == 0:
            return x0, t1_pre_noise
        t2_noise = self.sqrt_one_sub_alpha_bar[t2] * t1_pre_noise
        xt2 = self.sqrt_alpha_bar[t2] * x0 + t2_noise
        return xt2, t1_pre_noise

    @torch.no_grad()
    def __sample_dpm_2m(self, batch_size=4, x=None, txt=None, step=5):
        steps_arr = list(range(self.step_nums - 1, -1, -step))
        if x is None:
            x = torch.randn(size=(batch_size, *self.image_shape), device=self.device)
        t0_pre_noise = None
        for i in tqdm(steps_arr, desc="DPM++ Sampling"):
            next_t = max(i - step, 0)
            x, t0_pre_noise = self.__clean_noise_dpm_2m(xt=x, t0_pre_noise=t0_pre_noise, t1=i, t2=next_t, txt=txt)
            if next_t == 0:
                break
        return x

    @torch.no_grad()
    def sample(self, batch_size=4, txt=None, mode="ddpm", x=None, step=5, sigma=0.):
        assert mode in ["ddpm", "ddim", "dpm"], "支持 ddpm/ddim/dpm 采样"
        if mode == "ddpm":
            return self.__sample_ddpm(batch_size=batch_size, txt=txt)
        elif mode == "ddim":
            return self.__sample_ddim(batch_size=batch_size, x=x, txt=txt, step=step, sigma=sigma)
        else:
            return self.__sample_dpm_2m(batch_size=batch_size, x=x, txt=txt, step=step)
