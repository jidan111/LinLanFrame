from ..structs import *


class RectifiedFlow(ConfigModule):
    def __init__(self, model, image_shape=(3, 28, 28), step_dim=128):
        super(RectifiedFlow, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step_embedding = nn.Sequential(
            nn.Linear(1, step_dim),
            nn.SiLU(),
            nn.Linear(step_dim, step_dim)
        )
        self.model = model.to(self.device)
        self.image_shape = image_shape
        self.config["model"] = self.model.config
        self.loss = nn.MSELoss()

    def compute_flow_state(self, x0, x1, t):
        batch_size = t.shape[0]
        t = t.reshape(batch_size, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_target_velocity(self, x0, x1):
        return x1 - x0

    def generator_time_step(self, batch_size):
        return torch.rand(size=(batch_size,), device=self.device)

    def t_embed(self, t):
        t = t.unsqueeze(1)
        return self.step_embedding(t)

    def forward(self, x0, condition=None):
        batch_size = x0.shape[0]
        x1 = torch.randn_like(x0)
        t = self.generator_time_step(batch_size=batch_size)
        xt = self.compute_flow_state(x0=x0, x1=x1, t=t)
        ut = self.compute_target_velocity(x0=x0, x1=x1)
        t_embed = self.t_embed(t)
        vt = self.model(xt, t_embed, condition)
        loss = self.loss(vt, ut)
        return loss

    @torch.no_grad()
    def sample_euler(self, x, step_nums=100, condition=None):
        batch_size = x.shape[0]
        dt = -1. / step_nums
        steps = torch.linspace(1.0, 0.0, step_nums + 1, device=self.device)
        for i in tqdm(range(step_nums), desc="Euler Sampling"):
            t = steps[i].expand(size=(batch_size,))
            t_embed = self.t_embed(t)
            vt = self.model(x, t_embed, condition)
            x = x + dt * vt
        return x

    @torch.no_grad()
    def sample_rk4(self, x, step_nums=100, condition=None):
        def model_predict(x_in, t_in, condition_in):
            t_embed = self.t_embed(t_in)
            return self.model(x_in, t_embed, condition_in)

        batch_size = x.shape[0]
        dt = -1. / step_nums
        steps = 1. - torch.arange(start=0, end=step_nums, step=1, device=self.device) / step_nums
        for i in tqdm(range(step_nums), desc="Rk4 Sampling"):
            t = steps[i].expand(size=(batch_size,))
            f1 = model_predict(x_in=x, t_in=t, condition_in=condition)
            f2 = model_predict(x_in=x + 0.5 * dt * f1, t_in=t + 0.5 * dt, condition_in=condition)
            f3 = model_predict(x_in=x + 0.5 * dt * f2, t_in=t + 0.5 * dt, condition_in=condition)
            f4 = model_predict(x_in=x + dt * f3, t_in=t + dt, condition_in=condition)
            x = x + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return x

    @torch.no_grad()
    def sample(self, batch_size, x=None, condition=None, step_nums=100, mode="euler"):
        assert mode in ["euler", "rk4"]
        if x is None:
            x = torch.randn(size=(batch_size, *self.image_shape), device=self.device)
        if mode == "euler":
            return self.sample_euler(x=x, step_nums=step_nums, condition=condition)
        elif mode == "rk4":
            return self.sample_rk4(x=x, step_nums=step_nums, condition=condition)
        else:
            raise Exception(f"未设置该类采样方式:{mode}")
