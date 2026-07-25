from .losses import *
from .ImageGenerator import *
from .CLIP import *
from .utils import *


class LogVar(nn.Module):
    def __init__(self):
        super(LogVar, self).__init__()
        self.log_var = nn.Parameter(torch.ones(size=()) * 0.0)

    def forward(self):
        return torch.clamp(self.log_var, min=-10, max=10)


class BaseTrainer(object):
    """主要重写validate,compute_loss函数"""

    def __init__(self, save_dir="./model", valid_dir="./valid", mid_valid_step=None, use_ema=False, ema_decay=0.999,
                 ema_step=10):
        self.model = None
        self.model_opt = None
        self.loss_func = None
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        self.save_dir = save_dir
        self.valid_dir = valid_dir
        self.mid_valid_flag = mid_valid_step is not None
        self.mid_valid_step = mid_valid_step
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_ema = use_ema
        self.ema_step = ema_step
        self.ema = None
        self.scale = GradScaler()

    def get_model_information(self, model: ConfigModule) -> tuple:
        return model.config, model.state_dict()

    def save(self) -> None:
        config, state_dict = self.get_model_information(model=self.model)
        name = list(config.keys())[0]
        config_path = os.path.join(self.save_dir, f"{name}.json")
        state_dict_path = os.path.join(self.save_dir, f"{name}.pth")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        torch.save(state_dict, state_dict_path)
        if self.use_ema:
            ema_state_dict_path = os.path.join(self.save_dir, f"ema_{name}.pth")
            torch.save(self.ema.shadow, ema_state_dict_path)

    def validate(self, **kwargs) -> None:
        ...

    def preprocessing_data(self, data) -> torch.Tensor:
        data = data.to(self.device)
        return data

    def _validate_loss(self, loss, optimizer) -> None:
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            raise Exception("训练出现空值，已终止训练")

    def compute_loss(self, data) -> torch.Tensor:
        fake_sample = self.model(data)
        loss = self.loss_func(fake_sample=fake_sample, true_sample=data)
        self._validate_loss(loss=loss, optimizer=self.model_opt)
        return loss

    def train_one_batch(self, **kwargs) -> None:
        self.model.train()
        self.model_opt.zero_grad()
        data = kwargs["data"]
        with autocast():
            loss = self.compute_loss(data)
        self.scale.scale(loss).backward()
        self.scale.unscale_(self.model_opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scale.step(self.model_opt)
        self.scale.update()
        if self.use_ema:
            if kwargs["cnt"] % self.ema_step == 0:
                self.ema.update(model=self.model)

    def train_one_epoch(self, **kwargs) -> None:
        for cnt, data in enumerate(tqdm(kwargs["data_loader"], desc=f"{kwargs['epoch']}/{kwargs['epoch_nums']}")):
            data = self.preprocessing_data(data)
            self.train_one_batch(data=data, cnt=cnt)
            if self.mid_valid_flag:
                if cnt % self.mid_valid_step == 0:
                    self.validate(file_name="valid", valid_data=kwargs["valid_data"],
                                  valid_batch_size=kwargs["valid_batch_size"])
                    self.save()

    def run(self, data_loader, epoch_nums, valid_step, valid_batch_size, valid_data, **kwargs) -> None:
        for epoch in range(epoch_nums):
            self.train_one_epoch(data_loader=data_loader, epoch=epoch, epoch_nums=epoch_nums, valid_data=valid_data,
                                 valid_batch_size=valid_batch_size)
            if epoch % valid_step == 0:
                self.validate(file_name=epoch, valid_data=valid_data, valid_batch_size=valid_batch_size, **kwargs)
                self.save()


class DiffusionTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule = lambda x_, y, o=None: x_, lr=1e-4, save_dir="./model", valid_dir="./valid",
                 mid_valid_step=None, compile_model=False, use_ema=False, ema_decay=0.999, ema_step=10):
        super(DiffusionTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir, valid_dir=valid_dir,
                                               use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        if compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1e-8)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def compute_loss(self, data) -> torch.Tensor:
        loss = self.model(data)
        self._validate_loss(loss, self.model_opt)
        return loss

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        mode = kwargs.get("mode", "ddpm")
        condition = kwargs.get("condition", None)
        x = kwargs.get("x", None)
        step = kwargs.get("step", 5)
        sigma = kwargs.get("sigma", 0.)
        with torch.no_grad(), autocast():
            out = self.model.sample(batch_size=kwargs["valid_batch_size"], mode=mode, condition=condition, x=x,
                                    step=step, sigma=sigma)
        valid_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()


class FlowTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule = lambda x_, y, o=None: x_, lr=1e-4, save_dir="./model", valid_dir="./valid",
                 mid_valid_step=None, compile_model=False, use_ema=False, ema_decay=0.999, ema_step=10):
        super(FlowTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir, valid_dir=valid_dir,
                                          use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        if compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1e-8)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def compute_loss(self, data) -> torch.Tensor:
        loss = self.model(data)
        self._validate_loss(loss, self.model_opt)
        return loss

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        mode = kwargs.get("mode", "euler")
        condition = kwargs.get("condition", None)
        step_nums = kwargs.get("step_nums", 100)
        x = kwargs.get("x", None)
        with torch.no_grad(), autocast():
            out = self.model.sample(batch_size=kwargs["valid_batch_size"], mode=mode, condition=condition,
                                    step_nums=step_nums, x=x)
        valid_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()


class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule = lambda x_: x_, lr=1e-4, save_dir="./model", valid_dir="./valid",
                 mid_valid_step=500, compile_model=False, kl_weight=1e-6, perception_net="vgg", perception_weight=1.,
                 have_perception=False, book_weight=1., vae_mode="va", use_ema=False, ema_decay=0.999, ema_step=10):
        super(AutoEncoderTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir, valid_dir=valid_dir,
                                                 use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        assert vae_mode in ["va", "vq"], "只支持va,vq两种方式"
        if vae_mode == "va":
            assert lr <= 1e-5, "va训练方式，损失函数是sum，学习率建议小于1e-5"
        if compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        self.log_var = LogVar()
        self.model_opt = torch.optim.Adam(params=list(self.model.parameters()) + list(self.log_var.parameters()), lr=lr)
        self.vae_mode = vae_mode
        if vae_mode == "va":
            self.loss_func = AutoEncoderKLLoss(have_perception=have_perception, perception_weight=perception_weight,
                                               perception_net=perception_net, kl_weight=kl_weight)
        else:
            self.loss_func = VQAutoEncoderLoss(book_weight=book_weight, have_perception=have_perception,
                                               perception_weight=perception_weight,
                                               perception_net=perception_net)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def compute_loss(self, data) -> torch.Tensor:
        out, other = self.model(data)
        log_var = self.log_var()
        if self.vae_mode == "va":
            loss = self.loss_func(true_sample=data, fake_sample=out, latent=other, log_var=log_var)
        else:
            loss = self.loss_func(true_sample=data, fake_sample=out, book_loss=other, log_var=log_var)
        self._validate_loss(loss=loss, optimizer=self.model_opt)
        return loss

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        valid_data = self.preprocessing_data(kwargs["valid_data"])
        with torch.no_grad(), autocast():
            out, *_ = self.model(valid_data)
        valid_fake_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_fake.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_fake_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        valid_true_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_true.png')
        save_image(tensor=valid_data.clamp(min=-1, max=1), fp=valid_true_path,
                   nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()


class GanTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule, discriminator: ConfigModule, model_lr=0.0002, dis_lr=0.0002, n_critic=2,
                 valid_dir="./valid", save_dir="./model/gan", loss_type="hinge", lambda_gp=10, mid_valid_step=None,
                 use_ema=False, ema_decay=0.999, ema_step=10):
        super(GanTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir, valid_dir=valid_dir,
                                         use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        assert loss_type in ["hinge", "wgp", "dc"], "只支持hinge,wgp,dc三种损失"
        self.n_critic = n_critic
        self.model = model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=model_lr, betas=(0.5, 0.99))
        self.dis_opt = torch.optim.AdamW(params=self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.99))
        self.loss_type = loss_type
        self.gen_scale = GradScaler()
        self.dis_scale = GradScaler()
        if loss_type == "hinge":
            self.loss_func = GAN_HingeLoss()
        elif loss_type == "wgp":
            self.loss_func = WGAN_GP_Loss(lambda_gp=lambda_gp)
        else:
            self.loss_func = nn.BCEWithLogitsLoss()
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        with torch.no_grad(), autocast():
            out = self.model.sample(batch_size=kwargs["valid_batch_size"])
        valid_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()

    def compute_loss(self, true_sample, fake_sample, mode) -> torch.Tensor:
        if mode == "gen":
            if self.loss_type == "dc":
                dis_label = self.discriminator(fake_sample)
                target_label = torch.ones_like(dis_label, device=true_sample.device)
                loss = self.loss_func(dis_label, target_label)
            else:
                loss = -self.discriminator(fake_sample).mean()
            self._validate_loss(loss=loss, optimizer=self.model_opt)
        else:
            if self.loss_type == "hinge":
                true_sample = self.discriminator(true_sample)
                fake_sample = self.discriminator(fake_sample.detach())
                loss = self.loss_func(fake_sample=fake_sample, true_sample=true_sample)
            elif self.loss_type == "dc":
                dis_true = self.discriminator(true_sample)
                dis_fake = self.discriminator(fake_sample.detach())
                true_label = torch.ones_like(dis_true, device=true_sample.device)
                fake_label = torch.zeros_like(dis_fake, device=true_sample.device)
                true_loss = self.loss_func(dis_true, true_label)
                fake_loss = self.loss_func(dis_fake, fake_label)
                loss = 0.5 * (true_loss + fake_loss)
            else:
                loss = self.loss_func(model=self.discriminator, fake_sample=fake_sample.detach(),
                                      true_sample=true_sample)
            self._validate_loss(loss=loss, optimizer=self.dis_opt)
        return loss

    def train_one_batch(self, **kwargs) -> None:
        true_sample = kwargs["data"]
        with torch.no_grad():
            noise = torch.randn(size=(true_sample.shape[0], self.model.in_dim), device=self.device)
            fake_sample = self.model(noise)
        self.discriminator.train()
        self.dis_opt.zero_grad()
        with autocast():
            loss = self.compute_loss(true_sample=true_sample, fake_sample=fake_sample, mode="dis")
        self.dis_scale.scale(loss).backward()
        self.dis_scale.unscale_(self.dis_opt)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.dis_scale.step(self.dis_opt)
        self.dis_scale.update()
        if kwargs["cnt"] % self.n_critic == 0:
            self.model.train()
            self.model_opt.zero_grad()
            noise = torch.randn(size=(true_sample.shape[0], self.model.in_dim), device=self.device)
            fake_sample = self.model(noise)
            with autocast():
                loss = self.compute_loss(true_sample=true_sample, fake_sample=fake_sample, mode="gen")
            self.gen_scale.scale(loss).backward()
            self.gen_scale.unscale_(self.model_opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.gen_scale.step(self.model_opt)
            self.gen_scale.update()
            if self.use_ema:
                if kwargs["cnt"] % self.ema_step == 0:
                    self.ema.update(model=self.model)


class AutoEncoderWithDiscriminatorTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule, discriminator: ConfigModule, model_lr=0.0002, dis_lr=0.0002, n_critic=2,
                 valid_dir="./valid", save_dir="./model/gan", mid_valid_step=500, vae_mode="va", have_perception=False,
                 perception_net="vgg", perception_weight=1., kl_weight=1e-6, book_weight=1., dis_start=5001,
                 use_ema=False, ema_decay=0.999, ema_step=10):
        super(AutoEncoderWithDiscriminatorTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir,
                                                                  valid_dir=valid_dir, use_ema=use_ema,
                                                                  ema_decay=ema_decay, ema_step=ema_step)
        assert vae_mode in ["va", "vq"], "只支持va,vq两种方式"
        if vae_mode == "va":
            assert model_lr <= 1e-5, "va训练方式，损失函数是sum，学习率建议小于1e-5"
        self.n_critic = n_critic
        self.dis_start = dis_start
        self.dis_cnt = 0
        self.dis_flag = False
        self.log_var = LogVar()
        self.model = model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=model_lr, betas=(0.5, 0.99))
        self.dis_opt = torch.optim.AdamW(params=self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.99))
        self.dis_loss_func = GAN_HingeLoss()
        self.vae_mode = vae_mode
        self.dis_scale = GradScaler()
        if vae_mode == "va":
            self.vae_loss_func = AutoEncoderKLLoss(have_perception=have_perception, perception_weight=perception_weight,
                                                   perception_net=perception_net, kl_weight=kl_weight)
        else:
            self.vae_loss_func = VQAutoEncoderLoss(book_weight=book_weight, have_perception=have_perception,
                                                   perception_weight=perception_weight,
                                                   perception_net=perception_net)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        valid_data = self.preprocessing_data(kwargs["valid_data"])
        with torch.no_grad(), autocast():
            out, *_ = self.model(valid_data)
        valid_fake_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_fake.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_fake_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        valid_true_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_true.png')
        save_image(tensor=valid_data.clamp(min=-1, max=1), fp=valid_true_path,
                   nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()

    def compute_loss(self, true_sample, fake_sample, other, mode) -> torch.Tensor:
        if mode == "ordinary":
            if self.vae_mode == "va":
                loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, latent=other,
                                          log_var=self.log_var())
            else:
                loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, book_loss=other,
                                          log_var=self.log_var())
            self._validate_loss(loss=loss, optimizer=self.model_opt)
        elif mode == "gen":
            if self.vae_mode == "va":
                nll_loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, latent=other,
                                              log_var=self.log_var())
            else:
                nll_loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, book_loss=other,
                                              log_var=self.log_var())
            g_loss = -self.discriminator(fake_sample).mean()
            g_weight = self.calculate_adaptive_weight(a_loss=nll_loss, b_loss=g_loss,
                                                      model_last_layer=self.model.get_last_layer_weight())
            loss = nll_loss + g_weight * g_loss
            self._validate_loss(loss=loss, optimizer=self.model_opt)
        else:
            true_sample = self.discriminator(true_sample)
            fake_sample = self.discriminator(fake_sample.detach())
            loss = self.dis_loss_func(fake_sample=fake_sample, true_sample=true_sample)
            self._validate_loss(loss=loss, optimizer=self.dis_opt)
        return loss

    def train_one_batch(self, **kwargs) -> None:
        true_sample = kwargs["data"]
        self.model.train()
        self.model_opt.zero_grad()
        fake_sample, other = self.model(true_sample)
        if not self.dis_flag:
            with autocast():
                loss = self.compute_loss(true_sample=true_sample, fake_sample=fake_sample, other=other,
                                         mode="ordinary")
            self.scale.scale(loss).backward()
            self.scale.unscale_(self.model_opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.model_opt)
            self.scale.update()
            self.dis_cnt += 1
            self.dis_flag = self.dis_cnt >= self.dis_start
            if self.dis_flag:
                print("重构训练结束，开始对抗训练")
            if self.use_ema:
                if kwargs["cnt"] % self.ema_step == 0:
                    self.ema.update(model=self.model)
        else:
            self.discriminator.train()
            self.dis_opt.zero_grad()
            with autocast():
                loss = self.compute_loss(true_sample=true_sample, fake_sample=fake_sample, other=other,
                                         mode="dis")
            self.dis_scale.scale(loss).backward()
            self.dis_scale.unscale_(self.dis_opt)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.dis_scale.step(self.dis_opt)
            self.dis_scale.update()
            if kwargs["cnt"] % self.n_critic == 0:
                with autocast():
                    loss = self.compute_loss(true_sample=true_sample, fake_sample=fake_sample, other=other, mode="gen")
                self.scale.scale(loss).backward()
                self.scale.unscale_(self.model_opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scale.step(self.model_opt)
                self.scale.update()
                if self.use_ema:
                    if kwargs["cnt"] % self.ema_step == 0:
                        self.ema.update(model=self.model)


class ESRTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule, model_lr=1e-5, valid_dir="./valid", save_dir="./model/",
                 mid_valid_step=500, have_perception=False,
                 perception_net="vgg", perception_weight=1., compile_model=False, use_ema=False, ema_decay=0.999,
                 ema_step=10):
        super(ESRTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir,
                                         valid_dir=valid_dir, use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        assert model_lr <= 1e-5, "损失函数是sum，学习率建议小于1e-5"
        if compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        self.log_var = LogVar()
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=model_lr, betas=(0.5, 0.99))
        self.rec_loss_func = ESRLoss(have_perception=have_perception, perception_weight=perception_weight,
                                     perception_net=perception_net)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        lr, hr = self.preprocessing_data(kwargs["valid_data"])
        with torch.no_grad(), autocast():
            out = self.model(lr)
        valid_fake_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_fake.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_fake_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        valid_true_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_true.png')
        save_image(tensor=hr.clamp(min=-1, max=1), fp=valid_true_path,
                   nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()

    def compute_loss(self, true_sample, fake_sample) -> torch.Tensor:
        loss = self.rec_loss_func(fake_sample=fake_sample, true_sample=true_sample)
        self._validate_loss(loss, self.model_opt)
        return loss

    def train_one_batch(self, **kwargs) -> None:
        self.model.train()
        self.model_opt.zero_grad()
        lr, hr = kwargs["data"]
        with autocast():
            fake_sample = self.model(lr)
            loss = self.compute_loss(fake_sample=fake_sample, true_sample=hr)
        self.scale.scale(loss).backward()
        self.scale.unscale_(self.model_opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scale.step(self.model_opt)
        self.scale.update()
        if self.use_ema:
            if kwargs["cnt"] % self.ema_step == 0:
                self.ema.update(model=self.model)

    def preprocessing_data(self, data) -> tuple:
        lr = data[0].to(self.device)
        hr = data[1].to(self.device)
        return lr, hr


class ESRWithDiscriminatorTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule, discriminator: ConfigModule, model_lr=0.0002, dis_lr=0.0002, n_critic=2,
                 valid_dir="./valid", save_dir="./model/gan", mid_valid_step=500, have_perception=False,
                 perception_net="vgg", perception_weight=1., dis_start=5001, use_ema=False, ema_decay=0.999,
                 ema_step=10):
        super(ESRWithDiscriminatorTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir,
                                                          valid_dir=valid_dir, use_ema=use_ema, ema_decay=ema_decay,
                                                          ema_step=ema_step)
        assert model_lr <= 1e-5, "损失函数是sum，学习率建议小于1e-5"
        self.n_critic = n_critic
        self.dis_start = dis_start
        self.dis_cnt = 0
        self.dis_flag = False
        self.log_var = LogVar()
        self.model = model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=model_lr, betas=(0.5, 0.99))
        self.dis_opt = torch.optim.AdamW(params=self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.99))
        self.dis_loss_func = GAN_HingeLoss()
        self.dis_scale = GradScaler()
        self.rec_loss_func = ESRLoss(have_perception=have_perception, perception_weight=perception_weight,
                                     perception_net=perception_net)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        lr, hr = self.preprocessing_data(kwargs["valid_data"])
        with torch.no_grad(), autocast():
            out = self.model(lr)
        valid_fake_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_fake.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_fake_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        valid_true_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}_true.png')
        save_image(tensor=hr.clamp(min=-1, max=1), fp=valid_true_path,
                   nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()

    def preprocessing_data(self, data) -> tuple:
        lr = data[0].to(self.device)
        hr = data[1].to(self.device)
        return lr, hr

    def compute_loss(self, true_sample, fake_sample, mode) -> torch.Tensor:
        if mode == "ordinary":
            loss = self.rec_loss_func(true_sample=true_sample, fake_sample=fake_sample)
            self._validate_loss(loss=loss, optimizer=self.model_opt)
        elif mode == "gen":
            nll_loss = self.rec_loss_func(true_sample=true_sample, fake_sample=fake_sample)
            g_loss = -self.discriminator(fake_sample).mean()
            g_weight = self.calculate_adaptive_weight(a_loss=nll_loss, b_loss=g_loss,
                                                      model_last_layer=self.model.get_last_layer_weight())
            loss = nll_loss + g_weight * g_loss
            self._validate_loss(loss=loss, optimizer=self.model_opt)
        else:
            true_sample = self.discriminator(true_sample)
            fake_sample = self.discriminator(fake_sample.detach())
            loss = self.dis_loss_func(fake_sample=fake_sample, true_sample=true_sample)
            self._validate_loss(loss=loss, optimizer=self.dis_opt)
        return loss

    def train_one_batch(self, **kwargs) -> None:
        lr, hr = kwargs["data"]
        self.model.train()
        self.model_opt.zero_grad()
        fake_sample, other = self.model(lr)
        if not self.dis_flag:
            with autocast():
                loss = self.compute_loss(true_sample=hr, fake_sample=fake_sample, mode="ordinary")
            self.scale.scale(loss).backward()
            self.scale.unscale_(self.model_opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.model_opt)
            self.scale.update()
            self.dis_cnt += 1
            self.dis_flag = self.dis_cnt >= self.dis_start
            if self.dis_flag:
                print("重构训练结束，开始对抗训练")
            if self.use_ema:
                if kwargs["cnt"] % self.ema_step == 0:
                    self.ema.update(model=self.model)
        else:
            self.discriminator.train()
            self.dis_opt.zero_grad()
            with autocast():
                loss = self.compute_loss(true_sample=hr, fake_sample=fake_sample, mode="dis")
            self.dis_scale.scale(loss).backward()
            self.dis_scale.unscale_(self.dis_opt)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.dis_scale.step(self.dis_opt)
            self.dis_scale.update()
            if kwargs["cnt"] % self.n_critic == 0:
                with autocast():
                    loss = self.compute_loss(true_sample=hr, fake_sample=fake_sample, mode="gen")
                self.scale.scale(loss).backward()
                self.scale.unscale_(self.model_opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scale.step(self.model_opt)
                self.scale.update()
                if self.use_ema:
                    if kwargs["cnt"] % self.ema_step == 0:
                        self.ema.update(model=self.model)


class CLIPTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule, lr=1e-4, valid_dir="./valid/", save_dir="./model/", compile_model=False,
                 mid_valid_step=500, use_ema=False, ema_decay=0.999, ema_step=10):
        super(CLIPTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir,
                                          valid_dir=valid_dir, use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        if compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        self.model_opt = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)
        self.text_loss = []
        self.image_loss = []

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        texts = self.preprocessing_data(kwargs["valid_data"][0])
        images = self.preprocessing_data(kwargs["valid_data"][1])
        text_acc, image_acc = self.model.accuracy(text=texts, image=images)
        path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}.png')
        fig, axis = plt.subplots(1, 1)
        text = """
        accuracy image2text={:.2%} 
        accuracy text2image={:.2%}""".format(image_acc, text_acc)
        axis.plot(self.image_loss, label="image_loss")
        axis.plot(self.text_loss, label="text_loss")
        axis.set_title(text)
        axis.legend()
        fig.savefig(path)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()

    def compute_loss(self, texts, images) -> tuple:
        loss, loss_text, loss_image = self.model(text=texts, images=images)
        self._validate_loss(loss=loss, optimizer=self.model_opt)
        return loss, loss_text, loss_image

    def train_one_batch(self, **kwargs) -> None:
        texts = self.preprocessing_data(kwargs["data"][0])
        images = self.preprocessing_data(kwargs["data"][1])
        self.model.train()
        self.model_opt.zero_grad()
        with autocast():
            loss, loss_text, loss_image = self.compute_loss(texts=texts, images=images)
            self.text_loss.append(loss_text)
            self.image_loss.append(loss_image)
        self.scale.scale(loss).backward()
        self.scale.unscale_(self.model_opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scale.step(self.model_opt)
        self.scale.update()
        if self.use_ema:
            if kwargs["cnt"] % self.ema_step == 0:
                self.ema.update(model=self.model)

    def train_one_epoch(self, **kwargs) -> None:
        for cnt, data in enumerate(tqdm(kwargs["data_loader"], desc=f"{kwargs['epoch']}/{kwargs['epoch_nums']}")):
            self.train_one_batch(data=data, cnt=cnt)
            if self.mid_valid_flag:
                if cnt % self.mid_valid_step == 0:
                    self.validate(file_name="valid", valid_data=kwargs["valid_data"],
                                  valid_batch_size=kwargs["valid_batch_size"])
                    self.save()

    def run(self, data_loader, epoch_nums, valid_step, valid_batch_size, valid_data) -> None:
        for epoch in range(epoch_nums):
            self.train_one_epoch(data_loader=data_loader, epoch=epoch, epoch_nums=epoch_nums, valid_data=valid_data,
                                 valid_batch_size=valid_batch_size)
            if epoch % valid_step == 0:
                self.validate(file_name=epoch, valid_data=valid_data, valid_batch_size=valid_batch_size)
                self.save()
                self.text_loss = []
                self.image_loss = []


class LDMTrainer(BaseTrainer):
    def __init__(self, model: ConfigModule = lambda x_, y, o=None: x_, lr=1e-4, save_dir="./model", valid_dir="./valid",
                 mid_valid_step=None, compile_model=False, use_ema=False, ema_decay=0.999, ema_step=10,
                 std_mean_path="./pretrain/state_dict/AutoEncoder_x8_params_mean_std_compute_on_250K.json",
                 vae_pretrain_path="./pretrain/state_dict/003e2ba889653274bde0f086f5c318c3"):
        super(LDMTrainer, self).__init__(mid_valid_step=mid_valid_step, save_dir=save_dir, valid_dir=valid_dir,
                                         use_ema=use_ema, ema_decay=ema_decay, ema_step=ema_step)
        if compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        self.model_opt = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1e-8)
        with open(vae_pretrain_path + ".json", "r") as f:
            vae_config = json.load(f)
        self.vae = AutoEncoder(**vae_config["AutoEncoder"])
        self.vae.load_state_dict(get_load_state_dict_from_compile(vae_pretrain_path + ".pth", device=self.device))
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        with open(std_mean_path, "r") as f:
            params = json.load(f)
        self.global_std = params["global_std"]
        if self.use_ema:
            self.ema = EMA(model=self.model, decay=ema_decay)

    def preprocessing_data(self, data) -> torch.Tensor:
        data = data.to(self.device)
        data = data / self.global_std
        return data

    def compute_loss(self, data) -> torch.Tensor:
        loss = self.model(data)
        self._validate_loss(loss, self.model_opt)
        return loss

    def validate(self, **kwargs) -> None:
        self.model.eval()
        if self.use_ema:
            backup = self.ema.apply_shadow(model=self.model)
        mode = kwargs.get("mode", "ddpm")
        with torch.no_grad(), autocast():
            latent = self.model.sample(batch_size=kwargs["valid_batch_size"], mode=mode)
            flag = """
            latent_max:{:.3f}
            latent_min:{:.3f}
            latent_mean:{:.3f}
            latent_std:{:.3f}
            """.format(latent.max(), latent.min(), latent.mean(), latent.std())
            print(flag)
            latent = latent * self.global_std
            out = self.vae.latent2image(latent)
        valid_path = os.path.join(self.valid_dir, f'{kwargs["file_name"]}.png')
        save_image(tensor=out.clamp(min=-1, max=1), fp=valid_path, nrow=int(math.sqrt(kwargs["valid_batch_size"])),
                   normalize=True,
                   padding=1)
        if self.use_ema:
            self.ema.restore(model=self.model, backup=backup)
        self.model.train()
