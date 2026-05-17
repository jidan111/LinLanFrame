from .structs import *
from .losses import *


class DiffusionTrainer(Trainer):
    def __init__(self, model: ConfigModule, lr=1e-5, valid_dir="./valid", save_model_dir="./model/diffusion",
                 compile_model=False, mid_save_step=500, **kwargs):
        super(DiffusionTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                               mid_save_step=mid_save_step, file_name=self.__class__.__name__)
        if compile_model:
            self.model = torch.compile(model.to(self.device),
                                       mode="default",  # 最大程度优化（训练专用）
                                       dynamic=False,  # 固定输入尺寸关闭动态（更快）
                                       backend="inductor"  # 默认最优后端
                                       )
        else:
            self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1e-8)

    def get_model_config(self):
        return {self.model.model.__class__.__name__: self.model.model.config,
                self.model.__class__.__name__: self.model.config}

    def loss(self, data):
        loss = self.model(data)
        return loss

    def valid(self, valid_batch_size=4, file_name="valid", **kwargs):
        self.model.eval()
        row = int(math.sqrt(valid_batch_size))
        out = self.model.sample(batch_size=valid_batch_size).clamp(0, 1)
        save_image(out, nrow=row, fp=os.path.join(self.valid_dir, f"{file_name}.png"), normalize=False, padding=1)
        self.model.train()


class GANTrainer(Trainer):
    def __init__(self, generator: ConfigModule, discriminator: ConfigModule, gen_lr=0.0002, dis_lr=0.0002, n_critic=1,
                 valid_dir="./valid", save_model_dir="./model/gan", loss_type="hinge", lambda_gp=10, mid_save_step=500,
                 **kwargs):
        super(GANTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                         mid_save_step=mid_save_step, file_name=self.__class__.__name__)
        assert loss_type in ["hinge", "wgp", "dc"], "只支持hinge和wgp还有dc三种方式"
        self.loss_type = loss_type
        self.n_critic = n_critic
        self.discriminator = discriminator.to(self.device)
        self.model = generator.to(self.device)
        self.gen_opt = torch.optim.Adam(params=self.model.parameters(), lr=gen_lr, betas=(0.5, 0.99))
        self.dis_opt = torch.optim.Adam(params=self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.99))
        if loss_type == "hinge":
            self.loss_func = GAN_HingeLoss()
        elif loss_type == "dc":
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = WGAN_GP_Loss(lambda_gp=lambda_gp)

    def get_model_config(self):
        return {self.model.__class__.__name__: self.model.config,
                self.discriminator.__class__.__name__: self.discriminator.config}

    def loss(self, fake_sample, true_sample, mode="gen"):
        if mode == "gen":
            if self.loss_type == "dc":
                dis_label = self.discriminator(fake_sample)
                target_label = torch.ones_like(dis_label, device=true_sample.device)
                loss = self.loss_func(dis_label, target_label)
            else:
                loss = -self.discriminator(fake_sample).mean()
            return loss
        else:
            if self.loss_type == "hinge":
                true_sample = self.discriminator(true_sample)
                fake_sample = self.discriminator(fake_sample.detach())
                loss = self.loss_func(fake_sample=fake_sample, true_sample=true_sample)
            elif self.loss_type == "dc":
                dis_true = self.discriminator(true_sample)
                dis_fake = self.discriminator(fake_sample)
                true_label = torch.ones_like(dis_true, device=true_sample.device)
                fake_label = torch.zeros_like(dis_fake, device=true_sample.device)
                true_loss = self.loss_func(dis_true, true_label)
                fake_loss = self.loss_func(dis_fake, fake_label)
                loss = true_loss + fake_loss
            else:
                loss = self.loss_func(model=self.discriminator, fake_sample=fake_sample, true_sample=true_sample)
            return loss

    def valid(self, valid_batch_size=4, file_name="valid"):
        self.model.eval()
        row = int(math.sqrt(valid_batch_size))
        out = self.model.sample(batch_size=valid_batch_size)
        save_image(out, nrow=row, fp=os.path.join(self.valid_dir, f"{file_name}.png"), normalize=True, padding=1)
        self.model.train()

    def train_one_batch(self, data, index=0):
        self.model.train()
        self.discriminator.train()
        true_sample = data.to(self.device)
        with torch.no_grad():
            noise = torch.randn(size=(true_sample.shape[0], self.model.in_dim), device=self.device)
            fake_sample = self.model(noise)
        self.dis_opt.zero_grad()
        with autocast():
            loss = self.loss(true_sample=true_sample, fake_sample=fake_sample.detach(), mode="dis")
        if not torch.isfinite(loss):
            self.dis_opt.zero_grad()
            raise Exception("训练出现空值，已终止训练")
        self.scale.scale(loss).backward()
        self.scale.unscale_(self.dis_opt)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.scale.step(self.dis_opt)
        self.scale.update()
        if index % self.n_critic == 0:
            self.gen_opt.zero_grad()
            noise = torch.randn(size=(true_sample.shape[0], self.model.in_dim), device=self.device)
            fake_sample = self.model(noise)
            with autocast():
                loss = self.loss(true_sample=true_sample, fake_sample=fake_sample, mode="gen")
            if not torch.isfinite(loss):
                self.gen_opt.zero_grad()
                raise Exception("训练出现空值，已终止训练")
            self.scale.scale(loss).backward()
            self.scale.unscale_(self.gen_opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.gen_opt)
            self.scale.update()


class AutoEncoderTrainer(Trainer):
    def __init__(self, model: ConfigModule, lr=5e-6, valid_dir="./valid", save_model_dir="./model/autoencoder",
                 perception_net="vgg", kl_weight=1e-6, perception_weight=1.,
                 valid_batch_size=9, have_perception=False, compile_model=False, mid_save_step=500, book_weight=1.,
                 vae_mode="va", **kwargs):
        super(AutoEncoderTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                                 mid_save_step=mid_save_step, file_name=self.__class__.__name__)
        assert vae_mode in ["va", "vq"], "只支持vq和va两种训练方式"
        if lr >= 1e-5 and vae_mode == "va":
            warnings.warn("损失函数用的是sum，建议学习率小于1e-5，否则会出现nan", RuntimeWarning)
        if compile_model:
            self.model = torch.compile(model.to(self.device),
                                       mode="default",  # 最大程度优化（训练专用）
                                       dynamic=False,  # 固定输入尺寸关闭动态（更快）
                                       backend="inductor"  # 默认最优后端
                                       )
        else:
            self.model = model.to(self.device)
        self.mode = vae_mode
        self.log_var = LogVar()
        self.opt = torch.optim.Adam(params=list(self.model.parameters()) + list(self.log_var.parameters()),
                                    lr=lr)
        if vae_mode == "va":
            self.loss_func = AutoEncoderKLLoss(have_perception=have_perception, perception_weight=perception_weight,
                                               perception_net=perception_net, kl_weight=kl_weight)
        else:
            self.loss_func = VQAutoEncoderLoss(book_weight=book_weight, have_perception=have_perception,
                                               perception_weight=perception_weight, perception_net=perception_net)
        self.valid_image = None
        self.valid_batch_size = valid_batch_size

    def get_model_config(self):
        return {self.model.__class__.__name__: self.model.config}

    def loss(self, data):
        self.valid_image = data[:self.valid_batch_size]
        out, z = self.model(data)
        if self.mode == "va":
            loss = self.loss_func(true_sample=data, fake_sample=out, latent=z, log_var=self.log_var())
        else:
            loss = self.loss_func(true_sample=data, fake_sample=out, book_loss=z, log_var=self.log_var())
        return loss

    def valid(self, valid_batch_size=4, file_name="valid"):
        self.model.eval()
        row = int(math.sqrt(self.valid_batch_size))
        with torch.no_grad():
            out, z = self.model(self.valid_image)
        image = torch.cat((self.valid_image, out), dim=0)
        save_image(image, nrow=row, fp=os.path.join(self.valid_dir, f"{file_name}.png"), normalize=True, padding=1)
        self.model.train()


class AutoEncoderWithDiscriminatorTrainer(Trainer):
    def __init__(self, model: ConfigModule, discriminator: ConfigModule, gen_lr=5e-5, dis_lr=5e-5, n_critic=2,
                 valid_dir="./valid", save_model_dir="./model/gan", perception_net="vgg", perception_weight=1.,
                 valid_batch_size=9, have_perception=True, mid_save_step=500, kl_weight=1e-6, book_weight=1.,
                 vae_mode="va", dis_start=5001, **kwargs):
        super(AutoEncoderWithDiscriminatorTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                                                  mid_save_step=mid_save_step,
                                                                  file_name=self.__class__.__name__)
        assert vae_mode in ["va", "vq"], "只支持vq和va两种训练方式"
        if gen_lr >= 1e-5 and vae_mode == "va":
            warnings.warn("损失函数用的是sum，建议学习率小于1e-5，否则会出现nan", RuntimeWarning)
        self.vae_mode = vae_mode
        self.n_critic = n_critic
        self.dis_start = dis_start
        self.dis_cnt = 0
        self.dis_flag = False
        self.log_var = LogVar()
        self.discriminator = discriminator.to(self.device)
        self.model = model.to(self.device)
        self.gen_opt = torch.optim.AdamW(params=list(self.model.parameters()) + list(self.log_var.parameters()),
                                         lr=gen_lr)
        self.dis_opt = torch.optim.AdamW(params=self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.99))
        self.gan_loss_func = GAN_HingeLoss()
        if vae_mode == "va":
            self.vae_loss_func = AutoEncoderKLLoss(have_perception=have_perception, perception_weight=perception_weight,
                                                   perception_net=perception_net, kl_weight=kl_weight)
        else:
            self.vae_loss_func = VQAutoEncoderLoss(book_weight=book_weight, have_perception=have_perception,
                                                   perception_weight=perception_weight, perception_net=perception_net)
        self.valid_image = None
        self.valid_batch_size = valid_batch_size

    def get_model_config(self):
        return {self.model.__class__.__name__: self.model.config,
                self.discriminator.__class__.__name__: self.discriminator.config}

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def loss(self, fake_sample, true_sample, other, mode="gen"):
        self.valid_image = true_sample[:self.valid_batch_size]
        if mode == "ordinary":
            if self.vae_mode == "va":
                vae_loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, latent=other,
                                              log_var=self.log_var())
            else:
                vae_loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, book_loss=other,
                                              log_var=self.log_var())
            return vae_loss
        elif mode == "gen":
            if self.vae_mode == "va":
                vae_loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, latent=other,
                                              log_var=self.log_var())
            else:
                vae_loss = self.vae_loss_func(true_sample=true_sample, fake_sample=fake_sample, book_loss=other,
                                              log_var=self.log_var())
            g_loss = -self.discriminator(fake_sample).mean()
            g_weight = self.calculate_adaptive_weight(a_loss=vae_loss, b_loss=g_loss,
                                                      model_last_layer=self.model.get_last_layer_weight())
            loss = vae_loss + g_weight * g_loss
            return loss
        else:
            true_sample = self.discriminator(true_sample)
            fake_sample = self.discriminator(fake_sample.detach())
            loss = self.gan_loss_func(fake_sample=fake_sample, true_sample=true_sample)
            return loss

    def valid(self, valid_batch_size=4, file_name="valid"):
        self.model.eval()
        row = int(math.sqrt(self.valid_batch_size))
        with torch.no_grad():
            out, z = self.model(self.valid_image)
        image = torch.cat((self.valid_image, out), dim=0)
        save_image(image, nrow=row, fp=os.path.join(self.valid_dir, f"{file_name}.png"), normalize=True, padding=1)
        self.model.train()

    def train_one_batch(self, data, index=0):
        true_sample = data.to(self.device)
        fake_sample, z = self.model(true_sample)
        if not self.dis_flag:
            self.model.train()
            self.gen_opt.zero_grad()
            with autocast():
                loss = self.loss(true_sample=true_sample, fake_sample=fake_sample, other=z, mode="ordinary")
            if not torch.isfinite(loss):
                self.gen_opt.zero_grad()
                raise Exception("训练出现空值，已终止训练")
            self.scale.scale(loss).backward()
            self.scale.unscale_(self.gen_opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.gen_opt)
            self.scale.update()
            self.dis_cnt += 1
            self.dis_flag = self.dis_cnt >= self.dis_start
            if self.dis_flag:
                print("重构阶段结束，开始进入对抗阶段")
        else:
            self.discriminator.train()
            self.dis_opt.zero_grad()
            with autocast():
                loss = self.loss(true_sample=true_sample, fake_sample=fake_sample.detach(), other=z, mode="dis")
            if not torch.isfinite(loss):
                self.dis_opt.zero_grad()
                raise Exception("训练出现空值，已终止训练")
            self.scale.scale(loss).backward()
            self.scale.unscale_(self.dis_opt)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.scale.step(self.dis_opt)
            self.scale.update()
            if index % self.n_critic == 0:
                self.model.train()
                self.gen_opt.zero_grad()
                with autocast():
                    loss = self.loss(true_sample=true_sample, fake_sample=fake_sample, other=z, mode="gen")
                if not torch.isfinite(loss):
                    self.gen_opt.zero_grad()
                    raise Exception("训练出现空值，已终止训练")
                self.scale.scale(loss).backward()
                self.scale.unscale_(self.gen_opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scale.step(self.gen_opt)
                self.scale.update()
