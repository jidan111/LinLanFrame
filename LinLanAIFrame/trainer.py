"""
本项目还未完成
"""
from .structs import ConfigModule, EMA
from .losses import *
from .ImageGenerator import *
from .CLIP.TokenizerModels import Tokenizer


class NoneConfig(ConfigModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Trainer(object):
    def __init__(self, device, middle_validate_step=None,
                 gradient_accumulation_step=1, using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid", compile_model=False,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.valid_path = valid_path
        self.compile_model = compile_model
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        self.middle_validate_flag = middle_validate_step is not None
        self.middle_validate_step = middle_validate_step
        self.gradient_accumulation_step = gradient_accumulation_step
        self.gradient_accumulation_step_cur = 0
        self.using_ema = using_ema
        self.ema_update_step = ema_update_step
        self.ema_decay = ema_decay
        self.ema = None
        self.model = NoneConfig()
        self.optimizer = None
        self.scale = GradScaler()

    def set_model(self, model: ConfigModule) -> None:
        if self.compile_model:
            self.model = torch.compile(model.to(self.device))
        else:
            self.model = model.to(self.device)
        if self.using_ema:
            self.ema = EMA(model=self.model, decay=self.ema_decay)
        return None

    def set_optimizer(self, lr: float) -> None:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0)
        return None

    def set_log_file(self) -> None:
        log_file_path = os.path.join(self.save_path, "training.log")
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.close()

    def write_log(self, content, running_state: bool) -> None:
        log_file_path = os.path.join(self.save_path, "training.log")
        with open(log_file_path, "a+") as f:
            f.write(json.dumps({"loss": content, "state": running_state}) + "\n")

    def compute_loss(self, data) -> torch.Tensor:
        loss = self.model(data)
        self._validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def preprocessing_data(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        return data

    def _validate_loss(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            raise Exception("训练出现空值，已终止训练")

    def train_one_batch(self, data, iter_cnt) -> None:
        self.model.train()
        with autocast():
            loss = self.compute_loss(data) / self.gradient_accumulation_step
        self.scale.scale(loss).backward()
        self.gradient_accumulation_step_cur += 1
        if self.gradient_accumulation_step_cur == self.gradient_accumulation_step:
            self.scale.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scale.step(self.optimizer)
            self.scale.update()
            self.optimizer.zero_grad()
            self.gradient_accumulation_step_cur = 0
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.model)

    def train_one_epoch(self, data_loader, epoch, epoch_cnt, **kwargs) -> None:
        self.gradient_accumulation_step_cur = 0
        for iter_cnt, data in enumerate(tqdm(data_loader, desc=f"{epoch_cnt}/{epoch}")):
            data = self.preprocessing_data(data)
            self.train_one_batch(data=data, iter_cnt=iter_cnt)
            if self.middle_validate_flag:
                if iter_cnt % self.middle_validate_step == 0:
                    self.validate(**kwargs)
                    self.save_checkpoint()

    def run(self, data_loader, epoch, valid_step, valid_batch_size, valid_data, **kwargs) -> None:
        for epoch_cnt in range(epoch):
            self.train_one_epoch(data_loader=data_loader, epoch=epoch, epoch_cnt=epoch_cnt,
                                 valid_batch_size=valid_batch_size,
                                 valid_data=valid_data, **kwargs)
            if epoch_cnt % valid_step == 0:
                self.validate(valid_data=valid_data, valid_batch_size=valid_batch_size,
                              valid_file_name=f"{epoch_cnt}", **kwargs)
                self.save_checkpoint()

    def validate(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save_checkpoint(self) -> None:
        config = self.model.config
        if self.compile_model:
            state_dict = self.model._orig_mod.state_dict()
        else:
            state_dict = self.model.state_dict()
        model_name = list(config.keys())[0]
        config_path = os.path.join(self.save_path, f"{model_name}.json")
        state_dict_path = os.path.join(self.save_path, f"{model_name}.pth")
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))
        torch.save(state_dict, state_dict_path)
        if self.using_ema:
            ema_path = os.path.join(self.save_path, f"ema_{model_name}.pth")
            torch.save(self.ema.shadow, ema_path)


class GANTrainer(Trainer):
    def __init__(self, generator_train_step, device, loss_func=hinge, middle_validate_step=None,
                 gradient_accumulation_step=1, using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid"):
        super(GANTrainer, self).__init__(device=device, middle_validate_step=middle_validate_step,
                                         gradient_accumulation_step=gradient_accumulation_step, using_ema=using_ema,
                                         ema_update_step=ema_update_step, ema_decay=ema_decay,
                                         save_path=save_path, valid_path=valid_path)
        self.generator_train_step = generator_train_step
        self.generator_train_step_cur = 0
        self.generator = NoneConfig()
        self.generator_optimizer = None
        self.generator_scale = GradScaler()
        self.discriminator = NoneConfig()
        self.discriminator_optimizer = None
        self.discriminator_scale = GradScaler()
        self.loss_func = loss_func

    def set_model(self, generator: Generator, discriminator: Discriminator) -> None:
        self.generator = generator.to(self.device)
        if self.using_ema:
            self.ema = EMA(model=self.generator, decay=self.ema_decay)
        self.discriminator = discriminator.to(self.device)

    def set_optimizer(self, generator_lr: float = 1e-4, discriminator_lr: float = 1e-4) -> None:
        self.generator_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=generator_lr, betas=(0.9, 0.999),
                                                     eps=1e-08, weight_decay=0)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=discriminator_lr,
                                                         betas=(0.9, 0.999),
                                                         eps=1e-08, weight_decay=0)

    def __train_generator_once(self, batch_size: int):
        self.generator.train()
        self.generator_optimizer.zero_grad()
        noise = torch.randn(size=(batch_size, self.generator.in_dim), device=self.device)
        with autocast():
            fake_sample = self.generator(noise)
            loss = -self.discriminator(fake_sample).mean()
        self._validate_loss(loss, self.generator_optimizer)
        self.generator_scale.scale(loss).backward()
        self.generator_scale.unscale_(self.generator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.generator_scale.step(self.generator_optimizer)
        self.generator_scale.update()

    def __train_discriminator_once(self, fake_sample, true_sample):
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with autocast():
            fake_score = self.discriminator(fake_sample.detach())
            true_score = self.discriminator(true_sample)
            loss = self.loss_func(fake_sample=fake_score, true_sample=true_score)
            self._validate_loss(loss, self.discriminator_optimizer)
        self.discriminator_scale.scale(loss).backward()
        self.discriminator_scale.unscale_(self.discriminator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.discriminator_scale.step(self.discriminator_optimizer)
        self.discriminator_scale.update()

    def train_one_batch(self, data, iter_cnt) -> None:
        noise = torch.randn(size=(data.shape[0], self.generator.in_dim), device=self.device)
        fake_sample = self.generator(noise)
        self.__train_discriminator_once(fake_sample=fake_sample, true_sample=data)
        self.generator_train_step_cur += 1
        if self.generator_train_step_cur == self.generator_train_step:
            self.__train_generator_once(batch_size=data.shape[0])
            self.generator_train_step_cur = 0
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.generator)

    def validate(self, *args, **kwargs) -> None:
        valid_file = kwargs.get("valid_file_name", "valid")
        valid_file_path = os.path.join(self.valid_path, f"{valid_file}.jpg")
        valid_batch_size = kwargs.get("valid_batch_size", 16)
        self.generator.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.generator)
        with torch.no_grad(), autocast():
            noise = torch.randn(size=(valid_batch_size, self.generator.in_dim), device=self.device)
            out = self.generator(noise).clamp(min=-1, max=1)
            save_image(tensor=out, normalize=True, padding=1, nrow=int(math.sqrt(valid_batch_size)), fp=valid_file_path)
        if self.using_ema:
            self.ema.restore(model=self.generator, backup=back_params)
        self.generator.train()

    def save_checkpoint(self) -> None:
        config = self.generator.config
        state_dict = self.generator.state_dict()
        model_name = list(config.keys())[0]
        config_path = os.path.join(self.save_path, f"{model_name}.json")
        state_dict_path = os.path.join(self.save_path, f"{model_name}.pth")
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))
        torch.save(state_dict, state_dict_path)
        if self.using_ema:
            ema_path = os.path.join(self.save_path, f"ema_{model_name}.pth")
            torch.save(self.ema.shadow, ema_path)


class AutoEncoderTrainer(Trainer):
    def __init__(self, device,
                 loss_func=AutoEncoderKLLoss(have_perception=False, perception_weight=1., perception_net="alex",
                                             kl_weight=1e-6), middle_validate_step=None, compile_model=False,
                 gradient_accumulation_step=1, using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid"):
        super().__init__(device=device, middle_validate_step=middle_validate_step,
                         gradient_accumulation_step=gradient_accumulation_step, using_ema=using_ema,
                         ema_update_step=ema_update_step, ema_decay=ema_decay,
                         save_path=save_path, valid_path=valid_path, compile_model=compile_model)
        self.loss_func = loss_func

    def set_optimizer(self, lr: float) -> None:
        if type(self.loss_func) == AutoEncoderKLLoss:
            assert lr <= 2e-5, f"AutoEncoderKLLoss 使用sum统计损失值，学习率过大会导致NAN"
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0)

    def compute_loss(self, data) -> torch.Tensor:
        fake_sample, other = self.model(data)
        loss = self.loss_func(fake_sample, data[0], other)
        self._validate_loss(loss, self.optimizer)
        return loss

    def validate(self, *args, **kwargs) -> None:
        assert "valid_data" in kwargs, "需要在validate显式传入valid_data参数"
        assert kwargs["valid_data"] is not None, "valid_data不能为空"
        valid_data = kwargs["valid_data"].to(self.device)
        valid_file = kwargs.get("valid_file_name", "valid")
        valid_file_path = os.path.join(self.valid_path, f"{valid_file}.jpg")
        valid_batch_size = valid_data.shape[0]
        valid_row = int(math.sqrt(valid_batch_size))
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad(), autocast():
            latent = self.model.image2latent(valid_data)
            out = self.model.latent2image(latent).clamp(min=-1, max=1)
            out = make_grid(out, nrow=valid_row, padding=1, normalize=True).permute(1, 2, 0).cpu().numpy().astype(
                np.float32)
            latent = \
                make_grid(latent, nrow=valid_row, padding=1, normalize=False).permute(1, 2, 0).cpu().numpy().astype(
                    np.float32)[:, :, :3]
            in_ = make_grid(valid_data, nrow=valid_row, padding=1, normalize=True).permute(1, 2,
                                                                                           0).cpu().numpy().astype(
                np.float32)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax in axes:
                ax.axis("off")
            axes[0].imshow(in_)
            axes[0].title.set_text(f"Real:{in_.shape}")
            axes[1].imshow(latent)
            axes[1].title.set_text(f"Latent:{latent.shape}")
            axes[2].imshow(out)
            axes[2].title.set_text(f"Decode:{out.shape}")
            plt.tight_layout()
            plt.savefig(valid_file_path, dpi=300, bbox_inches='tight')
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()


class AutoEncoderKLTrainer(Trainer):
    def __init__(self, device,
                 autoencoder_loss_func=AutoEncoderKLLoss(have_perception=False, perception_weight=1.,
                                                         perception_net="alex",
                                                         kl_weight=1e-6), dis_loss_func=hinge,
                 middle_validate_step=None, dis_start=5001,
                 gradient_accumulation_step=1, using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid"):
        super().__init__(device=device, middle_validate_step=middle_validate_step,
                         gradient_accumulation_step=gradient_accumulation_step, using_ema=using_ema,
                         ema_update_step=ema_update_step, ema_decay=ema_decay,
                         save_path=save_path, valid_path=valid_path)
        self.autoencoder_loss_func = autoencoder_loss_func
        self.dis_loss_func = dis_loss_func
        self.dis_train_flag = False
        self.dis_start = dis_start
        self.dis_start_cnt_cur = 0
        self.autoencoder = AutoEncoder
        self.autoencoder_optimizer = None
        self.autoencoder_scale = GradScaler()
        self.discriminator = PatchDiscriminator
        self.discriminator_optimizer = None
        self.discriminator_scale = GradScaler()

    def set_model(self, autoencoder: AutoEncoder, discriminator: PatchDiscriminator) -> None:
        self.autoencoder = autoencoder.to(self.device)
        if self.using_ema:
            self.ema = EMA(model=self.autoencoder, decay=self.ema_decay)
        self.discriminator = discriminator.to(self.device)
        return None

    def set_optimizer(self, autoencoder_lr: float = 1e-4, discriminator_lr: float = 1e-4) -> None:
        if type(self.autoencoder_loss_func) == AutoEncoderKLLoss:
            assert autoencoder_lr <= 2e-5, f"AutoEncoderKLLoss 使用sum统计损失值，学习率过大会导致NAN"
        self.autoencoder_optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=autoencoder_lr,
                                                       betas=(0.9, 0.999),
                                                       eps=1e-08,
                                                       weight_decay=0)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=discriminator_lr,
                                                         betas=(0.9, 0.999), eps=1e-08,
                                                         weight_decay=0)
        return None

    def calculate_adaptive_weight(self, a_loss, b_loss, model_last_layer):
        a_grads = autograd.grad(outputs=a_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_grads = autograd.grad(outputs=b_loss, inputs=model_last_layer, retain_graph=True)[0]
        b_weight = torch.norm(a_grads) / (torch.norm(b_grads) + 1e-4)
        b_weight = torch.clamp(b_weight, 0.0, 1e4).detach()
        return b_weight

    def __train_autoencoder_once(self, data):
        self.autoencoder.train()
        self.autoencoder_optimizer.zero_grad()
        if not self.dis_train_flag:
            self.dis_start_cnt_cur += 1
            self.dis_train_flag = self.dis_start_cnt_cur >= self.dis_start
            with autocast():
                out, other = self.autoencoder(data)
                loss = self.autoencoder_loss_func(out, data, other)
        else:
            self.discriminator.eval()
            with autocast():
                out, other = self.autoencoder(data)
                nll_loss = self.autoencoder_loss_func(out, data, other)
                g_loss = -self.discriminator(out).mean()
                g_weight = self.calculate_adaptive_weight(a_loss=nll_loss, b_loss=g_loss,
                                                          model_last_layer=self.model.get_last_layer_weight())
                loss = nll_loss + g_weight * g_loss
        self._validate_loss(loss, self.autoencoder_optimizer)
        self.autoencoder_scale.scale(loss).backward()
        self.autoencoder_scale.unscale_(self.autoencoder_optimizer)
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
        self.autoencoder_scale.step(self.autoencoder_optimizer)
        self.autoencoder_scale.update()
        if self.dis_train_flag:
            self.discriminator.train()

    def __train_discriminator_once(self, data):
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with autocast():
            fake_sample, *_ = self.autoencoder(data)
            true_score = self.discriminator(data)
            fake_score = self.discriminator(fake_sample.detach())
            loss = self.dis_loss_func(true_sample=true_score, fake_sample=fake_score)
        self._validate_loss(loss, self.discriminator_optimizer)
        self.discriminator_scale.scale(loss).backward()
        self.discriminator_scale.unscale_(self.discriminator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.discriminator_scale.step(self.discriminator_optimizer)
        self.discriminator_scale.update()

    def train_one_batch(self, data, iter_cnt) -> None:
        self.__train_autoencoder_once(data=data)
        if self.dis_train_flag:
            self.__train_discriminator_once(data=data)
        if self.using_ema:
            if iter_cnt % self.ema_update_step == 0:
                self.ema.update(self.autoencoder)

    def validate(self, *args, **kwargs) -> None:
        assert "valid_data" in kwargs, "需要在validate显式传入valid_data参数"
        assert kwargs["valid_data"] is not None, "valid_data不能为空"
        valid_data = kwargs["valid_data"].to(self.device)
        valid_file = kwargs.get("valid_file_name", "valid")
        valid_file_path = os.path.join(self.valid_path, f"{valid_file}.jpg")
        valid_batch_size = valid_data.shape[0]
        valid_row = int(math.sqrt(valid_batch_size))
        self.autoencoder.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.autoencoder)
        with torch.no_grad(), autocast():
            latent = self.autoencoder.image2latent(valid_data)
            out = self.autoencoder.latent2image(latent).clamp(min=-1, max=1)
            out = make_grid(out, nrow=valid_row, padding=1, normalize=True).permute(1, 2, 0).cpu().numpy().astype(
                np.float32)
            latent =  make_grid(latent, nrow=valid_row, padding=1, normalize=False).permute(1, 2, 0).cpu().numpy().astype(
                np.float32)[:, :, :3]
            in_ = make_grid(valid_data, nrow=valid_row, padding=1, normalize=True).permute(1, 2,
                                                                                           0).cpu().numpy().astype(
                np.float32)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax in axes:
                ax.axis("off")
            axes[0].imshow(in_)
            axes[0].title.set_text(f"Real:{in_.shape}")
            axes[1].imshow(latent)
            axes[1].title.set_text(f"Latent:{latent.shape}")
            axes[2].imshow(out)
            axes[2].title.set_text(f"Decode:{out.shape}")
            plt.tight_layout()
            plt.savefig(valid_file_path, dpi=300, bbox_inches='tight')
        if self.using_ema:
            self.ema.restore(model=self.autoencoder, backup=back_params)
        self.autoencoder.train()

    def save_checkpoint(self) -> None:
        config = self.autoencoder.config
        state_dict = self.autoencoder.state_dict()
        model_name = list(config.keys())[0]
        config_path = os.path.join(self.save_path, f"{model_name}.json")
        state_dict_path = os.path.join(self.save_path, f"{model_name}.pth")
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))
        torch.save(state_dict, state_dict_path)
        if self.using_ema:
            ema_path = os.path.join(self.save_path, f"ema_{model_name}.pth")
            torch.save(self.ema.shadow, ema_path)


class DiffusionTrainer(Trainer):
    def validate(self, *args, **kwargs) -> None:
        assert "image_shape" in kwargs, "需要在validate显式传入image_shape参数"
        valid_file = kwargs.get("valid_file_name", "valid")
        valid_file_path = os.path.join(self.valid_path, f"{valid_file}.jpg")
        self.model.eval()
        valid_batch_size = kwargs.get("valid_batch_size", 16)
        model_name = self.model.__class__.__name__
        if model_name == "Diffusion":
            mode = kwargs.get("mode", "dpm")
        else:
            mode = kwargs.get("mode", "euler")
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad(), autocast():
            out = self.model.sample(batch_size=valid_batch_size, image_shape=kwargs["image_shape"], mode=mode).clamp(
                min=-1, max=1)
            save_image(tensor=out, normalize=True, padding=1, nrow=int(math.sqrt(valid_batch_size)), fp=valid_file_path)
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()
        return None


class ESRTrainer(Trainer):
    def __init__(self, device,
                 loss_func=ESRLoss(have_perception=False, perception_weight=1., perception_net="vgg"),
                 middle_validate_step=None, compile_model=False,
                 gradient_accumulation_step=1, using_ema=False, ema_update_step=10, ema_decay=0.999,
                 save_path="./model", valid_path="./valid"):
        super().__init__(device=device, middle_validate_step=middle_validate_step,
                         gradient_accumulation_step=gradient_accumulation_step, using_ema=using_ema,
                         ema_update_step=ema_update_step, ema_decay=ema_decay,
                         save_path=save_path, valid_path=valid_path, compile_model=compile_model)
        self.loss_func = loss_func

    def preprocessing_data(self, data: torch.Tensor) -> tuple:
        low_img = data[0].to(self.device)
        high_img = data[1].to(self.device)
        return low_img, high_img

    def compute_loss(self, data):
        fake_out = self.model(data[0])
        loss = self.loss_func(fake_out, data[1])
        self._validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def validate(self, *args, **kwargs) -> None:
        assert "valid_data" in kwargs, "需要在validate显式传入valid_data参数"
        assert kwargs["valid_data"] is not None, "valid_data不能为空"
        valid_data = kwargs["valid_data"].to(self.device)
        valid_file = kwargs.get("valid_file_name", "valid")
        valid_file_path = os.path.join(self.valid_path, f"{valid_file}.jpg")
        valid_batch_size = valid_data.shape[0]
        valid_row = int(math.sqrt(valid_batch_size))
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad(), autocast():
            esr_out = self.model(valid_data).clamp(min=-1, max=1)
            out = make_grid(esr_out, nrow=valid_row, padding=1, normalize=True).permute(1, 2, 0).cpu().numpy().astype(
                np.float32)
            in_ = make_grid(valid_data, nrow=valid_row, padding=1, normalize=True).permute(1, 2,
                                                                                           0).cpu().numpy().astype(
                np.float32)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax in axes:
                ax.axis("off")
            axes[0].imshow(in_)
            axes[0].title.set_text(f"Input:{in_.shape}")
            axes[1].imshow(out)
            axes[1].title.set_text(f"Esr_output:{out.shape}")
            plt.tight_layout()
            plt.savefig(valid_file_path, dpi=300, bbox_inches='tight')
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()


class CLIPTrainer(Trainer):
    def preprocessing_data(self, data: torch.Tensor) -> tuple:
        text = data[0].to(self.device)
        image = data[1].to(self.device)
        return image, text

    def compute_loss(self, data) -> torch.Tensor:
        loss = self.model(data[0], data[1])
        self._validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def validate(self, *args, **kwargs) -> None:
        assert "valid_data" in kwargs, "需要在validate显式传入valid_data参数"
        assert kwargs["valid_data"] is not None, "valid_data不能为空"
        valid_text, valid_image = self.preprocessing_data(kwargs["valid_data"])
        self.model.eval()
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad(), autocast():
            recall_result = self.model.recall_at_k(image=valid_image, text=valid_text, top_k_arr=(1, 5, 10))
            for recall in recall_result:
                key = list(recall.keys())[0]
                item = list(recall[key].items())
                display_str = f"{key}: {item[0][0]}={item[0][1]:.2%}, {item[1][0]}:{item[1][1]:.2%}"
                print(display_str)
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()


class Text2ImageTrainer(Trainer):
    def __init__(self, device, autoencoder, clip, autoencoder_std=4.18157, compile_model=False,
                 tokenizer=Tokenizer(dim=200, numpy=True),
                 middle_validate_step=None,
                 gradient_accumulation_step=1,
                 using_ema=False,
                 ema_update_step=10,
                 ema_decay=0.999,
                 save_path="./model",
                 valid_path="./valid"):
        super().__init__(device=device, middle_validate_step=middle_validate_step,
                         gradient_accumulation_step=gradient_accumulation_step, using_ema=using_ema,
                         ema_update_step=ema_update_step, ema_decay=ema_decay,
                         save_path=save_path, valid_path=valid_path, compile_model=compile_model)
        self.autoencoder = autoencoder.to(device)
        self.tokenizer = tokenizer
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        self.clip = clip.to(device)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()
        self.autoencoder_std = autoencoder_std

    def preprocessing_data(self, data: torch.Tensor) -> tuple:
        latent = data[0].to(self.device)
        latent = latent / self.autoencoder_std
        tokens = data[1].to(self.device)
        with torch.no_grad(), autocast():
            pool_token, global_token = self.clip.text_encoder.encode_text(tokens)
        return latent, global_token, pool_token

    def compute_loss(self, data) -> torch.Tensor:
        loss = self.model(data[0], data[1], data[2])
        self._validate_loss(loss=loss, optimizer=self.optimizer)
        return loss

    def validate(self, *args, **kwargs) -> None:
        assert "valid_data" in kwargs, "需要在validate显式传入valid_data参数"
        assert "latent_shape" in kwargs, "需要显式指定latent形状"
        assert kwargs["valid_data"] is not None, "valid_data不能为空"
        valid_file = kwargs.get("valid_file_name", "valid")
        valid_file_path = os.path.join(self.valid_path, f"{valid_file}.jpg")
        valid_batch_size = kwargs.get("valid_batch_size", 9)
        valid_data = [kwargs["valid_data"]] * valid_batch_size
        token = self.tokenizer(valid_data)
        token = torch.tensor(token, device=self.device)
        self.model.eval()
        model_name = self.model.__class__.__name__
        if model_name == "Diffusion":
            mode = kwargs.get("mode", "dpm")
        else:
            mode = kwargs.get("mode", "euler")
        back_params = None
        if self.using_ema:
            back_params = self.ema.apply_shadow(self.model)
        with torch.no_grad(), autocast():
            token_pool, token_global = self.clip.text_encoder.encode_text(token)
            latent = self.model.sample(batch_size=valid_batch_size, image_shape=kwargs["latent_shape"], mode=mode,
                                       condition1=token_global, condition2=token_pool)
            latent = latent * self.autoencoder_std
            out = self.autoencoder.latent2image(latent).clamp(min=-1, max=1)
        save_image(tensor=out, normalize=True, padding=1, nrow=int(math.sqrt(valid_batch_size)), fp=valid_file_path)
        if self.using_ema:
            self.ema.restore(model=self.model, backup=back_params)
        self.model.train()
