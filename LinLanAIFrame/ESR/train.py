from ..structs import *
from ..import_package import *
from .losses import *


class ESRTrainer(Trainer):
    """
    data_loader_in:[low_r, high_r]
    """

    def __init__(self, model: ConfigModule, lr: float = 1e-6, valid_dir="valid", save_model_dir="model",
                 mid_save_step=500, have_perception: bool = False, perception_weight=1., perception_net="alex",
                 compile_model=False, valid_batch_size=4):
        super(ESRTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                         mid_save_step=mid_save_step, file_name=self.__class__.__name__)
        assert lr < 5e-5, "使用的损失函数是sun，学习率建议小于5e-5，否则会产生NAN"
        self.loss_func = ESRLoss(have_perception=have_perception, perception_weight=perception_weight,
                                 perception_net=perception_net).to(self.device)
        if compile_model:
            self.model = torch.compile(model.to(self.device),
                                       mode="default",  # 最大程度优化（训练专用）
                                       dynamic=False,  # 固定输入尺寸关闭动态（更快）
                                       backend="inductor"  # 默认最优后端
                                       )
        else:
            self.model = model.to(self.device)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.valid_image = None
        self.valid_true_image = None
        self.valid_batch_size = valid_batch_size

    def get_model_config(self):
        return {self.model.__class__.__name__: self.model.config}

    def loss(self, data):
        low_r = data[0].to(self.device)
        self.valid_image = low_r[:self.valid_batch_size]
        high_r = data[1].to(self.device)
        self.valid_true_image = high_r[:self.valid_batch_size]
        out = self.model(low_r)
        loss = self.loss_func(fake_sample=out, true_sample=high_r)
        return loss

    def train_one_batch(self, data, index=0):
        self.model.train()
        self.opt.zero_grad()
        with autocast():
            loss = self.loss(data)
        if torch.isnan(loss).any():
            self.opt.zero_grad()
            raise Exception("训练出现空值，已终止训练")
        self.scale.scale(loss).backward()
        self.scale.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scale.step(self.opt)
        self.scale.update()

    def valid(self, valid_batch_size=4, file_name="valid", **kwargs):
        self.model.eval()
        row = int(math.sqrt(self.valid_batch_size))
        with torch.no_grad():
            out = self.model(self.valid_image)
        save_image(out, nrow=row, fp=os.path.join(self.valid_dir, f"esr_{file_name}.png"), normalize=True, padding=1)
        save_image(self.valid_true_image, nrow=row, fp=os.path.join(self.valid_dir, f"true_{file_name}.png"),
                   normalize=True,
                   padding=1)
        self.model.train()


class ESRGANTrainer(Trainer):
    """
    data_loader_in:[low_r, high_r]
    """

    def __init__(self, model: ConfigModule, discriminator: ConfigModule, gen_lr=5e-6, dis_lr=5e-5, n_critic=2,
                 valid_dir="./valid", save_model_dir="./model/esrgan_trainer", perception_net="vgg",
                 perception_weight=1., valid_batch_size=9, have_perception=True, mid_save_step=500, dis_start=5001):
        super(ESRGANTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                            mid_save_step=mid_save_step, file_name=self.__class__.__name__)
        assert gen_lr < 5e-5, "使用的损失函数是sun，学习率建议小于5e-5，否则会产生NAN"
        self.n_critic = n_critic
        self.dis_start = dis_start
        self.dis_cnt = 0
        self.dis_flag = False
        self.discriminator = discriminator.to(self.device)
        self.model = model.to(self.device)
        self.gen_opt = torch.optim.AdamW(params=self.model.parameters(), lr=gen_lr)
        self.dis_opt = torch.optim.AdamW(params=self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.99))
        self.gan_loss_func = GAN_HingeLoss()
        self.rec_loss_func = ESRLoss(have_perception=have_perception, perception_weight=perception_weight,
                                     perception_net=perception_net).to(self.device)
        self.valid_image = None
        self.valid_true_image = None
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

    def loss(self, fake_sample, true_sample, mode="gen"):
        if mode == "ordinary":
            rec_loss = self.rec_loss_func(true_sample=true_sample, fake_sample=fake_sample)
            return rec_loss
        elif mode == "gen":
            rec_loss = self.rec_loss_func(true_sample=true_sample, fake_sample=fake_sample)
            g_loss = -self.discriminator(fake_sample).mean()
            g_weight = self.calculate_adaptive_weight(a_loss=rec_loss, b_loss=g_loss,
                                                      model_last_layer=self.model.get_last_layer_weight())
            loss = rec_loss + g_weight * g_loss
            return loss
        else:
            true_sample = self.discriminator(true_sample)
            fake_sample = self.discriminator(fake_sample.detach())
            loss = self.gan_loss_func(fake_sample=fake_sample, true_sample=true_sample)
            return loss

    def valid(self, valid_batch_size=4, file_name="valid", **kwargs):
        self.model.eval()
        row = int(math.sqrt(self.valid_batch_size))
        with torch.no_grad():
            out = self.model(self.valid_image)
        save_image(out, nrow=row, fp=os.path.join(self.valid_dir, f"esr_{file_name}.png"), normalize=True, padding=1)
        save_image(self.valid_true_image, nrow=row, fp=os.path.join(self.valid_dir, f"true_{file_name}.png"),
                   normalize=True,
                   padding=1)
        self.model.train()

    def train_one_batch(self, data, index=0):
        low_r = data[0].to(self.device)
        true_sample = data[1].to(self.device)
        self.valid_image = low_r[:self.valid_batch_size]
        self.valid_true_image = true_sample[:self.valid_batch_size]
        fake_sample = self.model(low_r)
        if not self.dis_flag:
            self.model.train()
            self.gen_opt.zero_grad()
            with autocast():
                loss = self.loss(true_sample=true_sample, fake_sample=fake_sample, mode="ordinary")
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
                self.model.train()
                self.gen_opt.zero_grad()
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
