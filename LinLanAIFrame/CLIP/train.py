from ..import_package import *
from .BPEModels import *
from .CLIPModels import *


class CLIPTrainer(Trainer):
    def __init__(self, model: CLIP, lr=1e-4, valid_dir="./valid/", save_model_dir="./model/clip", compile_model=False,
                 mid_save_step=500, file_name="CLIPTrainer"):
        super(CLIPTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                          mid_save_step=mid_save_step, file_name=file_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mid_save_step = mid_save_step
        self.device = device
        if compile_model:
            self.model = torch.compile(model.to(self.device),
                                       mode="default",  # 最大程度优化（训练专用）
                                       dynamic=False,  # 固定输入尺寸关闭动态（更快）
                                       backend="inductor"  # 默认最优后端
                                       )
        else:
            self.model = model.to(self.device)
        self.valid_dir = valid_dir
        self.max_seq_length = model.config["max_seq_length"]
        self.save_model_dir = save_model_dir
        self.scale = GradScaler()
        self.model = model.to(device)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(save_model_dir, exist_ok=True)
        self.mid_cnt = 0
        self.image_acc = 0
        self.text_acc = 0
        self.img_arr = []
        self.text_arr = []

    def loss(self, caption, images):
        loss, loss_text, loss_image = self.model(text=caption, images=images)
        return loss, loss_text, loss_image

    def valid(self, epoch) -> None:
        path = os.path.join(self.valid_dir, f"valid_epoch={epoch}.png")
        fig, axis = plt.subplots(1, 1)
        text = "image2text_acc={:.2%}, text2image_acc={:.2%}".format(self.image_acc, self.text_acc)
        axis.plot(self.img_arr, label="image_loss")
        axis.plot(self.text_arr, label="text_loss")
        axis.set_title(text)
        axis.legend()
        fig.savefig(path)

    def train_one_batch(self, data, index=0):
        caption = data[0].to(self.device)
        images = data[1].to(self.device)
        self.model.train()
        self.opt.zero_grad()
        with autocast():
            loss, text_loss, image_loss = self.loss(caption=caption, images=images)
            self.img_arr.append(image_loss)
            self.text_arr.append(text_loss)
        if torch.isnan(loss).any():
            self.opt.zero_grad()
            raise Exception("训练出现空值，已终止训练")
        self.scale.scale(loss).backward()
        self.scale.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scale.step(self.opt)
        self.scale.update()

    def train_one_epoch(self, data_loader, epoch, epoch_nums):
        for data in tqdm(data_loader, desc=f"{epoch}/{epoch_nums}"):
            self.train_one_batch(data=data)
            if self.mid_cnt != 0 and self.mid_cnt % self.mid_save_step == 0:
                self.image_acc, self.text_acc = self.model.accuracy(text=data[0].to(self.device),
                                                                    image=data[1].to(self.device))
                self.save()
                self.valid(epoch="valid")
                self.mid_cnt = 0
            self.mid_cnt += 1

    def run(self, data_loader, epoch_nums, valid_step=5, valid_batch_size=16, **kwargs) -> None:
        for epoch in range(epoch_nums):
            self.train_one_epoch(data_loader=data_loader, epoch=epoch, epoch_nums=epoch_nums)
            if epoch % valid_step == 0:
                self.save()
                self.valid(epoch=epoch)
                self.img_arr = []
                self.text_arr = []
