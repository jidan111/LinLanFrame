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
        self.config = {
            k: local_vars[k] for k in params if k in local_vars and isinstance(local_vars[k], support_types)
        }

    def save_config(self, file_name: str = None):
        with open(file_name, "w") as f:
            json.dump({self.__class__.__name__: self.config}, f)


class Trainer(nn.Module):
    """
    单个模型需要重写loss, save, valid方法
    多个模型需要重写train_one_batch方法
    """

    def __init__(self, valid_dir="valid", save_model_dir="model", mid_save_step=500, file_name="trainer"):
        super(Trainer, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        subclass = self.__class__
        sig = inspect.signature(subclass.__init__)
        frame = inspect.currentframe().f_back  # 获取调用栈（子类 __init__）
        support_types = (str, int, float, list, dict, tuple, bool)
        local_vars = frame.f_locals
        params = list(sig.parameters.keys())[1:]
        self.config = {
            k: local_vars[k] for k in params if k in local_vars and isinstance(local_vars[k], support_types)
        }
        self.mid_save_step = mid_save_step
        self.valid_dir = valid_dir
        self.save_model_dir = save_model_dir
        self.scale = GradScaler()
        self.device = device
        self.model = None
        self.opt = None
        self.mid_cnt = 0
        self.file_name = file_name
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(save_model_dir, exist_ok=True)

    def get_model_config(self) -> list:
        return ["该模块未实现模型参数记录"]

    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("需要手动实现损失计算")

    def save(self) -> None:
        config = dict()
        config[self.__class__.__name__] = self.config
        state = self.state_dict()
        models_config = self.get_model_config()
        config["models"] = models_config
        config_path = os.path.join(self.save_model_dir, f"{self.file_name}.json")
        dict_path = os.path.join(self.save_model_dir, f"{self.file_name}.pth")
        with open(config_path, "w") as f:
            json.dump(config, f)
        torch.save(state, dict_path)
        result_config, result_state_dict = self.get_target_params_state_dict()
        result_config_path = os.path.join(self.save_model_dir, f"{self.file_name}_result.json")
        result_dict_path = os.path.join(self.save_model_dir, f"{self.file_name}_result.pth")
        with open(result_config_path, "w") as f:
            json.dump(result_config, f)
        torch.save(result_state_dict, result_dict_path)

    def valid(self, *args, **kwargs) -> None:
        raise NotImplementedError("需要手动实现验证代码")

    def get_target_params_state_dict(self):
        return {self.model.__class__.__name__: self.model.config}, self.model.state_dict()

    def train_one_batch(self, data, index=0):
        self.model.train()
        self.opt.zero_grad()
        data = data.to(self.device)
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

    def train_one_epoch(self, data_loader, epoch, epoch_nums):
        index = 0
        for data in tqdm(data_loader, desc=f"{epoch}/{epoch_nums}"):
            self.train_one_batch(data=data, index=index)
            if self.mid_cnt != 0 and self.mid_cnt % self.mid_save_step == 0:
                self.save()
                self.valid()
                self.mid_cnt = 0
            index += 1
            self.mid_cnt += 1

    def run(self, data_loader, epoch_nums, valid_step=5, valid_batch_size=16) -> None:
        for epoch in range(epoch_nums):
            self.train_one_epoch(data_loader=data_loader, epoch=epoch, epoch_nums=epoch_nums)
            if epoch % valid_step == 0:
                self.save()
                self.valid(valid_batch_size=valid_batch_size, epoch=epoch, file_name=epoch)


class OrdinaryTrainer(Trainer):
    def __init__(self, model: ConfigModule, loss_func: nn.Module, lr: float = 1e-4, valid_dir="./valid",
                 save_model_dir="。/model",
                 mid_save_step=500, compile_model=False):
        super(OrdinaryTrainer, self).__init__(valid_dir=valid_dir, save_model_dir=save_model_dir,
                                              mid_save_step=mid_save_step, file_name=self.__class__.__name__)
        self.loss_func = loss_func.to(self.device)
        if compile_model:
            self.model = torch.compile(model.to(self.device),
                                       mode="default",  # 最大程度优化（训练专用）
                                       dynamic=False,  # 固定输入尺寸关闭动态（更快）
                                       backend="inductor"  # 默认最优后端
                                       )
        else:
            self.model = model.to(self.device)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=lr)

    def get_model_config(self):
        return {self.model.__class__.__name__: self.model.config}

    def loss(self, data):
        out = self.model(data)
        loss = self.loss_func(fake_sample=out, true_sample=data)
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
        ...
