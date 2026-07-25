from .ImageGenerator import *
from .CLIP import *
from .text2image import *
from .ESR import *
from .utils import *

"""
云环境 !pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
kagglehub环境：!pip install kagglehub==1.0.1 kagglesdk==0.1.23
"""
__author__ = "lin lan"
__email__ = "2339654498@qq.com"


def set_deterministic_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时也生效
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn_only避免报错
    os.environ["PYTHONHASHSEED"] = str(seed)


# set_deterministic_seeds(seed=2001)  # 确保能够复现，在drop和Norm高频使用时，防止有误差


def init_model_by_params(config: dict):
    """
    :param config: {class_name:{a:1, b:2, c:3}}
    :return:
    """
    model_name = list(config.keys())[0]
    model_params = config[model_name]
    try:
        obj = globals()[model_name]
        return obj(**model_params)
    except KeyError:
        print(model_name, "未进行预设")
        return -1


def init_model_by_params_pretrain(config: dict, state_dict_path: str, strict=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = init_model_by_params(config)
    model.load_state_dict(get_load_state_dict_from_compile(file=state_dict_path, device=device), strict=strict)
    return model


def init_diffusion_by_params_pretrain(config, state_dict_path, strict=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model_by_params(config["model"])
    diffusion = Diffusion(model=model, **config["Diffusion"])
    diffusion.load_state_dict(get_load_state_dict_from_compile(file=state_dict_path, device=device), strict=strict)
    return diffusion
