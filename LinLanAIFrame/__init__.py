from .ImageGenerator import *
from .CLIP import *
from .import_package import *
from .config_example import *
from .text2image import *
from .ImageGenerator.functions import *
from .ESR import *
"""
云环境 !pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
"""
__author__ = "lin lan"
__email__ = "2339654498@qq.com"


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


def init_model_by_params_pretrain(config: dict, state_dict_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model_by_params(config)
    model.load_state_dict(get_load_state_dict_from_compile(file=state_dict_path, device=device))
    return model


def init_trainer_by_params_pretrain(config: dict, state_dict_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = list(config.items())
    trainer_name = items[0][0]
    trainer_params = items[0][1]
    models = []
    for name, params in list(items[1][1].items()):
        models.append(init_model_by_params({name: params}))
    models[1].apply(add_sn)
    try:
        obj = globals()[trainer_name]
        model = obj(*models, **trainer_params)
    except KeyError:
        print(trainer_name, "未进行预设")
        return -1
    model.load_state_dict(get_load_state_dict_from_compile(file=state_dict_path, device=device))
    return model
