from .import_package import *


def get_load_state_dict_from_compile(file, device="cuda"):
    new_dict = OrderedDict()
    for k, v in torch.load(file, map_location=device).items():
        key = k.replace("_orig_mod.", "")
        new_dict[key] = v
    return new_dict


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_matched_state_dict(model, pretrained_state_dict, strict=False, silent=False):
    """
    加载形状匹配的预训练参数到模型中。
    Args:
        model (torch.nn.Module): 需要加载参数的新模型。
        pretrained_state_dict (dict): 预训练模型的 state_dict。
        strict (bool): 是否在加载未匹配的键时抛出异常（通常设为False）。
        silent (bool): 是否静默模式（不打印跳过的参数信息）。
    """
    model_state_dict = model.state_dict()
    matched_state_dict = {}

    # 遍历预训练参数
    for name, param in pretrained_state_dict.items():
        if name in model_state_dict:
            # 检查形状是否一致
            if param.shape == model_state_dict[name].shape:
                matched_state_dict[name] = param
            else:
                if not silent:
                    print(f"[Skip] Shape mismatch for {name}: "
                          f"pretrained {param.shape} vs current {model_state_dict[name].shape}")
        else:
            if not silent:
                print(f"[Skip] Key '{name}' not found in current model.")
    model.load_state_dict(matched_state_dict, strict=strict)
    return model
