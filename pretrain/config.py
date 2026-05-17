import hashlib
import time
import os

curr_path = os.path.dirname(os.path.abspath(__file__))
pretrain_config_dir = os.path.join(curr_path, "./config")
pretrain_state_dict_dir = os.path.join(curr_path, "./state_dict")


def config2md5(the_config: dict):
    the_config["time"] = time.time()
    string = str(the_config)
    return hashlib.md5(string.encode("utf-8")).hexdigest()


pretrain_models_id = {
    "clip": "0b379874add9a8384fc55d128ab03867",
    "vae_large": "bff5e583bb5c0dbf038ef2765e4000a0",
    "vqvae_large": "50b0d715bd23fb2fb435612a00bf7c18",
    "esr_x2": "5f04fcd6df7e3902f3b5718672e86e89",
    "esr_x4": "168dfc4d736af7bd2289a8c146938cf1",
    "anime_gan": "0846d25a77e8b90edfc9db8ed1db4c1b",
    "unet_clean": "4212c52f1c86f57ef52569beee36b632"
}
