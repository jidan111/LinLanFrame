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
    "clip-512": "e60c4508462c61947f023f2513e5f701",
    "clip-224": "a542ac33c39d1e148feefdcf8f7288a7",
    "vae_x8": "003e2ba889653274bde0f086f5c318c3",
    "vqvae_x8": "e8884148895e444e6d21e51b3bf51bf5",
    "esr_x2": "5f04fcd6df7e3902f3b5718672e86e89",
    "esr_x4": "168dfc4d736af7bd2289a8c146938cf1",
    "ldm_dit": "848f967b258f1c3120aaca1357944a1a",
    "ldm_unet": "e1973f641a57e04b5e7315ef1999fb7b"
}
