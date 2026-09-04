from .import_package import *

def config2md5(the_config: dict):
    the_config["time"] = time.time()
    string_ = str(the_config)
    return hashlib.md5(string_.encode("utf-8")).hexdigest()


pretrain_models_id = {
    "clip-224": "a542ac33c39d1e148feefdcf8f7288a7",
    "vae-x8": "003e2ba889653274bde0f086f5c318c3",
    "vqvae-x8": "e8884148895e444e6d21e51b3bf51bf5",
    "esr-x2": "5f04fcd6df7e3902f3b5718672e86e89",
    "esr-x4": "168dfc4d736af7bd2289a8c146938cf1",
    "dit_rope": "06cacca83145b0812df400501cd96f57",
    "dit": "47086c62189969620a8e7690b757ce8f"
}


def get_model_path(model_name, download_path, huggingface_repo_id="hjjiao/LinLanFrame",
                   huggingface_file_base_path_in_repo="pretrain/"):
    if model_name in pretrain_models_id:
        model_id = pretrain_models_id[model_name]
        state_dict_id = f"state_dict/{model_id}.pth"
        state_dict_path = down_model_from_huggingface(file_name=state_dict_id, download_path=download_path,
                                                      repo_id=huggingface_repo_id,
                                                      file_base_path_in_repo=huggingface_file_base_path_in_repo)
        config_id = f"config/{model_id}.json"
        config_path = down_model_from_huggingface(file_name=config_id, download_path=download_path,
                                                  repo_id=huggingface_repo_id,
                                                  file_base_path_in_repo=huggingface_file_base_path_in_repo)
        return model_id, config_path, state_dict_path
    else:
        print("没有改预训练模型:", model_name)
        return "", "", ""


def down_model_from_huggingface(file_name=r"state_dict/AutoEncoder_x8_mean_std.json",
                                repo_id="hjjiao/LinLanFrame",
                                file_base_path_in_repo="pretrain/",
                                download_path=os.path.dirname(os.path.abspath(__file__))):
    assert file_name[0] != "/", "file_name不能以/开头"
    if file_base_path_in_repo[-1] == "/":
        file_path_in_repo = f"{file_base_path_in_repo}{file_name}"
    else:
        file_path_in_repo = f"{file_base_path_in_repo}/{file_name}"
    target_path_local = os.path.join(download_path, file_path_in_repo)
    if os.path.exists(target_path_local):
        print("文件已经存在, File Exists:", target_path_local)
        return target_path_local
    else:
        try:
            print("Model Download Begin:", target_path_local)
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=file_path_in_repo,
                local_dir=download_path,
            )
            print("文件下载完成，本地完整路径 Download successful:", local_file)
            return local_file
        except httpx.ConnectError as e:
            print(f"请关闭代理服务，或手动下载模型，模型地址 VPN Error:{file_path_in_repo}")
            return "-1"
