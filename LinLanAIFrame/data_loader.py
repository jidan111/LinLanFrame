from .import_package import *
from .CLIP.TokenizerModels import BPE


class H5DataBase(object):
    def __init__(self, file_name, data_shape=(8, 64, 64), dtype=np.uint8):
        self.file_name = file_name
        self.dtype = dtype
        self.compression = "gzip"
        self.data_shape = data_shape
        chunk_params = self.recommend_chunk_size_per_sample()
        self.chunk_size = chunk_params["recommended_chunk_size"]
        if not os.path.exists(file_name):
            self.create_h5()

    def recommend_chunk_size_per_sample(self):
        # 1. 计算数据类型字节数
        dtype_size = {
            "float32": 4,
            "float16": 2,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            np.float32: 4,
            np.float16: 2,
            np.int32: 4,
            np.int16: 2,
            np.uint8: 1
        }.get(self.dtype, 4)
        # 2. 计算单样本总字节数 (KB)
        total_bytes = np.prod(self.data_shape) * dtype_size
        single_sample_kb = total_bytes / 1024
        # 3. 计算理想chunk的样本数范围 (100KB ≤ chunk ≤ 1MB)
        min_chunk_bytes = 100 * 1024
        max_chunk_bytes = 1 * 1024 * 1024
        min_samples = max(1, int(np.ceil(min_chunk_bytes / total_bytes)))
        max_samples = max(1, int(np.floor(max_chunk_bytes / total_bytes)))
        # 4. 计算512KB目标对应的样本数
        target_samples = max(1, int(round(512 * 1024 / total_bytes)))
        recommended = min(max_samples, max(min_samples, target_samples))  # 限制在[min, max]区间
        # 5. 生成解释
        rationale = (
            f"单样本体积={single_sample_kb:.2f}KB | "
            f"100KB需≥{min_samples}样本 | "
            f"1MB需≤{max_samples}样本 | "
            f"推荐{recommended}样本/块（≈{recommended * single_sample_kb:.0f}KB）"
        )
        # 特殊情况处理
        if single_sample_kb > 1024:  # 单样本 >1MB
            rationale = f"单样本体积({single_sample_kb:.2f}KB) > 1MB，强制chunk_size=1（避免碎片化）"
            recommended = 1
        elif min_samples > max_samples:  # 体积矛盾
            rationale = f"单样本体积过大({single_sample_kb:.2f}KB)，chunk_size限制为1"
            recommended = 1

        return {
            "recommended_chunk_size": recommended,
            "single_sample_bytes": single_sample_kb,
            "min_samples_per_chunk": min_samples,
            "max_samples_per_chunk": max_samples,
            "rationale": rationale
        }

    def create_h5(self):
        with h5py.File(self.file_name, "w") as f:
            f.create_dataset(
                "data",
                shape=(0, *self.data_shape),
                maxshape=(None, *self.data_shape),  # 允许动态扩展
                chunks=(self.chunk_size, *self.data_shape),  # 关键：按批量对齐chunks
                dtype=self.dtype,
                compression=self.compression,  # 启用压缩（节省50%+空间）
                compression_opts=6,  # 压缩级别（1-9）
                shuffle=True  # 额外缩体积
            )
            # 初始化有效样本计数器（原子操作的关键）
            f.attrs["cnt"] = 0  # 安全写入起点

    def append_batch(self, data):
        batch_size = data.shape[0]
        if batch_size % self.chunk_size != 0:
            warnings.warn("batch_size必须能被chunk_size整除，否则存储时间与文件体积将会变大")
        assert batch_size > 0, "批次必须大于0"
        assert data.shape == (batch_size, *self.data_shape), f"Latents 形状和预设不同: {data.shape}"
        with h5py.File(self.file_name, "a") as f:  # 标准读写模式
            current_idx = f.attrs["cnt"]
            new_idx = current_idx + batch_size
            try:
                # === 原子操作：一次性扩展并写入 ===
                f["data"].resize(new_idx, axis=0)
                # 严格保持配对关系（HDF5保证写入原子性）
                f["data"][current_idx:new_idx] = data
                # 仅当全部成功才更新计数器
                f.attrs["cnt"] = new_idx
                f.flush()  # 强制落盘
            except Exception as e:
                # 事务回滚：恢复到安全状态
                f["data"].resize(current_idx, axis=0)
                raise RuntimeError(f"写入失败! 指针回滚到安全状态: {current_idx}") from e

    def read_data(self, index, batch_size=None):
        with h5py.File(self.file_name, "r") as f:
            if batch_size is None:
                return f["data"][index]
            else:
                return f["data"][index:index + batch_size]

class H5Caption2LatentDataBase(object):
    """
    # FAT = ...  # dtype is np.int64, shape is [N, ], FAT[caption_index] = latent_index
    # latent = ...  # dtype is np.float16, shape is [N, 16, 64, 64]
    # caption_pool = ...  # dtype is np.float16, shape is [N, 512]
    # caption_global = ...  # dtype is np.float16, shape is [N, 77, 512]
    支持 with H5Caption2LatentDataBase(file_name) as db:
            db operator
    """

    def __init__(self, file_name: str = "H5Caption2LatentDataBase.h5", latent_shape: tuple = (16, 64, 64),
                 caption_pool_shape: tuple = (512,), caption_global_shape: tuple = (77, 512), dtype=np.float16,
                 compression="gzip"):
        self.dtype = dtype
        self.file_name = file_name
        self.compression = compression
        self.latent_shape = latent_shape
        self.caption_pool_shape = caption_pool_shape
        self.caption_global_shape = caption_global_shape
        latent_chunk_params = self.recommend_chunk_size_per_sample(data_shape=latent_shape)
        self.latent_chunk_size = latent_chunk_params["recommended_chunk_size"]
        caption_pool_chunk_params = self.recommend_chunk_size_per_sample(data_shape=caption_pool_shape)
        self.caption_pool_chunk_size = caption_pool_chunk_params["recommended_chunk_size"]
        caption_global_chunk_params = self.recommend_chunk_size_per_sample(data_shape=caption_global_shape)
        self.caption_global_chunk_size = caption_global_chunk_params["recommended_chunk_size"]
        self.write_this = None
        if not os.path.exists(file_name):
            self.__create_h5()
            self.write_this = h5py.File(file_name, "a")
        self.read_this = h5py.File(file_name, "r")

    def close(self):
        self.read_this.close()
        if self.write_this is not None:
            self.write_this.close()

    def __enter__(self):
        return self

    def __del__(self):
        self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def recommend_chunk_size_per_sample(self, data_shape):
        # 1. 计算数据类型字节数
        dtype_size = {
            "float32": 4,
            "float16": 2,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            np.float32: 4,
            np.float16: 2,
            np.int32: 4,
            np.int16: 2,
            np.uint8: 1
        }.get(self.dtype, 4)
        # 2. 计算单样本总字节数 (KB)
        total_bytes = np.prod(data_shape) * dtype_size
        single_sample_kb = total_bytes / 1024
        # 3. 计算理想chunk的样本数范围 (100KB ≤ chunk ≤ 1MB)
        min_chunk_bytes = 100 * 1024
        max_chunk_bytes = 1 * 1024 * 1024
        min_samples = max(1, int(np.ceil(min_chunk_bytes / total_bytes)))
        max_samples = max(1, int(np.floor(max_chunk_bytes / total_bytes)))
        # 4. 计算512KB目标对应的样本数
        target_samples = max(1, int(round(512 * 1024 / total_bytes)))
        recommended = min(max_samples, max(min_samples, target_samples))  # 限制在[min, max]区间
        # 5. 生成解释
        rationale = (
            f"单样本体积={single_sample_kb:.2f}KB | "
            f"100KB需≥{min_samples}样本 | "
            f"1MB需≤{max_samples}样本 | "
            f"推荐{recommended}样本/块（≈{recommended * single_sample_kb:.0f}KB）"
        )
        # 特殊情况处理
        if single_sample_kb > 1024:  # 单样本 >1MB
            rationale = f"单样本体积({single_sample_kb:.2f}KB) > 1MB，强制chunk_size=1（避免碎片化）"
            recommended = 1
        elif min_samples > max_samples:  # 体积矛盾
            rationale = f"单样本体积过大({single_sample_kb:.2f}KB)，chunk_size限制为1"
            recommended = 1
        return {
            "recommended_chunk_size": recommended,
            "single_sample_bytes": single_sample_kb,
            "min_samples_per_chunk": min_samples,
            "max_samples_per_chunk": max_samples,
            "rationale": rationale
        }

    def __create_h5(self):
        with h5py.File(self.file_name, "w") as f:
            f.create_dataset(
                "caption_pool",
                shape=(0, *self.caption_pool_shape),
                maxshape=(None, *self.caption_pool_shape),  # 允许动态扩展
                chunks=(self.caption_pool_chunk_size, *self.caption_pool_shape),  # 关键：按批量对齐chunks
                dtype=self.dtype,
                compression=self.compression,  # 启用压缩（节省50%+空间）
                compression_opts=6,  # 压缩级别（1-9）
                shuffle=True  # 额外缩体积
            )
            f.create_dataset(
                "caption_global",
                shape=(0, *self.caption_global_shape),
                maxshape=(None, *self.caption_global_shape),  # 允许动态扩展
                chunks=(self.caption_global_chunk_size, *self.caption_global_shape),  # 关键：按批量对齐chunks
                dtype=self.dtype,
                compression=self.compression,  # 启用压缩（节省50%+空间）
                compression_opts=6,  # 压缩级别（1-9）
                shuffle=True  # 额外缩体积
            )
            f.create_dataset(
                "latent",
                shape=(0, *self.latent_shape),
                maxshape=(None, *self.latent_shape),  # 允许动态扩展
                chunks=(self.latent_chunk_size, *self.latent_shape),  # 关键：按批量对齐chunks
                dtype=self.dtype,
                compression=self.compression,  # 启用压缩（节省50%+空间）
                compression_opts=6,  # 压缩级别（1-9）
                shuffle=True  # 额外缩体积
            )
            f.create_dataset(
                "FAT",
                shape=(0,),
                maxshape=(None,),  # 允许动态扩展
                chunks=(1000,),  # 关键：按批量对齐chunks
                dtype=np.int64,
                compression=self.compression,  # 启用压缩（节省50%+空间）
                compression_opts=6,  # 压缩级别（1-9）
                shuffle=True  # 额外缩体积
            )
            # 初始化有效样本计数器（原子操作的关键）
            f.attrs["latent_cnt"] = 0  # 安全写入起点
            f.attrs["caption_pool_cnt"] = 0
            f.attrs["caption_global_cnt"] = 0
            f.attrs["FAT_cnt"] = 0

    def append_latent_batch(self, data):
        if self.write_this is None:
            self.write_this = h5py.File(self.file_name, "a")
        batch_size = data.shape[0]
        if batch_size % self.latent_chunk_size != 0:
            warnings.warn("batch_size必须能被chunk_size整除，否则存储时间与文件体积将会变大")
        assert batch_size > 0, "批次必须大于0"
        assert data.shape == (batch_size, *self.latent_shape), f"Latents 形状和预设不同: {data.shape}"
        current_idx = self.write_this.attrs["latent_cnt"]
        new_idx = current_idx + batch_size
        try:
            # === 原子操作：一次性扩展并写入 ===
            self.write_this["latent"].resize(new_idx, axis=0)
            # 严格保持配对关系（HDF5保证写入原子性）
            self.write_this["latent"][current_idx:new_idx] = data
            # 仅当全部成功才更新计数器
            self.write_this.attrs["latent_cnt"] = new_idx
            self.write_this.flush()  # 强制落盘
        except Exception as e:
            # 事务回滚：恢复到安全状态
            self.write_this["latent"].resize(current_idx, axis=0)
            raise RuntimeError(f"Latent write error! seek backward the safe state: {current_idx}") from e

    def append_caption_pool_batch(self, data):
        if self.write_this is None:
            self.write_this = h5py.File(self.file_name, "a")
        batch_size = data.shape[0]
        if batch_size % self.caption_pool_chunk_size != 0:
            warnings.warn("batch_size必须能被chunk_size整除，否则存储时间与文件体积将会变大")
        assert batch_size > 0, "批次必须大于0"
        assert data.shape == (batch_size, *self.caption_pool_shape), f"Latents 形状和预设不同: {data.shape}"
        current_idx = self.write_this.attrs["caption_pool_cnt"]
        new_idx = current_idx + batch_size
        try:
            # === 原子操作：一次性扩展并写入 ===
            self.write_this["caption_pool"].resize(new_idx, axis=0)
            # 严格保持配对关系（HDF5保证写入原子性）
            self.write_this["caption_pool"][current_idx:new_idx] = data
            # 仅当全部成功才更新计数器
            self.write_this.attrs["caption_pool_cnt"] = new_idx
            self.write_this.flush()  # 强制落盘
        except Exception as e:
            # 事务回滚：恢复到安全状态
            self.write_this["caption_pool"].resize(current_idx, axis=0)
            raise RuntimeError(f"Caption write error! seek backward the safe state: {current_idx}") from e

    def append_caption_global_batch(self, data):
        if self.write_this is None:
            self.write_this = h5py.File(self.file_name, "a")
        batch_size = data.shape[0]
        if batch_size % self.caption_global_chunk_size != 0:
            warnings.warn("batch_size必须能被chunk_size整除，否则存储时间与文件体积将会变大")
        assert batch_size > 0, "批次必须大于0"
        assert data.shape == (batch_size, *self.caption_global_shape), f"Latents 形状和预设不同: {data.shape}"
        current_idx = self.write_this.attrs["caption_global_cnt"]
        new_idx = current_idx + batch_size
        try:
            # === 原子操作：一次性扩展并写入 ===
            self.write_this["caption_global"].resize(new_idx, axis=0)
            # 严格保持配对关系（HDF5保证写入原子性）
            self.write_this["caption_global"][current_idx:new_idx] = data
            # 仅当全部成功才更新计数器
            self.write_this.attrs["caption_global_cnt"] = new_idx
            self.write_this.flush()  # 强制落盘
        except Exception as e:
            # 事务回滚：恢复到安全状态
            self.write_this["caption_global"].resize(current_idx, axis=0)
            raise RuntimeError(f"Caption write error! seek backward the safe state: {current_idx}") from e

    def append_fat_batch(self, data):
        if self.write_this is None:
            self.write_this = h5py.File(self.file_name, "a")
        batch_size = data.shape[0]
        assert batch_size > 0, "批次必须大于0"
        current_idx = self.write_this.attrs["FAT_cnt"]
        new_idx = current_idx + batch_size
        try:
            # === 原子操作：一次性扩展并写入 ===
            self.write_this["FAT"].resize(new_idx, axis=0)
            # 严格保持配对关系（HDF5保证写入原子性）
            self.write_this["FAT"][current_idx:new_idx] = data
            # 仅当全部成功才更新计数器
            self.write_this.attrs["FAT_cnt"] = new_idx
            self.write_this.flush()  # 强制落盘
        except Exception as e:
            # 事务回滚：恢复到安全状态
            self.write_this["FAT"].resize(current_idx, axis=0)
            raise RuntimeError(f"FAT write error! seek backward the safe state: {current_idx}") from e

    def read_index(self, index, caption_mode="pool") -> tuple:
        assert self.read_this.attrs["FAT_cnt"] > 0, "The Table of FAT is NULL"
        assert index < self.read_this.attrs[
            "FAT_cnt"], f"index out of the range, {index}>{self.read_this.attrs['FAT_cnt']}"
        assert caption_mode in ["pool", "global"], f"the mode is not supposed of {caption_mode}"
        latent_index = self.read_this["FAT"][index]
        if caption_mode == "pool":
            caption = self.read_this['caption_pool'][index]
        else:
            caption = self.read_this['caption_global'][index]
        return caption, self.read_this["latent"][latent_index]


class H5DataLoader(Dataset):
    def __init__(self, h5_path):
        super(H5DataLoader, self).__init__()
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, "r")
        self.cnt = self.h5_file.attrs["cnt"]

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        # h5_file = h5py.File(self.h5_path, "r")
        return torch.from_numpy(self.h5_file["data"][idx])  # shape (8, 64, 64)

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        if transform is None:
            self.transform = transforms.Compose([
                # transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.img_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(('jpg', 'png', 'jpeg'))
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class SuperResolutionDataset(Dataset):
    def __init__(
            self,
            folder,
            unscale=4,
            crop_size=128,
            norm_mean=(0.5, 0.5, 0.5),
            norm_std=(0.5, 0.5, 0.5),
    ):
        self.hr_paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        self.unscale = unscale
        self.crop_size = crop_size
        self.random_crop = transforms.RandomCrop(crop_size)
        self.resize = transforms.Resize(
            (crop_size // unscale, crop_size // unscale),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.normalize = transforms.Normalize(norm_mean, norm_std)
        self.to_tensor = transforms.ToTensor()
        self.blur = kornia.augmentation.RandomGaussianBlur(
            kernel_size=(7, 7),
            sigma=(0.2, 3.0),
            p=0.8,
        )
        self.jpeg = kornia.augmentation.RandomJPEG(
            jpeg_quality=(30, 95),
            same_on_batch=False,
            p=0.5,
        )

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        hr_img = self.random_crop(hr_img)
        hr = self.to_tensor(hr_img)
        with torch.no_grad():
            lr = hr.unsqueeze(0)
            lr = self.blur(lr)
            lr = self.resize(lr)
            if random.random() < 0.5:
                noise = torch.randn_like(lr) * random.uniform(0.005, 0.01)
                lr = lr + noise
            else:
                poisson_rate = lr * random.uniform(2.0, 5.0)
                poisson_rate = torch.clamp(poisson_rate, min=1e-6)
                poisson_noise = torch.poisson(poisson_rate) / 10.0
                lr = lr + poisson_noise
            lr = torch.clamp(lr, 0.0, 1.0)
            if random.random() < 0.5:
                lr = self.jpeg(lr)
            lr = lr.squeeze(0)
        lr = self.normalize(lr)
        hr = self.normalize(hr)
        return lr, hr


class ClearResolutionDataset(Dataset):
    def __init__(
            self,
            folder,
            crop_size=256,
            norm_mean=(0.5, 0.5, 0.5),
            norm_std=(0.5, 0.5, 0.5),
    ):
        self.hr_paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        self.normalize = transforms.Normalize(norm_mean, norm_std)
        self.to_tensor = transforms.ToTensor()
        self.random_crop = transforms.RandomCrop(crop_size)
        self.blur = kornia.augmentation.RandomGaussianBlur(
            kernel_size=(7, 7),
            sigma=(0.2, 3.0),
            p=0.8,
        )
        self.jpeg = kornia.augmentation.RandomJPEG(
            jpeg_quality=(30, 95),
            same_on_batch=False,
            p=0.5,
        )

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        hr_img = self.random_crop(hr_img)
        hr = self.to_tensor(hr_img)
        with torch.no_grad():
            lr = hr.unsqueeze(0)
            lr = self.blur(lr)
            if random.random() < 0.7:
                noise = torch.randn_like(lr) * random.uniform(0.005, 0.01)
                lr = lr + noise
            else:
                poisson_rate = lr * random.uniform(2.0, 5.0)
                poisson_rate = torch.clamp(poisson_rate, min=1e-6)
                poisson_noise = torch.poisson(poisson_rate) / 10.0
                lr = lr + poisson_noise
            lr = torch.clamp(lr, 0.0, 1.0)
            if random.random() < 0.5:
                lr = self.jpeg(lr)
            lr = lr.squeeze(0)
        lr = self.normalize(lr)
        hr = self.normalize(hr)
        return lr, hr


class BufferedShuffleDataset(IterableDataset):
    """
    # !pip install modelscope==1.38.0
    dataset = MsDataset.load(
        "hjjiao/mini_test2image_dataset_webdataset",
        use_streaming=True,
        split="train",
        cache_dir="/your/dedicated/disk/cache_dir",
        # data_files=data_files
    )
    """
    def __init__(self, ms_dataset, buffer_size):
        super().__init__()
        self.buffer_size = buffer_size
        self.dataset = ms_dataset

    def preprocessing_data(self, item):
        caption_global = torch.from_numpy(np.array(item["cap_global.npy"]))
        caption_pool = torch.from_numpy(np.array(item["cap_pool.npy"]))
        latent = torch.from_numpy(np.array(item["latent.npy"]))
        return latent, caption_global, caption_pool

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        try:
            for _ in range(self.buffer_size):
                buffer.append(self.preprocessing_data(next(iterator)))
        except StopIteration:
            pass
        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            try:
                buffer[idx] = self.preprocessing_data(next(iterator))
            except StopIteration:
                buffer.pop(idx)

# buffered_dataset = BufferedShuffleDataset(ms_dataset=..., buffer_size=8192)
# data_loader = DataLoader(buffered_dataset, batch_size=512, num_workers=4)
# dataset = ImageDataset(folder="你的图片文件夹", transform=transform)
# data_loader = DataLoader(dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True,
#     persistent_workers=True,
#     )
