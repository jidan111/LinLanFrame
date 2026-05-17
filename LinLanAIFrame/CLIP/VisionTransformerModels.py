from .attention import *
from ..structs import *


class VIT(ConfigModule):
    def __init__(self, image_shape=(3, 64, 64), patch_size=16, d_model=512, head_num=8, dropout=.1,
                 pool_type="mean", layer_num=8, hidden_dim=64):
        super(VIT, self).__init__()
        assert pool_type in ["mean", "top"], "只支持mean池化输出和取cls标记输出"
        assert image_shape[1] % patch_size == 0, f"图片大小无法被平均切块{image_shape[1]}%{patch_size}!=0"
        assert d_model % head_num == 0, f"{d_model}无法被均分为{head_num}组, {d_model}%{head_num}!=0"
        self.pool_type = pool_type
        flatten_dim = image_shape[0] * (patch_size ** 2)
        seq_dim = (image_shape[1] // patch_size) * (image_shape[2] // patch_size)
        self.d_model = d_model
        self.image2sequence = nn.Sequential(
            nn.Conv2d(in_channels=image_shape[0], out_channels=flatten_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),  # 后续必须有transpose(1,2)或者permute(0,2,1)，将维度变为(batch_size, seq_dim ,flatten_dim)
        )
        self.init = nn.Sequential(
            nn.LayerNorm(flatten_dim),
            nn.Linear(flatten_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, seq_dim + 1, d_model)))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = TransformerEncoder(query_dim=d_model, d_model=d_model, hidden_dim=hidden_dim, head_num=head_num,
                                              layer_num=layer_num, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.image2sequence(x).transpose(1, 2).contiguous()
        x = self.init(x)
        batch_size, seq_length, d_model = x.shape
        cls_token = self.cls_token.expand(size=(batch_size, 1, d_model))
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding[:, :seq_length + 1]
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1) if self.pool_type == "mean" else x[:, 0]
        x = self.out(x)
        return x

    def image2patches(self, image, patch_size):
        batch_size, channels, high, width = image.shape
        patch_nums_h = high // patch_size
        patch_nums_w = width // patch_size
        patch_nums = patch_nums_h * patch_nums_w
        patch_dim = channels * (patch_size ** 2)
        image = image.reshape(batch_size, channels, patch_nums_h, patch_size, patch_nums_w, patch_size)
        image = image.permute(0, 2, 4, 1, 3, 5)
        image = image.reshape(batch_size, patch_nums, patch_dim)
        return image
