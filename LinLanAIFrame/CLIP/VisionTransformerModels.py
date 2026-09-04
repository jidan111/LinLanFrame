from ..attention import *
from ..structs import *


class VIT(ConfigModule):
    def __init__(self, image_shape=(3, 64, 64), patch_size=16, d_model=512, head_nums=8, dropout=.1,
                 layer_num=8, mlp_ratio=4, out_dim=512, checkpoint_enable=True):
        super(VIT, self).__init__()
        assert image_shape[1] % patch_size == 0, f"图片大小无法被平均切块{image_shape[1]}%{patch_size}!=0"
        assert d_model % head_nums == 0, f"{d_model}无法被均分为{head_nums}组, {d_model}%{head_nums}!=0"
        self.patch_size = patch_size
        self.patch_h, self.patch_w = image_shape[1] // patch_size, image_shape[2] // patch_size
        self.seq_length = self.patch_h * self.patch_w
        self.query_dim = image_shape[0] * patch_size * patch_size
        self.d_model = d_model
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=d_model, kernel_size=patch_size,
                              stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(size=(1, self.seq_length + 1, d_model)))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = TransformerEncoder(query_dim=d_model, d_model=d_model, head_nums=head_nums,
                                              layer_num=layer_num, dropout=dropout, mlp_ratio=mlp_ratio,
                                              checkpoint_enable=checkpoint_enable)
        self.norm_post = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x = self.init(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        cls_token = self.cls_token.expand(size=(batch_size, 1, self.d_model))
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm_post(x)
        x = x[:, 0, :]
        x = self.out(x)
        return x

    @torch.no_grad()
    def encode_image(self, x):
        x = self(x)
        return x
