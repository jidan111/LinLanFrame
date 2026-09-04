from .structs import *
from ..position_embedding import *


class SelfAttentionBlock(nn.Module):
    def __init__(self, query_dim=512, d_model=512, head_nums=8, dropout=.1,
                 rope_max_rotary_freq=10, rope_theta=1000, rope_start_index=0,
                 rope_amplitude_scale=1., have_text=True):
        super(SelfAttentionBlock, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_nums均分"
        self.have_text = have_text
        self.d_k = d_model // head_nums
        self.dropout_p = dropout
        self.d_model = d_model
        self.head_nums = head_nums
        self.QKV = nn.Linear(query_dim, 3 * d_model)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, query_dim),
            nn.Dropout(dropout)
        )
        self.image_rope = RotaryEmbedding(dim=d_model // head_nums//2, max_rotary_freq=rope_max_rotary_freq,
                                          mode="pixel", start_index=rope_start_index,
                                          amplitude_scale=rope_amplitude_scale)
        if self.have_text:
            self.text_rope = RotaryEmbedding(dim=d_model // head_nums, theta=rope_theta, mode="lang",
                                             start_index=rope_start_index, amplitude_scale=rope_amplitude_scale)

    def apply_rotary(self, x, split_index, pixel_shape):
        assert split_index == pixel_shape[0] * pixel_shape[1], "图片token长度必须等于H*W"
        image = x[:, :, :split_index, :]
        image_rotary = self.image_rope(image, pixel_shape)
        if self.have_text:
            text = x[:, :, split_index:, :]
            text_rotary = self.text_rope(text)
            out = torch.cat([image_rotary, text_rotary], dim=-2)
        else:
            out = image_rotary
        return out.contiguous()

    def forward(self, query, split_index, pixel_shape):
        batch_size, query_seq_length, query_dim = query.shape
        qkv = self.QKV(query)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        q = self.apply_rotary(q, split_index=split_index, pixel_shape=pixel_shape)
        k = self.apply_rotary(k, split_index=split_index, pixel_shape=pixel_shape)
        v = v.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        x = x.transpose(1, 2).reshape(batch_size, query_seq_length, self.d_model).contiguous()
        x = self.proj_out(x)
        return x


def modulate(x, shift, scale):
    """自适应调制: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=.1):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        return self.layer(x)


class DiTBlock(nn.Module):
    def __init__(self, query_dim, condition_dim, d_model=512, head_nums=8, dropout=.1, mlp_ratio=4,
                 rope_max_rotary_freq=10, rope_theta=1000, rope_start_index=0,
                 rope_amplitude_scale=1., have_text=False, **kwargs):
        super(DiTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(query_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(query_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttentionBlock(query_dim=query_dim, d_model=d_model, head_nums=head_nums, dropout=dropout,
                                            rope_max_rotary_freq=rope_max_rotary_freq, rope_theta=rope_theta,
                                            rope_start_index=rope_start_index,
                                            rope_amplitude_scale=rope_amplitude_scale, have_text=have_text)
        self.mlp = MLP(in_dim=query_dim, hidden_dim=query_dim * mlp_ratio, dropout=dropout)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * query_dim)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, condition, split_index, pixel_shape):
        """
        :param x:
        :param condition:
        :param split_index:
        :param pixel_shape: [H, W]
        :return:
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(condition).chunk(6, dim=1)
        x_norm1 = self.norm1(x)
        x_norm1 = modulate(x=x_norm1, scale=scale_msa, shift=shift_msa)
        attn_out = self.self_attn(x_norm1, split_index=split_index, pixel_shape=pixel_shape)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x=x_norm2, scale=scale_mlp, shift=shift_mlp)
        mlp = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp
        return x


class DiTWithRope(ConfigModule):
    def __init__(self, in_channels, patch_size=8, condition_dim=128, d_model=384, head_nums=6, layer_num=12,
                 text_dim=None, dropout=.1, mlp_ratio=2, checkpoint_enable=False,
                 rope_max_rotary_freq=10, rope_theta=1000,
                 rope_start_index=0,
                 rope_amplitude_scale=1.):
        super(DiTWithRope, self).__init__()
        self.checkpoint_enable = checkpoint_enable
        self.patch_size = patch_size
        query_dim = in_channels * patch_size * patch_size
        assert query_dim <= d_model, "in_channel*patch*patch > d_model， 模型收敛困难"
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size,
                                   stride=patch_size)
        self.have_text = text_dim is not None
        if self.have_text:
            if text_dim != d_model:
                self.text_init = nn.Linear(text_dim, d_model)
            else:
                self.text_init = nn.Identity()
            if text_dim != condition_dim:
                self.pool_text2condition = nn.Linear(text_dim, condition_dim)
            else:
                self.pool_text2condition = nn.Identity()
        self.transformer = nn.ModuleList([DiTBlock(query_dim=d_model, condition_dim=condition_dim,
                                                   d_model=d_model, head_nums=head_nums, dropout=dropout,
                                                   mlp_ratio=mlp_ratio, rope_max_rotary_freq=rope_max_rotary_freq,
                                                   rope_theta=rope_theta,
                                                   rope_start_index=rope_start_index,
                                                   rope_amplitude_scale=rope_amplitude_scale,
                                                   have_text=self.have_text) for i in
                                          range(layer_num)])
        self.post_condition_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * d_model)
        )
        self.post_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(d_model, in_channels * patch_size * patch_size)
        nn.init.zeros_(self.post_condition_adaLN[-1].weight)
        nn.init.zeros_(self.post_condition_adaLN[-1].bias)

    def __using_checkpoint(self, x, condition, split_index, pixel_shape, model):
        x = model(x, condition, split_index, pixel_shape)
        return x

    def forward(self, x, condition, other_condition1=None, other_condition2=None):
        batch_size, c, h, w = x.shape
        x = self.init_conv(x)
        _, _, ph, pw = x.shape
        split_index = ph * pw
        pixel_shape = [ph, pw]
        x = x.flatten(2).transpose(1, 2).contiguous()
        if self.have_text:
            other_condition2 = self.pool_text2condition(other_condition2)
            condition = condition + other_condition2
            other_condition1 = self.text_init(other_condition1)
            x = torch.cat([x, other_condition1], dim=1)
        for model in self.transformer:
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, condition, split_index, pixel_shape, model,
                    use_reentrant=False
                )
            else:
                x = model(x, condition, split_index, pixel_shape)
        post_scale, post_shift = self.post_condition_adaLN(condition).chunk(2, dim=1)
        x = self.post_norm(x)
        x = modulate(x, scale=post_scale, shift=post_shift)
        x = self.out(x)
        x = x[:, :split_index, :]
        x = x.reshape(batch_size, ph, pw, c, self.patch_size,
                      self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(batch_size, c, h, w).contiguous()
        return x
