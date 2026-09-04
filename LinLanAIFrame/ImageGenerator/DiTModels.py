from .structs import *


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
    def __init__(self, query_dim, condition_dim, d_model=512, head_nums=8, dropout=.1, mlp_ratio=4):
        super(DiTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(query_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(query_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttentionBlock(query_dim=query_dim, d_model=d_model, head_nums=head_nums, dropout=dropout,
                                            residual_out=False, query_norm=False)
        self.mlp = MLP(in_dim=query_dim, hidden_dim=query_dim * mlp_ratio, dropout=dropout)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * query_dim)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, condition):
        """
        :param x: [b,seq_len,dim]
        :param condition: [b,dim]
        :return:
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(condition).chunk(6, dim=1)
        x_norm1 = self.norm1(x)
        x_norm1 = modulate(x=x_norm1, scale=scale_msa, shift=shift_msa)
        attn_out = self.self_attn(x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x=x_norm2, scale=scale_mlp, shift=shift_mlp)
        mlp = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp
        return x


class DiT(ConfigModule):
    def __init__(self, image_shape, patch_size=8, condition_dim=128, d_model=384, head_nums=6, layer_num=12,
                 key_dim=None,
                 dropout=.1, mlp_ratio=2, key_seq_length=None, double_condition_control=True, checkpoint_enable=False,
                 **kwargs):
        super(DiT, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_num分组"
        assert image_shape[1] % patch_size == 0 and image_shape[2] % patch_size == 0, "image_size不能被patch分组"
        self.checkpoint_enable = checkpoint_enable
        self.key_dim = key_dim
        self.double_condition_control = double_condition_control
        self.patch_size = patch_size
        self.patch_h, self.patch_w = image_shape[1] // patch_size, image_shape[2] // patch_size
        self.seq_length = self.patch_h * self.patch_w
        self.query_dim = image_shape[0] * patch_size * patch_size
        assert self.query_dim < d_model, f"序列维度大于d_model,属于降维，模型大概率无法收敛，请修改patch_size或d_model,{self.query_dim}>={d_model}"
        if self.key_dim is not None:
            if key_dim != d_model:
                self.key_init = nn.Linear(key_dim, d_model)
            else:
                self.key_init = nn.Identity()
            if self.double_condition_control:
                self.pool_key2condition = nn.Linear(key_dim, condition_dim)
        self.key_seq_length = key_seq_length
        key_seq_len = 0
        if key_dim is not None:
            key_seq_len = 1 if key_seq_length is None else key_seq_length
        self.pos_embed = nn.Parameter(torch.randn(size=(1, self.seq_length + key_seq_len, d_model)))
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=d_model, kernel_size=patch_size,
                              stride=patch_size)
        self.transformer = nn.ModuleList([DiTBlock(query_dim=d_model, condition_dim=condition_dim,
                                                   d_model=d_model, head_nums=head_nums, dropout=dropout,
                                                   mlp_ratio=mlp_ratio) for i in
                                          range(layer_num)])
        self.post_condition_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * d_model)
        )
        self.post_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(d_model, self.query_dim)
        nn.init.zeros_(self.post_condition_adaLN[-1].weight)
        nn.init.zeros_(self.post_condition_adaLN[-1].bias)

    def __using_checkpoint(self, x, condition, model):
        x = model(x, condition)
        return x

    def forward(self, x, condition, other_condition1=None, other_condition2=None, **kwargs):
        """
        :param x: [b,c,h,w] it is data
        :param condition: [b,dim] it is time_steps
        :param other_condition1: [b,seq_len,dim] it is global caption embedding
        :param other_condition2: [b,dim] it is pool caption embedding
        :return:
        """
        batch_size, c, h, w = x.shape
        x = self.init(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        if self.key_dim is not None:
            assert other_condition1.shape[
                       1] == self.key_seq_length, \
                f"other_condition1输出条件维度{other_condition1.shape}与预设维度不同(-1,{self.key_seq_length},-1)"
            if other_condition1.dim() == 2:
                other_condition1 = other_condition1.unsqueeze(1)
            if self.double_condition_control:
                assert other_condition2.dim() == 2, \
                    f"other_condition2是池化特征应为(-1,{self.key_dim})，输入维度不符{other_condition2.shape}"
                other_condition2 = self.pool_key2condition(other_condition2)
                condition = condition + other_condition2
            text = self.key_init(other_condition1)
            x = torch.cat((x, text), dim=1)
        x = x + self.pos_embed
        for model in self.transformer:
            if self.checkpoint_enable:
                x = checkpoint(
                    self.__using_checkpoint,
                    x, condition, model,
                    use_reentrant=False
                )
            else:
                x = model(x, condition)
        post_scale, post_shift = self.post_condition_adaLN(condition).chunk(2, dim=1)
        x = self.post_norm(x)
        x = modulate(x, scale=post_scale, shift=post_shift)
        x = self.out(x)
        if self.key_dim is not None:
            x = x[:, :self.seq_length, :]
        x = x.reshape(batch_size, self.patch_h, self.patch_w, c, self.patch_size,
                      self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(batch_size, c, h, w).contiguous()
        return x
