from .attention import *
from .structs import *


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=.1):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layer(x)


class DITBlock(nn.Module):
    def __init__(self, query_dim, condition_dim, key_dim=None, d_model=512, head_num=8, dropout=.1):
        super(DITBlock, self).__init__()
        self.norm1 = nn.LayerNorm(query_dim)
        self.have_key = key_dim is not None
        self.self_attn = SelfAttentionBlock(query_dim=query_dim, d_model=d_model, head_nums=head_num, dropout=dropout)
        if self.have_key:
            self.cross_attn = CrossAttentionBlock(query_dim=query_dim, key_dim=key_dim, d_model=d_model,
                                                  head_nums=head_num, dropout=dropout)
        self.norm2 = nn.LayerNorm(query_dim)
        self.mlp = MLP(in_dim=query_dim, hidden_dim=d_model)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * query_dim)
        )

    def forward(self, x, condition, text=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(condition).chunk(6, dim=1)
        x = self.norm1(x)
        x = (1 + scale_msa.unsqueeze(1)) * x + shift_msa.unsqueeze(1)
        attn = self.self_attn(x)
        if self.have_key:
            attn = self.cross_attn(attn, text)
        x = x + gate_msa.unsqueeze(1) * attn
        x = self.norm2(x)
        x = (1 + scale_mlp.unsqueeze(1)) * x + shift_mlp.unsqueeze(1)
        mlp = self.mlp(x)
        x = x + gate_mlp.unsqueeze(1) * mlp
        return x


class AdaLN(nn.Module):
    def __init__(self, dim, other_dim):
        super(AdaLN, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(other_dim, 2 * dim)
        )

    def forward(self, x, condition):
        x = self.norm(x)
        scale, shift = self.adaLN(condition).chunk(2, dim=1)
        x = (1 + scale.unsqueeze(1)) * x + shift.unsqueeze(1)
        return x


class DIT(ConfigModule):
    def __init__(self, image_shape, patch_size, condition_dim=128, d_model=512, head_nums=8, layer_num=8, key_dim=None,
                 dropout=.1):
        super(DIT, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_num分组"
        assert image_shape[1] % patch_size == 0, "image_size不能被patch分组"
        self.image_shape = image_shape
        self.d_model = d_model
        self.patch_size = patch_size
        seq_length = (image_shape[1] // patch_size) * (image_shape[2] // patch_size)
        self.patch_h, self.patch_w = image_shape[1] // patch_size, image_shape[2] // patch_size
        self.seq_length = seq_length
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=d_model, kernel_size=patch_size,
                              stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(size=(seq_length, d_model)))
        self.transformer = nn.ModuleList([DITBlock(query_dim=d_model, condition_dim=condition_dim, key_dim=key_dim,
                                                   d_model=d_model, head_num=head_nums, dropout=dropout) for i in
                                          range(layer_num)])
        self.adaLN = AdaLN(dim=d_model, other_dim=condition_dim)
        self.out = nn.Linear(d_model, image_shape[0] * patch_size * patch_size)

    def forward(self, x, condition, text=None):
        batch_size, c, h, w = x.shape
        x = self.init(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [batch_size, seq_length, dim]
        x = x + self.pos_embed
        for model in self.transformer:
            x = model(x, condition, text)
        x = self.adaLN(x, condition)
        x = self.out(x)
        x = x.reshape(batch_size, self.patch_h, self.patch_w, self.image_shape[0], self.patch_size,
                      self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(batch_size, c, h, w).contiguous()
        return x
