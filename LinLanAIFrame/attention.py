from .import_package import *
from .position_embedding import RotaryEmbedding


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    """输入类型:[batch_size, head_num, seq_length, d_k]，且d_model=head_num*d_k"""
    assert key.shape[2] == value.shape[2], f"key和value的维度不同{key.shape[2]}!={value.shape[2]}"
    b, l, s = query.size(0), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if len(query.shape) == 3:
        attn_bias = torch.zeros(b, l, s, dtype=query.dtype)  # 输入为(batch, query_seq, d_model)
    else:
        attn_bias = torch.zeros(b, 1, l, s, dtype=query.dtype)  # 输入为(batch, head_num, query_seq, d_model)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(l, s, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        # assert attn_mask.shape[-2:] == (l, s), "掩码形状错误,应为(q_seq_length, k_seq_length)"
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class SelfAttentionBlock(nn.Module):
    def __init__(self, query_dim=512, d_model=512, head_nums=8, dropout=.1, residual_out=True, query_norm=True):
        super(SelfAttentionBlock, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_nums均分"
        self.d_k = d_model // head_nums
        self.residual_out = residual_out
        self.dropout_p = dropout
        self.d_model = d_model
        self.head_nums = head_nums
        self.QKV = nn.Linear(query_dim, 3 * d_model)
        self.have_query_orm = query_norm
        if self.have_query_orm:
            self.norm = nn.LayerNorm(query_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, mask=None):
        batch_size, query_seq_length, query_dim = query.shape
        h_ = query
        if self.have_query_orm:
            query = self.norm(query)
        qkv = self.QKV(query)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        x = x.transpose(1, 2).reshape(batch_size, query_seq_length, self.d_model).contiguous()
        x = self.proj_out(x)
        if self.residual_out:
            return h_ + x
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim=512, key_dim=512, d_model=512, head_nums=8, dropout=.1, residual_out=True,
                 query_norm=True):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_nums均分"
        self.d_k = d_model // head_nums
        self.d_model = d_model
        self.residual_out = residual_out
        self.head_nums = head_nums
        self.dropout_p = dropout
        self.Q = nn.Linear(query_dim, d_model)
        self.K = nn.Linear(key_dim, d_model)
        self.V = nn.Linear(key_dim, d_model)
        self.have_query_norm = query_norm
        if self.have_query_norm:
            self.q_norm = nn.LayerNorm(query_dim)
            self.k_norm = nn.LayerNorm(key_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value=None, mask=None):
        value = key if value is None else value
        batch_size, query_seq_length, query_dim = query.shape
        key_seq_length = key.shape[1]
        h_ = query
        if self.have_query_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
            value = self.k_norm(value)
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)
        q = q.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, key_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, key_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p)
        x = x.transpose(1, 2).reshape(batch_size, query_seq_length, self.d_model).contiguous()
        x = self.proj_out(x)
        if self.residual_out:
            return h_ + x
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=.1):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        h_ = x
        return h_ + self.layer(x)


class TransformerEncoder(nn.Module):
    """
    做CLIP或者LLM，开启梯度检查点，能将batch_size提升7到8倍
    """

    def __init__(self, query_dim, d_model, head_nums, layer_num, dropout, mlp_ratio=4, checkpoint_enable=False):
        super(TransformerEncoder, self).__init__()
        self.checkpoint_enable = checkpoint_enable
        self.layer = nn.ModuleList([])
        for num in range(layer_num):
            self.layer.append(
                nn.ModuleList(
                    [SelfAttentionBlock(query_dim=query_dim, d_model=d_model, head_nums=head_nums, dropout=dropout),
                     FeedForward(in_dim=query_dim, hidden_dim=query_dim * mlp_ratio, dropout=dropout)]))

    def __using_checkpoint(self, x, mask, atte, ffn):
        x = atte(x, mask)
        x = ffn(x)
        return x

    def forward(self, query, mask=None):
        for atte, ffn in self.layer:
            if self.checkpoint_enable:
                query = checkpoint(
                    self.__using_checkpoint,
                    query, mask, atte, ffn,
                    use_reentrant=False
                )
            else:
                query = atte(query, mask=mask)
                query = ffn(query)
        return query


class ImageSelfAttentionBlock(nn.Module):
    def __init__(self, channels, head_num, dropout=.1, query_norm=True, residual_out=True):
        super(ImageSelfAttentionBlock, self).__init__()
        assert channels % head_num == 0, "通道数需要支持多头拆分"
        self.residual_out = residual_out
        self.have_query_norm = query_norm
        if self.have_query_norm:
            self.norm = nn.LayerNorm(channels * 3)
        self.dropout_p = dropout
        self.channels = channels
        self.head_num = head_num
        self.d_k = channels // head_num
        self.QKV = nn.Conv2d(in_channels=channels, out_channels=channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, query):
        b, c, h, w = query.shape
        qkv = self.QKV(query)
        qkv = qkv.flatten(2).transpose(1, 2)
        if self.have_query_norm:
            qkv = self.norm(qkv)
        q, k, v = qkv.chunk(3, dim=2)
        b, seq_len, c = q.shape
        q = q.reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        out = out.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.residual_out:
            return query + self.proj_out(out)
        return self.proj_out(out)


class ImageCrossAttentionBlock(nn.Module):
    def __init__(self, channels, d_model, head_num, key_dim=None, dropout=.1, query_norm=True, residual_out=True):
        super(ImageCrossAttentionBlock, self).__init__()
        assert d_model % head_num == 0, "d_model不能被分组"
        self.d_model = d_model
        self.dropout_p = dropout
        self.residual_out = residual_out
        self.have_query_norm = query_norm
        if self.have_query_norm:
            self.q_norm = nn.LayerNorm(d_model)
            self.k_norm = nn.LayerNorm(key_dim)
            self.v_norm = nn.LayerNorm(key_dim)
        self.head_num = head_num
        self.d_k = d_model // head_num
        self.Q = nn.Conv2d(in_channels=channels, out_channels=d_model, kernel_size=1)
        self.K = nn.Linear(key_dim, d_model)
        self.V = nn.Linear(key_dim, d_model)
        self.proj_out = nn.Conv2d(in_channels=d_model, out_channels=channels, kernel_size=1)

    def forward(self, query, key, value=None):
        value = key if value is None else value
        b, c, h, w = query.shape
        key_seq_length = key.shape[1]
        q = self.Q(query)
        q = q.flatten(2).transpose(1, 2)
        if self.have_query_norm:
            q = self.q_norm(q)
            key = self.k_norm(key)
            value = self.v_norm(value)
        k = self.K(key)
        v = self.V(value)
        b, seq_len, c = q.shape
        q = q.reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(b, key_seq_length, self.head_num, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(b, key_seq_length, self.head_num, self.d_k).transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        out = out.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.residual_out:
            return query + self.proj_out(out)
        return self.proj_out(out)
