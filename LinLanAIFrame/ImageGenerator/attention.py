from .functions import *


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    """输入类型:[batch_size, head_num, seq_length, d_k]，且d_model=head_num*d_k"""
    assert key.shape[2] == value.shape[2], f"key和value的维度不同{key.shape[2]}!={value.shape[2]}"
    l, s = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(l, s, dtype=query.dtype)
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
    def __init__(self, query_dim=512, d_model=512, head_nums=8, dropout=.1):
        super(SelfAttentionBlock, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_nums均分"
        self.d_k = d_model // head_nums
        self.d_model = d_model
        self.head_nums = head_nums
        self.QKV = nn.Linear(query_dim, 3 * d_model)
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)
        self.v_norm = nn.LayerNorm(self.d_k)
        self.attn_dropout = nn.Dropout()
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query):
        batch_size, query_seq_length, query_dim = query.shape
        qkv = self.QKV(query)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.attn_dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, query_seq_length, self.d_model).contiguous()
        x = self.norm(x)
        x = self.proj_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=.1):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)


class TransformerEncoder(nn.Module):
    def __init__(self, query_dim, d_model, head_num, layer_num, dropout, hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList([])
        for num in range(layer_num):
            self.layer.append(
                nn.ModuleList(
                    [SelfAttentionBlock(query_dim=query_dim, d_model=d_model, head_nums=head_num, dropout=dropout),
                     FeedForward(in_dim=query_dim, hidden_dim=hidden_dim, dropout=dropout)]))

    def forward(self, query):
        for atte, ffn in self.layer:
            query = atte(query) + query
            query = ffn(query) + query
        return query


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim=512, key_dim=512, d_model=512, head_nums=8, dropout=.1):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % head_nums == 0, "d_model不能被head_nums均分"
        self.d_k = d_model // head_nums
        self.d_model = d_model
        self.head_nums = head_nums
        self.Q = nn.Linear(query_dim, d_model)
        self.K = nn.Linear(key_dim, d_model)
        self.V = nn.Linear(key_dim, d_model)
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)
        self.v_norm = nn.LayerNorm(self.d_k)
        self.attn_dropout = nn.Dropout()
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value=None):
        value = key if value is None else value
        batch_size, query_seq_length, query_dim = query.shape
        key_seq_length = key.shape[1]
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)
        q = q.reshape(batch_size, query_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, key_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, key_seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.attn_dropout(x)
        x = x.transpose(1, 2).reshape(batch_size, query_seq_length, self.d_model).contiguous()
        x = self.norm(x)
        x = self.proj_out(x)
        return x


class ImageSelfAttentionBlock(nn.Module):
    def __init__(self, channels, head_num):
        super(ImageSelfAttentionBlock, self).__init__()
        assert channels % head_num == 0, "通道数需要支持多头拆分"
        self.norm1 = channels_get_norms(channels)
        self.channels = channels
        self.head_num = head_num
        self.d_k = channels // head_num
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)
        self.v_norm = nn.LayerNorm(self.d_k)
        self.norm2 = channels_get_norms(channels)
        self.QKV = nn.Conv2d(in_channels=channels, out_channels=channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, query):
        q = self.norm1(query)
        q, k, v = self.QKV(q).chunk(3, dim=1)
        b, c, h, w = q.shape
        seq_len = h * w
        q = q.flatten(2).transpose(1, 2).reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        k = k.flatten(2).transpose(1, 2).reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        v = v.flatten(2).transpose(1, 2).reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        out = self.norm2(out)
        return query + self.proj_out(out)


class ImageCrossAttentionBlock(nn.Module):
    def __init__(self, channels, d_model, head_num, key_dim=None):
        super(ImageCrossAttentionBlock, self).__init__()
        assert d_model % head_num == 0, "d_model不能被分组"
        self.d_model = d_model
        self.norm1 = channels_get_norms(channels)
        self.head_num = head_num
        self.d_k = d_model // head_num
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)
        self.v_norm = nn.LayerNorm(self.d_k)
        self.Q = nn.Conv2d(in_channels=channels, out_channels=d_model, kernel_size=1)
        self.K = nn.Linear(key_dim, d_model)
        self.V = nn.Linear(key_dim, d_model)
        self.norm2 = channels_get_norms(d_model)
        self.proj_out = nn.Conv2d(in_channels=d_model, out_channels=channels, kernel_size=1)

    def forward(self, query, key, value=None):
        value = key if value is None else value
        key_seq_length = key.shape[1]
        q = self.norm1(query)
        q = self.Q(q)
        k = self.K(key)
        v = self.V(value)
        b, c, h, w = q.shape
        seq_len = h * w
        q = q.flatten(2).transpose(1, 2).reshape(b, seq_len, self.head_num, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(b, key_seq_length, self.head_num, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(b, key_seq_length, self.head_num, self.d_k).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        out = self.norm2(out)
        return query + self.proj_out(out)
