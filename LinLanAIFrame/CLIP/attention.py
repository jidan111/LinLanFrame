from ..import_package import *


def make_padding_mask(query):
    seq_length = query.shape[2]
    token_valid = ~torch.all(query == 0, dim=-1)
    mask = torch.any(token_valid, dim=1)
    return mask.unsqueeze(-1).expand(-1, -1, seq_length)[0]


def make_causal_mask(query, key):
    l, s = query.shape[2], key.shape[2]
    mask = torch.ones(l, s, dtype=torch.bool).tril(diagonal=0)
    return mask


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


class SelfAttentionBlock(nn.Module):
    def __init__(self, query_dim, d_model, head_num, dropout=.1):
        super(SelfAttentionBlock, self).__init__()
        assert d_model % head_num == 0, f"d_model无法被head_num均分, {d_model}%{head_num}!=0"
        self.head_nums = head_num
        self.d_k = d_model // head_num
        self.norm = nn.LayerNorm(query_dim)
        self.qkv = nn.Linear(query_dim, d_model * 3, bias=False)
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)
        self.v_norm = nn.LayerNorm(self.d_k)
        self.out = nn.Sequential(
            nn.Linear(d_model, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, mask=None):
        query = self.norm(query)
        q, k, v = self.qkv(query).chunk(3, dim=2)
        batch_size, seq_length, d_model = q.shape
        q = q.reshape(batch_size, seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        out = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_length, d_model).contiguous()
        out = self.out(out)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_in_dim=None, key_in_dim=None, value_in_dim=None, d_model=512, head_num=8, dropout=.1):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % head_num == 0, f"d_model无法被head_num均分, {d_model}%{head_num}!=0"
        key_in_dim = query_in_dim if key_in_dim is None else key_in_dim
        value_in_dim = key_in_dim if value_in_dim is None else value_in_dim
        self.head_nums = head_num
        self.d_k = d_model // head_num
        self.norm = nn.LayerNorm(query_in_dim)
        self.Q = nn.Linear(query_in_dim, d_model, bias=False)
        self.K = nn.Linear(key_in_dim, d_model, bias=False)
        self.V = nn.Linear(value_in_dim, d_model, bias=False)
        self.q_norm = nn.LayerNorm(self.d_k)
        self.k_norm = nn.LayerNorm(self.d_k)
        self.v_norm = nn.LayerNorm(self.d_k)
        self.out = nn.Sequential(
            nn.Linear(d_model, query_in_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value=None, padding_mask=None):
        value = key if value is None else value
        query = self.norm(query)
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)
        batch_size, seq_length, d_model = q.shape
        q = q.reshape(batch_size, seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, seq_length, self.head_nums, self.d_k).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        out = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=padding_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_length, d_model).contiguous()
        out = self.out(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, query_dim, d_model, head_num, layer_num, dropout, hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList([])
        for num in range(layer_num):
            self.layer.append(
                nn.ModuleList(
                    [SelfAttentionBlock(query_dim=query_dim, d_model=d_model, head_num=head_num, dropout=dropout),
                     FeedForward(in_dim=query_dim, hidden_dim=hidden_dim, dropout=dropout)]))

    def forward(self, query, mask=None):
        for atte, ffn in self.layer:
            query = atte(query, mask=mask) + query
            query = ffn(query) + query
        return query
