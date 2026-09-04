from ..structs import *
from ..attention import *


def get_sinusoidal_pos_encoding(max_seq_length=2048, d_model=512):
    """生成正余弦位置编码矩阵"""
    pos_encoding = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))
    pos_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pos_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    return pos_encoding.unsqueeze(0)  # 添加batch维度


class TextEmbedTransformer(ConfigModule):
    """
    依赖输入序列为</start>tokens</end></pad></pad>...</pad>, 且</end>的下标是词表中最大的，计算argmax来得到</end>的张量表示整个句子，
    从而避免</pad>产生的无效干扰
    """

    def __init__(self, max_seq_length=128, vocab_size=2048, d_model=512, head_nums=8, layer_num=8, dropout=.1,
                 mlp_ratio=4, out_dim=512, checkpoint_enable=True):
        super(TextEmbedTransformer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_model = d_model
        self.register_buffer(
            "pos_embed",
            get_sinusoidal_pos_encoding(max_seq_length, d_model)
        )
        self.vocab_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.init = nn.Linear(d_model, d_model)
        self.transformer = TransformerEncoder(query_dim=d_model, d_model=d_model, head_nums=head_nums,
                                              layer_num=layer_num,
                                              mlp_ratio=mlp_ratio, dropout=dropout, checkpoint_enable=checkpoint_enable)
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, out_dim)

    def padding_mask(self, tokens):
        """False 为pad ,True为有效"""
        attn_mask = (tokens != 0)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        # 后续接入多头注意力，形状为(batch, head_num, seq_len, d_k)， 需要(batch, 1, 1, seq_len)进行广播
        return attn_mask

    def forward(self, tokens):
        batch_size, seq_length = tokens.shape
        x = self.vocab_embed(tokens)
        x = x + self.pos_embed[:, :seq_length, :]
        x = self.init(x)
        mask = self.padding_mask(tokens).to(x.device)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        x = x[torch.arange(batch_size), tokens.argmax(-1)]
        x = self.proj_out(x)
        return x

    @torch.no_grad()
    def encode_text(self, tokens):
        batch_size, seq_length = tokens.shape
        x = self.vocab_embed(tokens)
        x = x + self.pos_embed[:, :seq_length, :]
        x = self.init(x)
        mask = self.padding_mask(tokens).to(x.device)
        x = self.transformer(x, mask=mask)
        features = self.norm(x)
        x = features[torch.arange(batch_size), tokens.argmax(-1)]
        x = self.proj_out(x)
        x = x / x.norm(dim=1, keepdim=True)
        return x, features
