from ..structs import *
from .attention import *


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
    def __init__(self, max_seq_length=128, vocab_size=2048, d_model=512, head_num=8, layer_num=8, dropout=.1,
                 hidden_dim=64):
        super(TextEmbedTransformer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer(
            "pos_embed",
            get_sinusoidal_pos_encoding(max_seq_length, d_model)
        )
        self.vocab_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.init = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(query_dim=d_model, d_model=d_model, head_num=head_num, layer_num=layer_num,
                                              hidden_dim=hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def padding_mask(self, tokens):
        """False 为pad ,True为有效"""
        padding_mask = (tokens != 0)
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        return attn_mask

    def forward(self, tokens):
        batch_size, seq_length = tokens.shape
        x = self.vocab_embed(tokens)
        x = x + self.pos_embed[:, :seq_length, :]
        x = self.init(x)
        mask = self.padding_mask(tokens).to(x.device)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
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
        x = self.norm(x)
        return x
