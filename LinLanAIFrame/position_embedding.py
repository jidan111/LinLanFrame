from .import_package import *


class RotaryEmbedding(nn.Module):
    """
    1D情况，dim就是[batch_size, heads, seq, dim]里面的dim, 频率全覆盖
    2D情况，dim就是[batch_size, heads, seq, dim]里面的dim//2， 平分频率到两个轴上
    3D以此类推，需要将频率平分到每一个轴(通道上)
    """
    def __init__(self, dim, max_rotary_freq=10, theta=1000, mode="lang", start_index=0, amplitude_scale=1.):
        super(RotaryEmbedding, self).__init__()
        assert mode in ["lang", "pixel"]
        self.mode = mode
        self.dim = dim
        self.half_dim = dim // 2
        self.amplitude_scale = amplitude_scale
        self.start_index = start_index
        if mode == "lang":
            base_freq = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        else:
            base_freq = torch.linspace(1., max_rotary_freq / 2, dim // 2) * math.pi
        self.register_buffer("base_freq", base_freq)

    def rotate_half(self, x):
        """
        x: [batch, head, seq_len, d]
        """
        b, h, s, d = x.shape
        assert d % 2 == 0, f"特征维度应该能被均分, {d}%2!=0"
        x = x.reshape(b, h, s, d // 2, 2)
        x1, x2 = x.unbind(dim=-1)  # 交错取值reshape+unbind，也可以用reshape+chunk+squeeze
        result = torch.stack((-x2, x1), dim=-1)  # 交错拼接stack+flatten，不能用cat
        return result.reshape(b, h, s, d)

    @autocast(enabled=False)
    def apply_rotary_emb(self, x, freq, start_index=0, amplitude_scale=1.):
        """
        x: [batch, head, seq_len, d]
        freq: [seq_len, d]
        """
        freq_cos = freq.cos()
        freq_sin = freq.sin()
        rot_dim = freq.shape[-1]
        end_index = start_index + rot_dim
        assert end_index <= x.shape[-1], "特征维度小于旋转维度，无法完成left-middle-right切分"
        x_left = x[..., :start_index]
        x_middle = x[..., start_index:end_index]
        x_right = x[..., end_index:]
        rotary_middle = (x_middle * freq_cos * amplitude_scale) + (
                self.rotate_half(x_middle) * freq_sin * amplitude_scale)
        rotary = torch.cat((x_left, rotary_middle, x_right), dim=-1)
        return rotary.contiguous()

    def get_device(self):
        return self.base_freq.device

    def get_rotary_freq_1d(self, seq_len):
        pos = torch.arange(seq_len, device=self.get_device())
        rotary_freq = pos[:, None] * self.base_freq[None, :]
        rotary_freq = rotary_freq.repeat_interleave(2, dim=-1)  # 两两逐元素复制，而非按块复制
        return rotary_freq

    def get_rotary_freq_2d(self, shape):
        """
        输入shape为patch分割后的形状，而非原始形状
        :param shape:
        :return:
        """
        assert len(shape) == 2, "只支持(h, w)形状"
        pos_h = torch.linspace(-1, 1, shape[0], device=self.get_device())
        pos_w = torch.linspace(-1, 1, shape[1], device=self.get_device())
        freq_h = pos_h[:, None] * self.base_freq[None, :]
        freq_h = freq_h.repeat_interleave(2, dim=-1)
        freq_w = pos_w[:, None] * self.base_freq[None, :]
        freq_w = freq_w.repeat_interleave(2, dim=-1)
        freq_h, freq_w = torch.broadcast_tensors(freq_h[:, None, :], freq_w[None, :, :])
        rotary_freq = torch.stack((freq_h, freq_w), dim=-1)
        rotary_freq = rotary_freq.reshape(math.prod(shape), -1)
        return rotary_freq

    def rotate_queries_or_keys_1d(self, x):
        """
        :param x: [batch_size, heads, seq, dim]
        :return:
        """
        seq_len = x.shape[2]
        rotary_freq = self.get_rotary_freq_1d(seq_len=seq_len)
        return self.apply_rotary_emb(x=x, freq=rotary_freq, start_index=self.start_index,
                                     amplitude_scale=self.amplitude_scale)

    def rotate_queries_or_keys_2d(self, x, shape):
        """
        :param x: [batch_size, heads, seq, dim]
        :param shape:
        :return:
        """
        rotary_freq = self.get_rotary_freq_2d(shape=shape)
        return self.apply_rotary_emb(x=x, freq=rotary_freq, start_index=self.start_index,
                                     amplitude_scale=self.amplitude_scale)

    def rotate_queries_and_keys(self, query, key, shape):
        """
        本人自用接口，query为图片token，key为外部编码结果
        :param query:
        :param key:
        :param shape:
        :return:
        """
        q_rotary_freq = self.get_rotary_freq_2d(shape=shape)
        q_rotary_emb = self.apply_rotary_emb(x=query, freq=q_rotary_freq, start_index=self.start_index,
                                             amplitude_scale=self.amplitude_scale)
        key_seq_len = key.shape[2]
        k_rotary_freq = self.get_rotary_freq_1d(seq_len=key_seq_len)
        k_rotary_emb = self.apply_rotary_emb(x=key, freq=k_rotary_freq, start_index=self.start_index,
                                             amplitude_scale=self.amplitude_scale)
        return q_rotary_emb, k_rotary_emb

    def forward(self, x, shape=None):
        """
        x:[batch_size, heads, seq, dim]
        """
        if self.mode == "lang":
            seq_len = x.shape[2]
            rotary_freq = self.get_rotary_freq_1d(seq_len=seq_len)
        else:
            rotary_freq = self.get_rotary_freq_2d(shape=shape)
        return self.apply_rotary_emb(x=x, freq=rotary_freq, start_index=self.start_index,
                                     amplitude_scale=self.amplitude_scale)
