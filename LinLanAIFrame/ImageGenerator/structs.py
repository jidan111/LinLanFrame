from ..structs import *
from ..attention import *
from .functions import *


class ResidualBlock(nn.Module):
    """
    norm+act+conv+norm+act+conv能提供更稳定的梯度流
    残差连接
    """

    def __init__(self, in_channels, out_channels, dropout=.1):
        super(ResidualBlock, self).__init__()
        self.equal = in_channels == out_channels
        norm1 = channels_get_norms(in_channels)
        norm2 = channels_get_norms(out_channels)
        self.conv1 = nn.Sequential(
            norm1,
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            norm2,
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        if not self.equal:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return x + out


class DownSample(nn.Module):
    """先卷积，后采样"""

    def __init__(self, in_channels, out_channels, resnet_num=1, dropout=.1):
        super(DownSample, self).__init__()
        layer = []
        tmp_channels = in_channels
        for num in range(resnet_num):
            tmp_channels = tmp_channels * 2
            layer.append(
                ResidualBlock(in_channels=tmp_channels // 2, out_channels=tmp_channels, dropout=dropout))
        self.resnet = nn.Sequential(*layer)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=tmp_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x = self.resnet(x)
        out = self.down(x)
        return out


class UpSample(nn.Module):
    """先采样，后卷积"""

    def __init__(self, in_channels, out_channels, resnet_num=2, mode="interpolate", dropout=.1):
        super(UpSample, self).__init__()
        assert mode in ["interpolate", "ConvTranspose2d"], "只支持ConvTranspose2d和interpolate共2种上采样"
        self.mode = mode
        self.in_ = in_channels
        self.out_ = out_channels
        if mode == "interpolate":
            self.up = nn.functional.interpolate
            self.change_channels = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                         padding=1)
        layer = []
        if resnet_num == 1:
            layer.append(
                ResidualBlock(in_channels=out_channels, out_channels=out_channels, dropout=dropout))
        else:
            tmp_channels = out_channels
            for num in range(resnet_num):
                tmp_channels *= 2
                if num == resnet_num - 1:
                    layer.append(
                        ResidualBlock(in_channels=tmp_channels // 2, out_channels=out_channels, dropout=dropout))
                else:
                    layer.append(
                        ResidualBlock(in_channels=tmp_channels // 2, out_channels=tmp_channels, dropout=dropout))
        self.resnet = nn.Sequential(*layer)

    def forward(self, x):
        if self.mode == "interpolate":
            x = self.up(x, scale_factor=2.0, mode="nearest")
            x = self.change_channels(x)
        else:
            x = self.up(x)
        out = self.resnet(x)
        return out


class Encoder(ConfigModule):
    def __init__(self, depth, hidden_channels, resnet_num=2, attention=[], dropout=.1, head_num=8):
        super(Encoder, self).__init__()
        if len(attention) == 0:
            attention = [False] * depth
        else:
            if len(attention) <= depth:
                attention.extend([False] * (depth - len(attention)))
            else:
                attention = attention[:depth]
        layer = []
        for i, atte in enumerate(attention):
            if atte:
                layer.append(
                    ImageSelfAttentionBlock(channels=hidden_channels * (2 ** i), head_num=head_num))
            layer.append(DownSample(
                in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i + 1)),
                resnet_num=resnet_num, dropout=dropout
            ))
        self.encoder = nn.Sequential(*layer)

    def forward(self, x):
        return self.encoder(x)


class Decoder(ConfigModule):
    def __init__(self, depth, hidden_channels, resnet_num=2, mode="interpolate", attention=[], dropout=.1, head_num=8):
        super(Decoder, self).__init__()
        if len(attention) == 0:
            attention = [False] * depth
        else:
            if len(attention) <= depth:
                attention.extend([False] * (depth - len(attention)))
            else:
                attention = attention[:depth]
        if type(mode) == str:
            mode = [mode] * depth
        layer = []
        for index, atte in enumerate(attention):
            i = depth - index
            layer.append(UpSample(
                in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i - 1)),
                resnet_num=resnet_num, mode=mode[index], dropout=dropout
            ))
            if atte:
                layer.append(
                    ImageSelfAttentionBlock(channels=hidden_channels * (2 ** (i - 1)), head_num=head_num))
        self.decoder = nn.Sequential(*layer)

    def forward(self, x):
        return self.decoder(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, num_channels, other_dim):
        super(AdaGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.norm = channels_get_norms(num_channels)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(other_dim, num_channels * 2)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, other, **kwargs):
        batch_size, *_ = other.shape
        other = self.mlp(other)
        other = other.reshape(batch_size, self.num_channels * 2, 1, 1)
        scale, shift = other.chunk(2, dim=1)
        x = self.norm(x)
        x = (1 + scale) * x + shift
        return x


class LogVar(nn.Module):
    def __init__(self):
        super(LogVar, self).__init__()
        self.log_var = nn.Parameter(torch.ones(size=()) * 0.0)

    def forward(self):
        return torch.clamp(self.log_var, min=-10, max=10)
