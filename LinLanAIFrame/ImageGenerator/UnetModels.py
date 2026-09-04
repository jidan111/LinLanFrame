from .structs import *
from ..attention import *


class ResidualBlockConditional(nn.Module):
    """残差连接"""

    def __init__(self, in_channels, out_channels, dropout=.1, condition_dim=128):
        super(ResidualBlockConditional, self).__init__()
        self.equal = in_channels == out_channels
        self.norm1 = AdaGroupNorm(num_channels=in_channels, other_dim=condition_dim)
        self.norm2 = AdaGroupNorm(num_channels=out_channels, other_dim=condition_dim)
        self.conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        if not self.equal:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x, condition):
        h = x
        h = self.norm1(h, condition)
        h = self.conv1(h)
        h = self.norm2(h, condition)
        h = self.conv2(h)
        if not self.equal:
            x = self.conv3(x)
        return x + h


class DownSampleConditional(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_num=1, dropout=.1, condition_dim=128):
        super(DownSampleConditional, self).__init__()
        layer = []
        tmp_channels = in_channels
        for num in range(resnet_num):
            tmp_channels = tmp_channels * 2
            layer.append(
                ResidualBlockConditional(in_channels=tmp_channels // 2, out_channels=tmp_channels, dropout=dropout,
                                         condition_dim=condition_dim))
        self.resnet = nn.ModuleList(layer)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=tmp_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1))

    def forward(self, x, condition):
        for model in self.resnet:
            x = model(x, condition)
        out = self.down(x)
        return out


class UpSample(nn.Module):
    """先采样，后卷积"""

    def __init__(self, in_channels, out_channels, resnet_num=2, mode="interpolate", dropout=.1):
        super(UpSample, self).__init__()
        assert mode in ["interpolate", "ConvTranspose2d"], "只支持ConvTranspose2d和interpolate两种上采样"
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
                ResidualBlock(in_channels=out_channels * 2, out_channels=out_channels, dropout=dropout))
        else:
            tmp_channels = out_channels * 2
            for num in range(resnet_num):
                tmp_channels *= 2
                if num == resnet_num - 1:
                    layer.append(
                        ResidualBlock(in_channels=tmp_channels // 2, out_channels=out_channels, dropout=dropout))
                else:
                    layer.append(
                        ResidualBlock(in_channels=tmp_channels // 2, out_channels=tmp_channels, dropout=dropout))
        self.resnet = nn.Sequential(*layer)

    def forward(self, x, y):
        if self.mode == "interpolate":
            x = self.up(x, scale_factor=2.0, mode="nearest")
            x = self.change_channels(x)
        else:
            x = self.up(x)
        x = torch.cat((x, y), dim=1)
        out = self.resnet(x)
        return out


class UpSampleConditional(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_num=2, mode="interpolate", dropout=.1, condition_dim=128):
        super(UpSampleConditional, self).__init__()
        assert mode in ["interpolate", "ConvTranspose2d"], "只支持ConvTranspose2d和interpolate肯三种上采样"
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
                ResidualBlockConditional(in_channels=out_channels * 2, out_channels=out_channels, dropout=dropout,
                                         condition_dim=condition_dim))
        else:
            tmp_channels = out_channels * 2
            for num in range(resnet_num):
                tmp_channels *= 2
                if num == resnet_num - 1:
                    layer.append(
                        ResidualBlockConditional(in_channels=tmp_channels // 2, out_channels=out_channels,
                                                 dropout=dropout, condition_dim=condition_dim))
                else:
                    layer.append(
                        ResidualBlockConditional(in_channels=tmp_channels // 2, out_channels=tmp_channels,
                                                 dropout=dropout, condition_dim=condition_dim))
        self.resnet = nn.ModuleList(layer)

    def forward(self, x, y, condition):
        if self.mode == "interpolate":
            x = self.up(x, scale_factor=2.0, mode="nearest")
            x = self.change_channels(x)
        else:
            x = self.up(x)
        x = torch.cat((x, y), dim=1)
        for model in self.resnet:
            x = model(x, condition)
        return x


class Unet(ConfigModule):
    def __init__(self, in_channels, hidden_channels, depth=4, attention=[], head_num=8,
                 dropout=.1, resnet_num=1, up_mode="interpolate"):
        super(Unet, self).__init__()
        self.encoder = None
        self.encoder_atte = None
        self.decoder = None
        self.decoder_atte = None
        self.depth = depth
        if type(up_mode) == str:
            up_mode = [up_mode] * depth
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.set_encoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, attention=attention,
                         dropout=dropout, head_num=head_num)
        self.set_decoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, mode=up_mode,
                         attention=attention[::-1], dropout=dropout, head_num=head_num)
        self.mid_layer = nn.Sequential(
            ResidualBlock(in_channels=hidden_channels * (2 ** depth), out_channels=hidden_channels * (2 ** depth)),
            ResidualBlock(in_channels=hidden_channels * (2 ** depth), out_channels=hidden_channels * (2 ** depth))
        )

    def set_encoder(self, depth, hidden_channels, resnet_num=2, attention=[], dropout=.1, head_num=8):
        if len(attention) == 0:
            attention = [False] * depth
        else:
            if len(attention) <= depth:
                attention.extend([False] * (depth - len(attention)))
            else:
                attention = attention[:depth]
        layer = []
        atte = []
        for i, a in enumerate(attention):
            if a:
                atte.append(
                    ImageSelfAttentionBlock(channels=hidden_channels * (2 ** i), head_num=head_num, query_norm=True))
            else:
                atte.append(nn.Identity())
            layer.append(DownSample(
                in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i + 1)),
                resnet_num=resnet_num, dropout=dropout
            ))
        self.encoder = nn.ModuleList(layer)
        self.encoder_atte = nn.ModuleList(atte)

    def set_decoder(self, depth, hidden_channels, resnet_num=2, mode="interpolate", attention=[], dropout=.1,
                    head_num=8):
        if len(attention) == 0:
            attention = [False] * depth
        else:
            if len(attention) <= depth:
                attention.extend([False] * (depth - len(attention)))
            else:
                attention = attention[:depth]
        layer = []
        atte = []
        for index, a in enumerate(attention):
            i = depth - index
            layer.append(UpSample(
                in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i - 1)),
                resnet_num=resnet_num, mode=mode[index], dropout=dropout
            ))
            if a:
                atte.append(
                    ImageSelfAttentionBlock(channels=hidden_channels * (2 ** (i - 1)), head_num=head_num,
                                            query_norm=True))
            else:
                atte.append(nn.Identity())
        self.decoder = nn.ModuleList(layer)
        self.decoder_atte = nn.ModuleList(atte)

    def forward(self, x):
        x = self.init_conv(x)
        stack = [x]
        for encode, atte in zip(self.encoder, self.encoder_atte):
            x = atte(x)
            x = encode(x)
            stack.append(x)
        x = stack.pop()
        out = self.mid_layer(x)
        for decode, atte in zip(self.decoder, self.decoder_atte):
            x = stack.pop()
            out = decode(out, x)
            out = atte(out)
        out = self.out_conv(out)
        return out


class UnetConditional(ConfigModule):
    def __init__(self, in_channels, hidden_channels, depth=4, attention=[], d_model=512, head_num=8,
                 dropout=.1, resnet_num=1, up_mode="ConvTranspose2d", condition_dim=128, key_dim=None,
                 double_condition_control=True):
        super(UnetConditional, self).__init__()
        self.encoder = None
        self.decoder = None
        self.double_condition_control = double_condition_control
        self.encoder_self_atte = None
        self.decoder_self_atte = None
        self.encoder_cross_atte = None
        self.decoder_cross_atte = None
        self.key_dim = key_dim
        self.depth = depth
        if type(up_mode) == str:
            up_mode = [up_mode] * depth
        if self.key_dim:
            if self.double_condition_control:
                self.pool_key2condition = nn.Linear(key_dim, condition_dim)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.set_encoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, attention=attention,
                         d_model=d_model, dropout=dropout, key_dim=key_dim, condition_dim=condition_dim,
                         head_num=head_num)
        self.set_decoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, mode=up_mode,
                         attention=attention[::-1], d_model=d_model, dropout=dropout, key_dim=key_dim,
                         condition_dim=condition_dim, head_num=head_num)
        norm = channels_get_norms(hidden_channels * (2 ** depth))
        self.mid_layer = nn.Sequential(
            ResidualBlock(in_channels=hidden_channels * (2 ** depth), out_channels=hidden_channels * (2 ** depth)),
            norm,
            ResidualBlock(in_channels=hidden_channels * (2 ** depth), out_channels=hidden_channels * (2 ** depth))
        )

    def set_encoder(self, depth, hidden_channels, resnet_num=2, attention=[], head_num=8,
                    d_model=512, dropout=.1, key_dim=None, condition_dim=128):
        if len(attention) == 0:
            attention = [False] * depth
        else:
            if len(attention) <= depth:
                attention.extend([False] * (depth - len(attention)))
            else:
                attention = attention[:depth]
        layer = []
        atte = []
        cross_atte = []
        for i, a in enumerate(attention):
            if a:
                atte.append(
                    ImageSelfAttentionBlock(channels=hidden_channels * (2 ** i), head_num=head_num, dropout=dropout,
                                            query_norm=True))
            else:
                atte.append(nn.Identity())
            if self.key_dim is not None:
                cross_atte.append(ImageCrossAttentionBlock(channels=hidden_channels * (2 ** i),
                                                           d_model=d_model, head_num=head_num, key_dim=key_dim,
                                                           dropout=dropout, query_norm=True))
            else:
                cross_atte.append(nn.Identity())
            layer.append(DownSampleConditional(
                in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i + 1)),
                resnet_num=resnet_num, dropout=dropout, condition_dim=condition_dim
            ))
        self.encoder = nn.ModuleList(layer)
        self.encoder_self_atte = nn.ModuleList(atte)
        self.encoder_cross_atte = nn.ModuleList(cross_atte)

    def set_decoder(self, depth, hidden_channels, resnet_num=2, mode="interpolate", attention=[], d_model=512,
                    dropout=.1, key_dim=None, condition_dim=128, head_num=8):
        if len(attention) == 0:
            attention = [False] * depth
        else:
            if len(attention) <= depth:
                attention.extend([False] * (depth - len(attention)))
            else:
                attention = attention[:depth]
        layer = []
        atte = []
        cross_atte = []
        for index, a in enumerate(attention):
            i = depth - index
            layer.append(UpSampleConditional(
                in_channels=hidden_channels * (2 ** i), out_channels=hidden_channels * (2 ** (i - 1)),
                resnet_num=resnet_num, mode=mode[index], dropout=dropout, condition_dim=condition_dim
            ))
            if a:
                atte.append(ImageSelfAttentionBlock(channels=hidden_channels * (2 ** (i - 1)), head_num=head_num,
                                                    dropout=dropout, query_norm=True))
            else:
                atte.append(nn.Identity())
            if self.key_dim is not None:
                cross_atte.append(ImageCrossAttentionBlock(channels=hidden_channels * (2 ** (i - 1)), d_model=d_model,
                                                           head_num=head_num, key_dim=key_dim, dropout=dropout,
                                                           query_norm=True))
            else:
                cross_atte.append(nn.Identity())
        self.decoder = nn.ModuleList(layer)
        self.decoder_self_atte = nn.ModuleList(atte)
        self.decoder_cross_atte = nn.ModuleList(cross_atte)

    def forward(self, x, condition, other_condition1=None, other_condition2=None, **kwargs):
        """
        :param x: [b,c,h,w]
        :param condition: [b,dim]
        :param other_condition1: [b,seq_len,dim]
        :param other_condition2: [b,dim]
        :return:
        """
        if self.key_dim is not None:
            if other_condition1.dim() == 2:
                other_condition1 = other_condition1.unsqueeze(1)
            if self.double_condition_control:
                other_condition2 = self.pool_key2condition(other_condition2)
                condition = condition + other_condition2
        x = self.init_conv(x)
        stack = [x]
        for encode, self_atte, cross_atte in zip(self.encoder, self.encoder_self_atte, self.encoder_cross_atte):
            x = self_atte(x)
            if self.key_dim is not None:
                x = cross_atte(x, other_condition1)
            x = encode(x, condition)
            stack.append(x)
        x = stack.pop()
        out = self.mid_layer(x)
        for decode, self_atte, cross_atte in zip(self.decoder, self.decoder_self_atte, self.decoder_cross_atte):
            x = stack.pop()
            out = decode(out, x, condition)
            out = self_atte(out)
            if self.key_dim is not None:
                out = cross_atte(out, other_condition1)
        out = self.out_conv(out)
        return out
