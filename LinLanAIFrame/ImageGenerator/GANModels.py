from .structs import *


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h_ = self.conv(x)
        x = x + h_
        out = self.down(x)
        return out


class Discriminator(ConfigModule):
    def __init__(self, image_shape=(3, 64, 64), hidden_channels=8, depth=None, use_spectral_norm=False):
        super(Discriminator, self).__init__()
        if depth is not None:
            self.depth = depth
            self.size = image_shape[1] // (2 ** depth)
            assert self.size * (2 ** depth) == image_shape[1], "输入图片大小与深度不匹配"
        else:
            self.depth, self.size = get_depth(image_shape[1])
        arr = []
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        for i in range(depth):
            arr.extend([DiscriminatorBlock(in_channels=hidden_channels * (2 ** i),
                                           out_channels=hidden_channels * (2 ** (i + 1)))])
        self.layer = nn.Sequential(*arr)
        self.out = nn.Sequential(
            nn.Linear(self.size * self.size * hidden_channels * (2 ** self.depth), 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )
        if use_spectral_norm:
            self.apply(add_sn)

    def forward(self, x):
        x = self.init(x)
        x = self.layer(x)
        x = x.flatten(1)
        x = self.out(x)
        return x


class PatchDiscriminatorDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchDiscriminatorDownBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.layer(x)


class PatchDiscriminator(ConfigModule):
    def __init__(self, in_channels: int, layer_num: int = 4, hidden_channels: int = 8, max_hidden_channels: int = 8,
                 use_spectral_norm=False):
        super(PatchDiscriminator, self).__init__()
        arr = []
        in_c = in_channels
        out_c = hidden_channels
        for n in range(0, layer_num, 1):
            arr.append(PatchDiscriminatorDownBlock(in_channels=in_c, out_channels=out_c))
            in_c = out_c
            out_c = min(max_hidden_channels, out_c * 2)
        arr.extend(
            [nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(in_c),
                           nn.LeakyReLU(0.2, inplace=True))]
        )
        arr.extend([nn.Conv2d(in_channels=in_c, out_channels=1, kernel_size=3, stride=1, padding=1)])
        self.layer = nn.Sequential(
            *arr
        )
        if use_spectral_norm:
            self.apply(add_sn)

    def forward(self, x):
        return self.layer(x)


class Generator(ConfigModule):
    def __init__(self, in_dim, image_shape, hidden_channels=8, depth=None, attention=[], head_num=8,
                 dropout=.1, resnet_num=1, up_mode="interpolate"):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.hidden_channels = hidden_channels
        if depth is not None:
            self.depth = depth
            self.size = image_shape[1] // (2 ** depth)
            assert self.size * (2 ** depth) == image_shape[1], "输入图片大小与深度不匹配"
        else:
            self.depth, self.size = get_depth(image_shape[1])
        if type(up_mode) == str:
            up_mode = [up_mode] * self.depth
        self.init_conv = nn.Sequential(
            nn.Linear(in_dim, self.size * self.size * hidden_channels * (2 ** self.depth)),
            nn.Unflatten(1, (hidden_channels * (2 ** self.depth), self.size, self.size)),
        )
        self.decoder = Decoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, mode=up_mode,
                               attention=attention, dropout=dropout, head_num=head_num)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=image_shape[0], kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, *_ = x.shape
        x = self.init_conv(x)
        x = self.decoder(x)
        x = self.out_conv(x)
        return x

    def get_last_layer_weight(self):
        return self.out_conv[0].weight

    @torch.no_grad()
    def sample(self, batch_size):
        device = self.init_conv[0].weight.device
        noise = torch.randn(size=(batch_size, self.in_dim), device=device)
        out = self(noise)
        return out


class UnetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(.2, inplace=True)
        )
        self.equal = in_channels == out_channels
        if not self.equal:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return h_ + x


class UnetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_num=2):
        super(UnetDownSample, self).__init__()
        resnet = [UnetResidualBlock(in_channels=in_channels, out_channels=out_channels)]
        for i in range(resnet_num - 1):
            resnet.append(UnetResidualBlock(in_channels=out_channels, out_channels=out_channels))
        self.resnet = nn.Sequential(*resnet)
        self.down = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.down(x)
        return x


class UnetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_num=2):
        super(UnetUpSample, self).__init__()
        self.up = lambda in_: F.interpolate(in_, scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.resnet = nn.Sequential(
            *[UnetResidualBlock(in_channels=out_channels, out_channels=out_channels) for i in range(resnet_num)]
        )

    def forward(self, x):
        x = self.up(x)
        x = self.up_conv(x)
        x = self.resnet(x)
        return x


class UnetDiscriminator(ConfigModule):
    def __init__(self, in_channels, depth=3, hidden_channels=4, resnet_num=2, use_spectral_norm=False):
        super(UnetDiscriminator, self).__init__()
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                                   padding=1)
        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                  padding=1)
        down_sample = []
        up_sample = []
        per = hidden_channels
        post = hidden_channels * 2
        for i in range(depth):
            down_sample.append(UnetDownSample(in_channels=per, out_channels=post, resnet_num=resnet_num))
            up_sample.insert(0, UnetUpSample(in_channels=post * 2, out_channels=per, resnet_num=resnet_num))
            per = post
            post *= 2
        self.mid_layer = UnetResidualBlock(in_channels=per, out_channels=2 * per)
        self.down_models = nn.ModuleList(down_sample)
        self.up_models = nn.ModuleList(up_sample)
        if use_spectral_norm:
            self.apply(add_sn)

    def forward(self, x):
        x = self.init_conv(x)
        skip = [x]
        for index, model in enumerate(self.down_models):
            x = model(x)
            skip.append(x)
        x = skip.pop()
        out = self.mid_layer(x)
        for index, model in enumerate(self.up_models):
            if index == 0:
                out = model(out)
            else:
                down_ = skip.pop()
                out = torch.cat((out, down_), dim=1)
                out = model(out)
        out = self.out_conv(out)
        return out
