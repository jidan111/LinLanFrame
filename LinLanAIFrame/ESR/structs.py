from ..import_package import *


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels + hidden_channels, out_channels=hidden_channels, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels + 2 * hidden_channels, out_channels=hidden_channels,
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels + 3 * hidden_channels, out_channels=hidden_channels,
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=in_channels + 4 * hidden_channels, out_channels=in_channels, kernel_size=3,
                               stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels=in_channels, hidden_channels=hidden_channels)
        self.rdb2 = ResidualDenseBlock(in_channels=in_channels, hidden_channels=hidden_channels)
        self.rdb3 = ResidualDenseBlock(in_channels=in_channels, hidden_channels=hidden_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode="interpolate"):
        super(UpSampleBlock, self).__init__()
        self.up_mode = up_mode
        if up_mode == "interpolate":
            self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2,
                                         padding=1)
        self.out_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  padding=1)

    def forward(self, x):
        if self.up_mode == "interpolate":
            x = self.up(x, scale_factor=2.0, mode="nearest")
        else:
            x = self.up(x)
        x = self.out_conv(x)
        return x
