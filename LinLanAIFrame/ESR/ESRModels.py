from .structs import *
from ..structs import *


class RRDBNet(ConfigModule):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, up_num: int = 1, grow_channels=32,
                 layer_num: int = 23, out_channels: int = None, up_mode="interpolate"):
        super(RRDBNet, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.rrd_layer = nn.Sequential(
            *[RRDB(in_channels=hidden_channels, hidden_channels=grow_channels) for i in range(layer_num)]
        )
        self.act = nn.LeakyReLU(.2, inplace=True)
        self.fussion = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                                 padding=1)
        up = []
        for i in range(up_num):
            up.append(RRDB(in_channels=hidden_channels, hidden_channels=grow_channels))
            up.append(UpSampleBlock(in_channels=hidden_channels, out_channels=hidden_channels, up_mode=up_mode))
            up.append(nn.LeakyReLU(.2, inplace=True))
        self.up = nn.Sequential(*up)
        self.hr_feature = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                                    padding=1)
        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  padding=1)

    def get_last_layer_weight(self):
        return self.out_conv.weight

    def forward(self, x):
        x = self.init_conv(x)
        h_ = self.rrd_layer(x)
        h_ = self.act(self.fussion(h_))
        h_ = self.act(x + h_)
        h_ = self.up(h_)
        h_ = self.act(self.hr_feature(h_))
        h_ = self.out_conv(h_)
        return F.tanh(h_)
