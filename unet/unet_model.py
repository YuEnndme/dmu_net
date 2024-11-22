from unet.unet_parts import *
import torch.nn as nn

from unet.unet_parts import _DAHead, _DAHead3, _DAHead4


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = U_down(64, 128)
        self.down2 = U_down(128, 256)
        self.down3 = U_down(256, 512)
        self.down4 = U_down(512, 512)
        self.up1 = U_up(1024, 256)
        self.up2 = U_up(512, 128)
        self.up3 = U_up(256, 64)
        self.up4 = U_up(128, 64)
        self.out = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Res_Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Res_Unet, self).__init__()
        self.down = RU_first_down(n_channels, 32)
        self.down1 = RU_down(32, 64)
        self.down2 = RU_down(64, 128)
        self.down3 = RU_down(128, 256)
        self.down4 = RU_down(256, 56)
        self.up1 = RU_up(512, 128)
        self.up2 = RU_up(256, 64)
        self.up3 = RU_up(128, 32)
        self.up4 = RU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Ringed_Res_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, **kwargs):
        super(Ringed_Res_Unet, self).__init__()
        self.down = RRU_first_down(n_channels, 32)
        self.down1 = RRU_down(32, 64)
        self.down2 = RRU_down(64, 128)
        self.down3 = RRU_down(128, 256)
        self.down4 = RRU_down(256, 256)         # 图中的RRU_down路径画的有问题 最后一个channel应该是256 拼接之后才会变成512
        self.up1 = RRU_up(512, 256)
        self.up2 = RRU_up(256, 128)
        self.up3 = RRU_up3(256, 64)
        self.up4 = RRU_up3(128, 32)
        self.out = outconv(32, n_classes)
        # self.up1 = RRU_up(1024, 256)
        # self.up2 = RRU_up(512, 128)
        # self.up3 = RRU_up(256, 64)
        # self.up4 = RRU_up(128, 32)
        # self.out = outconv(32, n_classes)

        self.head = _DAHead4(1280,  **kwargs)
        self.aspp = ASPP(512, [12, 24, 36])

    def forward(self, x):
        x01 = self.down(x[0])            # x[0]代表原图 x[1]代表噪声图 32
        x02 = self.down1(x01)            # 64
        x03 = self.down2(x02)            # 128
        x04 = self.down3(x03)            # 256
        x05 = self.down4(x04)            # 512

        x11 = self.down(x[1])            # x[0]代表原图 x[1]代表噪声图
        x12 = self.down1(x11)
        x13 = self.down2(x12)
        x14 = self.down3(x13)
        x15 = self.down4(x14)

        x5 = torch.cat([x05, x15], dim=1)       # 在纬度范围进行拼接 此时的纬度为 512 拼接最底层的RGB和Nosie
        x5 = self.aspp(x5)                      # 此时的纬度为1280
        x5 = self.head(x5)                      # in_channel=1280 out_channel=512
        # x5_ = self.head(x5)                     # 此时x5的纬度为256
        # x4 = torch.cat([x04, x14], dim=1)
        # x4 = self.head3(x4)
        # x3 = torch.cat([x03, x13], dim=1)
        # x3 = self.head3(x3)
        # x2 = torch.cat([x02, x12], dim=1)
        # x2 = self.head2(x2)
        # x1 = torch.cat([x01, x11], dim=1)
        # x1 = self.head1(x1)

        x = self.up1(x5, x04, x14)
        x = self.up2(x, x03, x13)
        x = self.up3(x, x02, x12)
        x = self.up4(x, x01, x11)
        x = self.out(x)
        return x