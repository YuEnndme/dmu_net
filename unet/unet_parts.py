import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# ~~~~~~~~~~ U-Net ~~~~~~~~~~

class U_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class U_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            U_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class U_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(U_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


# ~~~~~~~~~~ RU-Net ~~~~~~~~~~

class RU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class RU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_first_down, self).__init__()
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.maxpool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RU_up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


# ~~~~~~~~~~ RRU-Net ~~~~~~~~~~

class RRU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),  # dilation=2 表示卷积核的空洞率为 2（即使用空洞卷积） padding=2可以保持大小不变
            nn.GroupNorm(32, out_ch),  # 分组归一化层，对卷积输出进行归一化处理，32 表示分组的数量，out_ch 是归一化的特征通道数。
            nn.ReLU(inplace=True),  # inplace=True 表示直接在原始数据上进行操作而不是创建副本。
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch)
        )
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),  # dilation=2 表示卷积核的空洞率为 2（即使用空洞卷积） padding=2可以保持大小不变
            nn.GroupNorm(32, out_ch),  # 分组归一化层，对卷积输出进行归一化处理，32 表示分组的数量，out_ch 是归一化的特征通道数。
            nn.ReLU(inplace=True),  # inplace=True 表示直接在原始数据上进行操作而不是创建副本。

            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, groups=8, padding=2, dilation=2),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),  # inplace=True 表示直接在原始数据上进行操作而不是创建副本。

            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class RRU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_first_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(  # 残差结构 前向传播
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch)  # 按channel纬度分为32个组进行归一化
        )
        self.res_conv_back = nn.Sequential(  # 反向传播 第一个结构中in_ch=3 out_ch=32
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False)  # 所以这里 输入channel=32 输出channel=3
        )

    def forward(self, x):
        # the first ring conv
        ft1 = self.conv(x)  # 两个空洞卷积
        r1 = self.relu(ft1 + self.res_conv(x))  # 残差链接后Relu

        # the second ring conv
        ft2 = self.res_conv_back(r1)  # 残差反馈
        '''
            输入x与 1 + F.sigmoid(ft2) 相乘，然后将结果赋值给 x；
            1等价于原来的输入1*x，
            F.sigmoid(ft2)等价于残差反馈 F.sigmoid(ft2)*x
        '''
        x = torch.mul(1 + torch.sigmoid(ft2), x)  # 把F.sigmoid 换成了 torch.sigmoid

        # the third ring conv
        ft3 = self.conv(x)  # 再来一次空洞卷积
        r3 = self.relu(ft3 + self.res_conv(x))  # 再来一次残差连接

        return r3


class RRU_down(nn.Module):  # 和first_down几乎一样，就多了一个池化层
    def __init__(self, in_ch, out_ch):
        super(RRU_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 相比于Fist_down 多了一个MaxPool层  H、W变为原来的1/2

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.pool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)  # 把F.sigmoid 换成了 torch.sigmoid
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, **kwargs):
        super(RRU_up, self).__init__()
        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 转置卷积和组归一化
            self.up = nn.Sequential(
                # nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                # nn.GroupNorm(32, in_ch // 2))
                nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2),  # 由于这个时候的in_ch纬度就是x1的纬度 所以不用//2
                nn.GroupNorm(32, in_ch))

        self.conv = RRU_double_conv(in_ch, out_ch)  # 这里是拼接上原来的特征图后所以输入为in_ch
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

        self.head2 = _DAHead2(in_ch * 2, **kwargs)

    def forward(self, x1, x2, x3):  # x1、x2、x3 分别是合并流 RGB流和噪声流
        x1 = self.up(x1)  # 先进行上采样
        diffX = x2.size()[2] - x1.size()[2]  # diffX 和 diffY 计算了两个特征图在高度和宽度上的尺寸差异
        diffY = x2.size()[3] - x1.size()[3]  # 0-batch 1-channel 2-height 3-weight

        x1 = F.pad(x1, (diffY, 0,  # 在左右上下四个方向填充0
                        diffX, 0))

        x = torch.cat([x2, x3], dim=1)  # 拼接RGB和Noise
        # x = self.head2(x)                                               # 纬度变为1/2
        x = self.relu(self.head2(torch.cat([x1, x], dim=1)))  # 在channel纬度进行拼接 纬度1/2
        # x = self.relu(torch.cat([x1, x], dim=1))                         # 在channel纬度进行拼接

        # the first ring conv
        ft1 = self.conv(x)  # 第一次前向传播
        r1 = self.relu(self.res_conv(x) + ft1)  # 残差连接
        # the second ring conv
        ft2 = self.res_conv_back(r1)  # 残差反馈
        x = torch.mul(1 + torch.sigmoid(ft2), x)  # 这个x是原来拼接的输入 等于其实拼接了两次  # 把F.sigmoid 换成了 torch.sigmoid
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up2(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, **kwargs):
        super(RRU_up2, self).__init__()
        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 转置卷积和组归一化
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.GroupNorm(32, in_ch // 2))

        self.conv = RRU_double_conv(in_ch, out_ch)  # 这里是拼接上原来的特征图后所以输入为in_ch
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

        self.head2 = _DAHead2(in_ch, **kwargs)
        self.head3 = _DAHead3(in_ch, **kwargs)

    def forward(self, x1, x2):  # x1、x2、x3 分别是合并流 RGB流和噪声流
        x1 = self.up(x1)  # 先进行上采样
        diffX = x2.size()[2] - x1.size()[2]  # diffX 和 diffY 计算了两个特征图在高度和宽度上的尺寸差异
        diffY = x2.size()[3] - x1.size()[3]  # 0-batch 1-channel 2-height 3-weight

        x1 = F.pad(x1, (diffY, 0,  # 在左右上下四个方向填充0
                        diffX, 0))

        # x = torch.cat([x2, x3], dim=1)
        # x = self.head2(x)                                             # 使用DA2模块 channel变为原来1/2
        # x = self.relu(self.head3(torch.cat([x1, x], dim=1)))          # 在channel纬度进行拼接
        x = self.relu(torch.cat([x1, x2], dim=1))  # 在channel纬度进行拼接

        # the first ring conv
        ft1 = self.conv(x)  # 第一次前向传播
        r1 = self.relu(self.res_conv(x) + ft1)  # 残差连接
        # the second ring conv
        ft2 = self.res_conv_back(r1)  # 残差反馈
        x = torch.mul(1 + torch.sigmoid(ft2), x)  # 这个x是原来拼接的输入 等于其实拼接了两次  # 把F.sigmoid 换成了 torch.sigmoid
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up3(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, **kwargs):
        super(RRU_up3, self).__init__()
        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 转置卷积和组归一化
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.GroupNorm(32, in_ch // 2))
            # nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2),          # 由于这个时候的in_ch纬度就是x1的纬度 所以不用//2
            # nn.GroupNorm(32, in_ch))

        self.conv = RRU_double_conv(in_ch, out_ch)  # 这里是拼接上原来的特征图后所以输入为in_ch
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x1, x2, x3):  # x1、x2、x3 分别是合并流 RGB流和噪声流
        x1 = self.up(x1)  # 先进行上采样
        diffX = x2.size()[2] - x1.size()[2]  # diffX 和 diffY 计算了两个特征图在高度和宽度上的尺寸差异
        diffY = x2.size()[3] - x1.size()[3]  # 0-batch 1-channel 2-height 3-weight

        x1 = F.pad(x1, (diffY, 0,  # 在左右上下四个方向填充0
                        diffX, 0))

        x = torch.cat([x2, x3], dim=1)  # 拼接RGB和Noise
        # x = self.head2(x)  # 纬度变为1/2
        # x = self.relu(self.head3(torch.cat([x1, x], dim=1)))  # 在channel纬度进行拼接 纬度不变
        x = self.relu(torch.cat([x1, x], dim=1))  # 在channel纬度进行拼接

        # the first ring conv
        ft1 = self.conv(x)  # 第一次前向传播
        r1 = self.relu(self.res_conv(x) + ft1)  # 残差连接
        # the second ring conv
        ft2 = self.res_conv_back(r1)  # 残差反馈
        x = torch.mul(1 + torch.sigmoid(ft2), x)  # 这个x是原来拼接的输入 等于其实拼接了两次  # 把F.sigmoid 换成了 torch.sigmoid
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


# !!!!!!!!!!!! Universal functions !!!!!!!!!!!!

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


########################################
'''DA'''


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


# 最底层的RGB流和Noise融合 256+256->256  纬度变为1/2
class _DAHead(nn.Module):
    def __init__(self, in_channels, aux=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 **kwargs):  # in_channels=4096、nclass=1、aux =True
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 2
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 纬度变为256
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c  # 元素相加

        return feat_fusion


# 其他的RGB和Noise融合(记为fusion_R_N) 纬度变为原来的一半
class _DAHead2(nn.Module):
    def __init__(self, in_channels, aux=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 **kwargs):  # in_channels=4096、nclass=1、aux =True
        super(_DAHead2, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 2
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c  # 元素相加

        return feat_fusion


# fusion_R_N 和 上采样的结果融合 纬度不变
class _DAHead3(nn.Module):
    def __init__(self, in_channels, aux=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 **kwargs):  # in_channels=4096、nclass=1、aux =True
        super(_DAHead3, self).__init__()
        self.aux = aux
        inter_channels = in_channels
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c  # 元素相加

        return feat_fusion

# 融合ASPP的几个feature map
class _DAHead4(nn.Module):
    def __init__(self, in_channels, aux=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 **kwargs):  # in_channels=4096、nclass=1、aux =True
        super(_DAHead4, self).__init__()
        self.aux = aux
        inter_channels = 512
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 纬度变为256
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c  # 元素相加

        return feat_fusion


# 下面都是 ASPP
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return res