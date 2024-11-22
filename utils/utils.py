import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    # this function make img to square due to network architecture of UNet
    h = img.shape[0]        # 传入的图像其实是384*256 的图像 转换到HWC纬度 分别就是H:256  W:384
    if pos == 0:
        return img[:, :h]       # 截取左半个正方形
    else:
        return img[:, -h:]      # 截取右半个正方形


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):      # 将图像进行缩放和裁剪
    w = pilimg.size[0]                              # 0-weight
    h = pilimg.size[1]                              # 1-height
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))      # 四个参数依次是1、左边界的x坐标   2、上边界的y坐标。  3、右边界的x坐标  4、下边界的y坐标
    return np.array(img, dtype=np.float32)


def resize_and_crop2 (pilimg, scale=0.5, final_height=None):      # 将图像进行缩放和裁剪 测试的时候用
    w = pilimg.size[0]                              # 0-weight
    h = pilimg.size[1]                              # 1-height
    newW = int(w * scale)
    newH = int(h * scale)

    if 1000000 > h*w > 500000:
        newW = int(newW * 0.5)
        newH = int(newH * 0.5)
    elif h*w > 1000000:
        newW = int(newW * 0.2)
        newH = int(newH * 0.2)
    # elif 1000 < w < 2000 and 1000 < h < 2000:
    #     newW = int(newW * 0.5)
    #     newH = int(newH * 0.5)
    #
    # elif (2000 < w < 3000 and 1000 < h < 3000) or (2000 < h < 3000 and 1000 < w < 3000):
    #     newW = int(newW * 0.3)
    #     newH = int(newH * 0.3)
    #
    # elif 3000 <= w and 3000 <= h:
    #     newW = int(newW * 0.1)
    #     newH = int(newH * 0.1)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))      # 四个参数依次是1、左边界的x坐标   2、上边界的y坐标。  3、右边界的x坐标  4、下边界的y坐标
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):                    # 是 Python 内置函数，用于将一个可迭代对象组合为一个索引序列，同时返回索引和值  在循环中使用 enumerate 可以同时获取元素的值和索引
        b.append(t)                                     # i是索引  t是值
        if (i + 1) % batch_size == 0:
            yield b                                     # 生成当前批次
            b = []                                      # 清空批次列表，准备存储下一批次的元素

    if len(b) > 0:                                      # 如果还有剩余的元素未生成批次
        yield b                                         # 生成最后一个不足 batch_size 的批次

def batch2(iterable, iterable2, batch_size):
    """Yields lists by batch"""
    b = []
    # 处理SAN
    for i, t in enumerate(iterable):                    # 是 Python 内置函数，用于将一个可迭代对象组合为一个索引序列，同时返回索引和值  在循环中使用 enumerate 可以同时获取元素的值和索引
        b.append(t)                                     # i是索引  t是值
        if (i + 1) % batch_size == 0:
            yield b                                     # 生成当前批次
            b = []                                      # 清空批次列表，准备存储下一批次的元素

    if len(b) > 0:                                      # 如果还有剩余的元素未生成批次
        yield b                                         # 生成最后一个不足 batch_size 的批次
        b = []

    # 处理pscc—>splice
    for i2, t2 in enumerate(iterable2):                    # 是 Python 内置函数，用于将一个可迭代对象组合为一个索引序列，同时返回索引和值  在循环中使用 enumerate 可以同时获取元素的值和索引
        b.append(t2)                                     # i是索引  t是值
        if (i2 + 1) % batch_size == 0:
            yield b                                     # 生成当前批次
            b = []                                      # 清空批次列表，准备存储下一批次的元素

    if len(b) > 0:                                      # 如果还有剩余的元素未生成批次
        yield b                                         # 生成最后一个不足 batch_size 的批次



'''
[n: ]意思是取列表中第n项（从0数起）到最后一项，包含第0项和最后一项
[ :n]意思是取列表中第0项（从0数起）到第n项，包含第0项，不包含第n项。
[-n: ]意思是取列表中倒数第n项（从1数起）到最后一项，包含第n项，包含最后一项。
[:-n]意思是取列表中第0项（从0数起）到倒数第n项（从1数起），包含第0项，不包含倒数第n项。
'''
def split_train_val(dataset, val_percent=0.05):                  # dataset是传入的ids
    dataset = list(dataset)
    length = len(dataset)                                        # 获取训练机的总长度
    n = int(length * val_percent)
    random.shuffle(dataset)                                      # 打乱数据
    return {'train': dataset[:-n], 'val': dataset[-n:]}          # train从[0,length-n-1] val从[length-n,length-1]

def split_ft_test(dataset, val_percent=0.05):                  # dataset是传入的ids
    dataset = list(dataset)
    length = len(dataset)                                        # 获取训练机的总长度
    n = int(length * val_percent)
    random.shuffle(dataset)                                      # 打乱数据
    return {'ft': dataset[:-n], 'test': dataset[-n:]}          # train从[0,length-n-1] val从[length-n,length-1]


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()