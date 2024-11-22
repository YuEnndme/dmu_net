#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
from random import randrange

import numpy as np
from PIL import Image
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw, resize_and_crop2


def get_ids(dir):
    """
       Returns a list of the ids in the directory
       dir = "****./data_CASIA/train/tam/"
       get rid of data format (for example: abandon.jpg)
       从目录中获取文件名列表（不含后缀）例如1.jpg返回1
    """
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    """
        id is name of image, i is pos of image 
        (`pos` be used to define left or right square in function `get_square`, 
         pos == 0 is left,
         pos == 1 is right)
         
         eg：
         ids = ['image1', 'image2', 'image3']
         结果为  (0,1在后面用来截取图像的左侧和右侧 使图像成为一个正方形) 盲猜后面会根据名字组合一起
         ('image1', 0)
         ('image1', 1)
         ('image2', 0)
         ('image2', 1)
         ('image3', 0)
         ('image3', 1)     
    """
    return ((id, i) for id in ids for i in range(n))
    # return ((id, i) for i in range(n) for id in ids) # this order is wrong

def split_ids2(ids, n=1):
    return ((id, i) for id in ids for i in range(n))


def to_cropped_imgs_train(ids, dir, suffix, scale):                              # 训练集进行数据增强
    """
        From a list of tuples, returns the correct cropped img
        接受一组图像 ID 和相关参数，然后返回经过裁剪处理的图像数据
        ids 是一个包含图像 ID 和位置信息的列表或元组。
        dir 是图像文件所在的目录路径。
        suffix 是图像文件的后缀格式（例如 .jpg, _mask.png 等）。
        scale 是图像的缩放比例。

        使用 Image.open(dir + id + suffix) 打开对应的图像文件，得到图像的原始对象。
        调用 resize_and_crop() 函数对图像进行裁剪和缩放操作，返回处理后的图像对象。
        使用 get_square() 函数获取裁剪后的图像的左侧或右侧部分（根据位置信息 pos 确定）。
        使用 yield 将每个裁剪后的图像返回，以便在调用该函数的地方逐个获取图像数据。
        由于使用了 yield 关键字，它以生成器（generator）的形式逐个返回处理后的图像，而不是一次性返回所有图像数据。
    """
    # params suffix: is data format
    k = 0
    for id, pos in ids:
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)        # 调用了resize_and_crop()函数
        im = Image.open(dir + id + suffix)
        flag = k % 4                                                            # 因为ids的顺序是一样的 所以用一个变量k可以保证图像和掩码做了相同的变换
        im = data_aug(im, flag)                                                 # 而且 每个epoch中ids是不同的，所以等于再做随机变换
        k = k+1
        im = resize_and_crop(im, scale)
        yield im                                              # 调用了get square()函数

def to_cropped_imgs_val(ids, dir, suffix, scale):                                   # 验证集不需要了
    # params suffix: is data format
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)        # 调用了resize_and_crop()函数
        yield im                                              # 调用了get square()函数

# 裁剪测试集合的图片和掩码 不用分为左右两张图像
def to_cropped_imgs_test(ids, dir, suffix, scale):
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)        # 调用了resize_and_crop()函数
        yield im

# 裁剪测试集RGB类型的掩码
def to_cropped_imgs_test2(ids, dir, suffix, scale):
    # params suffix: is data format
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix).convert('L'), scale=scale)        # 调用了resize_and_crop()函数 与to_cropped_imgs不同在于加了一个convert('L')
        yield im


def get_imgs_and_masks_train(ids, dir_img, dir_mask, scale, dataset):
    """
        Return all the couples (img, mask)
        dataset = "CASIA"
        dir_img = "../data_CASIA/train/tam/"
        dir_mask = "../data_CASIA/train/mask/"
        scale = 1
        ids = 'train': [('10005', 0), ('10005', 1), ('10004', 1), ('10003', 0), ('10001', 0), ('10003', 1), ('10010', 0),
                        ('10009', 1), ('10002', 0), ('10004', 0), ('10009', 0), ('10002', 1), ('10006', 1), ('10010', 1),
                        ('.DS_S', 0)],
            'val': [('10001', 1), ('10006', 0),
    """
    if dataset == 'CASIA':
        format = 'jpg'
    elif dataset == 'COLUMB':
        format = 'jpg'

    imgs = to_cropped_imgs_train(ids, dir_img, '.{}'.format(format), scale)       # 调用to_cropped_imgs()函数 裁剪图片

    # need to transform from HWC to CHW
    # 将图像数据从 HWC 格式（Height、Width、Channels）转换为 CHW 格式（Channels、Height、Width）。
    imgs_switched = map(hwc_to_chw, imgs)                   # 调用了hwc_to_chw()
    imgs_normalized = map(normalize, imgs_switched)         # 对图像数据进行归一化处理。 调用了normalize

    masks = to_cropped_imgs_train(ids, dir_mask, '_mask.png', scale)  # 调用to_cropped_imgs()函数 裁剪GT真实掩码

    return zip(imgs_normalized, masks)


def get_imgs_and_masks_val(ids, dir_img, dir_mask, scale, dataset):
    """
        Return all the couples (img, mask)
        dataset = "CASIA"
        dir_img = "../data_CASIA/train/tam/"
        dir_mask = "../data_CASIA/train/mask/"
        scale = 1
        ids = 'train': [('10005', 0), ('10005', 1), ('10004', 1), ('10003', 0), ('10001', 0), ('10003', 1), ('10010', 0),
                        ('10009', 1), ('10002', 0), ('10004', 0), ('10009', 0), ('10002', 1), ('10006', 1), ('10010', 1),
                        ('.DS_S', 0)],
            'val': [('10001', 1), ('10006', 0),
    """
    if dataset == 'CASIA':
        format = 'jpg'
    elif dataset == 'COLUMB':
        format = 'jpg'

    imgs = to_cropped_imgs_val(ids, dir_img, '.{}'.format(format), scale)       # 调用to_cropped_imgs()函数 裁剪图片

    # need to transform from HWC to CHW
    # 将图像数据从 HWC 格式（Height、Width、Channels）转换为 CHW 格式（Channels、Height、Width）。
    imgs_switched = map(hwc_to_chw, imgs)                   # 调用了hwc_to_chw()
    imgs_normalized = map(normalize, imgs_switched)         # 对图像数据进行归一化处理。 调用了normalize

    masks = to_cropped_imgs_val(ids, dir_mask, '_mask.png', scale)  # 调用to_cropped_imgs()函数 裁剪GT真实掩码

    return zip(imgs_normalized, masks)

def get_imgs_and_masks2(ids, dir_img, dir_mask, scale, dataset):

    if dataset == 'CASIA':
        format = 'jpg'
    elif dataset == 'COLUMB':
        format = 'jpg'

    imgs = to_cropped_imgs_train(ids, dir_img, '.{}'.format(format), scale)       # 调用to_cropped_imgs()函数 裁剪图片

    # need to transform from HWC to CHW
    # 将图像数据从 HWC 格式（Height、Width、Channels）转换为 CHW 格式（Channels、Height、Width）。
    imgs_switched = map(hwc_to_chw, imgs)                   # 调用了hwc_to_chw()
    imgs_normalized = map(normalize, imgs_switched)         # 对图像数据进行归一化处理。 调用了normalize

    masks = to_cropped_imgs_train(ids, dir_mask, '.png', scale)      # 调用to_cropped_imgs()函数 裁剪GT真实掩码

    return zip(imgs_normalized, masks)

def get_imgs_and_masks3(ids, dir_img, dir_mask, scale, dataset):

    if dataset == 'CASIA':
        format = 'tif'
    elif dataset == 'COLUMB':
        format = 'tif'

    imgs = to_cropped_imgs_train(ids, dir_img, '.{}'.format(format), scale)       # 调用to_cropped_imgs()函数 裁剪图片

    # need to transform from HWC to CHW
    # 将图像数据从 HWC 格式（Height、Width、Channels）转换为 CHW 格式（Channels、Height、Width）。
    imgs_switched = map(hwc_to_chw, imgs)                   # 调用了hwc_to_chw()
    imgs_normalized = map(normalize, imgs_switched)         # 对图像数据进行归一化处理。 调用了normalize

    masks = to_cropped_imgs_train(ids, dir_mask, '.png', scale)      # 调用to_cropped_imgs()函数 裁剪GT真实掩码

    return zip(imgs_normalized, masks)


def get_imgs_and_masks_test(ids, dir_img, dir_mask, scale, dataset):    # 获取测试集的图片和掩码
    if dataset == 'CASIA':
        format = 'tif'
    elif dataset == 'COLUMB':
        format = 'jpg'

    imgs = to_cropped_imgs_test(ids, dir_img, '.{}'.format(format), scale)  # 调用to_cropped_imgs_test()函数 裁剪图片

    # need to transform from HWC to CHW
    # 将图像数据从 HWC 格式（Height、Width、Channels）转换为 CHW 格式（Channels、Height、Width）。
    imgs_switched = map(hwc_to_chw, imgs)  # 调用了hwc_to_chw()
    imgs_normalized = map(normalize, imgs_switched)  # 对图像数据进行归一化处理。 调用了normalize

    masks = to_cropped_imgs_test(ids, dir_mask, '_gt.png', scale)  # 调用to_cropped_imgs()函数 裁剪GT真实掩码

    return zip(imgs_normalized, masks)


def get_imgs_and_masks_test2(ids, dir_img, dir_mask, scale, dataset):    # 获取测试集的图片和掩码（掩码为RGB）
    if dataset == 'CASIA':
        format = 'TIF'
    elif dataset == 'COLUMB':
        format = 'jpg'

    imgs = to_cropped_imgs_test(ids, dir_img, '.{}'.format(format), scale)  # 调用to_cropped_imgs_test()函数 裁剪图片

    # need to transform from HWC to CHW
    # 将图像数据从 HWC 格式（Height、Width、Channels）转换为 CHW 格式（Channels、Height、Width）。
    imgs_switched = map(hwc_to_chw, imgs)  # 调用了hwc_to_chw()
    imgs_normalized = map(normalize, imgs_switched)  # 对图像数据进行归一化处理。 调用了normalize

    masks = to_cropped_imgs_test2(ids, dir_mask, '.PNG', scale)  # 调用to_cropped_imgs()函数 裁剪GT真实掩码

    return zip(imgs_normalized, masks)


# 数据增强
def data_aug(img, data_aug_ind):
    # img = Image.fromarray(img)
    if data_aug_ind == 0:                                               # 原始图像，无任何变化。
        return img
    elif data_aug_ind == 1:                                             # 顺时针旋转 180 度。
        return img.rotate(180, expand=True)
    elif data_aug_ind == 2:                                             # 上下翻转。
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif data_aug_ind == 3:                                             # 顺时针旋转 180 度后，再进行上下翻转。
        return img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM)

    else:
        raise Exception('Data augmentation index is not applicable.')

# def get_full_img_and_mask(id, dir_img, dir_mask):
#     im = Image.open(dir_img + id + '.jpg')
#     mask = Image.open(dir_mask + id + '_mask.gif')
#     return np.array(im), np.array(mask)