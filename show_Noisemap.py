# -*- coding = utf-8 -*-
# @Time : 2024/7/13 11:56
# @Author : Yu
# @File : show_Noisemap.py
# @Software: PyCharm
from tqdm import tqdm

from unet.unet_model import *
from utils import *
from utils.data_vis import plot_img_and_mask
from utils.srm import SRM

def truncate_2(x):
    neg = ((x + 2) + abs(x + 2)) / 2 - 2
    return -(-neg + 2 + abs(- neg + 2)) / 2 + 2


def space_transfer(imgs):
    assert imgs.dim() == 4

    # imgs_hed = rgb_to_hed(imgs)
    SRM_kernel = SRM()
    imgs_srm = SRM_kernel(imgs)
    imgs_srm = truncate_2(imgs_srm)

    return [imgs, imgs_srm]


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def show_noise(img):
    img = resize_and_crop(img, scale=1).astype(np.float32)  # 由于scale=1 所以图像的大小不会变
    img = np.transpose(normalize(img), (2, 0, 1))  # 归一化和 变换纬度
    img = torch.from_numpy(img).unsqueeze(dim=0)  # 将图像转为pytorch张量 并增加batch纬度
    img, img_srm= space_transfer(img)

    # 转换回NumPy数组
    img = img.squeeze().permute(1, 2, 0).numpy()
    img_srm = img_srm.squeeze().permute(1, 2, 0).numpy()

    # 显示原始图像和SRM处理后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("SRM Processed Image")
    plt.imshow(img_srm)
    plt.axis('off')

    plt.show()

    srm_img = Image.fromarray((img_srm * 255).astype(np.uint8))
    srm_img.save('./nosie.jpg')
    # 不用的时候记得将SRM.cuda()改回来


if __name__ == "__main__":
    img = Image.open('/Users/a1234/Desktop/Test/CASIA1.0/SP/forgery/Sp_D_NRN_A_pla0094_ani0025_0411.jpg')
    show_noise(img)



