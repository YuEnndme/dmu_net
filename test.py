# 37550用来测试
# -*- coding = utf-8 -*-
# @Time : 2024/5/17 19:16
# @Author : Yu
# @File : test.py
# @Software: PyCharm
from mertic import metric_net
from unet.unet_model import *
from utils import *
from utils.data_vis import plot_img_and_mask
from utils.srm import SRM

def test_net(net,
             gpu=False,                        # 是否使用GPU加速
             img_scale=1,                      # 图像缩放比例
             test_percent=1,
             dataset=None):

    ids = split_ids2(get_ids(dir_img))                      # 获取图像的ID  dir_img就是数据集的地址
    iddataset = split_ft_test(ids, test_percent)           # 将数据集分为微调和测试 如果不微调则把 test_percent = 1
    print(len(iddataset['test']))

    ft = get_imgs_and_masks_test2(iddataset['ft'], dir_img, dir_mask, img_scale, dataset)  # 获取微调的图像和mask并进行相应的处理 data_CASIA   2-是用来裁剪RGB类型的掩码的
    test = get_imgs_and_masks_test2(iddataset['test'], dir_img, dir_mask, img_scale, dataset)  # 获取测试集的图像和mask并进行相应的处理

    net.eval()
    metric_net(net, test, gpu)


if __name__ == "__main__":
    scale, mask_threshold, gpu,  viz, no_save = 0.35, 0.5, True, False, False
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    dataset = 'CASIA'
    network = 'Ringed_Res_Unet'

    model = './result/logs/CASIA/Ringed_Res_Unet/CASIA-[val_dice]-0.9593-[train_loss]-0.0167.pkl'

    '''本地测试时的路径'''
    # dir_img = '/Users/a1234/Desktop/output/imgs/'
    # dir_mask = '/Users/a1234/Desktop/output/masks/'

    '''gpu训练时的路径'''
    # CASIA1.0
    # dir_img = '../test_data/CASIA1.0/SP/forgery/'
    # dir_mask = '../test_data/CASIA1.0/SP/mask/'

    # Columbia
    # dir_img = '../test_data/Columbia/DRIVE_Columbia/val/images/'
    # dir_mask = '../test_data/Columbia/DRIVE_Columbia/val/mask/'

    # in_the_wild
    # dir_img = '../in_the_wild/label_in_wild/images/'
    # dir_mask = '../in_the_wild/label_in_wild/masks/'

    # NIST16
    # dir_img = '../NIST16/splice/im_aug/'
    # dir_mask = '../NIST16/splice/gt_aug/'

    # Realistic
    dir_img = '../Realistic/im_aug/'
    dir_mask = '../Realistic/gt_aug/'

    if network == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif network == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    else:
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if gpu:
        net.cuda()
        net.load_state_dict(torch.load(model))

        # 尝试多GPU解决显存问题
        # net = torch.nn.DataParallel(net).cuda()  # 使用DataParallel包装模型
        # state_dict = torch.load(model)
        # net.load_state_dict(state_dict)  # 直接加载权重

    else:
        net.cpu()
        net.load_state_dict(torch.load(model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    test_net(net=net,
             gpu=gpu,
             img_scale=scale,
             dataset=dataset)
