import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

from dice_loss import dice_coeff
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

    return list([imgs, imgs_srm])

def eval_net(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    Score = 0
    '''
         i 是一个整数，代表当前批次的索引号（从 0 开始）。
         b 是当前批次的数据。这个批次是 batch() 函数的输出，其中包含了训练数据的元组（图像和对应的掩码）。
         i[0] 和 i[1] 分别表示图像和mask 因为get_imgs_and_masks()的返回用zip()函数将图像数据和掩码数据组合成元组的集合
       '''
    for i, b in enumerate(dataset):
        img = b[0].astype(np.float32)                       # 获取图像并转换为float32类型、i[0]代表图像
        true_mask = b[1].astype(np.float32)/255             # 获取掩码并转换为float32类型并缩放到[0,1]之间  i[1]代表掩码

        '''
            将 numpy 数组转换为 PyTorch 张量，并添加一个批次维度

            unsqueeze(0) 是 PyTorch 中用于在张量的维度上添加一个额外的维度的操作。在这个上下文中，unsqueeze(0) 
             被用来为图像和真实掩码添加一个批次(batch)维度，因为神经网络的输入通常需要具有批次维度。这个操作将原始的二维图像张量（对于单个样本）转换为三维张量，
             其中第一个维度表示批次大小。

            训练时不这样做：在训练过程中，通常会以批次的形式输入数据，因此不需要在训练时使用 unsqueeze(0) 添加额外的维度，因为数据已经以批次的形式传递给网络。
            但在验证阶段，如果你处理的是单个图像或一组图像，而不是以批次的形式传递给网络，那么就需要在验证时手动添加批次维度，以保持输入数据的形状一致性。         
        '''
        img = torch.from_numpy(img).unsqueeze(0)

        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        img = space_transfer(img)

        mask_pred = net(img)[0]                                     # 训练过程中没有用0是因为 训练时是一个batch 这里只是一张图片
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()        # 阈值设置为0.5 这里可以调参

        '''
        true_mask2 = true_mask.int()
        true_masks_flat2 = true_mask2.cpu().clone().cpu().numpy().reshape(-1)
        mask_pred2 = (torch.sigmoid(mask_pred) > 0.5).int()                           # 阈值设置为0.5 这里可以调参
        masks_probs_flat2 = torch.unsqueeze(mask_pred2, 0)
        masks_probs_flat2 = masks_probs_flat2.cpu().clone().cpu().detach().numpy().reshape(-1)
        print(len(true_masks_flat2))
        print(len(masks_probs_flat2))
        F1 = metrics.f1_score(true_masks_flat2, masks_probs_flat2, average='macro', labels=[0, 1])
        AUC = 0
        try:
            AUC = metrics.roc_auc_score(true_masks_flat2, masks_probs_flat2)
        except ValueError:
            pass
        F1_total += F1
        Auc_total += AUC
        '''
        tot += dice_coeff(mask_pred, true_mask).item()              # 计算Dice系数

    return tot / i
