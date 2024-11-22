# 对应着18563     37546是从18536迁移过来的
import torch.backends.cudnn as cudnn
from torch import optim
import zipfile
from itertools import chain
from sklearn import metrics

from eval import eval_net
from unet.unet_model import *
from utils import *
from utils.srm import SRM

import matplotlib.pyplot as plt
import time

def save_zipfile(zip_file_path, files_to_zip):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in files_to_zip:
            zipf.write(file_path)


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

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-2,                          # 梯度下降学习律
              val_percent=0.05,                 # 验证集所占比例
              save_cp=True,                     # 是否保存检查点（模型权重）
              gpu=False,                        # 是否使用GPU加速
              img_scale=1,                      # 图像缩放比例
              dataset=None):                    # 数据集的名称            'CASIA'
    # training images are square
    ids = split_ids2(get_ids(dir_img))                       # 获取图像的ID  dir_img就是数据集的地址    换成了split_ids2 只返回(name,0)
    iddataset = split_train_val(ids, val_percent)           # 将数据集氛围训练集和验证集

    # ids2 = split_ids2(get_ids(dir_img2))                    # 获取图像的ID  dir_img就是数据集的地址
    # iddataset2 = split_train_val(ids2, val_percent)         # 将数据集氛围训练集和验证集
    #
    # ids3 = split_ids2(get_ids(dir_img3))
    # iddataset3 = split_train_val(ids3, val_percent)


    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs,
               batch_size,
               lr,
               len(iddataset['train']),
               len(iddataset['val']),
               str(save_cp),
               str(gpu)))

    N_train = len(iddataset['train'])                                                 # 训练集大小
    optimizer = optim.Adam(net.parameters(),                                            # Adam优化器
                           lr=lr,
                           weight_decay=0)
    criterion = nn.BCELoss()                                    # 二分类交叉熵损失函数

    Train_loss = []                                            # 记录训练损失
    Valida_dice = []                                            # 记录验证Dice系数
    EPOCH = []                                                  # 记录每个epoch

    for epoch in range(epochs):            # 遍历每个epoch
        # ids = split_ids2(get_ids(dir_img))
        # iddataset = split_train_val(ids, val_percent)  # 将数据集氛围训练集和验证集（重新加上这一条 保证每次划分的训练机和验证集不同）

        net.train()                        # 将模型设置为训练模式。在训练过程中，通常会使用该方法来启用训练相关的功能，例如启用 Dropout 或 Batch Normalization 层的训练模式，以及设置其他训练相关的标志。

        start_epoch = time.time()          # 记录开始时间
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # reset the generators
        train = get_imgs_and_masks_train(iddataset['train'], dir_img, dir_mask, img_scale, dataset)           # 获取训练集的图像和mask并进行相应的处理 data_CASIA
        val = get_imgs_and_masks_val(iddataset['val'], dir_img, dir_mask, img_scale, dataset)               # 获取验证集的图像和mask并进行相应的处理

        # train2 = get_imgs_and_masks2(iddataset2['train'], dir_img2, dir_mask2, img_scale, dataset)      # 获取训练集的图像和mask并进行相应的处理 pscc->splice
        # val2 = get_imgs_and_masks2(iddataset2['val'], dir_img2, dir_mask2, img_scale, dataset)          # 获取验证集的图像和mask并进行相应的处理
        #
        # train3 = get_imgs_and_masks3(iddataset3['train'], dir_img3, dir_mask3, img_scale, dataset)      # 获取训练集的图像和mask并进行相应的处理 pscc->splice_randmask
        # val3 = get_imgs_and_masks3(iddataset3['val'], dir_img3, dir_mask3, img_scale, dataset)          # 获取验证集的图像和mask并进行相应的处理

        # train_total = chain(train, train2)
        # val_total = chain(val, val2)
        # train_total = train
        # val_total = val


        '''
        for i, (img, mask) in enumerate(val_total, 1):  # 从1开始编号  使用前要将图片格式变换由CHW变换成HWC
            # 创建一个包含两个子图的图形窗口
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # 显示图像
            axes[0].imshow(img)
            axes[0].set_title(f"Image {i}")

            # 显示掩码
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f"Mask {i}")

            # 关闭坐标轴
            axes[0].axis('off')
            axes[1].axis('off')

            # 调整布局
            plt.tight_layout()

            # 显示图形窗口
            plt.show()
        '''

        epoch_loss = 0

        # for i, b in enumerate(batch2(train, train2,  batch_size)):                                    # 对每个batch进行训练  train是一个包含图片和掩码的zip
        for i, b in enumerate(batch(train, batch_size)):
            start_batch = time.time()                                                       # 记录batch开始时间
            imgs = np.array([i[0] for i in b]).astype(np.float32)                           # 获取图像并转换为float32类型 i[0]代表图像元素
            true_masks = np.array([i[1] for i in b]).astype(np.float32) / 255.              # 获取掩码并转换为float32类型并缩放到[0,1]之间  i[1]代表掩码

            imgs = torch.from_numpy(imgs)                                                   # 转换为PyTorch张量
            true_masks = torch.from_numpy(true_masks)                                       # 转换为PyTorch张量

            if gpu:                                                                         # 是否启用GPU训练
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            optimizer.zero_grad()                                                           # 梯度清零
            imgs = space_transfer(imgs)                                                     # 将图像转为RGB+Nosie
            masks_pred = net(imgs)                                                          # 前向传播获取预测结果
            masks_probs = torch.sigmoid(masks_pred)                                         # 对结果进行sigmoid处理
            masks_probs_flat = masks_probs.view(-1)                                         # 将预测结果展平
            true_masks_flat = true_masks.view(-1)                                           # 将真实标签展平

            loss = criterion(masks_probs_flat, true_masks_flat)                             # 计算损失(交叉熵损失)

            '''
                            i * batch_size / N_train 进度
                            loss  该batch损失                  
                            time.time()-start_batch)  该batch时间
             '''
            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / N_train, loss, time.time()-start_batch))

            epoch_loss += loss.item()

            loss.backward()                         # 计算当前批次的损失相对于模型参数的梯度
            optimizer.step()                        # 根据优化器的设置，使用损失函数的梯度来更新模型参数

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))           # 平均每个batch的损失

        # 以下是验证集
        # validate the performance of the model
        net.eval()                  # 将模型设置为评估模式。在评估或推断过程中，通常会使用该方法来禁用 Dropout 或 Batch Normalization 层的训练模式，并设置其他评估相关的标志，以确保评估过程的稳定性和一致性。

        val_dice = eval_net(net, val, gpu)                          # 评估Dice系数
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))

        Train_loss.append(epoch_loss / i)                   # Train_loss是一个列表，用于记录每个epoch的平均训练损失。
        Valida_dice.append(val_dice)                        # 是一个列表，用于记录每个epoch在验证集上计算得到的Dice系数。
        EPOCH.append(epoch)                                 # 是一个列表，用于记录每个epoch的编号。

        fig = plt.figure()

        plt.title('Training Process')                       # 设置了图表的标题为'Training Process'
        plt.xlabel('epoch')                                 # 横坐标为epoch
        plt.ylabel('value')                                 # 纵坐标为value
        l1, = plt.plot(EPOCH, Train_loss, c='red')          # 创建了一个图例对象l1，用红色绘制了训练损失随epoch的变化曲线。
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')        # 创建了另一个图例对象l2，用蓝色绘制了验证Dice系数随epoch的变化曲线。

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_dice'], loc='best')
        # 我后加的
        if not os.path.exists(dir_logs):
            os.makedirs(dir_logs)

        plt.savefig(dir_logs + 'Training Process for lr-{}.png'.format(lr), dpi=600)

        torch.save(net.state_dict(),
                   dir_logs + '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}.pkl'.format(dataset, val_dice, epoch_loss / i))
        print('Spend time: {:.3f}s'.format(time.time() - start_epoch))
        print()


if __name__ == '__main__':
    epochs, batchsize, scale, gpu = 50, 8, 1, True
    lr = 1e-3
    ft = False
    dataset = 'CASIA'

    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    model = 'Ringed_Res_Unet'

    # dir_img = '../data_{}/train/tam/'.format(dataset)
    # dir_mask = '../data_{}/train/mask/'.format(dataset)
    '''服务器训练时的路径'''
    dir_img = '../data_CASIA/tam/'                                      # 原来data_CASIA的路径
    dir_mask = '../data_CASIA/mask/'
    dir_logs = './result/logs/{}/{}/'.format(dataset, model)

    # dir_img2 = '../splice/fake/'                                        # PSCC数据集合的splice
    # dir_mask2 = '../splice/mask/'
    #
    # dir_img3 = '../splice_randmask/fake/'                                # PSCC数据集合的splice_randmask
    # dir_mask3 = '../splice_randmask/mask/'


    '''本地训练时的路径'''
    # dir_img = '/Users/a1234/Desktop/RRU_simplifyData/tam2/'                         # 原来data_CASIA的路径
    # dir_mask = '/Users/a1234/Desktop/RRU_simplifyData/mask2/'
    # dir_logs = './result/logs/{}/{}/'.format(dataset, model)
    #
    # dir_img2 = '/Users/a1234/Desktop/PSCC_simplifyData/splicing/fake/'              # PSCC数据集合的splice
    # dir_mask2 = '/Users/a1234/Desktop/PSCC_simplifyData/splicing/mask/'
    #
    # dir_img3 = '/Users/a1234/Desktop/PSCC_simplifyData/splicing_randmask/fake/'     # PSCC数据集合的splice_randmask
    # dir_mask3 = '/Users/a1234/Desktop/PSCC_simplifyData/splicing_randmask/mask/'

    if model == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif model == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif model == 'Ringed_Res_Unet':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if ft:                           # 根据名字是Fine tuning模型 但是也可以改成先前训练好的 然后做一个继续训练
        fine_tuning_model = './result/logs/{}/{}/CASIA-[val_dice]-0.9545-[train_loss]-0.0215.pkl'.format(dataset, model)
        net.load_state_dict(torch.load(fine_tuning_model))
        print('Model loaded from {}'.format(fine_tuning_model))  #

    if gpu:                                                                 # 如果使用GPU
        net.cuda()                                                          # 将网络移到GPU上
        cudnn.benchmark = True  # faster convolutions, but more memory      # 提高卷积操作的速度，但会增加内存占用

    train_net(net=net,                                                      # 开始训练模型
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              gpu=gpu,
              img_scale=scale,
              dataset=dataset)
