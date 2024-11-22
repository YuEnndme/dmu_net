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

    return list([imgs, imgs_srm])

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_gpu=True):
    net.eval()

    img = resize_and_crop(full_img, scale=scale_factor).astype(np.float32)      # 由于scale=1 所以图像的大小不会变
    img = np.transpose(normalize(img), (2, 0, 1))                               # 归一化和 变换纬度
    img = torch.from_numpy(img).unsqueeze(dim=0)                                # 将图像转为pytorch张量 并增加batch纬度

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        # 判断是否有 Alpha 通道
        if img.shape[1] == 4:
            # 如果有 Alpha 通道，去除 Alpha 通道
            img = img[:, :3, :, :]
        img = space_transfer(img)
        mask = net(img)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()

    return mask > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    scale, mask_threshold, cpu,  viz, no_save = 1, 0.5, True, False, False
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    network = 'Ringed_Res_Unet'

    img = Image.open('/Users/a1234/Desktop/Test/CASIA1.0/SP/forgery/Sp_D_NNN_A_cha0085_ani0037_0313.jpg')
    model = './result/logs/CASIA/Ringed_Res_Unet/CASIA-[val_dice]-0.9549-[train_loss]-0.0221.pkl'

    if network == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif network == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    else:
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if not cpu:
        net.cuda()
        net.load_state_dict(torch.load(model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=scale,
                       out_threshold=mask_threshold,
                       use_gpu=not cpu)

    if viz:
        print("Visualizing results for image {}, close to continue ...".format(j))
        plot_img_and_mask(img, mask)

    if not no_save:
        result = mask_to_image(mask)

        if network == 'Unet':
            result.save('predict_u.png')
        elif network == 'Res_Unet':
            result.save('predict_ru.png')
        else:
            result.save('predict_rru2.png')