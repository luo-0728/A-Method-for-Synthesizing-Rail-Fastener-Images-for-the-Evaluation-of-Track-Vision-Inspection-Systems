import argparse
import logging
import os.path
import sys
import cv2
import time
from collections import OrderedDict
from skimage import io, metrics
import torchvision.utils as tvutils
import torch.nn.functional as F
import numpy as np
import torch
#from IPython import embed
import lpips
from torchvision.utils import save_image
import options as option
from models import create_model
import random
sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
failed_info = []
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    #dataset_dir = '/data/luowei/tiekeyuan/tiekeyuan/diff_results/new_data_deco_noatt/'
    dataset_dir = '/data/luowei/tiekeyuan/tiekeyuan/diff_results/loss_ceshi/'
    util.mkdir(dataset_dir)

    test_times = []

    for i, test_data in enumerate(test_loader):
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(img_name)

        #### input dataset_LQ
        LQ, MA = test_data["LQ"], test_data["MA"]
        GT = test_data["GT"]
        MA = (MA > 0.5).to(torch.float32)
        is_binary = torch.all((MA == 0) | (MA == 1))


        #print(GT.shape)
        B, C, H, W = GT.shape
        img_LR = LQ
        img_GT_1 = GT
        img_GT = GT
        img_MA = MA
        ma_np = MA.squeeze().cpu().numpy()  # shape: (H, W)
        rows, cols = np.where(ma_np > 0)
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("Mask中没有有效的非空白区域")
        x_min,x_max = cols.min(), cols.max()
        #x_max = x + w
        y_min,y_max = rows.min(), rows.max()
        #y_max = y + h
        #print(y_min, x_min,y_max, x_max)
        if max(0, x_max - 512)>min(x_min, W - 512):
            print(y_min, x_min,y_max, x_max)
            print(max(0, x_max - 512), min(x_min, W - 512))
        rnd_h = (max(0, y_max - 512) + min(y_min, H - 512)) // 2
        # 计算宽度方向的中间起始位置
        rnd_w = (max(0, x_max - 512) + min(x_min, W - 512)) // 2
        LQ = img_LR[:, :, rnd_h : rnd_h + 512, rnd_w : rnd_w + 512]

        GT = img_GT[:, :, rnd_h : rnd_h + 512, rnd_w : rnd_w + 512]
        MA = img_MA[:, :, rnd_h : rnd_h + 512, rnd_w : rnd_w + 512]



        #print(is_binary)
        latent_LQ, hidden = model.encode(LQ.to(device))
        latent_GT, hidden1 = model.encode(GT.to(device))
        #latent_mse = torch.mean((latent_LQ - latent_GT).pow(2)).item()
        #print(f"[Latent MSE]: {latent_mse:.6f}")
        noisy_state = sde.noise_state(latent_LQ)
        target_size = latent_LQ.size()[-2:]
        latent_MA = F.interpolate(MA.to(device), size=target_size, mode='bilinear', align_corners=False)
        # print(latent_LQ.shape)
        # print(latent_MA.shape)
        # if latent_MA != None:
        #     print("yes")
        #latent_GT = None
        state_recons = None
        model.feed_data(noisy_state, state_recons, latent_LQ, latent_GT, latent_MA)
        tic = time.time()
        model.test(sde, hidden, save_states=False)
        toc = time.time()
        test_times.append(toc - tic)
        print(f"[Time]: {toc - tic:.6f}")
        visuals = model.get_current_visuals(need_GT=False)
        SR_img = visuals["Output"][None, ...]
        # SR_img_single_channel = SR_img.mean(dim=1, keepdim=True)
        # print(SR_img_single_channel.size())
        # save_image(SR_img_single_channel, 'input_image.png')
        output = util.tensor2img(SR_img.squeeze())  # uint8
        # save_img_path1 = os.path.join(dataset_dir, img_name + "_1.png")
        # util.save_img(output, save_img_path1)
        # output = torch.from_numpy(output).float()
        MA = MA.squeeze().unsqueeze(-1)  # 形状变为(2021,2048,1)  
        LQ = LQ.squeeze().permute(1, 2, 0)

        MA = MA.numpy()
        LQ = LQ.numpy()
        if LQ.dtype == np.float32:
            LQ = (LQ * 255).astype(np.uint8)

        # 确保MA的数据类型是uint8
        if MA.dtype == np.float32:
            MA = (MA * 255).astype(np.uint8)
        if output.dtype != np.uint8:
            output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # save_img_path2 = os.path.join(dataset_dir, img_name + "_2.png")
        # util.save_img(output, save_img_path2)
        MA = MA.astype(np.float32) / 255.0

        # 确保 LQ 的范围是 0-1，类型为 float32（用于融合操作）
        LQ = LQ.astype(np.float32) / 255.0
        GT = GT.squeeze().permute(1, 2, 0)
        GT = GT.numpy().astype(np.float32) / 255.0
        output = output.astype(np.float32) / 255.0
        # diff = output - GT
        # decode_mse = np.mean(diff ** 2)
        # #print(f"[Decoded MSE] Reconstruction error in image space: {decode_mse:.6f}")
        # diff = LQ - GT
        # decode_mse = np.mean(diff ** 2)
        #print(f"[Decoded MSE] : {decode_mse:.6f}")
        # save_img_path3 = os.path.join(dataset_dir, img_name + "_3.png")
        # util.save_img(output, save_img_path3)
        output = output * MA + LQ * (1 - MA)

        


        #output = (output * 255).astype(np.uint8)
        #print(output.shape)
        img_LR = img_LR.squeeze()
        img_GT_1 = img_GT_1.squeeze()
        output_tensor = torch.from_numpy(output).to(dtype=img_LR.dtype, device=img_LR.device)
        img_LR[:, rnd_h : rnd_h + 512, rnd_w : rnd_w + 512] = output_tensor.permute(2, 0, 1)
        # LQ = (LQ * 255).astype(np.uint8)
        # MA = (MA * 255).astype(np.uint8)
        
        # LQ = LQ.numpy()
        # MA = MA.numpy()
        #print(img_GT_1.shape,img_LR.shape)
        gt_img = img_GT_1.cpu().numpy()
        output_img = img_LR.cpu().numpy()
        gt_img = gt_img[0, :, :]
        output_img = output_img[0, :, :]
        # if gt_img.max() <= 1.0:
        #     print(output_img.shape,gt_img.shape)
        mse = metrics.mean_squared_error(gt_img*255, output_img*255)
        print(mse)
        if mse > 1:
            #output_path = os.path.join(output_folder, output_file)
            txt_file_name = 'failed_images.txt'
            txt_file_path = os.path.join(dataset_dir, txt_file_name)
            with open(txt_file_path, 'a') as f:
                f.write(f"{img_name} {mse}\n")
        #print(f"MSE: {mse}")
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
            # save_img_path1 = os.path.join(dataset_dir, img_name + "_LQ.png")
            # save_img_path2 = os.path.join(dataset_dir, img_name + "_MA.png")
        #util.save_img(img_LR, save_img_path)
        img_LR_np = img_LR.detach().cpu().permute(1, 2, 0).numpy()  # C,H,W -> H,W,C

        if img_LR_np.dtype == np.float32 or img_LR_np.max() <= 1.0:
            img_LR_np = (img_LR_np * 255).clip(0, 255).astype(np.uint8)

        cv2.imwrite(save_img_path, img_LR_np)

    print(f"average test time: {np.mean(test_times):.4f}")
# if failed_info:
#     txt_file_path = 'failed_images.txt'
#     with open(txt_file_path, 'w') as f:
#         for path, mse in failed_info:
#             f.write(f"{path} {mse}\n")
#     print(f"已将 MSE 大于阈值的图像路径和 MSE 值保存到 {txt_file_path}")
# else:
#     print("没有图像的 MSE 大于阈值。")

