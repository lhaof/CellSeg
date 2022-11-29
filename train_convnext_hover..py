#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os, sys

join = os.path.join
#sys.path.append('/data2/yuxinyi/stardist_pytorch')

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from stardist import star_dist, edt_prob
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap, ray_angles
import monai
from collections import OrderedDict
from compute_metric import eval_tp_fp_fn, remove_boundary_cells
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    # LoadImaged_modified,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
    apply_transform,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from skimage import io
from skimage.color import gray2rgb

from models.unetr2d import UNETR2D
from models.swin_unetr import SwinUNETR
from models.flexible_unet_convext import FlexibleUNet_hv

from utils import cropping_center, gen_targets, xentropy_loss, dice_loss, mse_loss, msge_loss

import warnings
warnings.filterwarnings("ignore")

print("Successfully imported all requirements!")
torch.backends.cudnn.enabled = False

def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class HoverDataset(Dataset):
    def __init__(self, data, transform, mask_shape):
        self.data = data
        self.transform = transform
        self.mask_shape = mask_shape

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index):
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        ret = self._transform(index)
        # print(target_dict['img'].dtype, target_dict['label'].dtype)
        # gen targets
        inst_map = np.squeeze(ret['label'].numpy()).astype('int32') # 1HW -> HW
        target_dict = gen_targets(inst_map, inst_map.shape[:2])  # original code: self.mask_shape -> current code: aug_size
        np_map, hv_map = target_dict['np_map'], target_dict['hv_map']
        np_map = cropping_center(np_map, self.mask_shape)  # HW
        hv_map = cropping_center(hv_map, self.mask_shape) # HW2
        target_dict['np_map'] = torch.tensor(np_map)
        target_dict['hv_map'] = torch.tensor(hv_map)
        # centercrop img
        img = cropping_center(ret['img'].permute(1,2,0), self.mask_shape).permute(2,0,1) # CHW -> HWC -> CHW
        ret['img'] = img
        ret.update(target_dict)
        return ret

def valid_step(model, batch_data):

    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # NCHW

    # HWC
    true_np = torch.squeeze(true_np).type(torch.int64)
    true_hv = torch.squeeze(true_hv).type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        preds = model(imgs_gpu)
        pred_dict = {'np': preds[1], 'hv': preds[0]}
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].numpy(),
            "true_hv": true_dict["hv"].numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }

    return result_dict

def proc_valid_step_output(raw_data, nr_types=None):

    track_dict = {}
    
    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["prob_np"]
    true_np = raw_data["true_np"]
    for idx in range(len(raw_data["true_np"])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_dict['np_acc'] = acc_np
    track_dict['np_dice'] = dice_np

    # * HV regression statistic
    pred_hv = raw_data["pred_hv"]
    true_hv = raw_data["true_hv"]

    over_squared_error = 0
    for idx in range(len(raw_data["true_np"])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_dict['hv_mse'] = mse

    return track_dict

def main():

    # class Args:
    #     def __init__(self, data_path, seed, num_workers, model_name, input_size, mask_size, batch_size, max_epochs,
    #                  val_interval, save_interval, initial_lr, gpu_id, n_rays):
    #         self.data_path = data_path
    #         self.seed = seed
    #         self.num_workers = num_workers
    #         self.model_name = model_name
    #         self.input_size = input_size
    #         self.mask_size = mask_size
    #         self.batch_size = batch_size
    #         self.max_epochs = max_epochs
    #         self.val_interval = val_interval
    #         self.save_interval = save_interval
    #         self.initial_lr = initial_lr
    #         self.gpu_id = gpu_id
    #         self.n_rays = n_rays

    # args = Args('/data2/yuxinyi/stardist_pytorch/dataset/class3_seed2', 2022, 4, 'efficientunet', 512, 256, 16, 600,
    #             1, 10, 1e-4, '4', 32)
    modelname = 'star-hover'
    strategy = 'aug256_out256'
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default=f"/mntnfs/med_data5/louwei/consep/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument("--seed", default=10, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)
    
    # Model parameters
    parser.add_argument(
        "--model_name", default="efficientunet", help="select mode: unet, unetr, swinunetr"
    )
    parser.add_argument("--input_size", default=512, type=int, help="after rand crop")
    parser.add_argument("--mask_size", default=256, type=int, help="after gen target")
    # Training parameters
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=800, type=int)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--initial_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    work_dir = f'/mntnfs/med_data5/louwei/hover_stardist/class_{modelname}_{strategy}'

    # monai.config.print_config()
    pre_trained = False
    # %% set training/validation split
    np.random.seed(args.seed)
    model_path = join(work_dir)
    rm_n_mkdir(model_path)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "Train/Images_3channels")
    gt_path = join(args.data_path, "Train/tif")
    val_img_path = join(args.data_path, "Test/Images_3channels")
    val_gt_path = join(args.data_path, "Test/tif")
    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.replace('.png', '.tif') for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.1
    val_img_names = sorted(os.listdir(val_img_path))
    val_gt_names = [img_name.replace('.png', '.tif') for img_name in val_img_names]

    train_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i]), 'name': img_names[i]}
        for i in range(len(img_names))
    ]
    val_files = [
        {"img": join(val_img_path, val_img_names[i]), "label": join(val_gt_path, val_gt_names[i]),
         'name': val_img_names[i]}
        for i in range(len(val_img_names))
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    
    def load_img(img):
        ret = io.imread(img)
        if len(ret.shape) == 2:
            ret = gray2rgb(ret)
        return ret.astype('float32')
    
    def load_ann(ann):
        ret = np.squeeze(io.imread(ann)).astype('float32')
        return ret
    
    # %% define transforms for image and segmentation
    train_transforms = Compose(
        [
            Lambdad(('img',), load_img),
            Lambdad(('label',), load_ann),
            # LoadImaged(
            #     keys=["img", "label"], reader=PILReader, dtype=np.float32
            # ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            # ScaleIntensityd(
            # keys=["img"], allow_missing_keys=True
            # ),  # Do not scale label
            # SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            # RandSpatialCropd(
            #     keys=["img", "label"], roi_size=args.input_size, random_size=False
            # ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=0.15,
                min_zoom=0.5,
                max_zoom=2.0,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            Lambdad(('img',), load_img),
            Lambdad(('label',), load_ann),
            # LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.float32),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            # ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            # CenterSpatialCropd(
            #     keys=["img", "label"], roi_size=args.input_size
            # ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    # % define dataset, data loader
    # check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_ds = HoverDataset(data=train_files, transform=train_transforms, mask_shape=(args.mask_size, args.mask_size))
    print(len(check_ds))
    tmp = check_ds[0]
    print(tmp['img'].shape, tmp['label'].shape, tmp['hv_map'].shape, tmp['np_map'].shape)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
        check_data["hv_map"].shape,
        torch.max(check_data["hv_map"]),
        check_data["np_map"].shape,
        torch.max(check_data["np_map"]),
    )

    # %% create a training data loader
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_ds = HoverDataset(data=train_files, transform=train_transforms, mask_shape=(args.mask_size, args.mask_size))
    print(len(train_ds))
    # example = train_ds[0]
    # plt.imshow(np.array(example['img']).transpose(1,2,0).astype('uint8'))
    # plt.imshow(np.squeeze(example['np_map'].numpy()).astype('uint8'), 'gray')
    # plt.imshow(example['hv_map'].numpy()[...,0])
    # plt.imshow(example['hv_map'].numpy()[..., 1])
    # plt.show()
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_ds = HoverDataset(data=val_files, transform=val_transforms, mask_shape=(args.mask_size, args.mask_size))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    model = FlexibleUNet_hv(
        in_channels=3,
        out_channels=2+2,
        backbone='convnext_small',
        pretrained=True,
        n_rays=2,
        prob_out_channels=2,
    )

    activatation = nn.ReLU()
    sigmoid = nn.Sigmoid()
    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    scheduler = StepLR(optimizer, 100, 0.1)
    #if pre_trained == True:
        #print('Load pretrained weights...')
        #checkpoint = torch.load('/data2/yuxinyi/stardist_pytorch/pretrained/overall/330.pth')
        #model.load_state_dict(checkpoint['model_state_dict'])
    # model = DataParallel(model)
    model = model.to('cuda')
    # start a typical PyTorch training
    max_epochs = args.max_epochs
    val_interval = args.val_interval
    save_interval = args.save_interval
    epoch_loss_values = []
    writer = SummaryWriter(model_path)

    #*# record loss and f1
    loss_file = f'{work_dir}/train_loss.txt'
    f1_file = f'{work_dir}/train_loss.txt'
    if os.path.exists(loss_file):
        os.remove(loss_file)
    if os.path.exists(f1_file):
        os.remove(f1_file)
    #*#

    for epoch in range(1, args.max_epochs):
        model.train()
        epoch_loss = 0
        running_np_1, running_np_2, running_hv_1, running_hv_2 = 0.0, 0.0, 0.0, 0.0
        stream = tqdm(train_loader)
        for step, batch_data in enumerate(stream, start=1):
            
            #*# hv map
            inputs, true_np, true_hv = batch_data["img"], batch_data["np_map"], batch_data['hv_map']
            true_np = true_np.to("cuda").type(torch.int64) # NHW
            true_hv = true_hv.to("cuda").type(torch.float32) # NHWC
            true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32) # NHWC
            inputs = torch.tensor(inputs).to('cuda')
            # print(inputs.shape, true_np.shape, true_hv.shape)
            
            optimizer.zero_grad()
            pred_hv, pred_np = model(inputs) # NCHW
            pred_hv = pred_hv.permute(0, 2, 3, 1).contiguous() # NHWC
            pred_np = pred_np.permute(0, 2, 3, 1).contiguous() # NHWC
            pred_np = F.softmax(pred_np, dim=-1)
            
            # losses
            loss_np_1 = xentropy_loss(true_np_onehot, pred_np) # bce
            loss_np_2 = dice_loss(true_np_onehot, pred_np) # dice
            loss_hv_1 = mse_loss(true_hv, pred_hv) # mse
            loss_hv_2 = msge_loss(true_hv, pred_hv, true_np_onehot[...,1]) # msge
            loss = loss_np_1 + loss_np_2 + loss_hv_1 + loss_hv_2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
    
            running_np_1 += loss_np_1.item()
            running_np_2 += loss_np_2.item()
            running_hv_1 += loss_hv_1.item()
            running_hv_2 += loss_hv_2.item()
            #*#
            
            stream.set_description(
                f'Epoch {epoch} | np bce: {running_np_1 / step:.4f}, np dice: {running_np_2 / step:.4f}, hv mse: {running_hv_1 / step:.4f}, hv msge: {running_hv_2 / step:.4f}')
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("np_bce", running_np_1 / step, epoch)
        writer.add_scalar("np_dice", running_np_2 / step, epoch)
        writer.add_scalar("hv_mse", running_hv_1 / step, epoch)
        writer.add_scalar("hv_msge", running_hv_2 / step, epoch)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")
    
        #*# record
        with open(loss_file, 'a') as f:
            f.write(f'Epoch{epoch}\tloss:{epoch_loss:.4f}\tnp_bce:{running_np_1/step:.4f}\tnp_dice:{running_np_2/step:.4f}\thv_mse:{running_hv_1/step:.4f}\thv_msge:{running_hv_2/step:.4f}\n')
        #*#
    
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }
        if epoch % save_interval == 0:
            torch.save(checkpoint, join(model_path, str(epoch) + ".pth"))
    
        running_np_acc, running_np_dice, running_hv_mse = 0.0, 0.0, 0.0
        stream_val = tqdm(val_loader)
        for step, batch_data in enumerate(stream_val, start=1):
            raw_data = valid_step(model, batch_data)['raw']
            track_dict = proc_valid_step_output(raw_data)
            running_np_acc += track_dict['np_acc']
            running_np_dice += track_dict['np_dice']
            running_hv_mse += track_dict['hv_mse']
            stream.set_description(f'Epoch {epoch} | np acc: {running_np_acc / step:.4f}, np dice: {running_np_dice / step:.4f}, hv mse: {running_hv_mse / step:.4f}')
        writer.add_scalar("np_acc", running_np_acc / step, epoch)
        writer.add_scalar("np_dice", running_np_dice / step, epoch)
        writer.add_scalar("hv_mse", running_hv_mse / step, epoch)
        print(f'Epoch {epoch} | np acc: {running_np_acc / step:.4f}, np dice: {running_np_dice / step:.4f}, hv mse: {running_hv_mse / step:.4f}')
        
        #*# record
        with open(loss_file, 'a') as f:
            f.write(f'Validation | Epoch{epoch}\tloss:{epoch_loss:.4f}\tnp_acc:{running_np_acc/step:.4f}\tnp_dice:{running_np_dice/step:.4f}\thv_mse:{running_hv_mse/step:.4f}\n')
        #*#
        
        scheduler.step()
                  
if __name__ == "__main__":
    main()
