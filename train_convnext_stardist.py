#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os

join = os.path.join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from stardist import star_dist,edt_prob
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap,ray_angles
import monai
from collections import OrderedDict
from compute_metric import eval_tp_fp_fn,remove_boundary_cells
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
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
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import tqdm
from models.unetr2d import UNETR2D
from models.swin_unetr import SwinUNETR
from models.flexible_unet import FlexibleUNet 
from models.flexible_unet_convext import FlexibleUNetConvext
print("Successfully imported all requirements!")
torch.backends.cudnn.enabled =False

def main():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/data2/liuchenyu/external_processed/split",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", default="/data/louwei/nips_comp/convnext_fold0", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--local_rank", type=int)
    # Model parameters
    parser.add_argument(
        "--model_name", default="efficientunet", help="select mode: unet, unetr, swinunetr"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=1e-4, help="learning rate")

    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    monai.config.print_config()
    n_rays = 32
    pre_trained = True
    #%% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_name + "_3class")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    # This must be change every runing time ! ! ! ! ! ! ! ! ! ! !
    model_file = "models/flexible_unet_convext.py"
    shutil.copyfile(
        __file__, join(model_path, os.path.basename(__file__))
    )
    shutil.copyfile(
        model_file, join(model_path, os.path.basename(model_file))
    )
    all_image_path = '/data/louwei/nips_comp/train_cellpose_multi0/'
    all_img_path = join(all_image_path, "train/images")
    all_gt_path = join(all_image_path, "train/tif")    
    
    all_img_names = sorted(os.listdir(all_img_path))
    all_gt_names = [img_name.split(".")[0] + ".tif" for img_name in all_img_names]
    all_img_files = [join(all_img_path, all_img_names[i]) for i in range(len(all_img_names))]
    all_gt_files = [join(all_gt_path, all_gt_names[i]) for i in range(len(all_img_names))]    
    img_path = join(args.data_path, "train/images")
    gt_path = join(args.data_path, "train/tif")
    val_img_path = join(args.data_path, "test/images")
    val_gt_path = join(args.data_path, "test/tif")
    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split(".")[0] + ".tif" for img_name in img_names]
    train_img_files = [join(img_path, img_names[i]) for i in range(len(img_names))]
    train_gt_files = [join(gt_path, gt_names[i]) for i in range(len(img_names))]
    cat_img_files = train_img_files + all_img_files
    cat_gt_files = train_gt_files + all_gt_files
    img_num = len(img_names)
    val_frac = 0.1
    val_img_names = sorted(os.listdir(val_img_path))
    val_gt_names = [img_name.split(".")[0] + ".tif" for img_name in val_img_names]
    #indices = np.arange(img_num)
    #np.random.shuffle(indices)
    #val_split = int(img_num * val_frac)
    #train_indices = indices[val_split:]
    #val_indices = indices[:val_split]

    train_files = [
        {"img": cat_img_files[i], "label": cat_gt_files[i]}
        for i in range(len(cat_img_files))
    ]
    val_files = [
        {"img": join(val_img_path, val_img_names[i]), "label": join(val_gt_path, val_gt_names[i])}
        for i in range(len(val_img_names))
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    #%% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.float32
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            #ScaleIntensityd(
                #keys=["img"], allow_missing_keys=True
            #),  # Do not scale label
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
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
                max_zoom=2,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.float32),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            #ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    #% define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
    )

    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name.lower() == "unet":
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

    if args.model_name.lower() == "efficientunet":
        model = FlexibleUNetConvext(
            in_channels=3,
            out_channels=n_rays+1,
            backbone='convnext_small',
            pretrained=True,
        ).to(device)
    
    if args.model_name.lower() == "swinunetr":
        model = SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=n_rays+1,
            feature_size=24,  # should be divisible by 12
            spatial_dims=2,
        ).to(device)
  
    #loss_masked_dice = monai.losses.DiceCELoss(softmax=True)
    loss_dice = monai.losses.DiceLoss(squared_pred=True,jaccard=True)
    loss_bce = nn.BCELoss()
    loss_dist_mae = nn.L1Loss()
    activatation = nn.ReLU()
    sigmoid = nn.Sigmoid()
    #loss_dist_mae = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr
    encoder = list(map(id, model.encoder.parameters()))
    base_params = filter(lambda p: id(p) not in encoder, model.parameters())
    params = [
        {"params": base_params, "lr":initial_lr},
        {"params": model.encoder.parameters(), "lr": initial_lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(params, initial_lr)
    #if pre_trained == True:
        #print('Load pretrained weights...')
        #checkpoint = torch.load('/mntnfs/med_data5/louwei/nips_comp/swin_stardist/swinunetr_3class/40.pth', map_location=torch.device(device))
        #model.load_state_dict(checkpoint['model_state_dict'])
    # start a typical PyTorch training
    #checkpoint = torch.load("/data2/liuchenyu/log/convnextsmall/efficientunet_3class/510.pth", map_location=torch.device(device))
    #model.load_state_dict(checkpoint['model_state_dict'])
    print('distributed model')
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)  
    print('successful model')
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)
    max_f1 = 0
    for epoch in range(0, max_epochs):
        model.train()
        epoch_loss = 0
        epoch_loss_prob = 0
        epoch_loss_dist_2 = 0
        epoch_loss_dist_1 = 0
        for step, batch_data in enumerate(tqdm.tqdm(train_loader), 1):
            inputs, labels = batch_data["img"],batch_data["label"]
            print(step)
            processes_labels = []
            
            for i in range(labels.shape[0]):
                label = labels[i][0]
                distances = star_dist(label,n_rays)
                distances = np.transpose(distances,(2,0,1))
                #print(distances.shape)
                obj_probabilities = edt_prob(label.astype(int))
                obj_probabilities = np.expand_dims(obj_probabilities,0)
                #print(obj_probabilities.shape)
                final_label = np.concatenate((distances,obj_probabilities),axis=0)
                #print(final_label.shape)
                processes_labels.append(final_label)
            
            labels = np.stack(processes_labels)

            #print(inputs.shape,labels.shape)
            inputs, labels = torch.tensor(inputs).to(device), torch.tensor(labels).to(device)
            #print(inputs.shape,labels.shape)
            optimizer.zero_grad()
            output_dist,output_prob = model(inputs)
            #print(outputs.shape)
            dist_output = output_dist
            prob_output = output_prob
            dist_label = labels[:,:n_rays,:,:]
            prob_label = torch.unsqueeze(labels[:,-1,:,:], 1)
            #print(dist_output.shape,prob_output.shape,dist_label.shape)
            #labels_onehot = monai.networks.one_hot(
                #labels, args.num_class
            #)  # (b,cls,256,256)
            #print(prob_label.max(),prob_label.min())
            loss_dist_1 = loss_dice(dist_output*prob_label,dist_label*prob_label)
            #print(loss_dist_1)
            loss_prob = loss_bce(prob_output,prob_label)
            #print(prob_label.shape,dist_output.shape)
            loss_dist_2 = loss_dist_mae(dist_output*prob_label,dist_label*prob_label)
            #print(loss_dist_2)
            loss = loss_prob + loss_dist_2*0.3 + loss_dist_1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_prob += loss_prob.item()
            epoch_loss_dist_2 += loss_dist_2.item()
            epoch_loss_dist_1 += loss_dist_1.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            
        epoch_loss /= step
        epoch_loss_prob /= step
        epoch_loss_dist_2 /= step
        epoch_loss_dist_1 /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        writer.add_scalar("train_loss", epoch_loss, epoch)
        print('dist dice: '+str(epoch_loss_dist_1)+' dist mae: '+str(epoch_loss_dist_2)+' prob bce: '+str(epoch_loss_prob))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }
        if epoch < 8:
            continue
        if epoch > 1 and epoch % val_interval == 0:
            torch.save(checkpoint, join(model_path, str(epoch) + ".pth"))
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                seg_metric = OrderedDict()
                seg_metric['F1_Score'] = []
                for val_data in tqdm.tqdm(val_loader):
                    val_images, val_labels = val_data["img"].to(device), val_data[
                        "label"
                    ].to(device)
                    roi_size = (512, 512)
                    sw_batch_size = 4
                    output_dist,output_prob = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                        )
                    val_labels = val_labels[0][0].cpu().numpy()
                    prob = output_prob[0][0].cpu().numpy()
                    dist = output_dist[0].cpu().numpy()
                    #print(val_labels.shape,prob.shape,dist.shape)
                    dist = np.transpose(dist,(1,2,0))
                    dist = np.maximum(1e-3, dist)
                    points, probi, disti = non_maximum_suppression(dist,prob,prob_thresh=0.5, nms_thresh=0.4)

                    coord = dist_to_coord(disti,points)
            
                    star_label = polygons_to_label(disti, points, prob=probi,shape=prob.shape)
                    gt = remove_boundary_cells(val_labels.astype(np.int32)) 
                    seg = remove_boundary_cells(star_label.astype(np.int32))           
                    tp, fp, fn = eval_tp_fp_fn(gt, seg, threshold=0.5)
                    if tp == 0:
                        precision = 0
                        recall = 0
                        f1 = 0
                    else:
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        f1 = 2*(precision * recall)/ (precision + recall)
                    f1 = np.round(f1, 4)
                    seg_metric['F1_Score'].append(np.round(f1, 4))
                avg_f1 = np.mean(seg_metric['F1_Score'])
                writer.add_scalar("val_f1score", avg_f1, epoch)
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    print(str(epoch) + 'f1 score: ' + str(max_f1))
                    torch.save(checkpoint, join(model_path, "best_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()
