
import os
join = os.path.join
import argparse
import numpy as np
import torch
import monai
import torch.nn as nn

from utils import sliding_window_inference
#from baseline.models.unetr2d import UNETR2D
import time
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap,ray_angles
from stardist import star_dist,edt_prob
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif
import cv2
from overlay import visualize_instances_map
from models.flexible_unet import FlexibleUNet 
from models.flexible_unet_convext import FlexibleUNetConvext
def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    #parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    #parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetr_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='efficientunet', help='select mode: unet, unetr, swinunetr')
    parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=512, type=int, help='segmentation classes')
    args = parser.parse_args()
    
    input_path = '/home/data/TuningSet/'
    output_path = '/home/data/output/'
    overlay_path = '/home/data/overlay/'


    img_names = sorted(os.listdir(join(input_path)))
    n_rays = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if args.model_name.lower() == "efficientunet":
        model = FlexibleUNetConvext(
            in_channels=3,
            out_channels=n_rays+1,
            backbone='convnext_small',
            pretrained=True,
        ).to(device)
    


    sigmoid = nn.Sigmoid()
    checkpoint = torch.load('/home/louwei/stardist_convnext/efficientunet_3class/best_model.pth', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    with torch.no_grad():
        for img_name in img_names:
            print(img_name)
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))
            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            t0 = time.time()
            #test_npy01 = pre_img_data/np.max(pre_img_data)
            test_npy01 = pre_img_data
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            output_dist,output_prob = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
            #test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            prob = output_prob[0][0].cpu().numpy()
            dist = output_dist[0].cpu().numpy()


            dist = np.transpose(dist,(1,2,0))
            dist = np.maximum(1e-3, dist)
            points, probi, disti = non_maximum_suppression(dist,prob,prob_thresh=0.5, nms_thresh=0.4)

            coord = dist_to_coord(disti,points)
            
            star_label = polygons_to_label(disti, points, prob=probi,shape=prob.shape)
            tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'), star_label)
            overlay = visualize_instances_map(pre_img_data,star_label)
            cv2.imwrite(join(overlay_path, img_name.split('.')[0]+'.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            
        
if __name__ == "__main__":
    main()





