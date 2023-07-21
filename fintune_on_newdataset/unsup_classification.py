#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import shutil
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
from skimage import io, segmentation, morphology, exposure
from skimage.color import rgb2hsv
img_to_tensor = transforms.ToTensor()
import random
import tifffile as tif
path = '/data1/partitionA/CUHKSZ/histopath_2022/grand_competition/Train_Labeled/images/'
files = os.listdir(path)
binary_path = '0/'
gray_path = '1/'
colored_path = 'colored/'
os.makedirs(binary_path, exist_ok=True)
os.makedirs(colored_path, exist_ok=True)
os.makedirs(gray_path, exist_ok=True)
for img_name in files:
    img_path = path + str(img_name)
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
        img_data = tif.imread(img_path)
    else:
        img_data = io.imread(img_path)
    if len(img_data.shape) == 2 or (len(img_data.shape) == 3 and img_data.shape[-1] == 1):
        shutil.copyfile(path + img_name, binary_path + img_name)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        shutil.copyfile(path + img_name, colored_path + img_name)
    else:
        hsv_img = rgb2hsv(img_data)
        s = hsv_img[:,:,1]
        v = hsv_img[:,:,2]
        print(img_name,s.mean(),v.mean())
        if s.mean() > 0.1 or (v.mean()<0.1 or v.mean() > 0.6):
            shutil.copyfile(path + img_name, colored_path + img_name)
        else:
            shutil.copyfile(path + img_name, gray_path + img_name)



# In[3]:


####Phrase 2 clustering by cell size
from skimage import measure
colored_path = 'colored/'
label_path = 'allimages/tif/'
big_path = '2/'
small_path = '3/'
files = os.listdir(colored_path)
os.makedirs(big_path, exist_ok=True)
os.makedirs(small_path, exist_ok=True)
for img_name in files:
    label =  tif.imread(label_path + img_name.split('.')[0]+'.tif')
    props = measure.regionprops(label)
    num_pix = []
    for idx in range(len(props)):
        num_pix.append(props[idx].area)
    max_area = max(num_pix)
    print(max_area)
    if max_area > 30000:
        shutil.copyfile(path + img_name, big_path + img_name)
    else:
        shutil.copyfile(path + img_name, small_path + img_name)








