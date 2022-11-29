# Solution of SRIBD-Med Team for NeurIPS2022-CellSeg Challenge
Institution: Shenzhen Research Institute of Big Data (SRIBD, http://www.sribd.cn/)  
Authors: Wei Lou\*, Xinyi Yu\*, Chenyu Liu\*, Xiang Wan, Guanbin Li, Siqi Liu, Haofeng Li\# (http://haofengli.net/)  
Emails: lhaof\@sribd.cn, weilou\@link.cuhk.edu.cn (Please contact us if you have any questions)

This repository provides the solution of team Sribd-med for [NeurIPS-CellSeg](https://neurips22-cellseg.grand-challenge.org/) Challenge. The details of our method are described in our paper [Multi-stream Cell Segmentation with Low-level Cues for Multi-modality Images]. Some parts of the codes are from the baseline codes of the [NeurIPS-CellSeg-Baseline](https://github.com/JunMa11/NeurIPS-CellSeg) repository,

You can reproduce our method as follows step by step:

## Environments and Requirements:
Install requirements by

```shell
python -m pip install -r requirements.txt
```

## Dataset
The competition training and tuning data can be downloaded from https://neurips22-cellseg.grand-challenge.org/dataset/
Besides, you can download three publiced data from the following link: 
Cellpose: https://www.cellpose.org/dataset 
Omnipose: http://www.cellpose.org/dataset_omnipose
Sartorius: https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview 

## Automatic cell classification
You can classify the cells into four classes in this step.
Put all the images (competition + Cellpose + Omnipose + Sartorius) in one folder (data/allimages).
Run classification code:

```shell
python classification/unsup_classification.py
```
The results can be stored in data/classification_results/

## CNN-base classification model training
Using the classified images in data/classification_results/. Stay connected to the Internet and the code may automatically download the necessary ImageNet-Pretrained weights. A resnet18 is trained:
```shell
python classification/train_classification.py
```
## Segmentation Training
Pre-training convnext-stardist using all the images (data/allimages).
```shell
python train_convnext_stardist.py
```
For class 0,2,3 finetune on the classified data (Take class1 as a example):
```shell
python finetune_convnext_stardist.py model_dir=(The pretrained convnext-stardist model) data_dir='data/classification_results/class1'
```
For class 1 train the convnext-hover from scratch using classified class 3 data.
```shell
python train_convnext_hover.py data_dir='data/classification_results/class3'
```

Finally, four segmentation models will be trained.

## Trained models
The models can be downloaded from this link:
https://drive.google.com/drive/folders/1MkEOpgmdkg5Yqw6Ng5PoOhtmo9xPPwIj?usp=sharing

## Inference
The inference process includes classification and segmentation.
```shell
python predict.py -i input_path -o output_path --model_path './models' 
```

## Evaluation
Calculate the F-score for evaluation:
```shell
python compute_metric.py --gt_path path_to_labels --seg_path output_path
```

## Results
The tuning set F1 score of our method is 0.8795. The Running time with tolerance of our method on all the 101 cases in the tuning set is 0 (within the time tolerance) in our local workstation. 

## Acknowledgement
We thank for the contributors of public datasets. We thank for the support from Shenzhen Research Institute of Big Data (SRIBD, http://www.sribd.cn/)

