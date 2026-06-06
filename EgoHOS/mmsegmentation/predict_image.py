import argparse
import glob
import os
import warnings

import mmcv
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from PIL import Image
from skimage.io import imsave
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
import segmentation_refinement as refine
import torch

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py', type=str)
parser.add_argument("--checkpoint_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth', type=str)
parser.add_argument("--img_dir", default='../data/train/image', type=str)
parser.add_argument("--pred_seg_dir", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train_seg', type=str)
args = parser.parse_args()

os.makedirs(args.pred_seg_dir, exist_ok = True)

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')
refiner = refine.Refiner(device='cuda:0')

alpha = 0.5
for file in tqdm(glob.glob(args.img_dir + '/*')):
    fname = os.path.basename(file).split('.')[0]
    img = np.array(Image.open(os.path.join(args.img_dir, fname + '.jpg')))
    seg_result = inference_segmentor(model, file)[0]
    seg_result_final = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if seg_result[seg_result == 1].sum() > 0:
        seg_result_left = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        seg_result_left[seg_result == 1] = 255
        seg_result_left = refiner.refine(img, seg_result_left.astype(np.uint8), fast=True, L=900)
        seg_result_final[seg_result_left > 0] = 1

    if seg_result[seg_result == 2].sum() > 0:
        seg_result_right = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        seg_result_right[seg_result == 2] = 255
        seg_result_right = refiner.refine(img, seg_result_right.astype(np.uint8), fast=True, L=900)
        seg_result_final[seg_result_right > 0] = 2

    if seg_result[seg_result == 3].sum() > 0:
        seg_result_object1_left = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        seg_result_object1_left[seg_result == 3] = 255
        seg_result_object1_left = refiner.refine(img, seg_result_object1_left.astype(np.uint8), fast=True, L=900)
        seg_result_final[seg_result_object1_left > 0] = 3

    if seg_result[seg_result == 4].sum() > 0:
        seg_result_object1_right = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        seg_result_object1_right[seg_result == 4] = 255
        seg_result_object1_right = refiner.refine(img, seg_result_object1_right.astype(np.uint8), fast=True, L=900)
        seg_result_final[seg_result_object1_right > 0] = 4

    if seg_result[seg_result == 5].sum() > 0:
        seg_result_object1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        seg_result_object1[seg_result == 5] = 255
        seg_result_object1 = refiner.refine(img, seg_result_object1.astype(np.uint8), fast=True, L=900)
        seg_result_final[seg_result_object1 > 0] = 5
    
    imsave(os.path.join(args.pred_seg_dir, fname + '.png'), seg_result_final, check_contrast=False)



