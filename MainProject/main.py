from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import natsort
import os
from PIL import Image
import json
import cv2

# from preprocess_model import *
from ColorCorrectionAgent import *
from GlassesAgent import *

IMAGE_SIZE = 256

CELEB_A_DIR = "/home/dcor/datasets/CelebAMask-HQ/CelebA-HQ-img/"
GLASSES_NPY_DIR = "/home/dcor/ronmokady/workshop21/team4/glasses.npy"
CELEB_A_ANNO = "/home/dcor/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
DEFAULT_CONFIG = """
{
"color-correction": {
    "epochs": 100,
    "batch-size": 64,
    "learning-rate": 0.002,
    "lr-scheduler": {
        "step-size": 800000,
        "gamma": 0.1
    }
}
}"""

BATCH_SIZE_GLASSES = 16
BATCH_SIZE_WITHOUT_GLASSES = 16

# TODO: move to different file, data.py

class CustomDataSet(Dataset):
    def __init__(self, main_dir, img_list):
        TRANSFORM_IMG = torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMAGE_SIZE),
            torchvision.transforms.CenterCrop(IMAGE_SIZE),
            torchvision.transforms.ToTensor()
        ])

        self.main_dir = main_dir
        self.transform = TRANSFORM_IMG
        img_list = img_list[:1000]  # TODO: temporary
        all_jpg = [i for i in img_list if i.endswith('.jpg')]
        self.total_imgs = natsort.natsorted(all_jpg)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

# class

def get_img_lists(file_path):
    glasses_on = []
    glasses_off = []
    skip_count = 2
    with open(file_path, "r") as f:
        for line in f:
            if skip_count > 0:  # skip first 2 lines
                skip_count -= 1
                continue
            attr_list = line.split(' ')
            if int(attr_list[17]) == 1:
                glasses_on.append(attr_list[0])
            else:
                glasses_off.append(attr_list[0])

    return glasses_on, glasses_off


def fix_glasses(glasses):
    # image scaling
    rtn_img = np.zeros((glasses.shape[0], IMAGE_SIZE, IMAGE_SIZE, glasses.shape[3]))
    for i in range(glasses.shape[0]):
        rtn_img[i] = cv2.resize(glasses[i], dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    # PIL compatibility
    rtn_img = rtn_img.transpose(0, 3, 1, 2)  # z-x-y
    # to float [0,1]
    rtn_img = rtn_img / 255
    # remove background according to alpha
    rtn_img_alpha = rtn_img[:, 3, :, :]
    to_delete = np.where(rtn_img_alpha == 0)
    rtn_img[to_delete[0], :, to_delete[1], to_delete[2]] = 0
    # rtn_img = rtn_img[:, :3, :, :]
    rtn_img = rtn_img.astype(np.float32)

    return rtn_img


if __name__ == "__main__":
    try:
        with open("./config.json", 'r') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print("Couldn't find config.json, running with default config")
        config_dict = json.loads(DEFAULT_CONFIG)

    glasses_on, glasses_off = get_img_lists(CELEB_A_ANNO)
    all_images = np.concatenate((glasses_on, glasses_off))
    dataset_all = CustomDataSet(CELEB_A_DIR, all_images)

    # Color correction model
    # agent_color_correction = ModelAgentColorCorrection(dataset_all, config_dict['color-correction'])
    # agent_color_correction.train()

    # Glasses model
    dataset_with_glasses = CustomDataSet(CELEB_A_DIR, glasses_on)
    dataset_without_glasses = CustomDataSet(CELEB_A_DIR, glasses_off)

    glasses = np.load(GLASSES_NPY_DIR)  # x-y-z
    glasses = fix_glasses(glasses)

    agent_glasses = GlassesModelAgent(dataset_with_glasses, dataset_without_glasses, glasses, BATCH_SIZE_GLASSES,
                                      BATCH_SIZE_WITHOUT_GLASSES)
    agent_glasses.train()

