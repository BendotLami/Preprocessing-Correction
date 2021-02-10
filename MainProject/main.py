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

from preprocess_model import *
from ModelAgent2 import *

CELEB_A_DIR = "/home/dcor/datasets/CelebAMask-HQ/CelebA-HQ-img/"
GLASSES_NPY_DIR = "/home/dcor/ronmokady/workshop21/team4/glasses.npy"

BATCH_SIZE_GLASSES = 16
BATCH_SIZE_WITHOUT_GLASSES = 16

# TODO: move to different file, data.py

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, img_list):
        self.main_dir = main_dir
        self.transform = transform
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
    glasses = glasses.transpose(0, 3, 1, 2)  # z-x-y
    glasses = glasses / 255
    glasses_alpha = glasses[:, 3, :, :]
    to_delete = np.where(glasses_alpha == 0)
    glasses[to_delete[0], :, to_delete[1], to_delete[2]] = 0
    # glasses = glasses[:, :3, :, :]
    glasses = glasses.astype(np.float32)

    return glasses

# if __name__ == "__main__":
#     # plt.ion()   # interactive mode
#     TRANSFORM_IMG = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(144),
#         torchvision.transforms.CenterCrop(144),
#         torchvision.transforms.ToTensor()
#     ])
#
#     glasses_on, glasses_off = get_img_lists('/home/dcor/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt')
#
#     dataset_with_glasses = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG, glasses_on)
#     dataset_without_glasses = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG, glasses_off)
#
#     glasses = np.load(GLASSES_NPY_DIR)  # x-y-z
#     glasses = fix_glasses(glasses)
#
#     agent = ModelAgent(dataset_with_glasses, dataset_without_glasses, glasses, BATCH_SIZE_GLASSES,
#                        BATCH_SIZE_WITHOUT_GLASSES)
#
#     agent.train()


if __name__ == "__main__":
    # plt.ion()   # interactive mode
    TRANSFORM_IMG = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor()
    ])
    TRANSFORM_IMG_COLOR = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),
        torchvision.transforms.ToTensor()
    ])

    glasses_on, glasses_off = get_img_lists('/home/dcor/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt')

    all_images = np.concatenate((glasses_on, glasses_off))

    dataset_original = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG, all_images)
    # dataset_color_change = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG, all_images)

    # dataset_without_glasses = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG, glasses_off)

    # glasses = np.load(GLASSES_NPY_DIR)  # x-y-z
    # glasses = fix_glasses(glasses)

    agent = ModelAgentRotationCorrection(dataset_original)

    agent.train()


