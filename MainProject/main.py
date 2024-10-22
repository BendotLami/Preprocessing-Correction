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

from SuperResolutionModel.train_srresnet import main as SRresnet_train
from SuperResolutionModel.train_srgan import main as SRgan_train, set_srresnet_checkpoint

import sys

sys.path.insert(0, './SuperResolutionModel')
from SuperResolutionModel.eval import *

from ColorCorrectionAgent import *
from GlassesAgent import *
from RotationAgent import *

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
        # img_list = img_list[:1000]  # TODO: temporary
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
        print("Couldn't find config.json, Aborting.")
        exit(1)

    IMAGE_SIZE = config_dict['run-settings']['image_size']
    CELEB_A_DIR = config_dict['run-settings']['celeb_a_dir']
    GLASSES_NPY_DIR = config_dict['run-settings']['glasses_npy_dir']
    CELEB_A_ANNO = config_dict['run-settings']['celeb_a_anno']

    glasses_on, glasses_off = get_img_lists(CELEB_A_ANNO)
    all_images = np.concatenate((glasses_on, glasses_off))
    dataset_all = CustomDataSet(CELEB_A_DIR, all_images)

    # Color correction model
    agent_color_correction = ModelAgentColorCorrection(dataset_all, config_dict['color-correction'])
    if config_dict['run-settings']['run-color-correction']:
        if config_dict["run-settings"]["train-color-correction"]:
            print("Starting color-correction training...")
            agent_color_correction.train()
        else:
            agent_color_correction.load_model_from_dict(config_dict['color-correction']['pre-trained-path'])

    # Glasses model
    dataset_with_glasses = CustomDataSet(CELEB_A_DIR, glasses_on)
    dataset_without_glasses = CustomDataSet(CELEB_A_DIR, glasses_off)

    glasses = np.load(GLASSES_NPY_DIR)  # x-y-z
    glasses = fix_glasses(glasses)

    agent_glasses = GlassesModelAgent(dataset_with_glasses, dataset_without_glasses, glasses, BATCH_SIZE_GLASSES,
                                      BATCH_SIZE_WITHOUT_GLASSES, config_dict['glasses'])

    if config_dict['run-settings']['run-glasses']:
        if config_dict["run-settings"]["train-glasses"]:
            print("Starting glasses training...")
            agent_glasses.train()
        else:
            agent_glasses.load_model_from_dict(config_dict['glasses']['generator']['pre-trained-path'],
                                               config_dict['glasses']['discriminator']['pre-trained-path'])

    # Rotation model
    agent_rotation = RotationCorrectionAgent(dataset_all, config_dict['rotation'])
    if config_dict['run-settings']['run-rotation']:
        if config_dict["run-settings"]["train-rotation"]:
            print("Starting rotation-correction training...")
            agent_rotation.train()
        else:
            agent_rotation.load_model_from_dict(config_dict['rotation']['pre-trained-path'])

    # Super resolution
    if config_dict['run-settings']['run-super-resolution']:
        if config_dict["run-settings"]["train-srresnet"]:
            print("Starting SRresnet training...")
            SRresnet_train()
        if config_dict["run-settings"]["train-srgan"]:
            set_srresnet_checkpoint(config_dict['super-resolution']['pre-trained-path-srresnet'])
            print("Starting SRgan training...")
            SRgan_train()
        else:
            srgan_load_model(config_dict['super-resolution']['pre-trained-path-srgan'])

    # run dataset through the network
    if config_dict['run-settings']['eval-network']:
        dataset_eval = CustomDataSet(config_dict['run-settings']['eval-dataset-path'], all_images)
        index = 0
        with torch.no_grad():
            test_data_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=1, shuffle=True)
            for batch_data_test in test_data_loader:
                index += 1
                glasses_batch = glasses[np.random.choice(glasses.shape[0], 1)]
                glasses_batch = torch.from_numpy(glasses_batch)
                glasses_batch = glasses_batch.to(device)
                batch_data_test = batch_data_test.to(device)

                if config_dict['run-settings']['run-color-correction']:
                    img_eval = agent_color_correction.forward_pass(batch_data_test)
                if config_dict['run-settings']['run-glasses']:
                    img_eval = agent_glasses.forward_pass(img_eval, glasses_batch)
                if config_dict['run-settings']['run-rotation']:
                    img_eval = agent_rotation.forward_pass(img_eval)
                if config_dict['run-settings']['run-super-resolution']:
                    img_eval = SRgan_forward_pass(img_eval)


                for img_index in range(int(len(batch_data_test.cpu()))):
                    output_img = img_eval[img_index].cpu().numpy().transpose(1, 2, 0)
                    output_img = np.clip(output_img, 0, 1)
                    plt.imsave(str("./" + str(index) + "_" + str(img_index) + "_eval.jpg"),
                               output_img)
