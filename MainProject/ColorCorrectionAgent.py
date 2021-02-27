import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ColorCorrectionModel import *

# BATCH_SIZE = 64
IMG_SIZE = 256

def unexpand_vector(vector, tensor):
    vector_unsqueezed = vector.unsqueeze(2).unsqueeze(3)
    return vector_unsqueezed.expand_as(tensor)


class ModelAgentColorCorrection(object):
    def __init__(self, dataset, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ColorCorrectionNet(IMG_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning-rate'])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr-scheduler']['step-size'],
                                                      gamma=config['lr-scheduler']['gamma'])

        self.batch_size = config['batch-size']
        self.epochs = config['epochs']

        self.dataset = dataset


    def random_contrast_and_brightness(self, in_tensor):
        batch_size = in_tensor.size()[0]
        alpha = 0.7 + (1.5 - 0.7) * torch.rand(batch_size, 1)
        beta = -0.5 + (0.5 + 0.5) * torch.rand(batch_size, 1)

        alpha = alpha.to(self.device)
        beta = beta.to(self.device)

        out_tensor = in_tensor * unexpand_vector(alpha, in_tensor) + unexpand_vector(beta, in_tensor)

        return out_tensor, torch.cat((1.0/alpha, -beta/alpha), 1)


    def fix_image(self, in_image, alpha_beta):
        alpha = alpha_beta[:, :1]
        beta = alpha_beta[:, 1:]

        out_image = in_image * unexpand_vector(alpha, in_image) + unexpand_vector(beta, in_image)

        return out_image

    def train(self):
        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            for batch_data in train_data_loader:
                batch_data = batch_data.to(self.device)
                data_augmented, alpha_beta_fixed = self.random_contrast_and_brightness(batch_data)
                alpha_beta = self.model(data_augmented).to(self.device)
                train_loss = criterion(alpha_beta, alpha_beta_fixed)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                print("Done batch!")

            with torch.no_grad():
                save_idx = 0
                for batch_data_test in test_data_loader:
                    batch_data_test = batch_data_test.to(self.device)
                    data_augmented, alpha_beta_fixed = self.random_contrast_and_brightness(batch_data_test)

                    alpha_beta = self.model(data_augmented).to(self.device)

                    print(criterion(alpha_beta, alpha_beta_fixed).data)

                    test_examples_cpu = batch_data_test.cpu()

                    reconstruction_cpu = self.fix_image(data_augmented, alpha_beta).cpu()

                    data_augmented = data_augmented.cpu()

                    for index in range(int(len(alpha_beta)/10)):
                        img = test_examples_cpu[index].numpy()
                        img_reconstruct = reconstruction_cpu[index].numpy()
                        img_color_augmented = data_augmented[index].numpy()
                        valid_reconstruct_img = np.clip(img_reconstruct, 0, 1)
                        img_color_augmented = np.clip(img_color_augmented, 0, 1)

                        plt.imsave(str("./test_output_color/" + str(epoch) + "_" + str(save_idx) + ".jpg"),
                                   np.concatenate((img.transpose(1, 2, 0), img_color_augmented.transpose(1, 2, 0),
                                                   valid_reconstruct_img.transpose(1, 2, 0)), axis=1))

                        save_idx += 1

            torch.save(self.model.state_dict(), str("./Model_Weights/" + "ColorCorrection" + "_" + str(epoch)))

            print("Done epoch", epoch, "!")

    def load_model_from_dict(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward_pass(self, img):
        with torch.no_grad():
            rtn_img = self.model(img)
        return rtn_img
