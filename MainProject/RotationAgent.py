import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

from RotationModel import *

BATCH_SIZE = 16


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - torch.abs(torch.abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(y_true, torch.argmax(y_pred))
    return torch.mean(torch.abs(diff).float())


def get_random_angles_array(batch_size):
    angles = ((torch.rand(batch_size) - 0.5) * 90)
    angles = angles.long() % 360

    return angles


class RotationCorrectionAgent(object):
    def __init__(self, dataset, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = RotationCorrectionNet().to(self.device)
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=config['lr'])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, step_size=config['step_size'], gamma=config['gamma'])

        global BATCH_SIZE
        BATCH_SIZE = config['batch-size']

        self.dataset = dataset


    def train(self):
        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.NLLLoss()

        for epoch in range(100):
            ## train
            for batch_data in train_data_loader:
                batch_data = batch_data.to(self.device)
                angles = get_random_angles_array(batch_data.shape[0]).to(self.device)
                for i in range(batch_data.shape[0]):
                    batch_data[i] = torchvision.transforms.functional.rotate(batch_data[i], float(angles[i])).to(self.device)

                feat = self.model(batch_data).to(self.device)

                # train_loss = torch.autograd.Variable(angle_error(angles, feat), requires_grad=True)
                # print(feat.dtype, angles_probability.dtype)
                train_loss = criterion(feat, angles)

                self.model_optimizer.zero_grad()
                print("angle error: ", angle_error(angles, feat))
                train_loss.backward()
                self.model_optimizer.step()

                print("Done batch!")


            ## test
            with torch.no_grad():
                save_idx = 0
                for batch_data_test in test_data_loader:
                    original_input = torch.clone(batch_data_test)
                    batch_data_test = batch_data_test.to(self.device)
                    angles = get_random_angles_array(batch_data_test.shape[0]).to(self.device)
                    for i in range(batch_data_test.shape[0]):
                        batch_data_test[i] = torchvision.transforms.functional.rotate(batch_data_test[i], float(angles[i])).to(self.device)

                    original_input_rotated = torch.clone(batch_data_test)

                    feat = self.model(batch_data_test).to(self.device)

                    print("test loss: ", criterion(feat, angles).data)

                    for i in range(batch_data_test.shape[0]):
                        batch_data_test[i] = torchvision.transforms.functional.rotate(batch_data_test[i], -1 * float(torch.argmax(feat[i]))).to(self.device)

                    test_examples_cpu = batch_data_test.cpu()

                    for index in range(int(len(test_examples_cpu)/10)):
                        img = test_examples_cpu[index].numpy()
                        orig_input = original_input[index].numpy()
                        orig_in_rotated = original_input_rotated[index].cpu().numpy()

                        plt.imsave(str("./test_output_rotation/" + str(epoch) + "_" + str(save_idx) + ".jpg"),
                                   np.concatenate((orig_input.transpose(1, 2, 0),
                                                   orig_in_rotated.transpose(1, 2, 0), img.transpose(1, 2, 0)), axis=1))

                        save_idx += 1

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), str("./Model_Weights/" + "RotationCorrection" + "_" + str(epoch)))

            print("Done epoch", epoch, "!")

    def load_model_from_dict(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward_pass(self, img):
        if len(img.size()) < 4:  # if we want to forward pass a single image
            img = img[None, :, :, :]
        with torch.no_grad():
            rot_img = self.model(img.to(self.device))
            for i in range(img.shape[0]):
                img[i] = torchvision.transforms.functional.rotate(img[i], -1 * float(torch.argmax(rot_img[i]))).to(self.device)
            return img
