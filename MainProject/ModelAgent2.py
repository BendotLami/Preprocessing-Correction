import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from preprocess_model import *

BATCH_SIZE = 64

class ModelAgentColorCorrection(object):
    def __init__(self, dataset):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ColorCorrectionNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=800000, gamma=0.1)

        self.dataset = dataset

        transforms = torch.nn.Sequential(
            torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7),
        )
        self.scripted_transforms = torch.jit.script(transforms)

    def train(self):
        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.L1Loss()

        for epoch in range(100):
            for batch_data in train_data_loader:
                batch_data = batch_data.to(self.device)
                data_augmented = self.scripted_transforms(batch_data).to(self.device)

                img_reconstructed = self.model(data_augmented).to(self.device)

                train_loss = criterion(img_reconstructed, batch_data)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                print("Done batch!")

            with torch.no_grad():
                save_idx = 0
                for batch_data_test in test_data_loader:
                    batch_data_test = batch_data_test.to(self.device)
                    data_augmented = self.scripted_transforms(batch_data_test).to(self.device)

                    img_reconstructed = self.model(data_augmented).to(self.device)

                    print(criterion(img_reconstructed, batch_data_test).data)

                    test_examples_cpu = batch_data_test.cpu()

                    reconstruction_cpu = img_reconstructed.cpu()

                    data_augmented = data_augmented.cpu()

                    for index in range(int(len(img_reconstructed)/10)):
                        img = test_examples_cpu[index].numpy()

                        img_reconstruct = reconstruction_cpu[index].numpy()

                        img_color_augmented = data_augmented[index].numpy()

                        valid_reconstruct_img = np.clip(img_reconstruct, 0, 1)

                        plt.imsave(str("./test_output_color/" + str(epoch) + "_" + str(save_idx) + ".jpg"),
                                   np.concatenate((img.transpose(1, 2, 0), img_color_augmented.transpose(1, 2, 0),
                                                   valid_reconstruct_img.transpose(1, 2, 0)), axis=1))

                        save_idx += 1

            torch.save(self.model.state_dict(), str("./Model_Weights/" + "weights" + "_" + str(epoch)))

            print("Done epoch", epoch, "!")
