import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

from models import *

# utils functions, move to diff file in future
image_counter = 0


def print_images_to_folder(real_img, fake_img_D, fake_img_G, glasses, path='./debug_outputs/'):
    global image_counter
    for i in range(real_img.shape[0]):
        real = real_img[i].numpy().transpose(1, 2, 0)
        real = np.clip(real, 0, 1)
        fake_d = fake_img_D[i].numpy().transpose(1, 2, 0)
        fake_d = np.clip(fake_d, 0, 1)
        fake_g = fake_img_G[i].numpy().transpose(1, 2, 0)
        fake_g = np.clip(fake_g, 0, 1)
        glass = glasses[i].numpy().transpose(1, 2, 0)
        # glass = glass / np.max(glass)
        glass = np.clip(glass, 0, 1)
        glass = glass[:, :, :3]
        plt.imsave(str(path + "output_" + str(i) + "_batch_" + str(image_counter) + ".jpg"),
                   np.concatenate((real, fake_d, fake_g, glass), axis=1))
    image_counter += 1
    print("Done printing!")


def concatenate_glasses_and_foreground(glasses, BG_images):
    pass


class ModelAgent(object):
    def __init__(self, dataset_with_glasses, dataset_without_glasses, glasses, batch_size_glasses, batch_size_without_glasses):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_G = GeneratorNet().to(self.device)
        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=0.01)
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=800000, gamma=0.1)

        self.model_D = DiscriminatorNet().to(self.device)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=0.01)
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=800000, gamma=0.1)

        self.dataset_with_glasses = dataset_with_glasses  # Dataset
        self.dataset_without_glasses = dataset_without_glasses  # Dataset
        self.glasses = glasses  # ndarray

        self.batch_size_glasses = batch_size_glasses
        self.batch_size_without_glasses = batch_size_without_glasses


    def preprocess_celebA(self, folder_path):
        pass

    def train_G(self, FG_images, BG_images):
        glasses_transformed, t_matrix = self.model_G(FG_images, BG_images)
        fake_images = BG_images + glasses_transformed[:, :3, :, :]
        out_src = self.model_D(fake_images)
        g_loss_adv = - out_src

        g_loss_geo = torch.linalg.norm(t_matrix)  # L2
        g_loss_geo = torch.pow(g_loss_geo, 2)  # L2 ** 2

        # TODO: add loss using [glasses_transformed, BG_images]

        # target-to-original domain
        # x_reconst = self.model_G(x_real, c_org - c_org)
        # g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

        # backward and optimize
        g_loss = torch.mean(g_loss_adv + g_loss_geo)  # + self.config.lambda3 * g_loss_rec + self.config.lambda2 * g_loss_cls
        print("train_G loss:", g_loss.item())
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        return fake_images

    def train_D(self, FG_images, BG_images, real_images):
        out_src = self.model_D(real_images)  # out_src should be 0 - all images are real
        d_loss_real = - torch.mean(out_src)  # mean of out_src is mean of loss

        # compute loss with fake images
        glasses_transformed, _ = self.model_G(FG_images, BG_images)
        # fake_images = concatenate_glasses_and_foreground(glasses_transformed.cpu().numpy(),
        #                                                  BG_images.cpu().numpy())

        fake_images = BG_images + glasses_transformed[:, :3, :, :]

        out_src = self.model_D(fake_images)
        d_loss_fake = torch.mean(out_src)

        # compute loss for gradient penalty # TODO: maybe will be useful in the future
        # alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        # x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        # out_src, _ = self.D(x_hat)
        # d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # backward and optimize
        d_loss_adv = d_loss_real + d_loss_fake  # + self.config.lambda_gp * d_loss_gp
        print("train_D loss:", d_loss_adv.item())
        self.optimizer_D.zero_grad()
        d_loss_adv.backward(retain_graph=True)
        self.optimizer_D.step()

        return fake_images

    def train(self):
        train_size = int(0.9 * len(self.dataset_with_glasses))
        test_size = len(self.dataset_with_glasses) - train_size
        train_dataset_with, test_dataset_with = torch.utils.data.random_split(self.dataset_with_glasses, [train_size, test_size])

        train_size = int(0.9 * len(self.dataset_without_glasses))
        test_size = len(self.dataset_without_glasses) - train_size
        train_dataset_without, test_dataset_without = torch.utils.data.random_split(self.dataset_without_glasses, [train_size, test_size])

        train_data_loader_with = data.DataLoader(train_dataset_with, batch_size=self.batch_size_glasses, shuffle=True)
        # test_data_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_data_loader_without = data.DataLoader(train_dataset_without, batch_size=self.batch_size_without_glasses,
                                                    shuffle=True)
        # test_data_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(10):
            for with_g, without_g in zip(train_data_loader_with, train_data_loader_without):
                with_g.to(self.device)
                without_g.to(self.device)
                glasses_batch = self.glasses[np.random.choice(self.glasses.shape[0], self.batch_size_without_glasses)]
                glasses_batch = torch.from_numpy(glasses_batch)
                glasses_batch.to(self.device)

                fake_img_D = self.train_D(glasses_batch, without_g, with_g)
                fake_img_G = self.train_G(glasses_batch, without_g)

                self.lr_scheduler_G.step()
                self.lr_scheduler_D.step()

                print_images_to_folder(without_g.detach(), fake_img_D.detach(), fake_img_G.detach(), glasses_batch.detach())
