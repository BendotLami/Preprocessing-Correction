import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCorrectionNet(nn.Module):
    def __init__(self):
        super(ColorCorrectionNet, self).__init__()
        in_layers = 3

        # geometric transform:
        self.conv1 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 5, 1, 2), nn.LeakyReLU(True))
        # in_layers += 3
        self.conv2 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 5, 1, 2), nn.LeakyReLU(True))
        # in_layers += 3
        self.conv3 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 5, 1, 2), nn.LeakyReLU(True))
        # in_layers += 3
        # self.conv4 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.ReLU(True))
        # in_layers += 3
        # self.conv5 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 2), nn.ReLU(True))
        # in_layers += 3

        # self.linear_layer_1 = nn.Sequential(nn.Linear(in_layers*5*5, 256), nn.ReLU(True))
        # self.linear_layer_2 = nn.Sequential(nn.Linear(256, 2))


    def forward(self, image):
        img = image
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        # img = self.linear_layer_1(img)
        # img = self.linear_layer_2(img)
        return img
