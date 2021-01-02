import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCorrectionNet(nn.Module):
    def __init__(self):
        super(ColorCorrectionNet, self).__init__()
        in_layers = 3

        # geometric transform:
        self.conv1 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        # in_layers += 3
        # self.conv2 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        # in_layers += 3
        # self.conv3 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        # in_layers += 3
        # self.conv4 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 5, 1, 2), nn.ReLU(True))
        # in_layers += 3
        # self.conv5 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 2), nn.ReLU(True))
        # in_layers += 3

        self.linear_layer_1 = nn.Sequential(nn.Linear(in_layers * 72 * 72, 256), nn.ReLU(True))
        self.linear_layer_2 = nn.Sequential(nn.Linear(256, 6))


    def forward(self, image):
        feat = image
        feat = self.conv1(feat)
        # feat = self.conv2(feat)
        # feat = self.conv3(feat)
        # feat = self.conv4(feat)
        feat = feat.view(feat.size()[0], -1)
        feat = self.linear_layer_1(feat)
        feat = self.linear_layer_2(feat)  # feat = [lambda_0, lambda_1, lambda_2, beta_0, beta_1, beta_2]
        img = image
        img = img * feat[:, :3, None, None] + feat[:, 3:, None, None]
        return img
