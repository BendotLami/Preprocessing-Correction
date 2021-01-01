import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCorrectionNet(nn.Module):
    def __init__(self):
        super(ColorCorrectionNet, self).__init__()
        in_layers = 3

        # geometric transform:
        self.color_conv1 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        # in_layers += 3
        self.color_conv2 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        # in_layers += 3
        self.color_conv3 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        # in_layers += 3
        # self.conv4 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 5, 1, 2), nn.ReLU(True))
        # in_layers += 3
        # self.conv5 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 2), nn.ReLU(True))
        # in_layers += 3

        self.rotation_conv1 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        self.rotation_conv2 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        self.rotation_conv3 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))
        self.rotation_conv4 = nn.Sequential(nn.Conv2d(in_layers, in_layers, 4, 2, 1), nn.LeakyReLU(True))

        self.rotation_linear_layer_1 = nn.Sequential(nn.Linear((in_layers)*9*9, 256), nn.ReLU(True))
        self.rotation_linear_layer_2 = nn.Sequential(nn.Linear(256, 6))

        self.linear_layer_1 = nn.Sequential(nn.Linear(in_layers*18*18, 256), nn.ReLU(True))
        self.linear_layer_2 = nn.Sequential(nn.Linear(256, 6))


    def forward(self, image):
        # Color Correction
        # feat = image
        # feat = self.color_conv1(feat)
        # feat = self.color_conv2(feat)
        # feat = self.color_conv3(feat)
        # # feat = self.conv4(feat)
        # feat = feat.view(feat.size()[0], -1)
        # feat = self.linear_layer_1(feat)
        # feat = self.linear_layer_2(feat)  # feat = [lambda_0, lambda_1, lambda_2, beta_0, beta_1, beta_2]
        # img = image
        # img = img * feat[:, :3, None, None] + feat[:, 3:, None, None]

        # Rotation Correction
        feat_rotation = image
        feat_rotation = self.rotation_conv1(feat_rotation)
        feat_rotation = self.rotation_conv2(feat_rotation)
        feat_rotation = self.rotation_conv3(feat_rotation)
        feat_rotation = self.rotation_conv4(feat_rotation)
        feat_rotation = feat_rotation.view(feat_rotation.size()[0], -1)
        feat_rotation = self.rotation_linear_layer_1(feat_rotation)
        feat_rotation = self.rotation_linear_layer_2(feat_rotation)

        theta = feat_rotation.view(-1, 2, 3)

        grid = F.affine_grid(theta, image)
        image_rotated = F.grid_sample(image, grid)

        return image_rotated, feat_rotation
