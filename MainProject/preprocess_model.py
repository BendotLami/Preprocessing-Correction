import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCorrectionNet(nn.Module):

    def create_layer_down(self, in_channels, out_channels):
        rtn_object = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        return rtn_object

    def __init__(self, in_image_size):
        super(ColorCorrectionNet, self).__init__()

        self.conv1 = self.create_layer_down(3, 32)
        self.conv2 = self.create_layer_down(32, 64)
        self.conv3 = self.create_layer_down(64, 128)
        self.conv4 = self.create_layer_down(128, 256)
        self.conv5 = self.create_layer_down(256, 512)

        img_size = int(in_image_size / 32)
        self.linear_1 = nn.Sequential(nn.Linear(img_size * img_size * 512, 512),
                                      nn.LeakyReLU(negative_slope=0.1))
        self.linear_2 = nn.Sequential(nn.Linear(512, 64), nn.LeakyReLU(negative_slope=0.1))

        self.linear_3 = nn.Linear(64, 2)

    def forward(self, image):
        feat = image
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)
        feat = self.conv5(feat)
        feat = feat.view(feat.size()[0], -1)
        feat = self.linear_1(feat)
        feat = self.linear_2(feat)
        feat = self.linear_3(feat)  # [batch_size, 2]
        # img = image
        # img = img * feat[:, :3, None, None] + feat[:, 3:, None, None]
        return feat
