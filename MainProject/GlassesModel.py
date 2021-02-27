import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorConvolutionLayer(nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size, stride, padding):
        super(GeneratorConvolutionLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_layers, out_layers, kernel_size, stride, padding),
            nn.ReLU(True))

    def downsample(self, x):
        padH, padW = int(x.shape[2]) % 2, int(x.shape[3]) % 2
        if padH != 0 or padW != 0: x = F.pad(x, [0, padH, 0, padW])  # TODO: make sure it works
        return torch.nn.AvgPool2d([2, 2], [2, 2])(x)

    def forward(self, feat, image_concat):
        feat = self.conv_layer(feat)
        image_concat = self.downsample(image_concat)
        feat = torch.cat([feat, image_concat], dim=1)
        return feat, image_concat

class GlassesGeneratorNet(nn.Module):
    def __init__(self):
        super(GlassesGeneratorNet, self).__init__()

        in_layers = 7

        # geometric transform:
        self.conv1 = GeneratorConvolutionLayer(in_layers, in_layers, 4, 2, 1)
        in_layers += 7
        self.conv2 = GeneratorConvolutionLayer(in_layers, in_layers, 4, 2, 1)
        in_layers += 7
        self.conv3 = GeneratorConvolutionLayer(in_layers, in_layers, 4, 2, 1)
        in_layers += 7
        self.conv4 = GeneratorConvolutionLayer(in_layers, in_layers, 4, 2, 1)
        in_layers += 7
        self.conv5 = GeneratorConvolutionLayer(in_layers, in_layers, 4, 2, 1)

        in_layers += 7
        # self.total_conv = nn.Sequential(*self.conv_layers)
        self.linear_layer_1 = nn.Sequential(nn.Linear((in_layers)*8*8, 256), nn.ReLU(True))
        self.linear_layer_2 = nn.Sequential(nn.Linear(256, 6))

    def downsample(self, x):
        padH, padW = int(x.shape[2]) % 2, int(x.shape[3]) % 2
        if padH != 0 or padW != 0: x = F.pad(x, [0, padH, 0, padW])  # TODO: make sure it works
        return torch.nn.AvgPool2d([2, 2], [2, 2])(x)

    def forward(self, FG, BG, start_feat):
        # transform the FG input
        theta = start_feat.view(-1, 2, 3)
        grid = F.affine_grid(theta, FG.size())
        FG_start = F.grid_sample(FG, grid)

        print(FG_start.size(), BG.size())

        image_concat = torch.cat([FG_start, BG], dim=1)
        feat = image_concat  # TODO: make sure this is a copy!

        feat, image_concat = self.conv1(feat, image_concat)
        feat, image_concat = self.conv2(feat, image_concat)
        feat, image_concat = self.conv3(feat, image_concat)
        feat, image_concat = self.conv4(feat, image_concat)
        feat, image_concat = self.conv5(feat, image_concat)

        feat = feat.view(feat.size()[0], -1)
        feat = self.linear_layer_1(feat)
        feat = self.linear_layer_2(feat)

        theta = theta + feat.view(-1, 2, 3)

        grid = F.affine_grid(theta, FG.size())
        FG_after_transform = F.grid_sample(FG, grid)

        return FG_after_transform, theta, feat


class GlassesDiscriminatorNet(nn.Module):
    def __init__(self):
        super(GlassesDiscriminatorNet, self).__init__()
        self.conv_layers = []
        in_layers = 3
        for i in range(4):
            self.conv_layers.append(nn.Conv2d(in_layers, in_layers, 4, 2, 1))
            self.conv_layers.append(nn.LeakyReLU(True))
        self.conv_layers.append(nn.Conv2d(in_layers, in_layers, 4, 2, 2))
        self.conv_layers.append(nn.LeakyReLU(True))

        self.conv_layers.append(nn.Conv2d(in_layers, 1, 5, 1, 0))

        self.total_layers = nn.Sequential(*self.conv_layers)

    def forward(self, img):
        feat = img
        feat = self.total_layers(feat)
        score = feat
        return score
