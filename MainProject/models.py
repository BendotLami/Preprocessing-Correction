import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorNet(nn.Module):
    def __init__(self, device):
        super(GeneratorNet, self).__init__()
        self.conv_layers = []
        in_layers = 7
        for i in range(4):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_layers, in_layers, 4, 2, 1),
                nn.ReLU(True)).to(device))
            in_layers += 7
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(in_layers, in_layers, 4, 2, 2),
            nn.ReLU(True)).to(device))
        in_layers += 7

        self.linear_layer_1 = nn.Sequential(nn.Linear((in_layers)*5*5, 256), nn.ReLU(True))
        self.linear_layer_2 = nn.Sequential(nn.Linear(256, 6))

    def downsample(self, x):
        padH, padW = int(x.shape[2]) % 2, int(x.shape[3]) % 2
        if padH != 0 or padW != 0: x = F.pad(x, [0, padH, 0, padW])  # TODO: make sure it works
        return torch.nn.AvgPool2d([2, 2], [2, 2])(x)

    def forward(self, FG, BG):
        # transform the FG input
        image_concat = torch.cat([FG, BG], dim=1)
        feat = image_concat # TODO: make sure this is a copy!

        for i in range(len(self.conv_layers)):
            feat = self.conv_layers[i](feat)
            image_concat = self.downsample(image_concat)
            feat = torch.cat([feat, image_concat], dim=1)

        feat = feat.view(feat.size()[0], -1)
        feat = self.linear_layer_1(feat)
        feat = self.linear_layer_2(feat)

        theta = feat.view(-1, 2, 3)

        grid = F.affine_grid(theta, FG.size())
        FG_after_transform = F.grid_sample(FG, grid)

        # FG_after_transform, affine_matrix = self.stn(FG, BG)

        # concat FG to BG # TODO: fix this sum
        # concat_img = BG + FG_after_transform[:,:3,:,:]

        return FG_after_transform, theta


class DiscriminatorNet(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, fc_dim=1024, n_layers=5):
        super(DiscriminatorNet, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1),
                nn.InstanceNorm2d(conv_dim * 2 ** i, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i
        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2**n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, img):
        # transform the FG input
        y = self.conv(img)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        return logit_adv
