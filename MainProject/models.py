import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(6, 40, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(40, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, FG, BG):
        full_img = torch.cat([FG, BG], dim=3)
        xs = self.localization(full_img)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, FG.size())
        FG = F.grid_sample(FG, grid)

        return FG, theta

    def forward(self, FG, BG):
        # transform the FG input
        FG_after_transform, affine_matrix = self.stn(FG, BG)

        # concat FG to BG # TODO: fix this sum
        concat_img = BG + FG_after_transform

        return FG_after_transform, affine_matrix


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
