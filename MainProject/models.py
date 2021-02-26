import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class RotationCorrectionNet(nn.Module):
    def __init__(self):
        super(RotationCorrectionNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)

        self.linear_1 = nn.Sequential(nn.Linear(1000, 360), nn.Softmax())


    def forward(self, image):
        img = self.resnet(image)
        print(img.size())
        angles = self.linear_1(img)

        return angles
