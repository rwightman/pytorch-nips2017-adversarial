import torch.nn as nn
import torchvision.models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

        # Patch over the torchvision imagenet model to make sense for CIFAR
        # The resulting architecture is consistent with the paper
        # https://arxiv.org/abs/1512.03385
        self.resnet18.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.resnet18.maxpool = Identity()
        self.resnet18.avgpool = nn.AvgPool2d(4)
    def forward(self, x):
        return self.resnet18(x)