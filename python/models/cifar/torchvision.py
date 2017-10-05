import torch.nn as nn
import torchvision.models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Resnet(nn.Module):
    def __init__(self, resnet):
        super(Resnet, self).__init__()
        self.resnet = resnet

        # Patch over the torchvision imagenet model to make sense for CIFAR
        # The resulting architecture is consistent with the paper
        # https://arxiv.org/abs/1512.03385
        self.resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.resnet.maxpool = Identity()
        self.resnet.avgpool = nn.AvgPool2d(4)

    def forward(self, x):
        return self.resnet(x)

class Resnet18(Resnet):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__(torchvision.models.resnet18(pretrained=False, num_classes=num_classes))

class Resnet34(Resnet):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__(torchvision.models.resnet18(pretrained=False, num_classes=num_classes))


