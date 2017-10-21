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

class Squeezenet(nn.Module):
    def __init__(self, squeezenet, num_classes):
        super(Squeezenet, self).__init__()
        self.squeezenet = squeezenet

        # Patch over the imagenet model to make sense for CIFAR
        self.squeezenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self, x):
        return self.squeezenet(x)

class Squeezenet1_0(Squeezenet):
    def __init__(self, num_classes):
        super(Squeezenet1_0, self).__init__(torchvision.models.squeezenet1_0(pretrained=False, num_classes=num_classes), num_classes)

class Squeezenet1_1(Squeezenet):
    def __init__(self, num_classes):
        super(Squeezenet1_1, self).__init__(torchvision.models.squeezenet1_1(pretrained=False, num_classes=num_classes), num_classes)