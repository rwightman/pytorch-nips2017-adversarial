"""

Architectures from:

Ensemble Adversarial Training: Attacks and Defenses
Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Dan Boneh, Patrick McDaniel
https://arxiv.org/abs/1705.07204

See Appendix B
For details also see:
https://github.com/ftramer/ensemble-adv-training/blob/master/mnist.py

"""

import torch.nn as nn
import torch.nn.functional as F


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.fc1 = nn.Linear(20*20*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5)
        self.fc = nn.Linear(3*3*128, 10)

    def forward(self, x):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view((x.size(0), -1))
        return self.fc(x)


class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.fc1 = nn.Linear(24*24*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.view((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.view((x.size(0),-1))
        x = F.dropout(p=0.5, input=F.relu(self.fc1(x)))
        x = F.dropout(p=0.5, input=F.relu(self.fc2(x)))
        x = F.dropout(p=0.5, input=F.relu(self.fc3(x)))
        x = F.dropout(p=0.5, input=F.relu(self.fc4(x)))
        x = x.view((x.size(0), -1))
        return self.fc5(x)
