from .normalizer import Normalizer

import torch
import torch.autograd as autograd

import numpy as np


class DualPathNet(Normalizer):
    def __init__(self):
        super(DualPathNet, self).__init__()
        self.normalizer_mean = autograd.Variable(torch.FloatTensor(np.array([124.0/255.0, 117.0/255.0, 104.0/255.0])[:, None, None])).cuda()
        self.normalizer_multiplier = autograd.Variable(torch.FloatTensor(np.array([0.0167*255]))).cuda()

    def forward(self, x):
        normalized = (x - self.normalizer_mean) * self.normalizer_multiplier
        return super(DualPathNet, self).forward(normalized)