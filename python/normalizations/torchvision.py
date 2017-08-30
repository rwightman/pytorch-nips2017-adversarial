from .normalizer import Normalizer

import torch
import torch.autograd as autograd

import numpy as np


class Torchvision(Normalizer):
    def __init__(self):
        super(Torchvision, self).__init__()
        self.normalizer_mean = autograd.Variable(torch.FloatTensor(np.array([0.485, 0.456, 0.406])[:, None, None])).cuda()
        self.normalizer_std = autograd.Variable(torch.FloatTensor(np.array([0.229, 0.224, 0.225])[:, None, None])).cuda()

    def forward(self, x):
        normalized = (x - self.normalizer_mean) / self.normalizer_std
        return super(Torchvision, self).forward(normalized)