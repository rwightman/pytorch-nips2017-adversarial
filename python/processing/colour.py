# Colour because we're Canadian, damn it!

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

# From http://docs.opencv.org/2.4/doc/tutorials/core/basic_linear_transform/basic_linear_transform.html
class RandomBrightnessContrast(nn.Module):
    def __init__(self, gain_min, gain_max, bias_min, bias_max):
        super(RandomBrightnessContrast, self).__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.bias_min = bias_min
        self.bias_max = bias_max

    def forward(self, x):
        gain = np.random.uniform(self.gain_min, self.gain_max)
        bias = np.random.uniform(self.bias_max, self.bias_max)
        rescaled = x * gain + bias
        return torch.clamp(rescaled, 0.0, 1.0)

# From http://www.laurenscorijn.com/articles/colormath-basics
class RandomSaturation(nn.Module):
    # sat_min should be between 0 and 1
    # sat_max should be above 1
    def __init__(self, sat_min=0.75, sat_max=3.0):
        super(RandomSaturation, self).__init__()
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.grayscale_multiplier = autograd.Variable(torch.FloatTensor(np.array([0.299, 0.587, 0.114])[:, None, None])).cuda()

    def forward(self, x):
        # Drawing from a uniform(sat_min, sat_max) would bias too much towards increase
        # Instead we flip a coin and then draw an alpha
        increase_saturation = np.random.rand() < 0.5
        if increase_saturation:
            alpha = np.random.uniform(1.0, self.sat_max)
        else:
            alpha = np.random.uniform(self.sat_min,1.0)

        gray = torch.sum(x * self.grayscale_multiplier, dim=1, keepdim=True)
        sat_changed = alpha * x + (1 - alpha) * gray
        return torch.clamp(sat_changed, 0.0, 1.0)