import numpy as np
import torch.nn as nn
from models.median_pool import MedianPool2d


class RandomBlur(nn.Module):
    def __init__(self, blur_prob, blur2x2_prob):
        super(RandomBlur, self).__init__()
        self.blur_prob = blur_prob
        self.blur2x2_prob = blur2x2_prob

        self.median_pool_2x2 = MedianPool2d(kernel_size=2, same=True).cuda()
        self.median_pool_3x3 = MedianPool2d(kernel_size=3, same=True).cuda()

    def forward(self, x):
        perform_blur = np.random.rand() < self.blur_prob
        if perform_blur:
            blur2x2 = np.random.rand() < self.blur2x2_prob
            if blur2x2:
                blurred = self.median_pool_2x2(x)
            else:
                blurred = self.median_pool_3x3(x)
        else:
            blurred = x

        return blurred


class Blur(nn.Module):
    def __init__(self, k=3):
        super(Blur, self).__init__()
        self.median_pool = MedianPool2d(kernel_size=k, same=True).cuda()

    def forward(self, x):
        blurred = self.median_pool(x)
        return blurred
