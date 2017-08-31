import numpy as np
import torch
import torch.autograd as autograd
from models.median_pool import MedianPool2d

from .augmentation import Augmentation


class Blur(Augmentation):
    def __init__(self, blur_prob, blur2x2_prob):
        super(Blur, self).__init__()
        self.blur_prob = blur_prob
        self.blur2x2_prob = blur2x2_prob

        self.median_pool_2x2 = MedianPool2d(kernel_size=2, same=True).cuda()
        self.median_pool_3x3 = MedianPool2d(kernel_size=3, same=True).cuda()

    def forward(self, x):
        input_size = x.size(2)

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