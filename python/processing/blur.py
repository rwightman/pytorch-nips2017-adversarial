import numpy as np
import torch
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


class RandomGaussianBlur(nn.Module):
    def __init__(self, prob_blur, size, sigma):
        super(RandomGaussianBlur, self).__init__()
        self.prob_blur = prob_blur
        self.gaussian_blur = GaussianBlur(size, sigma)

    def forward(self, x):
        do_blur = np.random.rand() < self.prob_blur
        if do_blur:
            return self.gaussian_blur(x)
        else:
            return x


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma, same=True):
        super(GaussianBlur, self).__init__()

        self.same = same
        if self.same:
            self.padding = nn.ReplicationPad2d(1).cuda()

        self.convolution = nn.Conv2d(3,3,kernel_size,stride=1,padding=0,groups=3,bias=False)

        kernel = GaussianBlur.matlab_style_gauss2D(
            shape=(kernel_size, kernel_size),
            sigma=sigma
        )
        convolution_weight_numpy = np.stack([kernel[None, :, :] for _ in range(3)])
        self.convolution.weight = nn.Parameter(torch.FloatTensor(convolution_weight_numpy))
        self.convolution.cuda()

    def forward(self, x):
        if self.same:
            padded = self.padding(x)
        else:
            padded = x
        blurred = self.convolution(padded)
        return blurred


    # From https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    # Not clear how this differs from other implementations
    @staticmethod
    def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h