import numpy as np
import torch
import torch.autograd as autograd

from .augmentation import Augmentation


class Mirror(Augmentation):
    def __init__(self, mirror_prob):
        super(Mirror, self).__init__()
        self.mirror_prob = mirror_prob
        self.inv_idx = None

    def forward(self, x):
        input_size = x.size(2)

        perform_mirror = np.random.rand() < self.mirror_prob
        if perform_mirror:
            if self.inv_idx is None:
                self.inv_idx = autograd.Variable(torch.arange(input_size-1, -1, -1).long()).cuda()
            mirrored = x.index_select(3, self.inv_idx)
        else:
            mirrored = x

        return mirrored