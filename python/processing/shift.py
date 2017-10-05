import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RandomShift(nn.Module):
    def __init__(self, shift_min, shift_max):
        super(RandomShift, self).__init__()
        self.shift_min = shift_min
        self.shift_max = shift_max

    def forward(self, x):
        #FIXME: Code interprets a negative shift as a shift to the right. Symmetry means no big deal.
        shift = np.random.randint(self.shift_min, self.shift_max, 2)

        blank_canvas = Variable(torch.FloatTensor(torch.zeros(x.size())).cuda())

        left = max(shift[0], 0)
        right = 28 - max(-shift[0], 0)
        top = max(shift[1], 0)
        bottom = 28 - max(-shift[1], 0)

        paste_left = max(-shift[0], 0)
        paste_right = paste_left + (right - left)
        paste_top = max(-shift[1], 0)
        paste_bottom = paste_top + (bottom - top)

        blank_canvas[:, :, paste_left:paste_right, paste_top:paste_bottom] = x[:, :, left:right, top:bottom]

        return blank_canvas
