import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from processing.resize import Resize
from processing.crop import CentreCrop


class RandomZoom(nn.Module):
    def __init__(self, zoom_min, zoom_max):
        super(RandomZoom, self).__init__()
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max

    def forward(self, x):
        scale = np.random.uniform(self.zoom_min, self.zoom_max)

        if scale > 1.0:
            cropper = CentreCrop(crop_ratio=1.0 / scale)
            resizer = Resize(28)

            out = resizer(cropper(x))
        else:
            resize_to = int(28*scale)
            resizer = Resize(resize_to)
            resized = resizer(x)

            blank_canvas = Variable(torch.FloatTensor(torch.zeros(x.size())).cuda())

            left_and_top = (blank_canvas.size(2) - resize_to) // 2

            blank_canvas[:, :,
                         left_and_top:(left_and_top + resize_to),
                         left_and_top:(left_and_top + resize_to)] = resized

            out = blank_canvas

        return out
