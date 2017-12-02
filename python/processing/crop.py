from numpy import random
import math
import torch.nn as nn


class RandomCrop(nn.Module):
    def __init__(self, crop_ratio=[0.8, 1.0], sticky_sides=None):
        super(RandomCrop, self).__init__()
        self.crop_ratio = crop_ratio
        self.sticky_sides = sticky_sides

    def forward(self, x):
        input_size = x.size(2)
        if self.crop_ratio[0] == self.crop_ratio[1]:
            crop_ratio = self.crop_ratio[0]
        else:
            crop_ratio = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        crop_size = min(input_size, round(input_size * crop_ratio))

        crop_left = random.randint(0, input_size - crop_size)
        crop_right = crop_left + crop_size
        crop_top = random.randint(0, input_size - crop_size)
        crop_bottom = crop_top + crop_size

        if self.sticky_sides:
            if random.random() < self.sticky_sides:
                stick = random.choice(['side', 'top', 'both'], p=[0.45,0.45,0.1])

                if stick in ['side', 'both']:
                    # Stick to sides
                    if random.random() < 0.5:
                        # stick to left
                        crop_left = 0
                        crop_right = crop_left + crop_size
                    else:
                        # stick to right
                        crop_right = input_size
                        crop_left = input_size - crop_size
                if stick in ['top', 'both']:
                    # stick to top/bot
                    if random.random() < 0.5:
                        # stick to top
                        crop_top = 0
                        crop_bottom = crop_top + crop_size
                    else:
                        # stick to bottom
                        crop_bottom = input_size
                        crop_top = crop_bottom - crop_size


        cropped = x[:, :, crop_left:crop_right, crop_top:crop_bottom]

        return cropped


class CentreCrop(nn.Module):
    def __init__(self, crop_ratio=1.0, crop_size=None):
        super(CentreCrop, self).__init__()
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size

    def forward(self, x):
        input_size = x.size(2)
        if self.crop_size is None:
            crop_size = min(input_size, self.crop_size or round(input_size * self.crop_ratio))
        crop_left = int(round((input_size - crop_size) / 2.))
        crop_right = crop_left + crop_size
        crop_top = int(round((input_size - crop_size) / 2.))
        crop_bottom = crop_top + crop_size

        cropped = x[:, :, crop_left:crop_right, crop_top:crop_bottom]

        return cropped