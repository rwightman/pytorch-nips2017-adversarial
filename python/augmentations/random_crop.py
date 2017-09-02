import numpy as np
from .augmentation import Augmentation


class RandomCrop(Augmentation):
    def __init__(self, crop_ratio=[0.875, 1.0], crop_size=None):
        super(RandomCrop, self).__init__()
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size

    def forward(self, x):
        input_size = x.size(2)
        if self.crop_size is None:
            if self.crop_ratio[0] == self.crop_ratio[1]:
                crop_ratio = self.crop_ratio[0]
            else:
                crop_ratio = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
            crop_size = min(input_size, round(input_size * crop_ratio))
        else:
            crop_size = min(input_size, self.crop_size)
        crop_left = np.random.randint(0, input_size - crop_size + 1)
        crop_top = np.random.randint(0, input_size - crop_size + 1)
        cropped = x[:, :, crop_left:crop_left + crop_size, crop_top:crop_top + crop_size]

        return cropped


class CentreCrop(Augmentation):
    def __init__(self, crop_ratio=1.0, crop_size=None):
        super(CentreCrop, self).__init__()
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size

    def forward(self, x):
        input_size = x.size(2)
        crop_size = min(input_size, self.crop_size or round(input_size * self.crop_ratio))
        crop_left = (input_size - crop_size) // 2
        crop_top = (input_size - crop_size) // 2
        cropped = x[:, :, crop_left:crop_left + crop_size, crop_top:crop_top + crop_size]

        return cropped
