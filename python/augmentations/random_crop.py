import numpy as np

from .augmentation import Augmentation


class RandomCrop(Augmentation):
    def __init__(self, crop_size):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        input_size = x.size(2)
        assert self.crop_size < input_size

        crop_out = input_size - self.crop_size

        crop_left = np.random.randint(0, crop_out)
        crop_right = crop_out - crop_left
        crop_top = np.random.randint(0, crop_out)
        crop_bottom = crop_out - crop_top

        cropped = x[:, :, crop_left:-crop_right, crop_top:-crop_bottom]

        return cropped