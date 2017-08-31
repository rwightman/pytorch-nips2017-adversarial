import torch.nn as nn

class AugmentationComposer(nn.Module):
    def __init__(self, augmentations):
        super(AugmentationComposer, self).__init__()
        self.augmentations = augmentations

    def forward(self, x):
        tensors = [x]
        for aug in self.augmentations:
            tensors.append(aug(tensors[-1]))
        return tensors[-1]