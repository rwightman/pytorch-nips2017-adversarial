import torch.nn as nn

class Augmentation(nn.Module):
    def forward(self, x):
        return x