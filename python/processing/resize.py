import torch.nn as nn


class Resize(nn.Module):
    def __init__(self, input_size):
        super(Resize, self).__init__()
        self.input_size = input_size
        self.resize_layer = nn.Upsample(size=(self.input_size, self.input_size), mode='bilinear').cuda()

    def forward(self, x):
        resized = self.resize_layer(x)
        return resized
