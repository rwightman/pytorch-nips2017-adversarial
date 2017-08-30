import torch.nn as nn


class SelfResizingModel(nn.Module):
    def __init__(self, model, input_size):
        super(SelfResizingModel, self).__init__()
        self.model = model
        self.input_size = input_size
        self.resize_layer = nn.Upsample(size=(self.input_size, self.input_size), mode='bilinear').cuda()

    def forward(self, x):
        resized_x = self.resize_layer(x)
        return self.model(resized_x)

    def get_core_model(self):
        if hasattr(self.model, 'get_core_model'):
            return self.model.get_core_model()
        else:
            return self.model