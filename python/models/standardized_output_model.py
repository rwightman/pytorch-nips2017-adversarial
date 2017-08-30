import torch.nn as nn


class StandardizedOutputModel(nn.Module):
    def __init__(self, model, drop_first_class=False):
        super(StandardizedOutputModel, self).__init__()
        self.model = model
        self.output_fn = nn.Softmax().cuda()
        self.drop_first_class = drop_first_class

    def forward(self, x):
        model_output = self.model(x)
        if self.drop_first_class:
            standard_output = model_output[:, 1:]
        else:
            standard_output = model_output
        return self.output_fn(standard_output)

    def get_core_model(self):
        if hasattr(self.model, 'get_core_model'):
            return self.model.get_core_model()
        else:
            return self.model