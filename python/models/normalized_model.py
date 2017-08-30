import torch.nn as nn


class NormalizedModel(nn.Module):
    def __init__(self, model, normalizer):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x):
        normalized_x = self.normalizer(x)
        return self.model(normalized_x)

    def get_core_model(self):
        if hasattr(self.model, 'get_core_model'):
            return self.model.get_core_model()
        else:
            return self.model