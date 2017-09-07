import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models, ensembling_weights=None):
        super(Ensemble, self).__init__()

        self.models = models

        if ensembling_weights is None:
            self.ensembling_weights = [1.0 for _ in models]
        else:
            self.ensembling_weights = ensembling_weights

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        o = torch.pow(outputs[0], self.ensembling_weights[0])
        for idx in range(len(outputs)-1):
            o = torch.mul(o, torch.pow(outputs[idx+1], self.ensembling_weights[idx+1]))
        geometric_mean_probs = torch.pow(o, 1 / sum(self.ensembling_weights))

        return geometric_mean_probs