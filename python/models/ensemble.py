import torch
import torch.nn as nn

import numpy as np
class Ensemble(nn.Module):
    def __init__(self, models, ensembling_weights=None, mean='geometric'):
        super(Ensemble, self).__init__()

        self.models = models

        if ensembling_weights is None:
            self.ensembling_weights = [1.0 for _ in models]
        else:
            self.ensembling_weights = ensembling_weights

        self.mean = mean

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        # Some ensembles were resulting in NaNs during the backwards pass
        # This was resulting in NaN updates to the w_matrix and ruining everything
        # I eventually managed to prevent this with the 1e-6 entries below
        # That guarantee no probability comes out zero
        # Note that 1e-8 wasn't sufficient
        if self.mean == 'geometric':
            o = torch.pow(outputs[0], self.ensembling_weights[0]) + 1e-6
            for idx in range(len(outputs) - 1):
                o = torch.mul(o, torch.pow(outputs[idx + 1], self.ensembling_weights[idx + 1]) + 1e-6)
            mean_probs = torch.pow(o, 1.0 / sum(self.ensembling_weights))
        else:
            o = self.ensembling_weights[0] * outputs[0]
            for idx in range(len(outputs)-1):
                o = o + self.ensembling_weights[idx + 1]*outputs[idx+1]
            mean_probs = o / sum(self.ensembling_weights)

        return mean_probs