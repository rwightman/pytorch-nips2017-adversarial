import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models, ensembling_weights=None):
        super(Ensemble, self).__init__()

        self.models = nn.ModuleList(models)

        if ensembling_weights is None:
            self.ensembling_weights = [1.0 for _ in models]
        else:
            self.ensembling_weights = ensembling_weights

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        o = self.ensembling_weights[0] * outputs[0]
        for idx in range(len(outputs)-1):
            o = o + self.ensembling_weights[idx + 1]*outputs[idx+1]
        mean_probs = o / sum(self.ensembling_weights)

        return mean_probs

    def classifier_parameters(self):
        params = []
        for m in self.models:
            params += m.get_classifier().parameters()
        return params