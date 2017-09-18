import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models, ensembling_weights=None, mean_method='arithmetic'):
        super(Ensemble, self).__init__()

        self.models = models

        if ensembling_weights is None:
            self.ensembling_weights = [1.0 for _ in models]
        else:
            self.ensembling_weights = ensembling_weights

        self.mean_method = mean_method

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        if self.mean_method == 'geometric':
            # When ensembling probabilities, the 1e-6 seems to prevent NaNs from going backwards through the network
            # 1e-8 was not effective, but 1e-6 seems to be.
            # Seemed only to be an issue with some models.
            # i.e. [ adv_inception_resnet_v2, inception_v3 ] seemed fine
            #      Add resnet18, possibility of instability and NaNs
            o = torch.pow(outputs[0], self.ensembling_weights[0]) + 1e-6
            for idx in range(len(outputs) - 1):
                o = torch.mul(o, torch.pow(outputs[idx + 1], self.ensembling_weights[idx + 1]) + 1e-6)
            mean_probs = torch.pow(o, 1.0 / sum(self.ensembling_weights))
        elif self.mean_method == 'arithmetic':
            o = self.ensembling_weights[0] * outputs[0]
            for idx in range(len(outputs)-1):
                o = o + self.ensembling_weights[idx + 1]*outputs[idx+1]
            mean_probs = o / sum(self.ensembling_weights)
        else:
            raise ValueError('Invalid mean_method {}. Geometric and arithmetic only.'.format(mean_method))

        return mean_probs