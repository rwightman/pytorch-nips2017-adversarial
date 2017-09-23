import numpy as np
import torch


class UniversalPerturbation(object):
    def __init__(self,
                 max_epsilon,
                 universal_perturbation_npy_file):
        super(UniversalPerturbation, self).__init__()

        self.max_epsilon = max_epsilon

        self.universal_perturbation_npy_file = universal_perturbation_npy_file
        self.universal_perturbation = torch.tanh(torch.FloatTensor((np.load(self.universal_perturbation_npy_file)))).cuda()

    def __call__(self, input, target, batch_idx, deadline_time):
        eps = self.max_epsilon / 255.0

        perturbed = input + eps * self.universal_perturbation
        clamped = torch.clamp(perturbed, 0.0, 1.0)

        return clamped, target
