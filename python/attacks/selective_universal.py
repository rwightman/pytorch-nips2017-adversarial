import os
import numpy as np
from scipy.misc import imsave

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision.transforms as transforms

from processing import Mirror

class SelectiveUniversal(object):
    def __init__(self,
                 target_ensemble,
                 w_matrix_files,
                 max_epsilon=16,
                 try_mirrors=False):
        super(SelectiveUniversal, self).__init__()
        self.target_ensemble = target_ensemble
        self.w_matrix_files = w_matrix_files
        self.max_epsilon = max_epsilon
        self.nllloss = torch.nn.NLLLoss().cuda()

        self.w_matrices = [torch.tanh(torch.FloatTensor((np.load(f))).cuda()) for f in self.w_matrix_files]
        if try_mirrors:
            self.mirrors = [lambda x: x, Mirror()]
            self.is_mirror = [False, True]
        else:
            self.mirrors = [lambda x: x]
            self.is_mirror = [False]

    def __call__(self, input, target, batch_idx, deadline_time):
        eps = self.max_epsilon / 255.0

        input = input.cuda()
        input_var = autograd.Variable(input, volatile=False, requires_grad=False)

        log_probs_var = self.target_ensemble(input_var)
        log_probs = log_probs_var.data.cpu().numpy()
        pred_class = np.argsort(log_probs, axis=1)[:, -1]
        pred_class_var = autograd.Variable(torch.LongTensor(pred_class)).cuda()

        best_loss = 9999.0
        best_perturbed = None
        best_is_fooled = False

        for w_id, w_matrix in enumerate(self.w_matrices):
            w_matrix_var = autograd.Variable(w_matrix, requires_grad=False)

            for func, is_mirrored in zip(self.mirrors, self.is_mirror):
                perturbed = input_var + func(eps * w_matrix_var)
                clamped = torch.clamp(perturbed, 0.0, 1.0)
                log_probs_perturbed_var = self.target_ensemble(clamped)
                loss = -self.nllloss(log_probs_perturbed_var, target=pred_class_var).data.cpu().numpy()
                if loss < best_loss:
                    best_loss = loss
                    best_perturbed = clamped.data.cpu().numpy()

                    log_probs = log_probs_perturbed_var.data.cpu().numpy()
                    top_class = np.argsort(log_probs, axis=1)[:, -1]
                    if top_class != pred_class:
                        best_is_fooled = True
                    else:
                        best_is_fooled = False


        return np.transpose(best_perturbed, axes=(0, 2, 3, 1)), None
        return best_perturbed, None
        return best_perturbed, None, best_is_fooled

