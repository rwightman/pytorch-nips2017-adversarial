import os
import numpy as np
from scipy.misc import imsave

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision.transforms as transforms

class SelectiveUniversal(object):
    def __init__(self,
                 max_epsilon,
                 target_ensemble,
                 w_matrix_files):
        super(SelectiveUniversal, self).__init__()
        self.max_epsilon = max_epsilon
        self.target_ensemble = target_ensemble
        self.w_matrix_files = w_matrix_files
        self.nllloss = torch.nn.NLLLoss().cuda()

        self.w_matrices = [torch.tanh(torch.FloatTensor((np.load(f))).cuda()) for f in self.w_matrix_files]

    def __call__(self, input, target, batch_idx, deadline_time):
        eps = self.max_epsilon / 255.0

        best_losses = []
        best_w_ids =[]
        pred_classes = []

        input = input.cuda()
        input_var = autograd.Variable(input, volatile=False, requires_grad=False)

        log_probs_var = self.target_ensemble(input_var)
        log_probs = log_probs_var.data.cpu().numpy()
        pred_class = autograd.Variable(torch.LongTensor(np.argsort(log_probs, axis=1)[:, -1])).cuda()

        best_loss = 9999.0
        best_w_id = -1
        best_perturbed = None
        for w_id, w_matrix in enumerate(self.w_matrices):
            w_matrix_var = autograd.Variable(w_matrix, requires_grad=False)
            perturbed = input_var + eps * w_matrix_var
            clamped = torch.clamp(perturbed, 0.0, 1.0)
            log_probs_perturbed_var = self.target_ensemble(clamped)
            loss = -self.nllloss(log_probs_perturbed_var, target=pred_class).data.cpu().numpy()
            if loss < best_loss:
                best_loss = loss
                best_w_id = w_id
                best_perturbed = clamped.data.cpu().numpy()

        return np.transpose(best_perturbed, axes=(0, 2, 3, 1)), None

