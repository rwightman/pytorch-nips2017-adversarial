import os
import sys
import time
import numpy as np
import math
from scipy.misc import imsave

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms


class PerturbationNet(nn.Module):
    def __init__(self, defense_ensemble, defense_augmentation, epsilon, prob_dont_augment):
        super(PerturbationNet, self).__init__()
        self.defense_ensemble = defense_ensemble
        self.defense_augmentation = defense_augmentation
        self.epsilon = epsilon
        self.prob_dont_augment = prob_dont_augment
        self.w_matrix = None

    def forward(self, x):
        perturbed = x + PerturbationNet.delta(self.w_matrix, x, self.epsilon)

        if np.random.rand() < self.prob_dont_augment:
            augmented = perturbed
        else:
            augmented = self.defense_augmentation(perturbed)

        output = self.defense_ensemble(augmented)
        return output

    @staticmethod
    def delta(wi, x, epsilon):
        constraint_min = torch.clamp(x, max=epsilon)
        constraint_max = torch.clamp((1 - x), max=epsilon)

        return torch.clamp(constraint_min * torch.tanh(wi), -999, 0) + \
               torch.clamp(constraint_max * torch.tanh(wi), 0, 999)

    def set_w_matrix(self, w_matrix):
        self.w_matrix = w_matrix


class CWInspired(object):
    def __init__(self,
                 target_ensemble,
                 defense_augmentation,
                 max_epsilon=16,
                 n_iter=100,
                 lr=0.02,
                 targeted=False,
                 target_nth_highest=6,
                 prob_dont_augment=0.0,
                 initial_w_matrix=None,
                 always_target=None,
                 random_start=None,
                 n_restarts=1):
        super(CWInspired, self).__init__()
        self.eps = max_epsilon / 255.0
        self.n_iter = n_iter
        self.lr = lr
        self.targeted = targeted
        self.target_nth_highest = target_nth_highest
        if initial_w_matrix is not None:
            self.initial_w_matrix = np.load(initial_w_matrix)
        else:
            self.initial_w_matrix = None
        self.always_target = always_target
        self.perturbation_model = PerturbationNet(
            target_ensemble,
            defense_augmentation,
            self.eps,
            prob_dont_augment,
        ).cuda()
        self.loss_fn = torch.nn.NLLLoss().cuda()
        self.random_start = random_start
        self.n_restarts = n_restarts

    def __call__(self, input, target, batch_idx, deadline_time):
        if deadline_time:
            time_remaining = deadline_time - time.time()
            assert time_remaining > 0
            time_thresh = time_remaining * .1

        input_var = autograd.Variable(input, volatile=False, requires_grad=True)

        # In case of the final batch not being complete
        this_batch_size = input_var.size(0)

        # Predict class
        self.perturbation_model.set_w_matrix(autograd.Variable(torch.zeros(input.size()).cuda(),requires_grad=True))
        if not self.targeted:
            if self.always_target is not None:
                target = torch.LongTensor(np.repeat(self.always_target, this_batch_size))
            else:
                log_probs_var = self.perturbation_model(input_var)
                log_probs = log_probs_var.data.cpu().numpy()

                target = torch.LongTensor(
                    np.argsort(log_probs, axis=1)[:, -self.target_nth_highest])
                del log_probs_var

        # target came either from the loader or above
        target_var = autograd.Variable(target).cuda()

        best_loss = torch.FloatTensor(np.repeat(9999.0, this_batch_size)).cuda()
        best_w_matrix = autograd.Variable(torch.zeros(input.size()).cuda())

        for _ in range(self.n_restarts):

            if self.initial_w_matrix is None:
                batch_w_matrix = autograd.Variable(
                    torch.zeros(input.size()).cuda(),
                    requires_grad=True)
            elif self.random_start:
                batch_w_matrix = autograd.Variable(
                    torch.FloatTensor(np.random.uniform(self.random_start, list(input.size()))).cuda(),
                    requires_grad=True)
            else:
                batch_w_matrix = autograd.Variable(
                    torch.FloatTensor(
                        np.stack([self.initial_w_matrix[0] for _ in range(this_batch_size)])).cuda(),
                    requires_grad=True)

            self.perturbation_model.set_w_matrix(batch_w_matrix)

            optimizer = optim.Adam([batch_w_matrix], lr=self.lr)

            for i in range(self.n_iter):
                log_probs_perturbed_var = self.perturbation_model(input_var)
                optimizer.zero_grad()
                loss = self.loss_fn(log_probs_perturbed_var, target=target_var)

                better = loss.data < best_loss
                for b in range(this_batch_size):
                    if better[b]:
                        best_w_matrix[b, :, :, :] = batch_w_matrix[b, :, :, :]

                loss.backward()
                optimizer.step()

            if deadline_time:
                time_remaining = deadline_time - time.time()
                if i > 10 and time_remaining < time_thresh // 2:
                    print("Warning: breaking early at %d, time critical, %s remaining in batch."
                          % (i, time_remaining))
                    sys.stdout.flush()
                    break
                elif time_remaining < time_thresh:
                    print("Warning: time critical, %s remaining in batch." % time_remaining)
                    sys.stdout.flush()

        final_change = PerturbationNet.delta(best_w_matrix, input_var, self.eps)
        final_change = torch.clamp(final_change, -self.eps, self.eps)  # Hygiene, math should mean this is already true

        final_image_tensor = input_var.data + final_change.data
        # Hygiene, math should mean this is already true
        final_image_tensor = torch.clamp(final_image_tensor, 0.0, 1.0)
        return final_image_tensor, target, None
