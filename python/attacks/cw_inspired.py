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

TIME_LIMIT_PER_100 = 450  #FIXME make arg


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
                 input_dir,
                 output_dir,
                 max_epsilon,
                 target_ensemble,
                 defense_augmentation,
                 dataset,
                 n_iter=100,
                 lr=0.02,
                 targeted=False,
                 target_nth_highest=6,
                 img_size=299,
                 batch_size=20,
                 prob_dont_augment=0.0,
                 initial_w_matrix=None,
                 outputs_are_logprobs=False):
        super(CWInspired, self).__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_epsilon = max_epsilon
        self.target_ensemble = target_ensemble
        self.defense_augmentation = defense_augmentation
        self.dataset = dataset
        self.n_iter = n_iter
        self.lr = lr
        self.targeted = targeted
        self.target_nth_highest = target_nth_highest
        self.img_size = img_size
        self.batch_size = batch_size
        self.prob_dont_augment = prob_dont_augment
        if initial_w_matrix is not None:
            self.initial_w_matrix = np.load(initial_w_matrix)
        else:
            self.initial_w_matrix = None
        self.outputs_are_logprobs = outputs_are_logprobs

    def run(self):
        attack_start = time.time()

        eps = self.max_epsilon / 256.0

        loader = data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False)

        tf = transforms.Compose([
            transforms.Scale(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()
        ])
        self.dataset.set_transform(tf)

        perturbation_model = PerturbationNet(
            self.target_ensemble,
            self.defense_augmentation,
            eps,
            self.prob_dont_augment
        ).cuda()

        nllloss = torch.nn.NLLLoss().cuda()

        time_limit = TIME_LIMIT_PER_100 * ((len(self.dataset) - 1) // 100 + 1)
        time_limit_per_batch = time_limit / len(loader)
        iter_time = AverageMeter()
        for batch_idx, (input, target) in enumerate(loader):
            iter_start = time.time()

            input = input.cuda()
            input_var = autograd.Variable(input, volatile=False, requires_grad=True)

            # In case of the final batch not being complete
            this_batch_size = input_var.size(0)

            if self.initial_w_matrix is None:
                batch_w_matrix = autograd.Variable(
                    torch.zeros(this_batch_size, 3, self.img_size, self.img_size).cuda(),
                    requires_grad=True)
            else:
                batch_w_matrix = autograd.Variable(
                    torch.FloatTensor(np.stack([self.initial_w_matrix[0] for _ in range(this_batch_size)])).cuda(),
                    requires_grad=True)
            perturbation_model.set_w_matrix(batch_w_matrix)

            # Predict class
            if not self.targeted:
                probs_perturbed_var = perturbation_model(input_var)
                probs_perturbed = probs_perturbed_var.data.cpu().numpy()

                # target = 6th why not since top 5% accuracy is so good
                target = torch.LongTensor(
                    np.argsort(probs_perturbed, axis=1)[:, -self.target_nth_highest])
                del probs_perturbed_var

            # target came either from the loader or above
            target_var = autograd.Variable(target).cuda()

            optimizer = optim.Adam([batch_w_matrix], lr=self.lr)

            best_loss = torch.FloatTensor(np.repeat(9999.0,this_batch_size)).cuda()
            best_w_matrix = autograd.Variable(torch.zeros(batch_w_matrix.size()).cuda())

            for i in range(self.n_iter):
                probs_perturbed_var = perturbation_model(input_var)
                optimizer.zero_grad()
                if not self.outputs_are_logprobs:
                    loss = nllloss(torch.log(probs_perturbed_var + 1e-8), target=target_var)
                else:
                    loss = nllloss(probs_perturbed_var, target=target_var)

                better = loss.data < best_loss
                for b in range(this_batch_size):
                    if better[b]:
                        best_w_matrix[b,:,:,:] = batch_w_matrix[b,:,:,:]

                loss.backward()
                optimizer.step()

                # measure elapsed time
                current = time.time()
                iter_time.update(current - iter_start)
                total_elapsed = current - attack_start
                if total_elapsed > (time_limit - 30):
                    print("Warning: time critical, %s" % total_elapsed)
                    if i > 10 and total_elapsed > (time_limit - 15):
                        print("Warning: breaking early at %d, time critical, %s"
                              % (i, total_elapsed))
                        sys.stdout.flush()
                        break
                if time_limit_per_batch and iter_time.count > 20:
                    iter_limit = math.floor(time_limit_per_batch / (iter_time.avg * 1.05))
                    if i >= iter_limit:
                        print('Breaking early at %d due to time constraints' % i)
                        sys.stdout.flush()
                        break
                iter_start = time.time()

            final_change = PerturbationNet.delta(best_w_matrix, input_var, eps)
            final_change = torch.clamp(final_change, -eps, eps)  # Hygiene, math should mean this is already true
            final_image_tensor = input_var + final_change
            final_image_tensor = torch.clamp(
                final_image_tensor, 0.0,
                1.0)  # Hygiene, math should mean this is already true

            start_index = self.batch_size * batch_idx
            indices = list(range(start_index, start_index + this_batch_size))
            for filename, o in zip(
                    self.dataset.filenames(indices, basename=True), final_image_tensor.cpu().data.numpy()):
                output_file = os.path.join(self.output_dir, filename)
                imsave(
                    output_file,
                    np.round(255.0 * np.transpose(o, axes=(1, 2, 0))).astype(np.uint8),
                    format='png')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
