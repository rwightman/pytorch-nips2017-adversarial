import os
import numpy as np
from scipy.misc import imsave

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients

class PerturbationNet(nn.Module):
    def __init__(self, defense_ensemble, defense_augmentation, epsilon):
        super(PerturbationNet, self).__init__()
        self.defense_ensemble = defense_ensemble
        self.defense_augmentation = defense_augmentation
        self.epsilon = autograd.Variable(torch.FloatTensor([epsilon]).cuda())
        self.w_matrix = None

    def forward(self, x):
        perturbed = x + PerturbationNet.delta(self.w_matrix, x, self.epsilon)

        augmented = self.defense_augmentation(perturbed)

        output = self.defense_ensemble(augmented)

        return output

    @staticmethod
    def delta(wi, x, epsilon):
        constraint_min = torch.min(x, epsilon.expand_as(x))
        constraint_max = torch.min((1.0 - x), epsilon.expand_as(x))

        return torch.clamp(constraint_min * (torch.tanh(wi)), -999, 0) + \
            torch.clamp(constraint_max * (torch.tanh(wi)), 0, 999)

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
                 batch_size=8,
                 gpu=True):
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
        self.gpu = gpu

    def run(self):
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
            eps
        )

        if self.gpu:
            perturbation_model.cuda()

        nllloss = torch.nn.NLLLoss()
        if self.gpu:
            nllloss = nllloss.cuda()

        for batch_idx, (input, target) in enumerate(loader):
            if self.gpu:
                input = input.cuda()

            input_var = autograd.Variable(input, volatile=False, requires_grad=True)

            # In case of the final batch not being complete
            this_batch_size = input_var.size(0)

            batch_w_matrix = autograd.Variable(
                #torch.FloatTensor(np.random.normal(loc=0,scale=0.33,size=(this_batch_size, 3, self.img_size, self.img_size))).cuda(),
                torch.FloatTensor(np.zeros((this_batch_size, 3, self.img_size, self.img_size))).cuda(),
                requires_grad=True)
            perturbation_model.set_w_matrix(batch_w_matrix)

            # Predict class
            if not self.targeted:
                probs_perturbed_var = perturbation_model(input_var)
                probs_perturbed = probs_perturbed_var.data.cpu().numpy()

                # target = 6th why not since top 5% accuracy is so good
                target = torch.LongTensor(np.argsort(probs_perturbed, axis=1)[:, -self.target_nth_highest])
                del probs_perturbed_var

            # target came either from the loader or above
            target_var = autograd.Variable(target).cuda()

            optimizer = optim.Adam([batch_w_matrix], lr=self.lr)

            for i in range(self.n_iter):
                probs_perturbed_var = perturbation_model(input_var)

                zero_gradients(batch_w_matrix)

                loss = nllloss(torch.log(probs_perturbed_var), target=target_var)

                loss.backward()

                optimizer.step()

            final_change = PerturbationNet.delta(batch_w_matrix,
                                                 input_var,
                                                 autograd.Variable(torch.FloatTensor([eps]).cuda()))
            final_change = torch.clamp(final_change, -eps, eps)  # Hygiene, math should mean this is already true
            assert torch.max(final_change).data.cpu().numpy() < eps
            assert torch.min(final_change).data.cpu().numpy() > -eps
            final_image_tensor = input_var + final_change
            final_image_tensor = torch.clamp(final_image_tensor, 0.0,
                                             1.0)  # Hygiene, math should mean this is already true

            start_index = self.batch_size * batch_idx
            indices = list(range(start_index, start_index + this_batch_size))
            for filename, o in zip(self.dataset.filenames(indices, basename=True), final_image_tensor.cpu().data.numpy()):
                output_file = os.path.join(self.output_dir, filename)
                imsave(output_file, np.transpose(o, axes=(1, 2, 0)), format='png')
