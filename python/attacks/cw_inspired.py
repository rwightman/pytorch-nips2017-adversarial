import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import numpy as np

from attacks.attack import DirectedAttack


class CWInspired(DirectedAttack):
    def __init__(self, target_model,
                 defense_augmentation,
                 max_epsilon=16,
                 n_iter=100,
                 lr=0.02,
                 prob_dont_augment=0.0,
                 debug=False,

                 # Targeting Args
                 targeted=True,  # target will be passed in
                 target_min=False,  # target least likely
                 target_rand=False,  # target randomly
                 target_nth_highest=None,  # target nth highest
                 always_target=None,  # target a single class always

                 # Initialization Args
                 random_start=None,
                 random_start_factor=0.2,
                 initial_w_matrix=None,
                 ):
        super(CWInspired, self).__init__(
            target_model,
            targeted=targeted,
            target_min=target_min,
            target_rand=target_rand,
            target_nth_highest=target_nth_highest,
            always_target=always_target)

        self.eps = max_epsilon / 255.0
        self.n_iter = n_iter
        self.lr = lr

        self.loss_fn = torch.nn.NLLLoss().cuda()

        self.random_start = random_start
        self.random_start_factor = random_start_factor
        if initial_w_matrix is not None:
            self.initial_w_matrix = np.load(initial_w_matrix)
        else:
            self.initial_w_matrix = None

        self.perturbation_model = PerturbationNet(self.target_model, defense_augmentation, self.eps,
                                                  prob_dont_augment).cuda()

    def __call__(self, input, target, batch_idx=None):
        input_var = autograd.Variable(input, volatile=False, requires_grad=True)

        this_batch_size = input_var.size(0)

        # Targeting
        if self.targeting_required:
            target_var = autograd.Variable(self.get_target(input_var)[0].cuda())
        else:
            target_var = autograd.Variable(target).cuda()

        # Initialization
        if self.random_start:
            # Initialize uniform [-2, 2] which maps pretty close to [-1 to 1] via tanh
            batch_w_matrix = autograd.Variable(self.random_start_factor * torch.rand(input.size()).cuda() * 4.0 - 2.0,
                                               requires_grad=True)
        elif self.initial_w_matrix is not None:
            batch_w_matrix = autograd.Variable(
                torch.FloatTensor(np.stack([self.initial_w_matrix[0] for _ in range(this_batch_size)])).cuda(),
                requires_grad=True)
        else:
            batch_w_matrix = autograd.Variable(torch.zeros(input.size()).cuda(), requires_grad=True)
        self.perturbation_model.set_w_matrix(batch_w_matrix)
        self.perturbation_model.clear_cache()

        # Main Iteration Loop
        optimizer = optim.Adam([batch_w_matrix], lr=self.lr)
        for i in range(self.n_iter):
            log_probs_perturbed_var = self.perturbation_model(input_var)
            optimizer.zero_grad()

            loss = self.loss_fn(log_probs_perturbed_var, target=target_var)

            if self.away:
                loss = -loss

            loss.backward()
            optimizer.step()

        # Final Output

        final_change = self.perturbation_model.delta()
        # Hygiene, math should mean this is already true
        final_change = torch.clamp(final_change, -self.eps, self.eps)

        final_image_tensor = input_var.data + final_change.data
        # Hygiene, math should mean this is already true
        final_image_tensor = torch.clamp(final_image_tensor, 0.0, 1.0)

        loss_per = [-log_probs_perturbed_var[i][target_var[i].data[0]].data.cpu().numpy() for i in
                    range(this_batch_size)]

        return final_image_tensor, target_var.data, loss_per

class PerturbationNet(nn.Module):
    def __init__(self, defense_ensemble, defense_augmentation, epsilon, prob_dont_augment):
        super(PerturbationNet, self).__init__()
        self.defense_ensemble = defense_ensemble
        self.defense_augmentation = defense_augmentation
        self.epsilon = epsilon
        self.prob_dont_augment = prob_dont_augment
        self.w_matrix = None

        # We will cache these the first time we calculate them when x is passed in
        self.constraint_min = None
        self.constraint_max = None

    def cache(self, x):
        if self.constraint_min is None:
            self.constraint_min = torch.clamp(x, 0, self.epsilon)
        if self.constraint_max is None:
            self.constraint_max = torch.clamp((1 - x), 0, self.epsilon)

    def clear_cache(self):
        self.constraint_min = None
        self.constraint_max = None

    def forward(self, x):
        self.cache(x)

        perturbed = x + self.delta()
        if np.random.rand() < self.prob_dont_augment:
            augmented = perturbed
        else:
            augmented = self.defense_augmentation(perturbed)
        output = self.defense_ensemble(augmented)
        return output

    def delta(self):
        return torch.clamp(self.constraint_min * torch.tanh(self.w_matrix), -999, 0) + \
               torch.clamp(self.constraint_max * torch.tanh(self.w_matrix), 0, 999)

    def set_w_matrix(self, w_matrix):
        self.w_matrix = w_matrix