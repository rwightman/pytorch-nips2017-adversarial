import torch
import numpy as np
from attacks.attack import Attack


class RestartAttack(Attack):
    def __init__(self, attack, n_restarts):
        self.attack = attack
        self.n_restarts = n_restarts

    def __call__(self, inputs, targets, batch_idx=0, deadline_time=None):
        this_batch_size = inputs.size(0)

        best_loss = np.repeat(9999.0, this_batch_size)
        best_adv = torch.zeros(inputs.size())
        best_tar = torch.zeros(inputs.size(0))

        for _ in range(self.n_restarts):
            input_adv, target_adv, loss_per_batch_item = self.attack(inputs, targets, batch_idx, deadline_time)

            for b in range(this_batch_size):
                if loss_per_batch_item[b] < best_loss[b]:
                    best_loss[b] = loss_per_batch_item[b]
                    best_adv[b, :, :, :] = input_adv[b, :, :, :]

                    if target_adv is not None:
                        best_tar[b] = target_adv[b]
                    else:
                        best_tar = None

        return best_adv, best_tar, best_loss