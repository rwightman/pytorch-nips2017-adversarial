"""Attack abstract base class
"""
from abc import abstractmethod
import torch
import numpy as np


class Attack:

    @abstractmethod
    def __call__(self, inputs, targets, batch_idx=0):
        assert False, "Not Implemented"


class DirectedAttack(Attack):
    def __init__(self, target_model,
                 # Targeting Args
                 targeted=True,  # target will be passed in
                 target_min=False,  # target least likely
                 target_rand=False,  # target randomly
                 target_nth_highest=None,  # target nth highest
                 always_target=None,  # target a single class always
                 ):

        self.target_model = target_model

        # Options are all mutually exclusive
        assert target_min + target_rand + target_nth_highest + (always_target is not None) + targeted <= 1

        self.targeted = targeted
        self.target_min = target_min
        self.target_rand = target_rand
        self.target_nth_highest = target_nth_highest
        self.always_target = always_target

        self.targeting_required = target_min or target_rand or target_nth_highest or always_target or (not targeted)

    def __call__(self, inputs, targets, batch_idx=0):
        assert False, "Not Implemented"

    def get_target(self, input_var):
        output = self.target_model(input_var)

        if self.target_min:
            return output.data.min(1)[1]
        elif self.target_rand:
            output_exp = 1. - torch.exp(output.data)
            return torch.multinomial(output_exp, 1).squeeze()
        elif self.target_nth_highest:
            return torch.LongTensor(np.argsort(output.data.cpu().numpy(), axis=1)[:, -self.target_nth_highest])
        elif self.always_target is not None:
            return torch.LongTensor(np.repeat(self.always_target, input_var.size(0)))
        else:
            # We'll move away from the predicted class
            return output.data.max(1)[1]