"""Attack abstract base class
"""
from abc import abstractmethod


class Attack:

    @abstractmethod
    def __call__(self, inputs, targets, batch_idx=0, deadline_time=None):
        assert False, "Not Implemented"
        # adv_input = torch.zeros(inputs.size()).cuda()
        # return adv_input, targets
