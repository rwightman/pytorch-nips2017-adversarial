"""Pytorch Iterative Fast-Gradient attack algorithm
"""
import sys
import torch
from torch import autograd
from torch.autograd.gradcheck import zero_gradients
from .helpers import *
from .attack import Attack


class AttackIterative(Attack):

    def __init__(
            self,
            target_model,
            max_epsilon=16,
            norm=float('inf'),
            step_alpha=None,
            num_steps=None,
            # Targeting Args
            targeted=True, target_min=False, target_rand=False,  # Target Args
            # Initialization Args
            random_start=False, random_start_method='signed', random_start_factor=0.2,
            debug=False):

        self.target_model = target_model
        self.targeted = targeted
        assert not target_min or not target_rand  # shouldn't both be set
        self.target_min = target_min
        self.target_rand = target_rand

        self.random_start = random_start
        self.random_start_method = random_start_method
        self.random_start_factor = random_start_factor

        self.eps = max_epsilon / 255.0
        self.num_steps = num_steps or 10
        self.norm = norm
        if not step_alpha:
            if norm == float('inf'):
                self.step_alpha = self.eps / self.num_steps
            else:
                # Different scaling required for L2 and L1 norms to get anywhere
                if norm == 1:
                    self.step_alpha = 500.0  # L1 needs a lot of (arbitrary) love
                else:
                    self.step_alpha = 1.0
        else:
            self.step_alpha = step_alpha
        self.loss_fn = torch.nn.NLLLoss().cuda()
        self.debug = debug

    def __call__(self, input, target, batch_idx=0, deadline_time=None):
        input_var = autograd.Variable(input)
        target_var = autograd.Variable(target)
        total_adv = autograd.Variable(torch.zeros(input.size()).cuda(), requires_grad=True)
        input_adv_var = input_var + total_adv

        eps = self.eps
        step_alpha = self.step_alpha

        # These limit the total change to keep a valid image
        lower_limit = -input_var
        upper_limit = 1.0 - input_var

        # Keep track of our first passes in order to possibly save compute
        done_fwd = False

        if (not self.targeted) or self.target_min or self.target_rand:
            # For non-targeted , we'll move away from most likely predicted target
            # and for targeted with min_target, we'll moe towards least likely target
            output = self.target_model(input_adv_var)
            done_fwd = True
            if self.target_min:
                target_var.data = output.data.min(1)[1]
            elif self.target_rand:
                # little more interesting than targeting least likely all the time,
                # pick random target
                output_exp = 1. - torch.exp(output.data)
                output_exp.scatter_(1, target_var.data.unsqueeze(1), 0.)
                target_var.data = torch.multinomial(output_exp, 1).squeeze()
            else:
                target_var.data = output.data.max(1)[1]

        if self.random_start:
            done_fwd = False  # We invalidate any forward pass we did during targeting by taking a random step
            if self.random_start_method == 'signed':
                total_adv.data += eps * self.random_start_factor * torch.sign(
                    torch.normal(means=torch.zeros(input.size()).cuda(), std=1.0))
            elif self.random_start_method == 'uniform':
                total_adv.data += eps * self.random_start_factor * ((torch.rand(input.size()).cuda() * 2.0) - 1.0)
            else:
                raise ValueError('Random start method {} not recognized.'.format(self.random_start_method))

            # We do not norm this because we assume the random start wil get it right
            input_adv_var = input_var + total_adv

        step = 0
        while step < self.num_steps:
            zero_gradients(total_adv)

            if step > 0 or (not done_fwd):  # Possibly save compute on step 0
                output = self.target_model(input_adv_var)

            loss = self.loss_fn(output, target_var)
            loss.backward()

            print(total_adv.grad.data)

            # normalize and scale gradient
            if self.norm == float('inf'):
                # infinity-norm
                normed_grad = torch.sign(total_adv.grad.data)
            else:
                # any other norm
                normed_grad = torch.renorm(total_adv.grad.data, self.norm, 0, 1)

            # step the total adversarial change
            if self.targeted:
                total_adv.data = total_adv.data - step_alpha * normed_grad
            else:
                total_adv.data = total_adv.data + step_alpha * normed_grad

            # limit the total adversarial change
            if self.norm == float('inf'):
                # infinity-norm
                total_adv = torch.clamp(total_adv, -eps, eps)
            else:
                # any other norm
                total_adv = torch.renorm(total_adv, self.norm, 0, eps)

            # force valid image
            lower_limited = torch.max(total_adv, lower_limit)
            upper_limited = torch.min(lower_limited, upper_limit)
            total_adv = autograd.Variable(upper_limited.data, requires_grad=True)  # If I don't do this, grads don't get calc'd a second time

            input_adv_var = input_var + total_adv

            if self.debug:
                print('batch:', batch_idx, 'step:', step, total_adv.mean(), total_adv.min(), total_adv.max())
                sys.stdout.flush()

            step += 1

        return input_adv_var.data, None if self.targeted else target_var.data, None
