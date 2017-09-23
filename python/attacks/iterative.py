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
            max_epsilon=16, targeted=True, random_start=False, norm=float('inf'),
            step_alpha=None, num_steps=None, target_min=False, target_rand=False,
            debug=False):

        self.target_model = target_model
        self.targeted = targeted
        assert not target_min or not target_rand  # shouldn't both be set
        self.target_min = target_min
        self.target_rand = target_rand
        self.random_start = random_start
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
        input_var = autograd.Variable(input, requires_grad=True)
        target_var = autograd.Variable(target)
        eps = self.eps
        step_alpha = self.step_alpha
        random_alpha = eps / 5

        step = 0
        while step < self.num_steps:
            zero_gradients(input_var)

            done_fwd = False
            if step == 0:
                if not self.targeted or self.target_min or self.target_rand:
                    # For non-targeted , we'll move away from most likely predicted target
                    # and for targeted with min_target, we'll moe towards least likely target
                    output = self.target_model(input_var)
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
                    done_fwd = True

                if self.random_start:
                    input_var.data += random_alpha * torch.sign(
                        torch.normal(means=torch.zeros(input_var.size()).cuda(), std=1.0))
                    done_fwd = False

            if not done_fwd:
                output = self.target_model(input_var)

            loss = self.loss_fn(output, target_var)
            loss.backward()

            # normalize and scale gradient
            if self.norm == 2:
                normed_grad = step_alpha * input_var.grad.data / l2_norm(input_var.grad.data)
            elif self.norm == 1:
                normed_grad = step_alpha * input_var.grad.data / l1_norm(input_var.grad.data)
            else:
                # infinity-norm
                normed_grad = step_alpha * torch.sign(input_var.grad.data)

            # perturb current input image by normalized and scaled gradient
            if self.targeted:
                step_adv = input_var.data - normed_grad
            else:
                step_adv = input_var.data + normed_grad

            # calculate total adversarial perturbation from original image and clip to epsilon constraints
            total_adv = step_adv - input
            if self.norm == 2:
                # total_adv = eps * total_adv / l2norm(total_adv)
                total_adv = torch.clamp(total_adv, -eps, eps)
            elif self.norm == 1:
                # total_adv = eps * total_adv / l1norm(total_adv)
                total_adv = torch.clamp(total_adv, -eps, eps)
            else:
                # infinity-norm
                total_adv = torch.clamp(total_adv, -eps, eps)

            if self.debug:
                print('batch:', batch_idx, 'step:', step, total_adv.mean(), total_adv.min(), total_adv.max())
                sys.stdout.flush()

            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_adv
            input_adv = torch.clamp(input_adv, 0., 1.0)
            input_var.data = input_adv
            step += 1

        return input_adv, None if self.targeted else target_var.data

