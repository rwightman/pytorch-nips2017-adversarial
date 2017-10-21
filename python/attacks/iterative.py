"""
Pytorch Iterative Fast-Gradient attack algorithm
"""
import sys
import torch
from torch import autograd
from torch.autograd.gradcheck import zero_gradients
from .helpers import *
from .attack import DirectedAttack


class AttackIterative(DirectedAttack):

    def __init__(
            self,
            target_model,
            max_epsilon=16,
            norm=float('inf'),

            # Iterations
            step_alpha=None,  # step size in range [0.0, 1.0]
            num_steps=None,  # number of iters

            # Targeting Args
            targeted=True,  # target will be passed in
            target_min=False,  # target least likely
            target_rand=False,  # target randomly
            target_nth_highest=None, # target nth highest
            always_target=None, # target a single class always

            # Initialization Args
            random_start=False,
            random_start_method='signed',  # signed or uniform
            random_start_factor=0.2,  # multiplied by epsilon to scale the random start

            # Other args
            debug=False):

        super(AttackIterative, self).__init__(target_model, targeted, target_min, target_rand, target_nth_highest, always_target)

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

    def __call__(self, input, target, batch_idx=0):
        input_var = autograd.Variable(input.clone(), requires_grad=True)
        target_var = autograd.Variable(target)
        eps = self.eps
        step_alpha = self.step_alpha

        # Keep track of our first passes in order to possibly save compute
        done_fwd = False

        # Targeting
        if self.targeting_required:
            target_data, output = self.get_target(input_var)
            target_var.data = target_data.cuda()
            done_fwd = True

        # Initialization
        # We do not norm this because we assume the random start wil get it right
        # In practice, that is most definitely not necessarily true
        if self.random_start:
            done_fwd = False  # We invalidate any forward pass we did during targeting by taking a random step
            if self.random_start_method == 'signed':
                input_var.data += eps * self.random_start_factor * torch.sign(
                    torch.normal(means=torch.zeros(input.size()).cuda(), std=1.0))
            elif self.random_start_method == 'uniform':
                input_var.data += eps * self.random_start_factor * ((torch.rand(input.size()).cuda() * 2.0) - 1.0)
            else:
                raise ValueError('Random start method {} not recognized.'.format(self.random_start_method))

        # Main gradient descent loop

        step = 0
        while step < self.num_steps:
            zero_gradients(input_var)

            if step > 0 or (not done_fwd):  # Possibly save compute on step 0
                output = self.target_model(input_var)

            loss = self.loss_fn(output, target_var)
            loss.backward()

            # normalize and scale gradient
            if self.norm == float('inf'):
                # infinity-norm
                normed_grad = step_alpha * torch.sign(input_var.grad.data)
            else:
                # any other norm
                normed_grad = step_alpha * torch.renorm(input_var.grad.data, self.norm, 0, 1)

            # perturb current input image by normalized and scaled gradient
            # calculate total adversarial perturbation from original image and clip to epsilon constraints
            if self.targeted:
                total_adv = input_var.data - normed_grad - input
            else:
                total_adv = input_var.data + normed_grad - input

            # limit the total adversarial change
            if self.norm == float('inf'):
                # infinity-norm
                total_adv = torch.clamp(total_adv, -eps, eps)
            else:
                # any other norm
                total_adv = torch.renorm(total_adv, self.norm, 0, eps)

            if self.debug:
                print('batch:', batch_idx, 'step:', step, total_adv.mean(), total_adv.min(), total_adv.max())
                sys.stdout.flush()

            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_adv
            input_adv = torch.clamp(input_adv, 0., 1.0)
            input_var.data = input_adv
            step += 1

        loss_per_batch_item = [-output[i][target_var[i].data[0]].data.cpu().numpy() for i in
                        range(input.size(0))]

        return input_adv, None if self.targeted else target_var.data, loss_per_batch_item
