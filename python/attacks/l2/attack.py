"""Carlini & Wagner L2 attack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from torch.autograd.gradcheck import zero_gradients
from scipy.misc import imsave
from dataset import Dataset

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--max_epsilon', type=int, default=16.0, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='enable targeted attack')
parser.add_argument('--no_gpu', action='store_true', default=False,
                    help='disables GPU training')


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def main():
    args = parser.parse_args()
    assert args.input_dir

    eps = 2.0 * args.max_epsilon / 255.0
    num_steps = 8
    step_eps = eps / num_steps

    if args.targeted:
        dataset = Dataset(args.input_dir)
    else:
        dataset = Dataset(args.input_dir, target_file=None)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    tf = transforms.Compose([
        transforms.Scale(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    dataset.set_transform(tf)

    model = torchvision.models.inception_v3(pretrained=False, transform_input=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    l2_fn = torch.nn.MSELoss(size_average=False)
    if not args.no_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
    model.eval()

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % args.checkpoint_path)

    for batch_idx, (input, target) in enumerate(loader):
        batch_size = input.size(0)
        input = input.cuda()
        target = target.cuda()
        modifier = torch.zeros(input.size()).cuda()
        target_var = autograd.Variable(target, volatile=False)
        modifier_var = autograd.Variable(modifier, volatile=False, requires_grad=True)

        new_img = torch.tanh(modifier_var + input) / 2

        output = model(new_img)

        l2_dist = l2_fn(new_img, torch.tanh(input) / 2)


        if not args.targeted:
            target_var.data = output.data.min(1)[1]

        lower_bound = np.zeros(batch_size)
        const_val = np.ones(batch_size) * args.initial_const
        upper_bound = np.ones(batch_size) * 1e10
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = [np.zeros(input.size()[1:])] * batch_size

        for step in range(args.search_steps):


            for iteration in range(args.max_iterations):


                loss = loss_fn(output, target_var)
                loss.backward()
                input_adv = input_var.data - step_eps * torch.sign(input_var.grad.data)
                #input_adv = input_var.data - step_eps * torch.normal(means=torch.sign(input_var.grad.data), std=0.1)
                input_adv = torch.clamp(input_adv, -1.0, 1.0)
                input_var.data = input_adv
                step += 1
                zero_gradients(input_var)
                output = model(input_var)

                l2s = l2_dist.cpu().numpy()
                nimg = new_img.cpu().numpy()

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    def compare(x, y):
                        if not isinstance(x, (float, int, np.int64)):
                            x = np.copy(x)
                            x[y] -= args.confidence
                            x = np.argmax(x)
                        if args.targeted:
                            return x == y
                        else:
                            return x != y

                    if l2 < best_l2[e] and compare(sc, np.argmax(batch_lab[e])):
                        best_l2[e] = l2
                        best_score[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batch_lab[e])):
                        o_best_l2[e] = l2
                        o_best_score[e] = np.argmax(sc)
                        o_best_attack[e] = ii

                # end inner 'max_iterations' loop

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

            # end binary search steps loop

        o_bestl2 = np.array(o_bestl2)

        # CONNECT THESE

        input_adv = input_adv.permute(0, 2, 3, 1)
        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input_var.size(0)))
        for filename, o in zip(dataset.filenames(indices, basename=True), input_adv.cpu().numpy()):
            output_file = os.path.join(args.output_dir, filename)
            imsave(output_file, (o + 1.0) * 0.5, format='png')


if __name__ == '__main__':
    main()
