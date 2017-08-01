"""Sample Pytorch attack.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
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

    dataset = Dataset(args.input_dir)

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
        if not args.no_gpu:
            input = input.cuda()
            target = target.cuda()
        input_var = autograd.Variable(input, volatile=False, requires_grad=True)
        target_var = autograd.Variable(target, volatile=False)

        step = 0
        while step < num_steps:
            zero_gradients(input_var)
            output = model(input_var)
            loss = loss_fn(output, target_var)
            loss.backward()
            input_adv = input_var.data - step_eps * torch.sign(input_var.grad.data)
            #input_adv = input_var.data - step_eps * torch.normal(means=torch.sign(input_var.grad.data), std=0.1)
            input_adv = torch.clamp(input_adv, -1.0, 1.0)
            input_var.data = input_adv
            step += 1

        input_adv = input_adv.permute(0, 2, 3, 1)
        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input_var.size(0)))
        for filename, o in zip(dataset.filenames(indices, basename=True), input_adv.cpu().numpy()):
            output_file = os.path.join(args.output_dir, filename)
            imsave(output_file, (o + 1.0) * 0.5, format='png')


if __name__ == '__main__':
    main()
