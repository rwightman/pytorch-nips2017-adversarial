"""Pytorch Carlini and Wagner L2 attack runner.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from attacks.attack_carlini_wagner_l2 import AttackCarliniWagnerL2
from attacks.attack_runner import run_attack
from dataset import Dataset, default_inception_transform

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')
parser.add_argument('--steps', type=int, default=None, metavar='N',
                    help='Number of optimization steps to run attack for (default: 1000)')
parser.add_argument('--search_steps', type=int, default=None, metavar='N',
                    help='Number of binary search steps to run attack for (default: 6)')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack')
parser.add_argument('--no_gpu', action='store_true', default=False,
                    help='Disable GPU training')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Enable verbose debug output')


def main():
    args = parser.parse_args()

    transform = default_inception_transform(args.img_size)
    if args.targeted:
        if args.random_target:
            dataset = Dataset(args.input_dir, target_file='', transform=transform, random_target=True)
        else:
            dataset = Dataset(args.input_dir, transform=transform)
    else:
        dataset = Dataset(args.input_dir, target_file='', transform=transform)

    attack = AttackCarliniWagnerL2(
        targeted=args.targeted,
        max_steps=args.steps,
        search_steps=args.search_steps,
        cuda=not args.no_gpu,
        debug=args.debug)

    run_attack(args, attack, dataset)

if __name__ == '__main__':
    main()
