"""Pytorch Iterate Fast-Gradient attack runner.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from attacks.iterative import AttackIterative
from attacks.image_save_runner import ImageSaveAttackRunner
from dataset import Dataset
from models import create_ensemble
from models.model_configs import config_from_string

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--ensemble', nargs='+', help='Class names for the defensive ensemble.')
parser.add_argument('--ensemble_weights', nargs='+', type=float,
                    help='Weights for weighted geometric mean of output probs')
parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to checkpoint files for each model.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')
parser.add_argument('--steps', type=int, default=10, metavar='N',
                    help='Number of steps to run attack for')
parser.add_argument('--step_alpha', type=float, default=0.0,
                    help='Per step scaling constant, defaults to epsilon/steps')
parser.add_argument('--norm', default='inf', type=float,
                    help='Gradient norm.')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack')
parser.add_argument('--random_start', action='store_true', default=False,
                    help='Random perturb starting point')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Enable verbose debug output')
parser.add_argument('--random_target', action='store_true', default=False,
                    help='Randomize target for attack')


def main():
    args = parser.parse_args()

    cfgs = [config_from_string(s) for s in args.ensemble]

    target_model = create_ensemble(cfgs, args.ensemble_weights, args.checkpoint_paths).cuda()
    target_model.eval()

    if args.targeted:
        dataset = Dataset(args.input_dir)
    else:
        dataset = Dataset(args.input_dir, target_file='')

    attack = AttackIterative(
        model=target_model,
        targeted=args.targeted,
        random_start=args.random_start,
        max_epsilon=args.max_epsilon,
        norm=args.norm,
        step_alpha=args.step_alpha,
        num_steps=args.steps,
        debug=args.debug)

    runner = ImageSaveAttackRunner(dataset, args.output_dir)
    runner.run(attack, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
