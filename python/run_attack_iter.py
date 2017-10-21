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

parser = argparse.ArgumentParser(description='Attack')

# NIPS 2017 Adversarial Interface
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')

# Target Model
parser.add_argument('--ensemble', nargs='+', help='Class names for the defensive ensemble.')
parser.add_argument('--ensemble_weights', nargs='+', type=float,
                    help='Weights for ensembling model outputs.')
parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to checkpoint files for each target_model.')


# Optimization
parser.add_argument('--steps', type=int, default=10, metavar='N',
                    help='Number of steps to run attack for')
parser.add_argument('--step_alpha', type=float, default=0.0,
                    help='Per step scaling constant, defaults to epsilon/steps')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Batch size (default: 20)')
parser.add_argument('--norm', default='inf', type=float,
                    help='Gradient norm.')

# Initialization
parser.add_argument('--random-start', action='store_true', default=False,
                    help='Randomize a starting point from the input image.')
parser.add_argument('--random-start-method', type=str, default='sign',
                    help='Method of selecting random values.')
parser.add_argument('--random-start-factor', type=float, default=0.5,
                    help='Proportion of max_epsilon to scale the random start.')

# Targeting
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack, targets will be read from the file.')
parser.add_argument('--target-min', action='store_true', default=False,
                    help='Least likely class will be the target after an intial forward pass.')
parser.add_argument('--target-rand', action='store_true', default=False,
                    help='Random class will be selected after an initial forward pass.')
parser.add_argument('--target-nth-highest', type=int, default=6,
                    help='The nth most likely class will be selected as a target after an initial forward pass.')
parser.add_argument('--always-target', type=int, default=None,
                    help='Alawys target this same class.')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Enable verbose debug output')

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
        target_model,
        max_epsilon=args.max_epsilon,
        norm=args.norm,
        step_alpha=args.step_alpha,
        num_steps=args.num_steps,
        targeted=args.targeted,
        target_min=args.target_min,
        target_rand=args.target_rand,
        target_nth_highest=args.target_nth_highest,
        always_target=args.always_target,
        random_start=args.random_start,
        random_start_method=args.random_start_method,
        random_start_factor=args.random_start_factor,
        debug=args.debug
    )

    runner = ImageSaveAttackRunner(dataset, args.output_dir)
    runner.run(attack, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
