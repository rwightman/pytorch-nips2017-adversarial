import argparse
import os
import torch
import torch.nn as nn

from attacks.cw_inspired import CWInspired
from dataset import Dataset

from models import create_ensemble
from models.model_configs import config_from_string
import processing

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')
parser.add_argument('--ensemble', nargs='+', help='Class names for the defensive ensemble.')
parser.add_argument('--ensemble_weights', nargs='+', type=float,
                    help='Weights for weighted geometric mean of output probs')
parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to checkpoint files for each model.')
parser.add_argument('--n_iter', type=int, default=100,
                    help='Number of iterations in optimization')
parser.add_argument('--target_nth_highest', type=int, default=6,
                    help='Number of iterations in optimization')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Learning rate for optimizer')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Batch size (default: 20)')
parser.add_argument('--initial_w_matrix', type=str, default=None)

# Augmentation Args
parser.add_argument('--no_augmentation', action='store_true', default=False,
                    help='No foveation or blurring.')
parser.add_argument('--gaus_blur_prob', type=float, default=0.5)
parser.add_argument('--gaus_blur_size', type=int, default=5)
parser.add_argument('--gaus_blur_sigma', type=float, default=3.0)
parser.add_argument('--brightness_contrast', action='store_true', default=False)
parser.add_argument('--saturation', action='store_true', default=False)
parser.add_argument('--prob_dont_augment', type=float, default=0.0)


def main():
    args = parser.parse_args()

    cfgs = [config_from_string(s) for s in args.ensemble]

    target_model = create_ensemble(cfgs, args.ensemble_weights, args.checkpoint_paths).cuda()
    target_model.eval()

    if args.no_augmentation:
        augmentation = lambda x: x
    else:
        modules = [
            processing.RandomMirror(0.5)
        ]
        if args.brightness_contrast:
            modules.append(processing.RandomBrightnessContrast())
        if args.saturation:
            modules.append(processing.RandomSaturation())
        modules.extend([
            processing.RandomGaussianBlur(args.gaus_blur_prob, args.gaus_blur_size, args.gaus_blur_sigma),
            processing.RandomCrop()
        ])
        augmentation = nn.Sequential(*modules)

    if args.targeted:
        dataset = Dataset(args.input_dir)
    else:
        dataset = Dataset(args.input_dir, target_file='')

    attack = CWInspired(
        args.input_dir,
        args.output_dir,
        args.max_epsilon,
        target_model,
        augmentation,
        dataset,
        n_iter=args.n_iter,
        lr=args.lr,
        targeted=args.targeted,
        target_nth_highest=args.target_nth_highest,
        batch_size=args.batch_size,
        prob_dont_augment=0.0,
        initial_w_matrix=args.initial_w_matrix
    )

    attack.run()


if __name__ == '__main__':
    main()
