import argparse
import torch.nn as nn


from attacks.cw_inspired import CWInspired
from attacks.image_save_runner import ImageSaveAttackRunner
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
parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to checkpoint files for each target_model.')
parser.add_argument('--n_iter', type=int, default=100,
                    help='Number of iterations in optimization')
parser.add_argument('--target_nth_highest', type=int, default=6,
                    help='Number of iterations in optimization')
parser.add_argument('--always_target', type=int, default=None)
parser.add_argument('--lr', type=float, default=0.02,
                    help='Learning rate for optimizer')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Batch size (default: 20)')
parser.add_argument('--time_limit_per_100', type=float, default=450)
parser.add_argument('--random_start', type=float)
parser.add_argument('--n_restarts', type=int, default=1)

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
        augmentation = processing.build_anp_augmentation_module(
            saturation=args.saturation,
            brightness_contrast=args.brightness_contrast,
            gaus_blur_prob=args.gaus_blur_prob,
            gaus_blur_size=args.gaus_blur_size,
            gaus_blur_sigma=args.gaus_blur_sigma
        ).cuda()

    if args.targeted:
        dataset = Dataset(args.input_dir)
    else:
        dataset = Dataset(args.input_dir, target_file='')

    attack = CWInspired(
        target_model,
        augmentation,
        max_epsilon=args.max_epsilon,
        n_iter=args.n_iter,
        lr=args.lr,
        targeted=args.targeted,
        target_nth_highest=args.target_nth_highest,
        prob_dont_augment=0.0,
        always_target=args.always_target,
        random_start=args.random_start,
        n_restarts = args.n_restarts
    )

    runner = ImageSaveAttackRunner(dataset, args.output_dir, time_limit_per_100=args.time_limit_per_100)
    runner.run(attack, args.batch_size)


if __name__ == '__main__':
    main()
