import argparse

from attacks.cw_inspired import CWInspired
from attacks.image_save_runner import ImageSaveAttackRunner
from dataset import Dataset
from models import create_ensemble
from models.model_configs import config_from_string
import processing

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
parser.add_argument('--n_iter', type=int, default=100,
                    help='Number of iterations in optimization')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Learning rate for optimizer')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Batch size (default: 20)')

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

# Initialization
parser.add_argument('--random-start', action='store_true', default=False,
                    help='Randomize a starting point from the input image.')
parser.add_argument('--random-start-factor', type=float, default=0.5,
                    help='Proportion of max_epsilon to scale the random start.')

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
        prob_dont_augment=args.prob_dont_augment,
        targeted=args.targeted,
        target_min=args.target_min,
        target_rand=args.target_rand,
        target_nth_highest=args.target_nth_highest,
        always_target=args.always_target,
        random_start=args.random_start,
        random_start_factor=args.random_start_factor,
    )

    runner = ImageSaveAttackRunner(dataset, args.output_dir)
    runner.run(attack, args.batch_size)


if __name__ == '__main__':
    main()
