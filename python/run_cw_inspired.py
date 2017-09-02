import argparse
import os
import torch

from attacks.cw_inspired import CWInspired
from dataset import Dataset

from models import create_ensemble
from experiments.model_configs import config_from_string
import augmentations

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
parser.add_argument('--no_augmentation', action='store_true', default=False,
                    help='No foveation or blurring.')
parser.add_argument('--no_augmentation_blurring', action='store_true', default=False,
                    help='No blurring.')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Batch size (default: 20)')


def main():
    args = parser.parse_args()

    cfgs = [config_from_string(s) for s in args.ensemble]

    target_model = create_ensemble(cfgs, args.ensemble_weights)

    for cfg, model, checkpoint_path in zip(cfgs, target_model.models, args.checkpoint_paths):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.get_core_model().load_state_dict(checkpoint['state_dict'])
        else:
            model.get_core_model().load_state_dict(checkpoint)
        model.get_core_model().cuda()
        model.get_core_model().eval()

    if args.no_augmentation:
        augmentation = lambda x: x
    else:
        if args.no_augmentation_blurring:
            augmentation = augmentations.AugmentationComposer([
                augmentations.Mirror(0.5),
                augmentations.RandomCrop(),  # augmentations.RandomCrop(269),
            ])
        else:
            augmentation = augmentations.AugmentationComposer([
                augmentations.Mirror(0.5),
                augmentations.Blur(0.5, 0.5),
                augmentations.RandomCrop(),  # augmentations.RandomCrop(269),
            ])

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
        batch_size=args.batch_size
    )

    attack.run()


if __name__ == '__main__':
    main()
