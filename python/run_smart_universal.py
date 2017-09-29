import argparse
import time
import torch.nn as nn
import sys

import numpy as np

from attacks.image_save_runner import ImageSaveAttackRunner
from attacks.selective_universal import SelectiveUniversal
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
parser.add_argument('--npy_files', nargs='+', type=str)
parser.add_argument('--ensemble', nargs='+', help='Class names for the defensive ensemble.')
parser.add_argument('--ensemble_weights', nargs='+', type=float,
                    help='Weights for weighted geometric mean of output probs')
parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to checkpoint files for each model.')
parser.add_argument('--try_mirrors', action='store_true', default=False)
parser.add_argument('--time_limit_per_100', type=float, default=470)
parser.add_argument('--no_augmentation', action='store_true', default=False,
                    help='No foveation or blurring.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Learning rate for optimizer')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Batch size (default: 20)')

import torch.utils.data as data
from PIL import Image
import torch
class Subset(data.Dataset):
    def __init__(self, dataset, indices):
        self.original_dataset = dataset
        self.subset_indices = indices
        self.root = dataset.root
        self.imgs = [dataset.imgs[i] for i in indices]
        self.transform = dataset.transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.original_dataset.transform is not None:
            img = self.original_dataset.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[], basename=False):
        original_indices = [self.subset_indices[i] for i in indices]
        return self.original_dataset.filenames(indices=original_indices, basename=basename)

def main():
    attack_start = time.time()

    args = parser.parse_args()

    dataset = Dataset(args.input_dir, target_file='')

    batches_of_100 = len(dataset) / 100.0
    print("Batches of 100: {}".format(batches_of_100))
    print("Time limit per 100: {}".format(args.time_limit_per_100))
    time_limit = max(batches_of_100 * args.time_limit_per_100,1.0)  # Helps pass the validation tool. Otherwise we see 0 images and therefore 0 seconds
    print("Time remaining: {}".format(time_limit))

    FINAL_DEADLINE = attack_start + time_limit

    sys.stdout.flush()

    cfgs = [config_from_string(s) for s in args.ensemble]

    target_model = create_ensemble(cfgs, args.ensemble_weights, args.checkpoint_paths).cuda()
    target_model.eval()

    attack = SelectiveUniversal(
        args.max_epsilon,
        target_model,
        args.npy_files,
        try_mirrors = args.try_mirrors
    )

    runner = ImageSaveAttackRunner(dataset, args.output_dir, time_limit_per_100=args.time_limit_per_100)
    performance = runner.run(attack, 1)

    del attack

    remaining_indices = [idx for idx, perf in enumerate(performance) if not performance[idx]]

    dataset2 = Subset(dataset, remaining_indices)

    if args.no_augmentation:
        augmentation = lambda x: x
    else:
        augmentation = nn.Sequential(
            processing.RandomMirror(0.5),
            processing.RandomGaussianBlur(0.5, 5, 3),
            processing.RandomCrop(),
        )

    attack = CWInspired(
        target_model,
        augmentation,
        max_epsilon=args.max_epsilon,
        n_iter=100,
        lr=args.lr,
        targeted=False,
        target_nth_highest=3,
        prob_dont_augment=0.0,
        initial_w_matrix=None
    )

    time_remaining = FINAL_DEADLINE - time.time()

    images_remaining = len(dataset2)
    time_remaining_per_100 = time_remaining / (images_remaining / 100.0)

    print("Images remaining for cw_inspired: {}".format(images_remaining))
    print("Time remaining: {}".format(time_remaining))
    print("Time remaining per 100: {}".format(time_remaining_per_100))
    sys.stdout.flush()

    runner = ImageSaveAttackRunner(dataset2, args.output_dir, time_limit_per_100=time_remaining_per_100)
    runner.run(attack, args.batch_size)


if __name__ == '__main__':
    main()
