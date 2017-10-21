import argparse

from attacks.image_save_runner import ImageSaveAttackRunner
from attacks.universal_perturbation import UniversalPerturbation
from dataset import Dataset

parser = argparse.ArgumentParser(description='Attack')

# NIPS 2017 Adversarial Interface
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')

# Args
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--npy_file', type=str)

def main():
    args = parser.parse_args()

    dataset = Dataset(args.input_dir, target_file='')

    attack = UniversalPerturbation(
        args.npy_file,
        max_epsilon=args.max_epsilon,
    )

    runner = ImageSaveAttackRunner(dataset, args.output_dir)
    runner.run(attack, args.batch_size)


if __name__ == '__main__':
    main()
