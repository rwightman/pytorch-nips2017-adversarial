import argparse

from attacks.universal_perturbation import UniversalPerturbation
from dataset import Dataset

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')
parser.add_argument('--npy_file', type=str)

def main():
    args = parser.parse_args()

    dataset = Dataset(args.input_dir, target_file='')

    attack = UniversalPerturbation(
        args.input_dir,
        args.output_dir,
        args.max_epsilon,
        args.npy_file,
        dataset
    )

    attack.run()

if __name__ == '__main__':
    main()
