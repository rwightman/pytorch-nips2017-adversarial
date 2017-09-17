import argparse

from attacks.selective_universal import SelectiveUniversal
from dataset import Dataset
from models import create_ensemble
from models.model_configs import config_from_string

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

def main():
    args = parser.parse_args()

    dataset = Dataset(args.input_dir, target_file='')

    cfgs = [config_from_string(s) for s in args.ensemble]

    target_model = create_ensemble(cfgs, args.ensemble_weights, args.checkpoint_paths)

    for transformed_model in target_model.models:
        transformed_model.model.cuda()
        transformed_model.model.eval()

    attack = SelectiveUniversal(
        args.input_dir,
        args.output_dir,
        args.max_epsilon,
        target_model,
        args.npy_files,
        dataset
    )

    attack.run()

if __name__ == '__main__':
    main()
