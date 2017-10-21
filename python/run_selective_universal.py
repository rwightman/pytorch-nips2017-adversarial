import argparse

from attacks.image_save_runner import ImageSaveAttackRunner
from attacks.selective_universal import SelectiveUniversal
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

# Args
parser.add_argument('--npy_files', nargs='+', type=str)
parser.add_argument('--try_mirrors', action='store_true', default=False)

def main():
    args = parser.parse_args()

    dataset = Dataset(args.input_dir, target_file='')

    cfgs = [config_from_string(s) for s in args.ensemble]

    target_model = create_ensemble(cfgs, args.ensemble_weights, args.checkpoint_paths).cuda()
    target_model.eval()

    attack = SelectiveUniversal(
        target_model,
        args.npy_files,
        max_epsilon=args.max_epsilon,
        try_mirrors = args.try_mirrors
    )

    runner = ImageSaveAttackRunner(dataset, args.output_dir)
    # Only supports batch size of 1
    runner.run(attack, 1)

if __name__ == '__main__':
    main()
