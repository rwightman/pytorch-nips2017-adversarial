import argparse
from models import create_ensemble
from models.model_configs import config_from_string
from defenses.base import Base
from dataset import Dataset
import torch
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Defence')

# NIPS 2017 Adversarial Interface
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')

# Defense Ensemble
parser.add_argument('--ensemble', nargs='+', help='Class names for the defensive ensemble.')
parser.add_argument('--ensemble_weights', nargs='+', type=float,
                    help='Weights for ensembling model outputs.')
parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to checkpoint files for each target_model.')

parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')


def main():
    args = parser.parse_args()

    cfgs = [config_from_string(s) for s in args.ensemble]

    ensemble = create_ensemble(cfgs, args.ensemble_weights, checkpoint_paths=args.checkpoint_paths).cuda()
    ensemble.eval()

    tf = transforms.Compose([
        transforms.Scale(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])
    dataset = Dataset(args.input_dir, transform=tf, target_file='')

    defense = Base(
        args.input_dir,
        args.output_file,
        ensemble,
        dataset,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    defense.run()


if __name__ == '__main__':
    main()
