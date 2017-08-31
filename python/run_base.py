import argparse

from models import create_ensemble
from experiments.model_configs import config_from_string
from defenses.base import Base
from dataset import Dataset
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--ensemble', nargs='+', help='Class names for the defensive ensemble.')
parser.add_argument('--ensemble_weights', nargs='+', type=float,
                    help='Weights for weighted geometric mean of output probs')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')


def main():
    args = parser.parse_args()

    cfgs = [config_from_string(s) for s in args.ensemble]

    ensemble = create_ensemble(cfgs, args.ensemble_weights)

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
        batch_size=args.batch_size,
        gpu=not args.no_gpu
    )

    defense.run()


if __name__ == '__main__':
    main()
