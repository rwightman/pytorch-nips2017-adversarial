# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from PIL import Image
import yaml

with open('local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
main_dir = local_config['results_dir']

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description='TODO: Add Description')
    parser.add_argument('attack_name', help='Name of attack')
    parser.add_argument('--targeted', action='store_true', default=False,
                        help='Targeted attack')
    return parser.parse_args()

def main():
    args = parse_args()

    print('Validating images for {}attack {}...'.format('targeted ' if args.targeted else '',args.attack_name))

    all_images = os.listdir(local_config['images_dir'])
    for im in all_images:
        if im.endswith('.png'):
            with open(os.path.join(local_config['images_dir'],im), 'rb') as f:
                original_image = Image.open(f).convert('RGB')
                original_image = np.array(original_image).astype(np.float)
            with open(os.path.join(main_dir,
                                   'targeted_attacks' if args.targeted else 'attacks',
                                   args.attack_name,
                                   im), 'rb') as f:
                perturbed_image = Image.open(f).convert('RGB')
                perturbed_image = np.array(perturbed_image).astype(np.float)

            dif = perturbed_image - original_image
            abs_dif = np.abs(dif)
            max_abs_dif = np.max(abs_dif)
            if max_abs_dif > 16.0:
                print('Invalid abs dif of {} for image{}'.format(max_abs_dif, im))

    print('Done!')

if __name__ == '__main__':
  main()