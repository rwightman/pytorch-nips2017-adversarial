"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from collections import OrderedDict
from dataset import Dataset
from defenses.multi_model import MultiModel
import processing

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE',
                    help='Output file to save labels.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='Batch size (default: 32)')

TIME_LIMIT = 2000

def main():
    args = parser.parse_args()
    assert args.input_dir

    dataset = Dataset(args.input_dir, target_file='')

    model_configs = OrderedDict()
    model_configs['inception_v3'] = {
        'processors': processing.build_8crop_augmentation(target_size=299, norm='le'),
        'num_samples': 8,
        'num_classes': 1001}
    model_configs['inception_resnet_v2'] = {
        'processors': processing.build_8crop_augmentation(target_size=299, norm='le'),
        'num_samples': 8,
        'num_classes': 1001}
    model_configs['dpn107'] = {
        'processors': processing.build_8crop_augmentation(target_size=299, norm='dpn'),
        'num_samples': 8,
        'num_classes': 1000}
    model_configs['dpn68'] = {
        'processors': processing.build_8crop_augmentation(target_size=320, norm='dpn'),
        'num_samples': 8,
        'num_classes': 1000}
    # model_args['densenet169'] = {
    #     'processors': [processing.build_random_augmentation(target_size=224, norm='torchvision')],
    #     'num_samples': 8,
    #     'num_classes': 1000}
    # model_args['wrn50'] = {
    #     'processors': processing.build_4crop_augmentation(target_size=224, norm='torchvision'),
    #     'num_samples': 4,
    #     'num_classes': 1000}
    # model_args['fbresnet200'] = {
    #     'processors': processing.build_2crop_augmentation(target_size=224, norm='torchvision'),
    #     'num_samples': 2,
    #     'num_classes': 1000}

    defense = MultiModel(
        dataset=dataset,
        model_configs=model_configs,
        batch_size=args.batch_size,
        output_file=args.output_file
    )

    defense.run()


if __name__ == '__main__':
    main()