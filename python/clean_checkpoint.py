""" Clean sparse masks and data-parallel prefix from checkpoint files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F

from models import create_model, dense_sparse_dense

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_file', metavar='FILE',
                    help='')
parser.add_argument('--output_file', metavar='FILE',
                    help='')
parser.add_argument('--model', '-m', metavar='MODEL', default='densenet121',
                    help='model architecture (default: densenet121)')


def main():
    args = parser.parse_args()

    model = create_model(args.model, pretrained=False)
    model = model.cuda()

    module_prefix = 'module.'
    module_prefx_len = len(module_prefix)

    checkpoint_file = args.input_file
    if os.path.isfile(args.input_file):
        print("=> loading checkpoint '{}'".format(args.input_file))
        checkpoint = torch.load(args.input_file)

        clean_checkpoint = {}
        for k, v in checkpoint['state_dict'].items():
            # skip extra training baggage
            if k.endswith('.sparsity_mask'):
                continue
            # remove extra 'module.' in key caused by using DataParallel to train
            k_clean = k[module_prefx_len:] if k.startswith(module_prefix) else k
            clean_checkpoint[k_clean] = v

        print("=> loaded checkpoint '{}' (epoch {})".format(args.input_file, checkpoint['epoch']))

        torch.save(clean_checkpoint, args.output_file)

        print("=> saved checkpoint '{}'".format(args.output_file))
    else:
        print("Error: no checkpoint found at '{}'".format(args.input_file))
        exit(-1)




if __name__ == '__main__':
    main()