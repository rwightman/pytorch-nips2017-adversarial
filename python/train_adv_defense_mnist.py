import sys
import os
import argparse
import math
import time
import torch
import torch.utils.data
import numpy as np
from copy import deepcopy
from torchvision import transforms
from torchvision import datasets
from torchvision import utils

from models import create_ensemble, create_model_from_cfg, Ensemble
from models.model_configs import config_from_string
from adversarial_generator import AdversarialGenerator
from mp_feeder import MpFeeder
from defenses import multi_task
from processing import Affine, RandomGaussianBlur, RandomShift, Normalize

import train_adv_defense

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mp', action='store_true', default=False,
                    help='multi-process training, attack and defense in separate processes')
parser.add_argument('--num-gpu', default=1, type=int, metavar='N',
                    help='number of gpus to use (default: 1)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=299, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--opt', default='sgd', type=str,
                    metavar='OPT', help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--mt', action='store_true', default=False,
                    help='multi-task defense objective')
parser.add_argument('--co', action='store_true', default=False,
                    help='optimize only defense classifier(s) parameters')
parser.add_argument('--df', action='store_true', default=False,
                    help='dogfood attack with defense model')
parser.add_argument('--model-name', type=str, help='Name of MNIST model to train.')

def main():
    args = parser.parse_args()

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.ToTensor())

    val_dataset = datasets.MNIST('../data', train=False, download=True,
                                 transform=transforms.ToTensor())

    """
    Example data augmentation that could be used.
    The Madry MNIST training uses none.
    https://github.com/MadryLab/mnist_challenge/blob/master/train.py
    
    augmentation = torch.nn.Sequential(RandomShift(-2, 2),
                                       Affine(np.pi / 8.0, -np.pi / 8.0,  # rotation
                                              np.pi / 8.0, -np.pi / 8.0,  # shear X
                                              np.pi / 8.0, -np.pi / 8.0,  # shear Y
                                              0.8, 1.2,  # scale X
                                              0.8, 1.2))  # scale Y
    augmentation.cuda()
    """
    augmentation = lambda x: x

    attack_cfgs = [
        {'attack_name': 'iterative', 'targeted': False, 'num_steps': 40, 'random_start': True, 'max_epsilon': 0.3*255.0, 'step_alpha':0.01,
            'random_start_method':'uniform', 'random_start_factor': 1.0},
    ]

    attack_model_cfgs = []

    defense_model = create_model_from_cfg({'model_name': args.model_name, 'checkpoint_file': None, 'dataset': 'mnist'})

    train_adv_defense.train_adv_defense(args, defense_model, train_dataset, val_dataset, augmentation, attack_model_cfgs, attack_cfgs)

if __name__ == '__main__':
    main()


