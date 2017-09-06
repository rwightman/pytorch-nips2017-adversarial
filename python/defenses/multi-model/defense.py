"""Baseline PyTorch multi-model defense.

A baseline defense using an ensemble of pretrained models as a starting point.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data

from dataset import Dataset
from models import create_model
from augmentations import *

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='Batch size (default: 32)')


TIME_LIMIT = 500

CHECKPOINTS = {
    'inceptionv3': 'inception_v3_google-1a9a5a14.pth',
    'densenet121': 'densenet121-fixed.pth',
    'densenet169': 'densenet169-clean.pth',
    'fbresnet200': 'fbresnet200.pth',
    'inception_resnet_v2': 'adv_inception_resnet_v2.pth',
    'dpn107': 'dpn107_extra-fc014e8ec.pth',
    'wrn50': 'wrn50.pth'
}


def reduce_mean(x, geom=False):
    if geom:
        if len(x) > 1:
            num_comb = float(len(x))
            o = x[0]
            for x in x[1:]:
                np.multiply(o, x, o)
            np.power(o, 1/num_comb, o)
        else:
            o = x[0]
    else:
        o = np.mean(x, axis=0)
    return o


def main():
    args = parser.parse_args()
    assert args.input_dir

    dataset = Dataset(args.input_dir)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    model_args = OrderedDict()
    model_args['dpn107'] = {
        'processors': build_2crop_augmentation(target_size=299, norm='dpn'),
        'num_samples': 2,
        'num_classes': 1000}
    model_args['inception_resnet_v2'] = {
        'processors': build_2crop_augmentation(target_size=299, norm='inception'),
        'num_samples': 2,
        'num_classes': 1001}
    model_args['densenet169'] = {
        'processors': [build_random_augmentation(target_size=224, norm='torchvision')],
        'num_samples': 8,
        'num_classes': 1000}
    model_args['wrn50'] = {
        'processors': build_4crop_augmentation(target_size=224, norm='torchvision'),
        'num_samples': 4,
        'num_classes': 1000}
    model_args['fbresnet200'] = {
        'processors': build_2crop_augmentation(target_size=224, norm='torchvision'),
        'num_samples': 2,
        'num_classes': 1000}

    outputs = []
    defense_start = time.time()
    for arch, margs in model_args.items():
        model_start = time.time()
        num_classes = margs['num_classes']
        processors = margs['processors']
        model = create_model(arch, pretrained=False, num_classes=num_classes).cuda()

        model.eval()

        checkpoint_path = CHECKPOINTS[arch]
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print('Loading checkpoint', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print("Error: No checkpoint found for %s at %s." % (arch, checkpoint_path))
            continue

        batch_time = AverageMeter()
        outputs_batch = []
        batch_start = time.time()
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            input_var = autograd.Variable(input, volatile=True)

            outputs_samples = []
            for i in range(margs['num_samples']):
                input_processed = processors[i % len(processors)](input_var)
                logits = model(input_processed)
                if num_classes > 1000:
                    logits = logits[:, 1:]
                label = F.log_softmax(logits)
                outputs_samples.append(label.data.cpu().numpy().astype(np.float64))

            outputs_batch.append(reduce_mean(outputs_samples, geom=False))

            # measure elapsed time
            current = time.time()
            batch_time.update(current - batch_start)
            batch_start = current

            if False:
                print('Batch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))
                sys.stdout.flush()

        model_outputs = np.concatenate(outputs_batch, axis=0)
        factor = 2 if arch == 'inception_resnet_v2' else 1
        for _ in range(factor):
            outputs.append(model_outputs)

        model_dur = time.time() - model_start
        print('Model {0} took {1:.3f} ({2:0.3f} per 100 sample)'.format(
            arch, model_dur, 100 * model_dur / len(dataset)))

        total_elapsed = time.time() - defense_start
        if total_elapsed > (TIME_LIMIT - 30):
            print("Warning: breaking defense due to time critical")
            break

    sys.stdout.flush()
    o = reduce_mean(outputs, geom=False)
    o = np.argmax(o, axis=1) + 1

    defense_dur = time.time() - defense_start
    print('Defense took {0:.3f} ({1:0.3f} per 100 sample)'.format(
        defense_dur, 100 * defense_dur / len(dataset)))

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, o):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
