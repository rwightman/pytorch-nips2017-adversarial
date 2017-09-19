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

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data

from dataset import Dataset
from models import create_model
from processing import *

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


TIME_LIMIT = 5000

CHECKPOINTS = {
    'inception_v3': 'adv_inception_v3_rw.pth',
    'densenet121': 'densenet121-fixed.pth',
    'densenet169': 'densenet169-clean.pth',
    'fbresnet200': 'fbresnet200.pth',
    'inception_resnet_v2': 'adv_inception_resnet_v2.pth',
    'dpn107': 'dpn107_extra-fc014e8ec.pth',
    'dpn68b': 'dpn68_extra.pth',
    'dpn68': 'dpn68-abcc47ae.pth',
    'wrn50': 'wrn50.pth'
}


def reduce_mean(x, geom=False):
    if geom:
        if len(x) > 1:
            num_comb = float(len(x))
            o = x[0]
            for xi in x[1:]:
                np.multiply(o, xi, o)
            np.power(o, 1/num_comb, o)
        else:
            o = x[0]
    else:
        o = np.mean(x, axis=0)
    return o


def mrr(x):
    ro = []
    for xi in x:
        r = pd.DataFrame(xi).rank(ascending=False, axis=1).as_matrix()
        ro.append(1 / r)
    return np.mean(ro, axis=0)


class MultiModel:

    def __init__(
            self,
            dataset,
            model_configs,
            batch_size,
            output_file,
            ):
        self.dataset = dataset
        self.model_configs = model_configs
        self.batch_size = batch_size
        self.output_file = output_file

    def run(self):
        loader = data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False)

        outputs = []
        defense_start = time.time()
        for arch, config in self.model_configs.items():
            model_start = time.time()
            num_classes = config['num_classes']
            processors = config['processors']
            model = create_model(
                arch, num_classes=num_classes, checkpoint_path=CHECKPOINTS[arch]).cuda()

            model.eval()

            batch_time = AverageMeter()
            outputs_batch = []
            batch_start = time.time()
            for batch_idx, (input, _) in enumerate(loader):
                input = input.cuda()
                input_var = autograd.Variable(input, volatile=True)

                outputs_samples = []
                for i in range(config['num_samples']):
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
            outputs.append(model_outputs)

            model_dur = time.time() - model_start
            print('Model {0} took {1:.3f} ({2:0.3f} per 100 sample)'.format(
                arch, model_dur, 100 * model_dur / len(self.dataset)))

            total_elapsed = time.time() - defense_start
            if total_elapsed > (TIME_LIMIT - 30):
                print("Warning: breaking defense due to time critical")
                break

        sys.stdout.flush()
        #o = reduce_mean(outputs, geom=False)
        o = mrr(outputs)
        o = np.argmax(o, axis=1) + 1

        defense_dur = time.time() - defense_start
        print('Defense took {0:.3f} ({1:0.3f} per 100 sample)'.format(
            defense_dur, 100 * defense_dur / len(self.dataset)))

        with open(self.output_file, 'w') as out_file:
            filenames = self.dataset.filenames()
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


