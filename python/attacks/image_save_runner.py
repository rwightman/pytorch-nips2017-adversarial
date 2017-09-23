"""Attack loop
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
import csv
import torch
import torch.utils.data as data

from scipy.misc import imsave


class ImageSaveAttackRunner:

    def __init__(self, dataset, output_dir, time_limit_per_100=470):
        self.dataset = dataset  # create dataset here instead?
        self.output_dir = output_dir
        self.write_output_targets = False
        self.time_limit_per_100 = time_limit_per_100

    def run(self, attack, batch_size):
        attack_start = time.time()

        loader = data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False)

        time_limit = self.time_limit_per_100 * ((len(self.dataset) - 1) // 100 + 1)
        time_limit_per_batch = time_limit / len(loader)
        batch_time = AverageMeter()

        batch_start = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            batch_deadline = batch_start + time_limit_per_batch

            input = input.cuda()
            target = target.cuda()
            input_adv, target_adv = attack(input, target, batch_idx, batch_deadline)
            if torch.is_tensor(input_adv):
                input_adv.permute_(0, 2, 3, 1)
                input_adv = input_adv.cpu().numpy()
            else:
                input_adv = np.transpose(input_adv,  axes=(0, 2, 3, 1))
            if target_adv is None:
                target_adv = target.cpu().numpy()
            elif torch.is_tensor(target_adv):
                target_adv = target_adv.cpu().numpy()

            start_index = batch_size * batch_idx
            indices = list(range(start_index, start_index + input.size(0)))
            output_targets = []
            for filename, o, t in zip(
                    self.dataset.filenames(indices, basename=True), input_adv, target_adv):

                filename = os.path.splitext(filename)[0] + '.png'
                output_targets.append((filename, t))
                output_file = os.path.join(self.output_dir, filename)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                imsave(output_file, np.round(255. * o).astype(np.uint8), format='png')

            if self.write_output_targets:
                with open(os.path.join(
                        self.output_dir, 'output_targets.csv'), mode='a') as cf:
                    dw = csv.writer(cf)
                    dw.writerows(output_targets)

            # measure elapsed time
            current = time.time()
            batch_time.update(current - batch_start)
            total_elapsed = current - attack_start
            if total_elapsed > (time_limit - 30):
                print("Warning: time critical, %s" % total_elapsed)
            elif total_elapsed > (time_limit - 10):
                print("Warning: breaking early at %d, time critical, %s" % (batch_idx, total_elapsed))
                sys.stdout.flush()
                break
            time_limit_per_batch = min(time_limit - total_elapsed, time_limit_per_batch)
            batch_start = time.time()


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
