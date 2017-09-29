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

        batches_of_100 = len(self.dataset) / 100.0
        print("Batches of 100: {}".format(batches_of_100))
        print("Time limit per 100: {}".format(self.time_limit_per_100))
        time_limit = max(batches_of_100 * self.time_limit_per_100, 1.0) # Helps pass the validation tool. Otherwise we see 0 images and therefore 0 seconds
        print("Time remaining: {}".format(time_limit))

        FINAL_DEADLINE = attack_start + time_limit

        n_batches = np.ceil(len(self.dataset) / batch_size)

        reports_returned = []

        for batch_idx, (input, target) in enumerate(loader):
            batch_start = time.time()
            time_remaining = FINAL_DEADLINE - batch_start
            batches_remaining = float(n_batches - batch_idx)
            time_per_batch_remaining = time_remaining / batches_remaining
            batch_deadline = min(batch_start + time_per_batch_remaining, FINAL_DEADLINE)

            input = input.cuda()
            target = target.cuda()
            input_adv, target_adv, report = attack(input, target, batch_idx, batch_deadline)

            if torch.is_tensor(input_adv):
                input_adv = input_adv.cpu().numpy()
            if target_adv is None:
                target_adv = target.cpu().numpy()
            elif torch.is_tensor(target_adv):
                target_adv = target_adv.cpu().numpy()
            reports_returned.append(report)

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
            total_elapsed = current - attack_start
            if total_elapsed > (time_limit - 30):
                print("Warning: time critical, %s" % total_elapsed)
            elif total_elapsed > (time_limit - 10):
                print("Warning: breaking early at %d, time critical, %s" % (batch_idx, total_elapsed))
                sys.stdout.flush()
                break

        return reports_returned


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
