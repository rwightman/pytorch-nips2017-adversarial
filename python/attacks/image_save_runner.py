"""Attack loop
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import csv
import torch
import torch.utils.data as data

from scipy.misc import imsave


class ImageSaveAttackRunner:

    def __init__(self, dataset, output_dir):
        self.dataset = dataset  # create dataset here instead?
        self.output_dir = output_dir
        self.write_output_targets = False

    def run(self, attack, batch_size):

        loader = data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False)

        reports_returned = []

        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            input_adv, target_adv, report = attack(input, target, batch_idx)

            if torch.is_tensor(input_adv):
                input_adv.permute_(0, 2, 3, 1)
                input_adv = input_adv.cpu().numpy()
            else:
                input_adv = np.transpose(input_adv,  axes=(0, 2, 3, 1))
            if target_adv is None:
                target_adv = target.cpu().numpy()
            elif torch.is_tensor(target_adv):
                target_adv = target_adv.cpu().numpy()
            reports_returned.append(report)

            start_index = batch_size * batch_idx
            indices = list(range(start_index, start_index + input.size(0)))
            output_targets = []
            for filename, o, t in zip(self.dataset.filenames(indices, basename=True), input_adv, target_adv):
                filename = os.path.splitext(filename)[0] + '.png'
                output_targets.append((filename, t))
                output_file = os.path.join(self.output_dir, filename)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                imsave(output_file, np.round(255. * o).astype(np.uint8), format='png')

            if self.write_output_targets:
                with open(os.path.join(self.output_dir, 'output_targets.csv'), mode='a') as cf:
                    dw = csv.writer(cf)
                    dw.writerows(output_targets)

        return reports_returned
