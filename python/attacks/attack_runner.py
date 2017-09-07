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
from models import model_factory


def run_attack(args, attack, dataset):

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    model = model_factory.create_model('inception_v3', checkpoint_path=args.checkpoint_path).cuda()

    model.eval()

    for batch_idx, (input, target) in enumerate(loader):
        print("Batch %d of %d" % (batch_idx, len(loader)))
        if not args.no_gpu:
            input = input.cuda()
            target = target.cuda()

        input_adv = attack.run(model, input, target, batch_idx)

        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input.size(0)))
        output_targets = []
        for filename, o, t in zip(dataset.filenames(indices, relname=True), input_adv, target):
            filename = os.path.splitext(filename)[0] + '.png'
            output_targets.append((filename, t))
            output_file = os.path.join(args.output_dir, filename)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            imsave(output_file, np.round(255 * (o + 1.0) * 0.5).astype(np.uint8), format='png')
            with open(os.path.join(args.output_dir, 'output_targets.csv'), mode='a') as cf:
                dw = csv.writer(cf)
                dw.writerows(output_targets)

