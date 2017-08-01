"""Baseline PyTorch multi-model defense.

A baseline defense using an ensemble of pretrained models as a starting point.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import sys

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from models import create_model
from dataset import Dataset

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--no_gpu', action='store_true', default=False,
                    help='disables GPU training')

CHECKPOINTS = {
    'inceptionv3': 'inception_v3_google-1a9a5a14.pth',
    'densenet121': 'densenet121-fixed.pth',
    'densenet169': 'densenet169-6f0f7f60.pth',
    'fbresnet200': 'fbresnet_200-37304a01b.pth',
    'inception_resnet_v2': 'inceptionresnetv2-d579a627.pth'
}


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def main():
    args = parser.parse_args()
    assert args.input_dir

    dataset = Dataset(
        args.input_dir,
        #test_aug=args.tta,
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    model_args = {}
    model_args['inception_resnet_v2'] = {'img_size': 299, 'iscale': 0.875, 'num_classes': 1001}
    model_args['densenet121'] = {'img_size': 224, 'iscale': 0.9,  'num_classes': 1000}
    model_args['densenet169'] = {'img_size': 224, 'iscale': 0.9125, 'num_classes': 1000}
    #model_args['fbresnet200'] = {'img_size': 224, 'num_classes': 1000}
    outputs = []

    for arch, margs in model_args.items():
        num_classes = margs['num_classes']
        tf = transforms.Compose([
            transforms.Scale(round(margs['img_size']/margs['iscale'])),
            transforms.CenterCrop(margs['img_size']),
            transforms.ToTensor(),
            LeNormalize() if 'inception' in arch else transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset.set_transform(tf)

        model = create_model(arch, pretrained=False, num_classes=num_classes)
        if not args.no_gpu:
            model = model.cuda()
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
        sys.stdout.flush()

        outputs_logit = []
        for batch_idx, (input, _) in enumerate(loader):
            if not args.no_gpu:
                input = input.cuda()
            input_var = autograd.Variable(input, volatile=True)
            labels = model(input_var)
            if num_classes > 1000:
                labels = labels[:, 1:]
            logits = F.softmax(labels)
            outputs_logit.append(logits.data.cpu().numpy())
        outputs.append(np.concatenate(outputs_logit, axis=0))

    assert outputs
    num_comb = float(len(outputs))
    o = outputs[0]
    for x in outputs[1:]:
        np.multiply(o, x, o)
    np.power(o, 1/num_comb, o)
    o = np.argmax(o, axis=1) + 1

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, o):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    main()
