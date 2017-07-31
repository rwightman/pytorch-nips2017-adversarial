"""Sample Pytorch defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from models import create_model
from dataset import Dataset

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')

CHECKPOINTS = {
    'inceptionv3': 'inception_v3_google-1a9a5a14.pth',
    'densenet169': 'densenet169-6f0f7f60.pth',
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

    model_params = {}
    model_params['inceptionv3'] = {'img_size': 299}
    model_params['densenet169'] = {'img_size': 224}
    outputs = []

    for arch, args in model_params.items():
        tf = transforms.Compose([
            transforms.Scale(round(args['img_size']/0.875)),
            transforms.CenterCrop(args['img_size']),
            transforms.ToTensor(),
            LeNormalize() if 'inception' in arch else transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset.set_transform(tf)

        model = create_model(arch, pretrained=False, *args)
        if not args.no_gpu:
            model = model.cuda()
        model.eval()

        checkpoint_path = CHECKPOINTS[arch]
        if checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print("Error: No checkpoint found for %s at %s." % (arch, checkpoint_path))

        outputs_raw = []
        #outputs_max = []
        for batch_idx, input in enumerate(loader):
            if not args.no_gpu:
                input = input.cuda()
            input_var = autograd.Variable(input, volatile=True)
            labels = model(input_var)
            outputs_raw.append(labels.data.cpu().numpy())
            #labels = labels.max(1)[1] + 1
            #outputs_max.append(labels.data.cpu().numpy())
        outputs.append(np.concatenate(outputs_raw, axis=0))

    assert outputs
    o = outputs[0]
    for x in outputs[1:]:
        np.multiply(o, x, o)
    np.power(o, 1/len(outputs), o)

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, outputs):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    main()
