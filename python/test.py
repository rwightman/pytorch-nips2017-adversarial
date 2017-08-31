import argparse
import os
import torch
from models import create_ensemble
import augmentations

import os
import argparse
import math
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')

def main():
    args = parser.parse_args()

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = Dataset(args.input_dir, transform=tf)
    loader = data.DataLoader(dataset, batch_size=8, shuffle=False)

    cfgs = [{
        'model_name' : 'resnet18',
        'pretrained' : False,
        'num_classes' : 1000,
        'normalize_inputs' : True,
        'resize_inputs' :  True,
        'input_size' : 224,
        'standardize_outputs' : True,
        'drop_first_class' : False,
        'checkpoint_file' : 'resnet18-5c106cde.pth'
    }]

    ensemble = create_ensemble(cfgs, [1.0])
    augment = augmentations.AugmentationComposer([
        augmentations.RandomCrop(269),
        augmentations.Mirror(0.5),
        augmentations.Blur(0.5, 0.5)
    ])

    for cfg, model in zip(cfgs, ensemble.models):
        checkpoint_path = os.path.join('/checkpoints/',cfg['checkpoint_file'])
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.get_core_model().load_state_dict(checkpoint['state_dict'])
        else:
            model.get_core_model().load_state_dict(checkpoint)
        model.get_core_model().cuda()
        model.get_core_model().eval()

    outputs = []
    for batch_idx, (input, _) in enumerate(loader):
        input = input.cuda()
        input_var = autograd.Variable(input, volatile=True)
        labels = ensemble(augment(input_var))
        labels = labels.max(1)[1]
        outputs.append(labels.data.cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, outputs):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    main()
