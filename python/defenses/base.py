import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import os


class Base(object):
    def __init__(self, input_dir, output_file, ensemble, dataset, img_size=299, batch_size=8, gpu=True):
        self.input_dir = input_dir
        self.output_file = output_file
        self.ensemble = ensemble
        self.dataset = dataset
        self.img_size = img_size
        self.batch_size = batch_size
        self.gpu = gpu

    def write_output_file(self, outputs):
        with open(self.output_file, 'w') as out_file:
            filenames = self.dataset.filenames()
            for filename, label in zip(filenames, outputs):
                filename = os.path.basename(filename)
                out_file.write('{0},{1}\n'.format(filename, (label + 1)))

    def run(self):

        tf = transforms.Compose([
            transforms.Scale(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()
        ])
        self.dataset.set_transform(tf)

        loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        outputs = []
        for batch_idx, (input, _) in enumerate(loader):
            if self.gpu:
                input = input.cuda()
            input_var = autograd.Variable(input, volatile=True)
            labels = self.ensemble(input_var)
            labels = labels.max(1)[1]
            outputs.append(labels.data.cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)

        self.write_output_file(outputs)