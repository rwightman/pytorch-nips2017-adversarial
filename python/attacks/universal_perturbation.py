import os
import numpy as np
from scipy.misc import imsave

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class UniversalPerturbation(object):
    def __init__(self,
                 input_dir,
                 output_dir,
                 max_epsilon,
                 universal_perturbation_npy_file,
                 dataset):
        super(UniversalPerturbation, self).__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_epsilon = max_epsilon
        self.universal_perturbation_npy_file = universal_perturbation_npy_file
        self.dataset = dataset

        self.universal_perturbation = torch.tanh(torch.FloatTensor((np.load(self.universal_perturbation_npy_file))))

    def run(self):
        eps = self.max_epsilon / 256.0

        loader = data.DataLoader(
            self.dataset,
            batch_size=10,
            shuffle=False)

        tf = transforms.Compose([
            transforms.Scale(299),
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ])
        self.dataset.set_transform(tf)

        for batch_idx, (input, target) in enumerate(loader):
            this_batch_size = input.size(0)

            perturbed = input + eps * self.universal_perturbation
            clamped = torch.clamp(perturbed, 0.0, 1.0)

            start_index = 10 * batch_idx
            indices = list(range(start_index, start_index + this_batch_size))
            for filename, o in zip(
                    self.dataset.filenames(indices, basename=True), clamped.numpy()):
                output_file = os.path.join(self.output_dir, filename)
                imsave(
                    output_file,
                    np.round(255.0 * np.transpose(o, axes=(1, 2, 0))).astype(np.uint8),
                    format='png')

