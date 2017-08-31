import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import os


class Base(object):
    def __init__(self, *args, **kwargs):
        super(Base, self).__init__( *args, **kwargs)

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