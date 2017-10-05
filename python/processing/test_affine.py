import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from processing.affine import Affine

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=4)

(data, target) = next(iter(train_loader))

Image.fromarray((256*data.numpy()[0,0,:,:]).astype(np.uint8)).show()

affiner = Affine(np.pi / 4.0, -np.pi / 4.0,
                 np.pi / 6.0, -np.pi / 6.0,
                 np.pi / 8.0, -np.pi / 8.0,
                 0.9, 1.1,
                 0.9, 1.1)

out = affiner(data)

Image.fromarray((256*out.data.cpu().numpy()[0,0,:,:]).astype(np.uint8)).show()