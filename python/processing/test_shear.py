import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from processing.shear import Shear

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=4)

(data, target) = next(iter(train_loader))

Image.fromarray((256*data.numpy()[0,0,:,:]).astype(np.uint8)).show()


shearer = Shear(torch.Size([1,1,28,28]), [(1,0),(0,1)], [0.5, 0.5])

out = shearer(data)


Image.fromarray((256*out.data.cpu().numpy()[0,0,:,:]).astype(np.uint8)).show()