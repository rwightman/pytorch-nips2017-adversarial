import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from processing.shift import RandomShift

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=4)

(data, target) = next(iter(train_loader))

Image.fromarray((256*data.numpy()[0,0,:,:]).astype(np.uint8)).show()

shifter = RandomShift(-2,2)
out = shifter(data)

Image.fromarray((256*out.cpu().numpy()[0,0,:,:]).astype(np.uint8)).show()