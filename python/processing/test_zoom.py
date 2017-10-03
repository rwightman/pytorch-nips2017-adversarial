import torch
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from processing.zoom import RandomZoom
from processing.resize import Resize

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=4)

(data, target) = next(iter(train_loader))
data = Variable(data)

Image.fromarray((256*data.data.numpy()[0,0,:,:]).astype(np.uint8)).show()

random_zoomer = RandomZoom(0.1,2.0)
out = random_zoomer(data)

Image.fromarray((256*out.data.cpu().numpy()[0,0,:,:]).astype(np.uint8)).show()