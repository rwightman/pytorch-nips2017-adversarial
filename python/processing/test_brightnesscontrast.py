import yaml
import os
from PIL import Image
import torch
import torch.autograd as autograd
from torchvision import transforms
import numpy as np

from processing.colour import RandomBrightnessContrast

with open('local_config.yaml', 'r') as f:
    local_config = yaml.load(f)

one_file = [f for f in os.listdir(local_config['images_dir']) if f.endswith('.png')][0]

img = Image.open(os.path.join(local_config['images_dir'], one_file)).convert('RGB')

img.show()

rbc = RandomBrightnessContrast()

img_tensor_batch = torch.stack([transforms.ToTensor()(img)])
img_tensor_batch.size()
img_tensor_batch_variable = autograd.Variable(img_tensor_batch)

for _ in range(10):

    output = rbc(img_tensor_batch_variable.cuda())

    output = output.data.cpu().numpy()[0]
    output = np.round(255.0 * np.transpose(output, axes=(1, 2, 0))).astype(np.uint8)

    Image.fromarray(output).show()



