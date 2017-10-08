import torch
from torch.autograd import Variable
from models import model_factory
from models.model_configs import config_from_string
import numpy as np
import torchvision.transforms

m = model_factory.create_model_from_cfg(config_from_string('inception_v3_tf'), checkpoint_path='/home/aleksey/code/nips2017-nw/shared/checkpoints/inception_v3_rw.pth')
m.eval()

from PIL import Image
img = Image.open('/home/aleksey/nips-100/attacks/round_4_nontargeted_eps16/2ba52bcf59097dc8.png')
img = Image.open('/home/aleksey/code/nips2017-nw/dataset/100/2ba52bcf59097dc8.png')

img = Image.open('/home/aleksey/nips-100/attacks/round_4_nontargeted_eps16/b0e0e06e647b478f.png')
img = Image.open('/home/aleksey/code/nips2017-nw/dataset/100/b0e0e06e647b478f.png')


img = np.array(img).astype(np.float32)

img = torchvision.transforms.ToTensor()(img)

o = m(Variable(img.unsqueeze(0)))

np.argsort(np.exp(o.data.numpy()))
np.sort(np.exp(o.data.numpy()))

o.data.numpy()[0,906]
