from dataset_imagenet import Dataset
from models import create_ensemble
from models.model_configs import config_from_string
import yaml
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import processing
import torchvision.utils
from scipy.misc import imsave
import torch.nn.functional as F

from meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer
from model import Model


with open('local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
CHECKPOINT_DIR = local_config['checkpoints_dir']

input_dir = '/media/stuff/ImageNet/train'

max_epsilon = 8.0
batch_size = 2

ensemble = ['inception_v3_tf']
ensemble_weights = [1.0]

checkpoint_paths = [os.path.join(CHECKPOINT_DIR, config_from_string(m)['checkpoint_file']) for m in ensemble]

dataset = Dataset(input_dir)

img_size = 299
cfgs = [config_from_string(s) for s in ensemble]
target_model = create_ensemble(cfgs, ensemble_weights, checkpoint_paths).cuda()
target_model.eval()

eps = max_epsilon / 255.0

loader = data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True)

tf = transforms.Compose([
    transforms.Scale(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])
dataset.set_transform(tf)

class PerturbationNet(nn.Module):
    def __init__(self, defense_ensemble, epsilon, batch_size):
        super(PerturbationNet, self).__init__()
        self.defense_ensemble = defense_ensemble
        self.defense_augmentation = nn.Sequential(
            processing.RandomMirror(0.5),
            processing.RandomGaussianBlur(0.5, 5, 3),
            processing.RandomCrop()
        )

        self.epsilon = epsilon
        self.w_matrix = nn.Parameter(torch.zeros(batch_size, 3, img_size, img_size).cuda(), requires_grad=True)

    def forward(self, x):
        perturbed = x + PerturbationNet.delta(self.w_matrix, x, self.epsilon)
        perturbed_clamped = torch.clamp(perturbed, 0.0, 1.0)
        augmented = self.defense_augmentation(perturbed_clamped)
        output = self.defense_ensemble(augmented)
        return output

    @staticmethod
    def delta(wi, x, epsilon):
        return epsilon * torch.tanh(wi)

    def parameters(self):
        return nn.ParameterList(parameters=[self.w_matrix])




"""
ABOVE HERE NICE CODE OF MINE
"""


meta_model = PerturbationNet(target_model, max_epsilon, batch_size)
meta_model.cuda()

meta_optimizer = MetaOptimizer(MetaModel(meta_model), 1, 10, 10)
meta_optimizer.cuda()

optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

MAX_EPOCH = 10
UPDATES_PER_EPOCH = 30
OPTIMIZER_STEPS = 30
TRUNCATED_BPTT_STEPS = 30

for epoch in range(MAX_EPOCH):
    decrease_in_loss = 0.0
    final_loss = 0.0
    train_iter = iter(loader)
    for i in range(UPDATES_PER_EPOCH):

        # Sample a new model
        model = PerturbationNet(target_model, max_epsilon, batch_size)

        x, y = next(train_iter)
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        # Compute initial loss of the model
        f_x = model(x)
        initial_loss = F.nll_loss(f_x, y)

        for k in range(OPTIMIZER_STEPS // TRUNCATED_BPTT_STEPS):
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model, use_cuda=True)

            loss_sum = 0
            prev_loss = torch.zeros(1)
            prev_loss = prev_loss.cuda()

            for j in range(TRUNCATED_BPTT_STEPS):
                x, y = next(train_iter)
                x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                # First we need to compute the gradients of the model
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward()

                # Perfom a meta update using gradients from model
                # and return the current meta model saved in the optimizer
                meta_model = meta_optimizer.meta_update(model, loss.data)

                # Compute a loss for a step the meta optimizer
                f_x = meta_model(x)
                loss = F.nll_loss(f_x, y)

                loss_sum += (loss - Variable(prev_loss))

                prev_loss = loss.data

            # Update the parameters of the meta optimizer
            meta_optimizer.zero_grad()
            loss_sum.backward()
            #for param in meta_optimizer.parameters():
            #    param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Compute relative decrease in the loss function w.r.t initial
        # value
        decrease_in_loss += loss.data[0] / initial_loss.data[0]
        final_loss += loss.data[0]

    print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch,
                                                                                  final_loss / UPDATES_PER_EPOCH,
                                                                                  decrease_in_loss / UPDATES_PER_EPOCH))