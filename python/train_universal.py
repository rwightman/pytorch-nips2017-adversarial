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
import torch.optim as optim
from tensorboardX import SummaryWriter
import processing
import torchvision.utils

with open('local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
CHECKPOINT_DIR = local_config['checkpoints_dir']

input_dir = '/media/stuff/ImageNet/train'
ensemble = ['adv_inception_resnet_v2', 'inception_v3', 'resnet18']
ensemble_weights = [1.0, 1.0, 0.5]
checkpoint_paths = [
    os.path.join(CHECKPOINT_DIR, 'adv_inception_resnet_v2.pth'),
    os.path.join(CHECKPOINT_DIR, 'inception_v3_google-1a9a5a14.pth'),
    os.path.join(CHECKPOINT_DIR, 'resnet18-5c106cde.pth')
]
max_epsilon = 16.0
batch_size = 12
img_size = 299

dataset = Dataset(input_dir)

cfgs = [config_from_string(s) for s in ensemble]

target_model = create_ensemble(cfgs, ensemble_weights, checkpoint_paths)

for transformed_model in target_model.models:
    transformed_model.model.cuda()
    transformed_model.model.eval()

eps = max_epsilon / 256.0

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
    def __init__(self, defense_ensemble, defense_augmentation, epsilon):
        super(PerturbationNet, self).__init__()
        self.defense_ensemble = defense_ensemble
        self.defense_augmentation = defense_augmentation
        self.epsilon = epsilon
        self.w_matrix = None

    def forward(self, x):
        perturbed = x + PerturbationNet.delta(self.w_matrix, x, self.epsilon)
        perturbed_clamped = torch.clamp(perturbed, 0.0, 1.0)
        augmented = self.defense_augmentation(perturbed_clamped)
        output = self.defense_ensemble(augmented)
        return output

    @staticmethod
    def delta(wi, x, epsilon):
        return epsilon * torch.tanh(wi)

    def set_w_matrix(self, w_matrix):
        self.w_matrix = w_matrix

augmentation = nn.Sequential(
    processing.RandomMirror(0.5),
    processing.RandomGaussianBlur(0.5, 5, 3),
    processing.RandomCrop()
)

perturbation_model = PerturbationNet(
    target_model,
    augmentation,
    eps
).cuda()

nllloss = torch.nn.NLLLoss().cuda()

w_matrix = autograd.Variable(
    torch.zeros(1, 3, img_size, img_size).cuda(),
    requires_grad=True)
perturbation_model.set_w_matrix(w_matrix)
optimizer = optim.Adam([w_matrix], lr=0.04)

writer = SummaryWriter()

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_reducers = ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, patience=5)

n_batches_per_epoch = 400
epoch_batch_idx = 0
losses = []
for batch_idx, (input, class_id) in enumerate(loader):
    if epoch_batch_idx == n_batches_per_epoch:
        mean_loss = np.mean(losses)
        lr_reducers.step(mean_loss)
        epoch_batch_idx = 0
        losses = []
        img = torch.tanh(perturbation_model.w_matrix).data.cpu()
        writer.add_image('Image', torchvision.utils.make_grid(img, normalize=True), batch_idx)
    epoch_batch_idx = epoch_batch_idx + 1
    input = input.cuda()
    input_var = autograd.Variable(input, volatile=False, requires_grad=True)
    class_var = autograd.Variable(torch.LongTensor(np.repeat(506,batch_size))).cuda()

    probs_perturbed_var = perturbation_model(input_var)
    optimizer.zero_grad()
    loss = nllloss(torch.log(probs_perturbed_var + 1e-8), target=class_var)
    losses.append(loss.data.cpu().numpy())
    loss.backward()

    writer.add_scalar('data/loss', float(loss.data.cpu().numpy()), batch_idx)

    optimizer.step()


np.save('{}_coil'.format(''.join(ensemble)), perturbation_model.w_matrix.data.cpu().numpy())


#from PIL import Image
#Image.fromarray(np.round(255.0 * np.sum(np.isnan(perturbation_model.w_matrix.data.cpu().numpy()), axis=(0,1))).astype(np.uint8)).show()