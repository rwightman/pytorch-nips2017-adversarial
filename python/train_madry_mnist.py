import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import models.mnist as mnist
from processing import Affine, RandomGaussianBlur, RandomShift, Normalize

seed = 0
batch_size = 64
test_batch_size = 64
lr = 0.01
momentum = 0.5
log_interval = (60000//batch_size) // 1
n_epochs = 30

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True)


model = mnist.MadryNet()
model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters())

augmentation = nn.Sequential(
    RandomShift(-2, 2),
    Affine(np.pi / 8.0, -np.pi / 8.0,  # rotation
           np.pi / 8.0, -np.pi / 8.0,  # shear X
           np.pi / 8.0, -np.pi / 8.0,  # shear Y
           0.9, 1.1,                   # scale X
           0.9, 1.1),                  # scale Y
    RandomGaussianBlur(0.5, 5, 0.5, n_channels=1),
    Normalize((0.1307,), (0.3081,))
)

best_test_loss = 9999999

for epoch in range(n_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        this_batch_size = data.size(0)

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(augmentation(data))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state_dict = model.state_dict()

torch.save(best_state_dict, 'my_mnist.pth')