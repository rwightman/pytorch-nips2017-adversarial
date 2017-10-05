import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time

from models import Ensemble
from models.mnist.model_factory import create_model as create_mnist_model

from processing import Affine, RandomGaussianBlur, RandomShift, Normalize
from attacks.iterative import AttackIterative
from attacks.cw_inspired import CWInspired


seed = 1
batch_size = 256
test_batch_size = batch_size
log_interval = (60000/batch_size) // 2
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
                   ])),
    batch_size=test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

augmentation = nn.Sequential(
    RandomShift(-2, 2),
    Affine(np.pi / 8.0, -np.pi / 8.0,  # rotation
           np.pi / 8.0, -np.pi / 8.0,  # shear X
           np.pi / 8.0, -np.pi / 8.0,  # shear Y
           0.9, 1.1,                   # scale X
           0.9, 1.1),                  # scale Y
    RandomGaussianBlur(0.5, 5, 0.5, n_channels=1),
)



def train_model(model, n_epochs, save_path, attack=lambda x, *args: (x.permute(0,2,3,1), None, None)):
    optimizer = optim.Adam(model.parameters())

    best_test_loss = 9999999
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            if np.random.rand() < 0.5:
                data, _, _ = attack(data, target)
                data = data.permute(0, 3, 1, 2)

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

        test_loss = test_model(model, attack=attack)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = model.state_dict()

    torch.save(best_state_dict, save_path)

def test_model(model, attack=lambda x, *args: (x.permute(0,2,3,1), None, None)):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, _, _ = attack(data, target, None, time.time()+99999)
        data = data.permute(0, 3, 1, 2)
        data, target = Variable(data.cuda(), volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss

def generate_numpy_file(target, attack):
    target.eval()

    output=[]
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, _, _ = attack(data, target, None, time.time() + 99999)
        output.append(data.cpu().numpy().squeeze())

    return np.concatenate(output)

train_model(create_mnist_model('madry'), n_epochs=30, save_path='madry_natural1.pth')



first_vanilla = mnist.MadryNet().cuda()
first_vanilla.load_state_dict(torch.load('my_mnist.pth'))
test_model(first_vanilla)
second_vanilla = mnist.MadryNet().cuda()
second_vanilla.load_state_dict(torch.load('my_mnist2.pth'))
test_model(second_vanilla)
first_adv = mnist.MadryNet().cuda()
first_adv.load_state_dict(torch.load('an_minist_adv.pth'))
test_model(first_adv)
second_adv = mnist.MadryNet().cuda()
second_adv.load_state_dict(torch.load('an_minist_adv2.pth'))
test_model(second_adv)

ensemble = Ensemble(models=[first_vanilla, second_vanilla, first_adv, second_adv], ensembling_weights=[1.0, 1.0, 3.0, 5.0])
test_model(ensemble)

attack = AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=1)
attack = AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=10)
attack = CWInspired(ensemble, augmentation, max_epsilon=255.0*0.3, n_iter=20, lr=2.0, target_nth_highest=3)
attack = CWInspired(ensemble, augmentation, max_epsilon=255.0*0.3, n_iter=100, lr=0.2, target_nth_highest=3, random_start=0.25)



test_model(ensemble, CWInspired(ensemble, augmentation, max_epsilon=255.0*0.3, n_iter=100, lr=0.2, target_nth_highest=2, random_start=0.25, n_restarts=50))



adv = generate_numpy_file(ensemble,
                          CWInspired(ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=100, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=0.25,
                                     n_restarts=20))
np.save('adv', adv)
np.save('adv_flat', np.reshape(adv,(10000,784)))

oring_adv = adv
adv = np.concatenate(adv)