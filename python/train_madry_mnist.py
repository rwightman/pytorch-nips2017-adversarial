import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time

from models import Ensemble
from models import create_model_from_cfg, create_ensemble
from models.mnist.networks import MadryNet, PytorchExampleNet

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
    RandomGaussianBlur(0.5, 5, 0.5, n_channels=1).cuda(),
)
augmentation.cuda()

def train_model(model, n_epochs, save_path, attack=lambda x, *args: (x, None, None)):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_test_loss = 9999999
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            if np.random.rand() < 0.5:
                data, _, _ = attack(data, target)

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            data = augmentation(data)
            output = model(data)
            #loss = F.cross_entropy(output, target)
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
            model.cpu()
            best_state_dict = model.state_dict()
            model.cuda()

    torch.save(best_state_dict, save_path)

def test_model(model, attack=lambda x, *args: (x, None, None)):
    model.eval()

    normalizer = Normalize((0.1307,), (0.3081,)).cuda()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, _, _ = attack(data, target, None, time.time()+99999)
        data, target = Variable(data.cuda(), volatile=True), Variable(target)
        data = normalizer(data)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        #test_loss += F.cross_entropy(output, target, size_average=False).data[0]  # sum up batch loss
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
        print(sum([len(x) for x in output]))

    return np.concatenate(output)

def train_natural_models():
    model = create_model_from_cfg({'model_name':'madry', 'checkpoint_file':None}, dataset='mnist')
    model.cuda()
    train_model(model, n_epochs=30, save_path='madry.pth')

    model = create_model_from_cfg({'model_name':'pytorch-example', 'checkpoint_file':None}, dataset='mnist')
    model.cuda()
    train_model(model, n_epochs=30, save_path='pytex.pth')

    for name in ['modela', 'modelb', 'modelc', 'modeld']:
        model = create_model_from_cfg({'model_name': name, 'checkpoint_file': None}, dataset='mnist')
        model.cuda()
        train_model(model, n_epochs=30, save_path='{}.pth'.format(name))

def test_model_numpy(model, nparray):
    model.eval()

    normalizer = Normalize((0.1307,), (0.3081,)).cuda()

    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = nparray[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
        data = data.reshape((data.shape[0], 1, 28, 28))
        data = torch.FloatTensor(data)

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data.cuda(), volatile=True), Variable(target)
        data = normalizer(data)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        #test_loss += F.cross_entropy(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


adv_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madry_df_fgiter_{}.pth'.format(i+1)}, dataset='mnist') for i in range(3)]
target_ensemble = Ensemble(adv_models)
target_ensemble.cuda()

test_model(target_ensemble)
test_model(target_ensemble, AttackIterative(target_ensemble, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(target_ensemble, AttackIterative(target_ensemble, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))

submission = generate_numpy_file(target_ensemble,
                          CWInspired(target_ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=400, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=0.25,
                                     n_restarts=5))


test_model_numpy(target_ensemble, submission)

for a in adv_models:
    print(test_model_numpy(a, submission))

natural_models = [
    create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madry.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'pytorch-example', 'checkpoint_file': 'pytex.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'modela.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'modelb.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelc', 'checkpoint_file': 'modelc.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modeld', 'checkpoint_file': 'modeld.pth'}, dataset='mnist'),
]

for m in natural_models:
    m.cuda()
    print(test_model_numpy(m, submission))

holdout_adv = create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madry_df_fgiter_{}.pth'.format(4)}, dataset='mnist')
holdout_adv.cuda()
print(test_model_numpy(holdout_adv, submission))

train_natural_models()

madry_natural = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry_natural1.pth', dataset='mnist')
madry_natural.cuda()
pytex_natural = create_model_from_cfg({'model_name':'pytorch-example'}, checkpoint_path='pt-ex1.pth', dataset='mnist')
pytex_natural.cuda()
madry_adv = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry-adv1.pth', dataset='mnist')
madry_adv.cuda()
madry_adv2 = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry-adv2.pth', dataset='mnist')
madry_adv2.cuda()
madry_adv3 = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry-adv3.pth', dataset='mnist')
madry_adv3.cuda()

madry_adv4 = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='checkpoint-49.pth.tar', dataset='mnist')
madry_adv4.cuda()

madry_adv5 = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='checkpoint-49.pth.tar', dataset='mnist')
madry_adv5.cuda()

test_model(madry_adv4)
test_model(madry_adv4, AttackIterative(madry_adv4, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_adv4, AttackIterative(madry_adv4, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))

test_model(madry_adv5)
test_model(madry_adv5, AttackIterative(madry_adv5, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_adv5, AttackIterative(madry_adv5, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))


print(test_model(madry_adv, AttackIterative(madry_adv, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
print(test_model(madry_adv, AttackIterative(madry_adv, max_epsilon=255.0*0.3, num_steps=10, targeted=False)))
print(test_model(madry_adv, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
print(test_model(madry_adv, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
print(test_model(madry_adv, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3)))
print(test_model(madry_adv, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3)))

print(test_model(madry_adv2, AttackIterative(madry_adv2, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
print(test_model(madry_adv2, AttackIterative(madry_adv2, max_epsilon=255.0*0.3, num_steps=10, targeted=False)))
print(test_model(madry_adv2, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
print(test_model(madry_adv2, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
print(test_model(madry_adv2, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3)))
print(test_model(madry_adv2, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3)))

test_model(madry_adv3)
test_model(madry_adv3, AttackIterative(madry_adv3, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_adv3, AttackIterative(madry_adv3, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))


adv = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='checkpoint-29.pth.tar', dataset='mnist')
adv.cuda()
adv2 = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='checkpoint-49.pth.tar', dataset='mnist')
adv2.cuda()

test_model(adv)
test_model(adv, AttackIterative(adv, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(adv, AttackIterative(adv, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))

test_model(adv2)
test_model(adv2, AttackIterative(adv2, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(adv2, AttackIterative(adv2, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))
test_model(adv2, AttackIterative(adv, max_epsilon=255.0*0.3, num_steps=1, targeted=False))


natural_models = [
    create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madry.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'pytorch-example', 'checkpoint_file': 'pytex.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'modela.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'modelb.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelc', 'checkpoint_file': 'modelc.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modeld', 'checkpoint_file': 'modeld.pth'}, dataset='mnist'),
]

for m in natural_models:
    m.cuda()
    test_model(adv, AttackIterative(m, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))

for m in natural_models:
    m.cuda()
    test_model(adv, AttackIterative(m, max_epsilon=255.0 * 0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))


super_ensemble_models = natural_models + [adv]
super_ensemble = Ensemble(super_ensemble_models, ensembling_weights=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0])

submission = generate_numpy_file(super_ensemble,
                          CWInspired(super_ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=100, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=0.25,
                                     n_restarts=20))


"""

madry_adv = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry-adv.pth', dataset='mnist')
madry_adv.cuda()


ensemble = Ensemble(models=[madry_natural, pytex_natural], ensembling_weights=[1.0, 1.0])

test_model(madry_natural)
test_model(pytex_natural)
test_model(madry_adv)
test_model(ensemble)

test_model(madry_natural, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_natural, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(pytex_natural, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(pytex_natural, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(ensemble, AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=1, targeted=False))

test_model(madry_adv, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_adv, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_adv, AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(madry_adv, AttackIterative(madry_adv, max_epsilon=255.0*0.3, num_steps=1, targeted=False))




test_model(madry_natural, AttackIterative(madry_natural, max_epsilon=255.0*0.3, num_steps=10, targeted=False))
test_model(pytex_natural, AttackIterative(pytex_natural, max_epsilon=255.0*0.3, num_steps=10, targeted=False))
test_model(ensemble, AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=10, targeted=False))

model = create_model_from_cfg({'model_name':'pytorch-example', 'checkpoint_file':None}, dataset='mnist')
model.cuda()
train_model(model, 30, save_path='pt-ex-adv1.pth', attack=AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=10, targeted=False))

model = create_model_from_cfg({'model_name':'madry', 'checkpoint_file':None}, dataset='mnist')
model.cuda()
train_model(model, 30, save_path='madry-adv1.pth', attack=AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=10, targeted=False))

from collections import OrderedDict
def clean_checkpoint(path):
    ckpt = torch.load(path)

    if 'state_dict' in ckpt:
        new_dict = OrderedDict()
        for k, v in ckpt.items():
            if k == 'state_dict':
                new_state_dict = OrderedDict()
                for name, value in v.items():
                    name = name.replace('model.','')
                    new_state_dict[name] = value
                new_dict['state_dict'] = new_state_dict
            else:
                new_dict[k] = v
        torch.save(new_dict, path)
    else:
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            name = k.replace('model.', '')
            new_state_dict[name] = v
        torch.save(new_state_dict, path)


#clean_checkpoint('madry-adv-nat1-df.pth')


madry_adv = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry-adv1.pth', dataset='mnist')
madry_adv.cuda()
pytex_adv = create_model_from_cfg({'model_name':'pytorch-example'}, checkpoint_path='pt-ex-adv1.pth', dataset='mnist')
pytex_adv.cuda()

test_model(madry_adv)
test_model(pytex_adv)

test_model(madry_adv, AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(pytex_adv, AttackIterative(ensemble, max_epsilon=255.0*0.3, num_steps=1, targeted=False))

test_model(madry_adv, AttackIterative(madry_adv, max_epsilon=255.0*0.3, num_steps=1, targeted=False))
test_model(pytex_adv, AttackIterative(pytex_adv, max_epsilon=255.0*0.3, num_steps=1, targeted=False))

test_model(madry_adv, AttackIterative(madry_adv, max_epsilon=255.0*0.3, num_steps=10, targeted=False))
test_model(pytex_adv, AttackIterative(pytex_adv, max_epsilon=255.0*0.3, num_steps=10, targeted=False))

ensemble2 =  Ensemble([madry_natural, pytex_natural, madry_adv, pytex_adv], ensembling_weights=[1.0, 1.0, 3.0, 2.0])

model = create_model_from_cfg({'model_name':'madry', 'checkpoint_file':None}, dataset='mnist')
model.cuda()
train_model(model, 30, save_path='madry-adv2.pth', attack=AttackIterative(ensemble2, max_epsilon=255.0*0.3, num_steps=10, targeted=False))
clean_checkpoint('madry-adv2.pth')

madry_adv2 = create_model_from_cfg({'model_name':'madry'}, checkpoint_path='madry-adv2.pth', dataset='mnist')
madry_adv2.cuda()

test_model(madry_adv2)
test_model(madry_adv2, AttackIterative(madry_adv2, max_epsilon=255.0*0.3, num_steps=1, targeted=False))

test_model(madry_adv2, AttackIterative(ensemble2, max_epsilon=255.0*0.3, num_steps=10, targeted=False))



model = create_model_from_cfg({'model_name':'madry', 'checkpoint_file':None}, dataset='mnist')
model.cuda()
train_model(model, 30, save_path='madry-test.pth', attack=AttackIterative(model, max_epsilon=255.0*0.3, num_steps=1, targeted=False))




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

"""