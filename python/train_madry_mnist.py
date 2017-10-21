import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time

from models import Ensemble
from models import create_model_from_cfg

from processing import Affine, RandomGaussianBlur, RandomShift, Normalize
from attacks.iterative import AttackIterative
from attacks.cw_inspired import CWInspired
from attacks.numpy_array_runner import NumpyArrayAttackRunner
from attacks.restart_attack import RestartAttack

seed = 1
batch_size = 256
test_batch_size = batch_size
log_interval = (60000/batch_size) // 2
n_epochs = 30

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
    transforms.ToTensor()
]))

test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

augmentation = nn.Sequential(
    RandomShift(-2, 2),
    Affine(np.pi / 8.0, -np.pi / 8.0,  # rotation
           np.pi / 8.0, -np.pi / 8.0,  # shear X
           np.pi / 8.0, -np.pi / 8.0,  # shear Y
           0.8, 1.2,                   # scale X
           0.8, 1.2),                  # scale Y
)
augmentation.cuda()

runner = NumpyArrayAttackRunner(test_dataset)

def test_model(model, attack=lambda x, *args: (x, None, None)):
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, _, _ = attack(data, target, None)
        data, target = Variable(data.cuda(), volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        #test_loss += F.cross_entropy(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct

def test_model_numpy(model, nparray):
    model.eval()

    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = nparray[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
        data = data.reshape((data.shape[0], 1, 28, 28))
        data = torch.FloatTensor(data)

        data, target = data.cuda(), target.cuda()
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

exact_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_{}.pth'.format(i)}, dataset='mnist') for i in range(7)]

exact_naturals = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadrynatural_{}.pth'.format(i)}, dataset='mnist') for i in range(10)]
modela_naturals = [ create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'modelanatural_{}.pth'.format(i)}, dataset='mnist') for i in range(10)]
modelb_naturals = [ create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'modelbnatural_{}.pth'.format(i)}, dataset='mnist') for i in range(10)]
modelc_naturals = [ create_model_from_cfg({'model_name': 'modelc', 'checkpoint_file': 'modelcnatural_{}.pth'.format(i)}, dataset='mnist') for i in range(10)]
modeld_naturals = [ create_model_from_cfg({'model_name': 'modeld', 'checkpoint_file': 'modeldnatural_{}.pth'.format(i)}, dataset='mnist') for i in range(10)]

madry_advesarials = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'ensadvnataug_madry_{}.pth'.format(i)}, dataset='mnist') for i in range(1)]
modela_adversarials = [ create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'ensadvnataug_modela_{}.pth'.format(i)}, dataset='mnist') for i in range(1)]
modelb_adversarials = [ create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'ensadvnataug_modelb_{}.pth'.format(i)}, dataset='mnist') for i in range(1)]
modelc_adversarials = [ create_model_from_cfg({'model_name': 'modelc', 'checkpoint_file': 'ensadvnataug_modelc_{}.pth'.format(i)}, dataset='mnist') for i in range(1)]
modeld_adversarials = [ create_model_from_cfg({'model_name': 'modeld', 'checkpoint_file': 'ensadvnataug_modeld_{}.pth'.format(i)}, dataset='mnist') for i in range(1)]

ensemble_naturals = [exact_naturals[0], modela_naturals[0], modelb_naturals[0], modelc_naturals[0], modeld_naturals[0]]
ensemble_adversarials = madry_advesarials[0:2] + modela_adversarials[0:2] + modelb_adversarials[0:2] + modelc_adversarials[0:2] + modeld_adversarials[0:2]
ensemble_weights = [1.0 for _ in ensemble_naturals] + [3.0 for _ in ensemble_adversarials]


target_ensemble = Ensemble(models= \
    [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_{}.pth'.format(i)}, dataset='mnist') for i in range(6)] + \
    [ create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'adv_modela_{}.pth'.format(i)}, dataset='mnist') for i in range(2)] + \
    [ create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'adv_modelb_{}.pth'.format(i)}, dataset='mnist') for i in range(2)] + \
    [ create_model_from_cfg({'model_name': 'modelc', 'checkpoint_file': 'adv_modelc_{}.pth'.format(i)}, dataset='mnist') for i in range(2)] + \
    [ create_model_from_cfg({'model_name': 'modeld', 'checkpoint_file': 'adv_modeld_{}.pth'.format(i)}, dataset='mnist') for i in range(2)] + \
    [ create_model_from_cfg({'model_name': 'pytorch-example', 'checkpoint_file': 'adv_pytorch-example_{}.pth'.format(i)}, dataset='mnist') for i in range(2)] \
                           )
target_ensemble.cuda()

"""
submission = runner.run(RestartAttack(
        CWInspired(target_ensemble,
            augmentation,
            max_epsilon=255.0*0.3,
            n_iter=100, lr=0.2,
            targeted=False,
            target_nth_highest=2,
            random_start=True,
            random_start_factor=0.75),4), batch_size)
#np.save('submission_20171013', submission)
#np.save('submission_20171013_flat', submission.reshape(10000,784))
"""
submission_best = np.load('submission_20171013.npy')

test_model_numpy(target_ensemble, submission)
test_model_numpy(target_ensemble, submission_best)

holdout_model = create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_6.pth'}, dataset='mnist')
holdout_model.cuda()
test_model_numpy(holdout_model, submission)
test_model_numpy(holdout_model, submission_best)


for m in target_ensemble.models:
    test_model_numpy(m, submission)


import matplotlib.pyplot as plt

def view(i):
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.imshow(np.array(255*submission[i], dtype='uint8'))
    plt.subplot(122)
    plt.imshow((255*next(iter(test_loader))[0][i][0].numpy()).astype('uint8'))

view(2)
view(3)
view(4)
view(5)
view(6)
view(7)
view(8)
view(9)




for m in exact_models:
    m.cuda()
    test_model(m, AttackIterative(m, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))


ens_adv_model = create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'checkpoint-6.pth.tar'}, dataset='mnist')
ens_adv_model.cuda()
test_model(ens_adv_model, AttackIterative(exact_advesarials[0], max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))
test_model(exact_advesarials[0], AttackIterative(Ensemble(models=[exact_advesarials[1],exact_advesarials[2]]), max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))



test_model(ens_adv_model, CWInspired(exact_advesarials[0],
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=100, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=True,
                                     random_start_factor=0.25,
                                     n_restarts=1))



natural_augmented = [
    create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'modelanaturalaugmented_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'modelbnaturalaugmented_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelc', 'checkpoint_file': 'modelcnaturalaugmented_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modeld', 'checkpoint_file': 'modeldnaturalaugmented_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madrynaturalaugmented_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'pytorch-example', 'checkpoint_file': 'pytexnaturalaugmented_0.pth'}, dataset='mnist'),
]
natural_augmented_ensemble = Ensemble(models=natural_augmented)
natural_augmented_ensemble.cuda()

ens_adv_aug = [
    create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'ensadvnataug_madry_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modela', 'checkpoint_file': 'ensadvnataug_modela_0.pth'}, dataset='mnist'),
    create_model_from_cfg({'model_name': 'modelb', 'checkpoint_file': 'checkpoint-7.pth.tar'}, dataset='mnist'),
]

better_ensemble = Ensemble(models=natural_augmented + ens_adv_aug, ensembling_weights=[1.0 for _ in natural_augmented] + [3.0 for _ in ens_adv_aug])
better_ensemble.cuda()

test_model(exact_advesarials[0], CWInspired(better_ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=50, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=True,
                                     random_start_factor=0.25,
                                     n_restarts=1))




for m in exact_models:
    m.cuda()

for m in exact_models:
    print(test_model(m))
# 9915
# 9905
# 9917
# 9921
# 9916
# 9910
# 9919

# Fixed Start
fgsm_matrix = np.zeros((len(exact_models), len(exact_models)))
for m_source_idx, m_source in enumerate(exact_models):
    for m_target_idx, m_target in enumerate(exact_models):
        fgsm_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))
print(fgsm_matrix)
"""
[[ 9707.  9726.  9743.  9758.  9761.  9748.  9755.]
 [ 9764.  9677.  9764.  9764.  9765.  9761.  9767.]
 [ 9767.  9738.  9706.  9766.  9748.  9748.  9757.]
 [ 9734.  9709.  9723.  9671.  9731.  9725.  9725.]
 [ 9734.  9707.  9724.  9737.  9671.  9717.  9716.]
 [ 9781.  9747.  9764.  9778.  9779.  9716.  9782.]
 [ 9762.  9714.  9741.  9758.  9748.  9737.  9688.]]
"""

# Random Start
fgsm_matrix = np.zeros((len(exact_models), len(exact_models)))
for m_source_idx, m_source in enumerate(exact_models):
    for m_target_idx, m_target in enumerate(exact_models):
        fgsm_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source,
                                                                                       max_epsilon=255.0 * 0.3,
                                                                                       num_steps=1,
                                                                                       targeted=False,
                                                                                       random_start=True,
                                                                                       random_start_method='uniform',
                                                                                       random_start_factor=1))
print(fgsm_matrix)
"""
[[ 9794.  9803.  9828.  9823.  9841.  9826.  9821.]
 [ 9832.  9771.  9820.  9847.  9835.  9846.  9827.]
 [ 9841.  9792.  9785.  9833.  9832.  9841.  9818.]
 [ 9812.  9807.  9814.  9793.  9814.  9806.  9801.]
 [ 9824.  9784.  9817.  9830.  9781.  9817.  9804.]
 [ 9851.  9817.  9830.  9840.  9829.  9776.  9841.]
 [ 9825.  9809.  9808.  9814.  9828.  9817.  9795.]]
"""

exact_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_{}.pth'.format(i)}, dataset='mnist') for i in range(7)]

target_ensemble = Ensemble(models=exact_models[:-1])
target_ensemble.cuda()

submission = generate_numpy_file(target_ensemble,
                          CWInspired(target_ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=400, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=True,
                                     n_restarts=1))

holdout_model = create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_6.pth', 'output_fn':'logsoftmax'}, dataset='mnist')
holdout_model.cuda()
test_model_numpy(holdout_model, submission)


test_model_numpy(target_ensemble, submission)





natural_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadrynatural_{}.pth'.format(i)}, dataset='mnist') for i in range(7)]
target_ensemble = Ensemble(models=natural_models + exact_models)
target_ensemble.cuda()

submission = generate_numpy_file(target_ensemble,
                          CWInspired(target_ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=100, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=True,
                                     n_restarts=1))
test_model_numpy(holdout_model, submission)
test_model_numpy(target_ensemble, submission)


pgd40_matrix = np.zeros((len(exact_models), len(exact_models)))
for m_source_idx, m_source in enumerate(exact_models):
    for m_target_idx, m_target in enumerate(exact_models):
        pgd40_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0*0.3, num_steps=40, targeted=False, step_alpha=0.01))
print(pgd40_matrix)
#[[ 9722.  9555.  9435.  9602.]
# [ 9486.  9491.  9555.  9489.]
# [ 9412.  9471.  9581.  9531.]
# [ 9505.  9509.  9485.  9542.]]






model = create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_0.pth'}, dataset='mnist')
model.cuda()
model.eval()

attack = AttackIterative(model, max_epsilon=255.0 * 0.3, num_steps=2, targeted=False)

data, target = next(iter(test_loader))
data, target = data.cuda(), target.cuda()
data_adv, _, _ = attack(data, target, None, time.time() + 99999)

torch.max(torch.max(data_adv - data, dim=3)[0], dim=2)[0]
torch.min(torch.min(data_adv - data, dim=3)[0], dim=2)[0]

test_model(model, AttackIterative(model, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))



adv_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madry_df_fgiter_{}.pth'.format(i+1)}, dataset='mnist') for i in range(4)]
target_ensemble = Ensemble(adv_models[:3])
target_ensemble.cuda()

holdout_adv = adv_models[-1]
holdout_adv.cuda()

for midx, m in enumerate(adv_models):
    print(test_model(m))
# 9924
# 9917
# 9924
# 9915

fgsm_matrix = np.zeros((len(adv_models), len(adv_models)))
for m_source_idx, m_source in enumerate(adv_models):
    for m_target_idx, m_target in enumerate(adv_models):
        fgsm_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))
print(fgsm_matrix)
#[[ 9814.  9367.  9465.  9275.]
# [ 9437.  9740.  9330.  9507.]
# [ 9434.  9425.  9852.  9424.]
# [ 9461.  9477.  9332.  9764.]]

pgd40_matrix = np.zeros((len(adv_models), len(adv_models)))
for m_source_idx, m_source in enumerate(adv_models):
    for m_target_idx, m_target in enumerate(adv_models):
        pgd40_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))
print(pgd40_matrix)
#[[ 9722.  9555.  9435.  9602.]
# [ 9486.  9491.  9555.  9489.]
# [ 9412.  9471.  9581.  9531.]
# [ 9505.  9509.  9485.  9542.]]

print(test_model(target_ensemble))
# 9923
print(test_model(target_ensemble, AttackIterative(target_ensemble, max_epsilon=255.0*0.3, num_steps=1, targeted=False)))
# 9697
print(test_model(target_ensemble, AttackIterative(target_ensemble, max_epsilon=255.0*0.3, num_steps=40, targeted=False, step_alpha=255.0 * 0.01)))
# 9238

"""
submission = generate_numpy_file(target_ensemble,
                          CWInspired(target_ensemble,
                                     augmentation,
                                     max_epsilon=255.0*0.3,
                                     n_iter=400, lr=0.2,
                                     target_nth_highest=2,
                                     random_start=0.25,
                                     n_restarts=5))
"""
submission = np.load('submission2.npy')

print(test_model_numpy(target_ensemble, submission))
# 8205

for a in adv_models:
    print(test_model_numpy(a, submission))
# 7766
# 7366
# 8777
# 9266 # Holdout

ens_adv_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'madry_dfens_fgiter_{}.pth'.format(i+1)}, dataset='mnist') for i in range(1)]
[ m.cuda() for m in ens_adv_models ]

for midx, m in enumerate(ens_adv_models):
    print(test_model(m))
# 9944

fgsm_matrix = np.zeros((len(adv_models), len(ens_adv_models)))
for m_source_idx, m_source in enumerate(adv_models):
    for m_target_idx, m_target in enumerate(ens_adv_models):
        fgsm_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))
print(fgsm_matrix)
#[[ 9372.]
# [ 9357.]
# [ 9440.]
# [ 9415.]]

fgsm_matrix = np.zeros((len(ens_adv_models), len(ens_adv_models)))
for m_source_idx, m_source in enumerate(ens_adv_models):
    for m_target_idx, m_target in enumerate(ens_adv_models):
        fgsm_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))
print(fgsm_matrix)
# [[ 6224.]]

pgd40_matrix = np.zeros((len(adv_models), len(ens_adv_models)))
for m_source_idx, m_source in enumerate(adv_models):
    for m_target_idx, m_target in enumerate(ens_adv_models):
        pgd40_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))
print(pgd40_matrix)
#[[ 9502.]
# [ 9482.]
# [ 9440.]
# [ 9444.]]


pgd40_matrix = np.zeros((len(ens_adv_models), len(ens_adv_models)))
for m_source_idx, m_source in enumerate(ens_adv_models):
    for m_target_idx, m_target in enumerate(ens_adv_models):
        pgd40_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0*0.3, num_steps=10, targeted=False, step_alpha=0.25*255.0*0.3))
print(pgd40_matrix)
# [[ 4662.]]






exact_models = [ create_model_from_cfg({'model_name': 'madry', 'checkpoint_file': 'exactmadry2_{}.pth'.format(i)}, dataset='mnist') for i in range(7) ]
for m in exact_models:
    m.cuda()

for m in exact_models:
    print(test_model(m))
# 9928

fgsm_matrix = np.zeros((len(exact_models), len(exact_models)))
for m_source_idx, m_source in enumerate(exact_models):
    for m_target_idx, m_target in enumerate(exact_models):
        fgsm_matrix[m_source_idx, m_target_idx] = test_model(m_target, AttackIterative(m_source, max_epsilon=255.0 * 0.3, num_steps=1, targeted=False))
print(fgsm_matrix)
#[[ 9372.]
# [ 9357.]
# [ 9440.]
# [ 9415.]]



"""


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