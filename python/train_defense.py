import os
import argparse
import math
import time
import torch
import torch.utils.data
import numpy as np
from copy import deepcopy
from torchvision import transforms
from torchvision import datasets
from torchvision import utils

from attacks.iterative import AttackIterative
from attacks.cw_inspired import CWInspired
from attacks.selective_universal import SelectiveUniversal
from models import create_ensemble, create_model
from models.model_configs import config_from_string
import processing

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--mp', action='store_true', default=False,
                    help='multi-process training, attack and defense in separate processes')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=299, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--drop', type=float, default=0., metavar='DROP',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='enable sparsity masking for DSD training')


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #utils.save_image(input_var.data, '%d-input.jpg' % i, padding=0)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if args.sparse:
        #    dense_sparse_dense.apply_sparsity_mask(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def attack_factory(model, cfg):
    cfg = deepcopy(cfg)
    attack_name = cfg.pop('attack_name')
    print('Creating attack (%s), with args: ' % attack_name, cfg)
    if attack_name == 'iterative':
        attack = AttackIterative(model, **cfg)
    elif attack_name == 'cw_inspired':
        augmentation = processing.build_anp_augmentation_module()
        augmentation = augmentation.cuda(0)
        attack = CWInspired(model, augmentation, **cfg)
    elif attack_name == 'selective_universal':
        attack = SelectiveUniversal(model, **cfg)
    else:
        assert False, 'Unknown attack'
    return attack


class AttackEnsemble:

    def __init__(self, loader, output_batch_size=8):
        self.loader = loader
        self.attack_cfgs = [
            {'attack_name': 'iterative', 'targeted': True, 'num_steps': 10, 'target_rand': True},
            {'attack_name': 'iterative', 'targeted': False, 'num_steps': 1, 'random_start': True},
            {'attack_name': 'cw_inspired', 'targeted': True, 'n_iter': 38},
            {'attack_name': 'cw_inspired', 'targeted': False, 'n_iter': 38},
        ]
        self.attack_probs = [0.4, 0.4, 0.1, 0.1]
        self.model_cfgs = [  # FIXME these are currently just test configs, need to setup properly
            {'models': ['inception_v3_tf']},
            {'models': ['inception_resnet_v2', 'resnet34'], 'weights': [1.0, .9]},
            {'models': ['adv_inception_resnet_v2', 'inception_v3_tf']},
        ]
        self.max_epsilons = np.array([8., 12., 16.])
        self.max_epsilon_probs = None
        self.models = []
        self.model_idx = None
        self.attack_idx = 0
        self.input_batch_size = loader.batch_size
        self.output_batch_size = output_batch_size
        self.input_device = 0
        self.output_device = 1
        self.img_size = 299

        self._load_models()

    def _load_models(self):
        for mc in self.model_cfgs:
            # pre-load all model params into system (CPU) memory
            cfgs = [config_from_string(x) for x in mc['models']]
            weights = mc['weights'] if 'weights' in mc and len(mc['weights']) else None
            ensemble = create_ensemble(cfgs, weights)
            self.models.append(ensemble)

    def _next_model(self):
        if self.model_idx is not None:
            self.models[self.model_idx].cpu()  # put model params back on CPU
        self.model_idx = inc_roll(self.model_idx, len(self.models))
        model = self.models[self.model_idx]
        model.cuda(self.input_device)   # move model params to GPU
        return model

    def _next_attack(self, model):
        attack_idx = np.random.choice(range(len(self.attack_cfgs)), p=self.attack_probs)
        cfg = self.attack_cfgs[attack_idx]
        if not 'max_epsilon' in cfg:
            cfg['max_epsilon'] = np.random.choice(self.max_epsilons, p=self.max_epsilon_probs)
        attack = attack_factory(model, cfg)
        return attack

    def _initialize_outputs(self):
        with torch.cuda.device(self.output_device):
            output_image = torch.zeros((self.output_batch_size, 3, self.img_size, self.img_size)).cuda()
            output_true_target = torch.zeros((self.output_batch_size,)).long().cuda()
            output_attack_target = torch.zeros((self.output_batch_size,)).long().cuda()
            return output_image, output_true_target, output_attack_target

    def _output_factor(self):
        return max(self.input_batch_size, self.output_batch_size) // self.output_batch_size

    def _input_factor(self):
        return max(self.input_batch_size, self.output_batch_size) // self.input_batch_size

    def __iter__(self):
        images, true_target, attack_target = self._initialize_outputs()
        output_ready = False
        out_idx = 0
        model = self._next_model()
        attack = self._next_attack(model)
        for i, (input, target) in enumerate(self.loader):
            input = input.cuda(self.input_device)
            target = target.cuda(self.input_device)
            in_idx = 0
            for j in range(self._output_factor()):
                # copy unperturbed samples from input to output
                num_u = self.input_batch_size // 2
                print(num_u, input.size(), images.size())
                images[out_idx:out_idx + num_u, :, :, :] = input[in_idx:in_idx + num_u, :, :, :]
                true_target[out_idx:out_idx + num_u] = target[in_idx:in_idx + num_u]
                attack_target[out_idx:out_idx + num_u] = target[in_idx:in_idx + num_u]
                out_idx += num_u
                in_idx += num_u

                # compute perturbed samples for current attack and copy to output
                num_p = self.input_batch_size - num_u

                perturbed, adv_targets = attack(
                    input[in_idx:in_idx + num_p, :, :, :],
                    target[in_idx:in_idx + num_p],
                    batch_idx=i,
                    deadline_time=None)
                if adv_targets is None:
                    adv_targets = target[in_idx:in_idx + num_p]

                images[out_idx:out_idx + num_p, :, :, :] = perturbed
                true_target[out_idx:out_idx + num_p] = target[in_idx:in_idx + num_p]
                attack_target[out_idx:out_idx + num_p] = adv_targets
                out_idx += num_p
                in_idx += num_p

                if out_idx == self.output_batch_size:
                    output_ready = True
                    break

                assert in_idx <= input.size(0)

            if output_ready:
                #FIXME I think we need a process/mult-thread break in this looop, too much wait time
                #and gpu inactivity, surprise surprise
                #print(images.mean(), true_target, attack_target)

                yield images, true_target, attack_target
                images, true_target, attack_target = self._initialize_outputs()
                output_ready = False
                out_idx = 0
                model = self._next_model()
                del attack
                attack = self._next_attack(model)

    def __len__(self):
        return len(self.loader) * self._input_factor()


def main():
    args = parser.parse_args()

    defense_models = ['adv_inception_resnet_v2', 'dpn68b_extra']

    defense_cfgs = [config_from_string(s) for s in defense_models]
    #defense_ensemble = create_ensemble(defense_cfgs, None)

    #FIXME stick with one known model for now to test
    defense_ensemble = create_model(
        'inception_v3', num_classes=1001, aux_logits=False, checkpoint_path='inception_v3_rw.pth',
        normalizer='le', output_fn='log_softmax', drop_first_class=True)

    defense_ensemble.cuda(1)

    optimizer = torch.optim.SGD(
        defense_ensemble.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'validation')

    train_dataset = datasets.ImageFolder(train_dir,  transforms.Compose([
            transforms.RandomSizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    attack_ensemble = AttackEnsemble(train_loader, output_batch_size=16)

    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Scale(int(math.floor(args.img_size / 0.875))),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor()
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = torch.nn.NLLLoss().cuda(1)

    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch, decay_epochs=args.decay_epochs)

        # train for one epoch
        train(args, attack_ensemble, defense_ensemble, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(args, val_loader, defense_ensemble, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'FIXME',
                'state_dict': defense_ensemble.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            filename='checkpoint-%d.pth.tar' % epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def inc_roll(index, length=1):
    if index is None:
        return 0
    else:
        return (index + 1) % length


def adjust_learning_rate(initial_lr, optimizer, epoch, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()


