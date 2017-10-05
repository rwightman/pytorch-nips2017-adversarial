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

from models import create_ensemble, create_model_from_cfg, Ensemble
from models.model_configs import config_from_string
from adversarial_generator import AdversarialGenerator
from mp_feeder import MpFeeder
from defenses import multi_task

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mp', action='store_true', default=False,
                    help='multi-process training, attack and defense in separate processes')
parser.add_argument('--num-gpu', default=1, type=int, metavar='N',
                    help='number of gpus to use (default: 1)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
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
parser.add_argument('--opt', default='sgd', type=str,
                    metavar='OPT', help='optimizer')
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
parser.add_argument('--mt', action='store_true', default=False,
                    help='multi-task defense objective')
parser.add_argument('--co', action='store_true', default=False,
                    help='optimize only defense classifier(s) parameters')
parser.add_argument('--df', action='store_true', default=False,
                    help='dogfood attack with defense model')
parser.add_argument('--batch_size', type=int, default=32)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        if isinstance(batch, Exception):
            print('Exception: ', str(batch))
            exit(1)
        elif batch is None:
            break
        input, target, target_adv, is_adv = batch

        # measure data loading time
        data_time.update(time.time() - end)

        if not input.is_cuda:
            input = input.cuda()
        if not target.is_cuda:
            target = target.cuda()
        if not is_adv.is_cuda:
            is_adv = is_adv.cuda()

        idx_perm = torch.randperm(input.size(0)).cuda()
        input = input[idx_perm, :, :, :]
        target = target[idx_perm]
        is_adv = is_adv[idx_perm]

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_adv_var = torch.autograd.Variable(target_adv)
        is_adv_var = torch.autograd.Variable(is_adv)

        #print(i, target_var.data)
        #utils.save_image(input_var.data, '%d-input.jpg' % i, padding=0)

        # compute output
        output = model(input_var)
        if isinstance(output, tuple):
            loss = multi_task.multi_loss(
                output, target_var, target_adv_var, is_adv_var, criterion)
            output = output[0]
        else:
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

        if i % 1000 == 0:
            save_checkpoint({
                     'epoch': epoch + 1,
                     'arch': 'FIXME',
                     'state_dict': model.state_dict(),
                     'best_prec1': top1.avg,
                     'optimizer': optimizer.state_dict(),
                 },
                 False,
                 filename='checkpoint-%d.pth.tar' % epoch)
 

def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        if isinstance(output, tuple):
            output = output[0]
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


def get_opt_params(module, classifier_only=False):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    if classifier_only:
        if isinstance(module, (multi_task.MultiTaskEnsemble, multi_task.MultiTask, Ensemble)):
            opt_params = module.classifier_parameters()
        else:
            opt_params = module.get_classifier().parameters()
    else:
        opt_params = module.parameters()
    return opt_params


def main():
    args = parser.parse_args()

    num_gpu = args.num_gpu
    if num_gpu == 1:
        input_devices = [0]
        output_devices = [0]
    elif num_gpu == 2:
        input_devices = [1]
        output_devices = [0]
    elif num_gpu == 3:
        input_devices = [0, 1]
        output_devices = [2]
    elif num_gpu == 4:
        input_devices = [0, 1]
        output_devices = [2, 3]
    else:
        assert False, 'Unsupported number of gpus'

    master_output_device = output_devices[0]

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    attack_cfgs = [
        {'attack_name': 'iterative', 'targeted': True, 'num_steps': 10, 'target_rand': True},
        {'attack_name': 'iterative', 'targeted': False, 'num_steps': 1, 'random_start': True},
        {'attack_name': 'cw_inspired', 'targeted': True, 'n_iter': 10, 'lr': 0.5, 'n_channels': 1},
        {'attack_name': 'cw_inspired', 'targeted': False, 'n_iter': 10, 'lr': 0.5, 'n_channels': 1},
    ]

    mnist_madry_untrained = create_model_from_cfg(
        {'model_name': 'madry', 'checkpoint_file': None}, dataset='mnist')

    adv_generator = AdversarialGenerator(
        train_loader,
        model_cfgs=[],
        attack_cfgs=attack_cfgs,
        attack_probs=[0.4, 0.4, 0.1, 0.1],
        output_batch_size=args.batch_size,
        input_devices=input_devices,
        master_output_device=master_output_device)

    adv_generator.models = [mnist_madry_untrained]

    if args.mp:
        adv_generator = MpFeeder(adv_generator, maxsize=8)

    with torch.cuda.device(master_output_device):
        defense_ensemble = [True]
        defense_model = create_model_from_cfg(
            {'model_name': 'madry', 'checkpoint_file': None}, dataset='mnist')

        if args.mt:
            if len(defense_ensemble) > 1:
                defense_model = multi_task.MultiTaskEnsemble(
                    defense_model.models,
                    use_features=False)
            else:
                defense_model = multi_task.MultiTask(defense_model)

        if len(output_devices) > 1:
            defense_model = torch.nn.DataParallel(defense_model, output_devices).cuda()
        else:
            defense_model.cuda()

        if args.df:
            adv_generator.set_dogfood(defense_model)

        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(
                get_opt_params(defense_model, classifier_only=args.co),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        elif args.opt =='adam':
            optimizer = torch.optim.Adam(
                get_opt_params(defense_model, classifier_only=args.co),
                args.lr,
                weight_decay=args.weight_decay)
        else:
            assert False, "Invalid optimizer specified"

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    args.start_epoch = checkpoint['epoch']
                    #best_prec1 = checkpoint['best_prec1']
                    defense_model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint['epoch']))
                else:
                    # load from a non-training state dict only checkpoint
                    defense_model.load_state_dict(checkpoint)
                    print("=> loaded checkpoint '{}'".format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                exit(-1)

        criterion = torch.nn.CrossEntropyLoss().cuda()

        best_prec1 = 0
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(args.lr, optimizer, epoch, decay_epochs=args.decay_epochs)

            # train for one epoch
            train(args, adv_generator, defense_model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(args, val_loader, defense_model, criterion)

            #FIXME run another validation on all adversarial examples?

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'FIXME',
                    'state_dict': defense_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename='checkpoint-%d.pth.tar' % epoch)

    if args.mp:
        adv_generator.shutdown()
        adv_generator.done()


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


