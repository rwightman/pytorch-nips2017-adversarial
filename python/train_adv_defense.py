import os
import argparse
import math
import time
import torch
import torch.utils.data
import torch.multiprocessing as mp
import numpy as np
from copy import deepcopy
from torchvision import transforms
from torchvision import datasets
from torchvision import utils

from models import create_ensemble, create_model
from models.model_configs import config_from_string
from adversarial_generator import AdversarialGenerator
from mp_feeder import MpFeeder

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--mp', action='store_true', default=False,
                    help='multi-process training, attack and defense in separate processes')
parser.add_argument('--num-gpu', default=2, type=int, metavar='N',
                    help='number of gpus to use (default: 2)')
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

        if not input.is_cuda:
            input = input.cuda()
        if not target.is_cuda:
            target = target.cuda()
        if True:
            idx_perm = torch.randperm(input.size(0)).cuda()
            input = input[idx_perm, :, :, :]
            target = target[idx_perm]
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #print(i, target_var.data)
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
        input = input.cuda()
        target = target.cuda()
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


def main():
    args = parser.parse_args()

    num_gpu = args.num_gpu
    if num_gpu == 2:
        input_device = 0
        output_device = 1
    elif num_gpu == 3:
        input_device = [0, 1]
        output_device = 2
    elif num_gpu == 4:
        input_device = [0, 1]
        output_device = [2, 3]
    else:
        assert False, 'Unsupported number of gpus'

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

    adv_dataset = AdversarialGenerator(
        train_loader,
        output_batch_size=args.batch_size,
        input_device=input_device,
        output_device=output_device)
    if args.mp:
        adv_dataset = MpFeeder(adv_dataset, maxsize=8)

    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Scale(int(math.floor(args.img_size / 0.875))),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor()
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    output_master_device = output_device[0] if isinstance(output_device, list) else output_device
    with torch.cuda.device(output_master_device):
        defense_models = ['adv_inception_resnet_v2', 'dpn68b_extra']
        defense_cfgs = [config_from_string(s) for s in defense_models]
        # defense_ensemble = create_ensemble(defense_cfgs, None)

        # FIXME stick with one known model for now to test
        defense_ensemble = create_model(
            'dpn68b', num_classes=1000, checkpoint_path='dpn68_extra.pth',
            normalizer='dpn', output_fn='log_softmax', drop_first_class=False)

        if isinstance(output_device, list):
            defense_ensemble = torch.nn.DataParallel(defense_ensemble, output_device).cuda()
        else:
            defense_ensemble.cuda()

        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(
                defense_ensemble.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        elif args.opt =='adam':
            optimizer = torch.optim.Adam(
                defense_ensemble.parameters(),
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
                    defense_ensemble.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint['epoch']))
                else:
                    # load from a non-training state dict only checkpoint
                    defense_ensemble.load_state_dict(checkpoint)
                    print("=> loaded checkpoint '{}'".format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                exit(-1)

        criterion = torch.nn.NLLLoss().cuda()

        best_prec1 = 0
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(args.lr, optimizer, epoch, decay_epochs=args.decay_epochs)

            # train for one epoch
            train(args, adv_dataset, defense_ensemble, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(args, val_loader, defense_ensemble, criterion)

            #FIXME run another validation on all adversarial examples?

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

    if args.mp:
        adv_dataset.shutdown()
        adv_dataset.done()


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


