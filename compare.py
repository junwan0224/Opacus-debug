import argparse
import os
import shutil
import time
import random
import resnet
import numpy
import math
from opacus import PrivacyEngine
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from opacus.utils import module_modification
#import pruneMask as prune

dev = 1
device = torch.device('cuda:1')

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

batch_select = 128
virtual_bn = 1
noise_scale = 0
clip_method = 1
clip_type = 2.0
clip_norm = 100000000000
start_lr = 0.1
gap_rate = 0
prune_percentage = 1
simple_prune = False
multi = noise_scale/clip_norm
num_epoch =200
print("Imagenet L NORM:",clip_type,"Gap",gap_rate,"CLIP NORM",clip_norm,"Grouping",batch_select * virtual_bn,"Noise",noise_scale,"Pruning",  prune_percentage, "Num of Epochs",  num_epoch, "Device", dev)


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=num_epoch, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=batch_select, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=start_lr, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
#parser.add_argument('--resume', default='save_temp/model100new.th', type=str, metavar='PATH', help ='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1, dev
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = resnet.__dict__[args.arch]()
    model2 = resnet.__dict__[args.arch]()
    temp_dict = model.state_dict()
    model2.load_state_dict(temp_dict)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #model = resnet.resnet0()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            temp_dict = model.state_dict()
            for name in temp_dict.keys():
                if (temp_dict[name].shape != checkpoint['state_dict'][name].shape):
                    checkpoint['state_dict'][name] = temp_dict[name]
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = module_modification.convert_batchnorm_modules(model)
    model2 = module_modification.convert_batchnorm_modules(model2)
    model.cuda(device)
    model2.cuda(device)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer2 = torch.optim.SGD(model2.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 250], last_epoch=args.start_epoch - 1)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [20, 30, 40 ,50, 60, 70 ,80, 90, 100, 110, 120, 130, 140,150,160,170,180, 190], gamma=0.8, last_epoch = args.start_epoch - 1)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97, last_epoch=args.start_epoch - 1)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=- 1,verbose=False)

    privacy_engine = PrivacyEngine(model, batch_size=batch_select,sample_size=50000, alphas=range(2, 32),noise_multiplier=multi, max_grad_norm=clip_norm)

    privacy_engine.attach(optimizer)
    #print(f"Using sigma={privacy_engine.noise_multiplier}")


    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print(epoch)
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        #train(train_loader, model, criterion, optimizer, privacy_engine, epoch)
        train(train_loader, model, model2, criterion, optimizer, optimizer2, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(best_prec1)
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


#def train(train_loader, model, criterion, optimizer, pe, epoch):
def train(train_loader, model, model2, criterion, optimizer, optimizer2, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    model2.train()

    #mask = prune.make_mask(model)
    #prune.prune_by_percentile(prune_percentage, mask, model, dev)
    #initial_state_dict = model.state_dict()
    #pe.update_mask(mask)

    end = time.time()
    lr_temp = optimizer.param_groups[0]['lr']
    tau = lr_temp * gap_rate
    print(tau)
    # tau =0
    optimizer.zero_grad()
    sum_grad = {}
    for i, (input, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(device)
        input_var = input.cuda(device)
        target_var = target
        if args.half:
            input_var = input_var.half()

        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()

        output2 = model2(input_var)
        loss2 = criterion(output2, target_var)
        loss2.backward()

        temp_dict = {}
        for name, param in model.named_parameters():
            temp_dict[name] = param.grad
        for name, param in model2.named_parameters():
            if not torch.equal(torch.isclose(param.grad, temp_dict[name]), torch.ones_like(param.grad)):
                print ("before step not equal: ", i, name, param.grad / temp_dict[name])

        optimizer.step()
        optimizer2.step()

        temp_dict = {}
        for name, param in model.named_parameters():
            temp_dict[name] = param.grad
        for name, param in model2.named_parameters():
            if not torch.equal(torch.isclose(param.grad, temp_dict[name]), torch.ones_like(param.grad)):
                print ("after step not equal: ", i, name, param.grad / temp_dict[name])
        optimizer.zero_grad()
        optimizer2.zero_grad()
        #prune.mask_initial(mask, model, initial_state_dict)

        '''
        temp_dict = model.state_dict()
        temp_dict2 = model2.state_dict()
        for name in temp_dict.keys():
            if not torch.equal(torch.isclose(temp_dict[name], temp_dict2[name]), torch.ones_like(temp_dict[name])):
                print(i, name, torch.cdist(temp_dict[name], temp_dict2[name], p = 1))
        '''

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq  == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device)
            input_var = input.cuda(device)
            target_var = target.cuda(device)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
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

