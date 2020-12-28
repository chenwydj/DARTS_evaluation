import os
import sys
import warnings
import builtins
import numpy as np
import time
import torch
import utils
from tqdm import tqdm
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes as genotypes
import torch.utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from thop import profile

from torch.autograd import Variable
from model import NetworkImageNet as Network


parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=768, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DrNAS_imagenet', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')

parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')

# args, unparsed = parser.parse_known_args()

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    PID = os.getpid()
    # global best_acc1
    args.gpu = gpu
    print("<< ============== JOB (PID = %d) @ GPU %d ============== >>"%(PID, gpu))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        # set up logs
        args.save = './experiments/imagenet/eval-{}-{}-{}-{}'.format(
            args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
        if args.auxiliary:
            args.save += '-auxiliary-' + str(args.auxiliary_weight)
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        writer = SummaryWriter(args.save)

    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    num_gpus = torch.cuda.device_count()
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = 0
    macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224), ), verbose=True)
    logging.info("param = %f, flops = %f", params, macs)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    train_acc = valid_acc_top1 = valid_acc_top5 = best_acc_top1 = best_acc_top5 = 0
    lr = args.learning_rate

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        ############ master process writes logs #####################
        epoch_bar = tqdm(range(args.epochs), position=0, leave=True)
        for epoch in epoch_bar:
            logging.info("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save))
            if args.distributed:
                train_sampler.set_epoch(epoch)

            if args.lr_scheduler == 'cosine':
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            elif args.lr_scheduler == 'linear':
                current_lr = adjust_lr(args, optimizer, epoch)
            else:
                print('Wrong lr type, exit')
                sys.exit(1)
            if epoch < 5 and args.batch_size > 32:
                current_lr = lr * (epoch + 1) / 5.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / 5.0
                # logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{}/{} | Best: {}/{}'.format(epoch+1, args.epochs, current_lr, train_acc, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            epoch_bar.set_description(description)

            if args.distributed or args.gpu is None:
                model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
            else:
                model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            epoch_start = time.time()
            train_acc, train_obj = train(args, train_queue, model, criterion_smooth, optimizer)
            logging.info('Train_acc: %f', train_acc)
            description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{}/{} | Best: {}/{}'.format(epoch+1, args.epochs, current_lr, train_acc, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            epoch_bar.set_description(description)

            valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
            logging.info('Valid_acc_top1: %f', valid_acc_top1)
            logging.info('Valid_acc_top5: %f', valid_acc_top5)
            description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{}/{} | Best: {}/{}'.format(epoch+1, args.epochs, current_lr, train_acc, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            epoch_bar.set_description(description)
            epoch_duration = time.time() - epoch_start
            # logging.info('Epoch time: %ds.', epoch_duration)

            is_best = False
            if valid_acc_top5 > best_acc_top5:
                best_acc_top5 = valid_acc_top5
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/valid_best_top1", best_acc_top1, epoch)
            writer.add_scalar("acc/valid_best_top5", best_acc_top5, epoch)
            writer.add_scalar("acc/valid_top1", valid_acc_top1, epoch)
            writer.add_scalar("acc/valid_top5", valid_acc_top5, epoch)
            description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{}/{} | Best: {}/{}'.format(epoch+1, args.epochs, current_lr, train_acc, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            epoch_bar.set_description(description)
            logging.info('Best_acc_top1: %f', best_acc_top1)
            logging.info('Best_acc_top5: %f', best_acc_top5)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
                }, is_best, args.save)
    else:
        ############ processes no logs #####################
        for epoch in range(args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            if args.lr_scheduler == 'cosine':
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            elif args.lr_scheduler == 'linear':
                current_lr = adjust_lr(args, optimizer, epoch)
            else:
                print('Wrong lr type, exit')
                sys.exit(1)
            if epoch < 5 and args.batch_size > 32:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / 5.0
            if args.distributed or args.gpu is None:
                model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
            else:
                model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
            train_acc, train_obj = train(args, train_queue, model, criterion_smooth, optimizer)


def adjust_lr(args, optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args, train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
