# -*- coding: utf-8 -*-

import os
import time
import logging
import numpy as np
import torch
import adabound
from tensorboardX import SummaryWriter


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


def check_config(hp, name):
    required = (
        # 'data.train',
        # 'data.val',
        'data.search.type',
        'data.augment.type',
    )
    
    flag = False
    for i in required:
        try:
            ddict = hp
            for a in i.split('.'):
                ddict = getattr(ddict, a)
            if ddict == '':
                print('ERROR: check_config: field {} requires non-empty value'.format(i))
                flag = True
        except:
            print('ERROR: check_config: field {} is missing'.format(i))
            flag = True

    defaults = {
        'data.search.root': './data',
        'data.augment.root': './data',
        'data.search.cutout': 0,
        'data.augment.cutout': 16,
    }

    for i in defaults:
        ddict = hp
        try:
            for a in i.split('.'):
                ddict = getattr(ddict, a)
        except:
            if a != i.split('.')[-1]:
                flag = True
                continue
            else:
                print('ERROR: check_config: setting field {} to default: {}'.format(i, defaults[i]))
                setattr(ddict, a, defaults[i])
    
    if flag:
        return True

    hp.train.path = os.path.join('searchs', name)
    hp.train.plot_path = os.path.join(hp.train.path, 'plot')
    print('check_config: OK')
    return False


def init_device(config, ovr_gpus):
    if not ovr_gpus:
        config.gpus = parse_gpus(config.gpus)
    else:
        config.gpus = parse_gpus(ovr_gpus)
    # set default gpu device id
    device = torch.device("cuda")
    torch.cuda.set_device(config.gpus[0])
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    return device, config.gpus


def get_logger(log_dir, name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (name, time.time()))),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def get_writer(log_dir):
    writer = SummaryWriter(log_dir)
    return writer


def get_optim(params, config):
    if config.type == 'adam':
        optimizer = torch.optim.Adam(params,
                            lr=config.lr)
    elif config.type == 'adabound':
        optimizer = adabound.AdaBound(params,
                            lr=config.lr,
                            final_lr=config.final_lr)
    elif config.type == 'sgd':
        optimizer = torch.optim.SGD(params,
                            lr=config.lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
    else:
        raise Exception("Optimizer not supported: %s" % config.optimizer)
    return optimizer


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return 4 * n_params / 1024. / 1024.

def param_count(model):
    """ Compute parameter count in million """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1000. / 1000.

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res
