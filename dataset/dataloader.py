# -*- coding: utf-8 -*-

import os
import torch
from torchvision import transforms, datasets
import numpy as np


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def load_data(config, validation=False):
    """ Get dataset """
    dataset = config.type.lower()

    data_root = config.root
    os.makedirs(data_root, exist_ok=True)
    
    if dataset == 'cifar10':
        dset = datasets.CIFAR10
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        trn_transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        val_transf = []
    elif dataset == 'cifar100':
        dset = datasets.CIFAR100
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        trn_transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        val_transf = []
    elif dataset == 'mnist':
        dset = datasets.MNIST
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        trn_transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
        val_transf = []
    elif dataset == 'fashionmnist':
        dset = datasets.FashionMNIST
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        trn_transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
        val_transf = []
    elif dataset == 'imagenet':
        dset = datasets.ImageFolder
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        trn_transf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        val_transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    elif dataset == 'image':
        dset = datasets.ImageFolder
        MEAN = [0, 0, 0]
        STD = [0, 0, 0]
        trn_transf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        val_transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    if config.cutout > 0:
        trn_transf.append(Cutout(cutout))
    
    if dset == datasets.ImageFolder:
        if validation:
            data = dset(config.val, transform=transforms.Compose(trn_transf + normalize))
        else:
            data = dset(config.train, transform=transforms.Compose(trn_transf + normalize))
    elif validation:
        data = dset(config.root, train = False, 
                transform=transforms.Compose(val_transf + normalize), download = True)
    else:
        data = dset(config.root, train = True,
                transform=transforms.Compose(trn_transf + normalize), download = True)
    
    # if config.split:
    #     n_train = len(data)
    #     split = int(n_train * config.sp_ratio)
    #     print('# of train/val data: {}/{}'.format(split, n_train-split))
    #     trn_data = data
    #     val_data = copy.deepcopy(data)
    #     trn_data.data = data.data[]

    return data

