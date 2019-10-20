# -*- coding: utf-8 -*-

import os
import torch
from torchvision import transforms, datasets
import numpy as np

def get_dataloader(config, metadata):
    if config.type == 'pytorch':
        from .torch_dataloader import get_torch_dataloader
        return get_torch_dataloader(config, metadata)
    else:
        raise ValueError('unsupported dataloader: {}'.format(config.type))

def load_data(config, validation):
    dataset = config.type.lower()

    root = config.valid_root if validation else config.train_root
    os.makedirs(root, exist_ok=True)
    
    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == 'cifar100':
        MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
    elif dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    elif dataset == 'image':
        MEAN = [0.5, 0.5, 0.5]
        STD = [0, 0, 0]
    else:
        raise ValueError('unsupported dataset: {}'.format(dataset))
    
    metadata = (dataset, root, MEAN, STD, validation)

    return get_dataloader(config.dloader, metadata)

