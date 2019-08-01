# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.models import ProxylessNASNet, DARTSLikeNet, NASController
import genotypes as gt

def get_model(config, device, dev_list):
    # criterion = ProxylessNASLossLayer().to(device)
    if config.type == 'proxyless-nas':
        criterion = nn.CrossEntropyLoss().to(device)
        model = NASController(config, ProxylessNASNet, criterion, gt.PRIMITIVES_DEFAULT, dev_list).to(device)
    elif config.type == 'darts-no-reduce':
        criterion = nn.CrossEntropyLoss().to(device)
        model = NASController(config, DARTSLikeNet, criterion, gt.PRIMITIVES_DEFAULT, dev_list).to(device)
    else:
        raise Exception("invalid model type")
    return model