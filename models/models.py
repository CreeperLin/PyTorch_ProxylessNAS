# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
from models.layers import DAGLayer, NASModule
from utils.profiling import profile_time
from profile.profiler import tprof

from torch.nn.parallel._functions import Broadcast

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, l)
    # l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies

class ProxylessNASNet(nn.Module):
    def __init__(self, config, net_kwargs):
        super().__init__()
        self.chn_in = config.channel_in
        self.chn = config.channel_init
        self.n_classes = config.classes
        self.n_layers = config.layers
        self.n_nodes = config.nodes
        self.n_samples = config.samples
        self.n_inputs_model = config.inputs_model
        self.n_inputs_layer = config.inputs_layer
        self.n_inputs_node = config.inputs_node

        chn_cur = self.chn * config.channel_multiplier
        self.conv_first = nn.Sequential(
            # nn.Conv2d(self.chn_in, chn_cur//2, 3, 1, 2),
            # nn.BatchNorm2d(chn_cur//2),
            # nn.ReLU(),
            # nn.Conv2d(chn_cur//2, chn_cur, 3, 1, 2),
            # nn.BatchNorm2d(chn_cur),
            nn.Conv2d(self.chn_in, chn_cur, 3, 1, 1),
            nn.BatchNorm2d(chn_cur),
        )

        chn_in, chn_cur = chn_cur, self.chn

        self.dag_layers = DAGLayer(**net_kwargs)
        
        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(self.dag_layers.chn_out, self.n_classes)

    # @profile_time
    def forward(self, x):
        tprof.begin_acc_item('model')
        x = self.conv_first(x)
        y = self.dag_layers((x, ) * self.n_inputs_model)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        # tprof.stat_acc('model')
        return y
    
    def to_genotype(self, k, ops):
        return self.dag_layers.to_genotype(k, ops)[1]


class DARTSLikeNet(nn.Module):
    def __init__(self, config, net_kwargs):
        super().__init__()
        self.chn_in = config.channel_in
        self.chn = config.channel_init
        self.n_classes = config.classes
        self.n_layers = config.layers
        self.n_nodes = config.nodes
        self.n_inputs_model = config.inputs_model
        self.n_inputs_layer = config.inputs_layer
        self.n_inputs_node = config.inputs_node

        chn_cur = self.chn * config.channel_multiplier
        self.conv_first = nn.Sequential(
            nn.Conv2d(self.chn_in, chn_cur, 3, 1, 1),
            nn.BatchNorm2d(chn_cur),
        )

        self.dag_layers = DAGLayer(**net_kwargs)
        
        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(self.dag_layers.chn_out, self.n_classes)

    
    def forward(self, x):
        x = self.conv_first(x)
        y = self.dag_layers((x, ) * self.n_inputs_model)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        return y
    

class NASController(nn.Module):
    def __init__(self, config, criterion, ops, device_ids=None, net=None, net_kwargs={}):
        super().__init__()
        self.n_nodes = config.nodes
        self.n_samples = config.samples
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.ops = ops
        self.net = net(config, net_kwargs)
        self.dag_layers = self.net.dag_layers


    def forward(self, x):
        
        NASModule.param_forward()
        
        if len(self.device_ids) == 1:
            return self.net(x)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        # w_dag_copies = Broadcast.apply(self.device_ids, w_dag)
        # s_dag_copies = Broadcast.apply(self.device_ids, s_dag)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(xs),
                                            #  list(zip(xs, w_dag_copies, s_dag_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info(torch.stack(tuple(F.softmax(a.detach(), dim=-1) for a in self.alphas()), dim=0))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self,k=2):
        gene_dag = self.net.to_genotype(k, ops=self.ops)
        return gt.Genotype(dag=gene_dag)

    def weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield p

    def named_weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield n, p

    def alphas(self):
        return NASModule.params()

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def alpha_grad(self, loss):
        return NASModule.params_grad()

    def alpha_backward(self, loss):
        NASModule.param_backward(loss)
    
    def mops(self):
        return NASModule.modules()
