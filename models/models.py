# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
from models.layers import DAGLayer

from torch.nn.parallel._functions import Broadcast

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

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

    
    def forward(self, x, w_dag):
        w_dag.detach_()
        x = self.conv_first(x)
        s_dag = w_dag.detach().view(-1, w_dag.shape[-1]).multinomial(self.n_samples).view(w_dag.shape[:-1]).unsqueeze(-1)
        y = self.dag_layers((x, ) * self.n_inputs_model, w_dag, s_dag)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        return y

    def freeze(self, freeze, w_dag=None, kwargs={}):
        self.dag_layers.freeze(freeze, w_dag=w_dag, **kwargs)

    def alphas_shape(self):
        return self.dag_layers.alphas_shape

    def alpha_grad(self, loss):
        return self.dag_layers.alpha_grad(loss)

    def mops(self):
        for m in self.dag_layers.mops():
            yield m


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

    
    def forward(self, x, w_dag):
        x = self.conv_first(x)
        y = self.dag_layers((x, ) * self.n_inputs_model, w_dag)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        return y
    
    def freeze(self, freeze, w_dag=None, kwargs={}):
        self.dag_layers.freeze(freeze, w_dag=None, **kwargs)

    def alphas_shape(self):
        return self.dag_layers.alphas_shape

    def alpha_grad(self, loss):
        return self.dag_layers.alpha_grad(loss)

    def mops(self):
        for m in self.dag_layers.mops():
            yield m


class NASController(nn.Module):
    def __init__(self, config, criterion, ops, device_ids=None, net=None, net_kwargs={}):
        super().__init__()
        self.n_nodes = config.nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.ops = ops
        self.net = net(config, net_kwargs)

        # initialize architect parameters: alphas
        self.alpha = nn.Parameter(1e-3*torch.randn(self.net.alphas_shape()))
        print(self.alpha.shape)

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))


    def forward(self, x):
        w_dag = F.softmax(self.alpha, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, w_dag)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        w_dag_copies = broadcast_list(w_dag, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, w_dag_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def freeze(self, freeze, kwargs={}):
        w_dag = F.softmax(self.alpha, dim=-1)
        self.net.freeze(freeze, w_dag=w_dag, **kwargs)

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
        for a in self.alpha:
            logger.info(F.softmax(a, dim=-1))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_dag = gt.parse(self.alpha, k=1, ops=self.ops)
        # gene_merge_filter = 
        # return gt.Genotype(dag=gene_normal, merge_out=gene_merge_filter)
        return gt.Genotype(dag=gene_dag)

    def weights(self, check_grad=False):
        # return self.net.parameters()
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                # print('dis: {}'.format(n))
                continue
            yield p

    def named_weights(self, check_grad=False):
        # return self.net.named_parameters()
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield n, p

    def alphas(self):
        # return self.alpha
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def alpha_grad(self, loss):
        return self.net.alpha_grad(loss)
    
    def mops(self):
        for m in self.net.mops():
            yield m
