# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import DAGLayer
import models.ops as ops
from models.layers import PreprocLayer

class BinGateNet(nn.Module):
    def __init__(self, config, groups, dag_kwargs):
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

        self.dag_layers = nn.ModuleList()
        for i in range(groups):
            stride = 1 if i==0 else 2
            dag_kwargs['stride'] = stride
            dag_kwargs['chn_in'] = (chn_in, ) * self.n_inputs_model
            dag = DAGLayer(**dag_kwargs)
            self.dag_layers.append(dag)
            chn_in = dag.chn_out
        
        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(self.dag_layers[-1].chn_out, self.n_classes)

    def forward(self, x):
        y = self.conv_first(x)
        for dag in self.dag_layers:
            y = dag((y, ) * self.n_inputs_model)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        return y
    
    def build_from_genotype(self, gene, drop_path=True):
        assert len(self.dag_layers) == len(gene.dag)
        for dag, g in zip(self.dag_layers, gene.dag):
            dag.build_from_genotype(g, )
    
    def to_genotype(self, ops):
        gene = []
        for dag in self.dag_layers:
            gene.append(dag.to_genotype(k=2, ops=ops)[1])
        return gene
    
    def dags(self):
        for dag in self.dag_layers:
            yield dag


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class DARTSLikeNet(nn.Module):
    def __init__(self, config, n_layers, shared_a, cell_cls, cell_kwargs):
        super().__init__()
        self.chn_in = config.channel_in
        self.chn = config.channel_init
        self.n_classes = config.classes
        self.shared_a = shared_a
        self.n_inputs_model = config.inputs_model
        assert self.n_inputs_model == 1
        self.n_inputs_layer = config.inputs_layer
        assert self.n_inputs_layer == 2
        self.n_inputs_node = config.inputs_node
        assert self.n_inputs_node == 1
        self.aux_pos = 2*n_layers//3 if config.auxiliary and config.augment else -1

        chn_cur = self.chn * config.channel_multiplier
        self.conv_first = nn.Sequential(
            nn.Conv2d(self.chn_in, chn_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chn_cur),
        )

        chn_pp, chn_p, chn_cur = chn_cur, chn_cur, self.chn

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            stride = 1
            reduction = False
            cell_kwargs['preproc'] = (PreprocLayer, PreprocLayer)
            if i in [n_layers//3, 2*n_layers//3]:
                reduction = True
                stride = 2
                chn_cur *= 2
            if reduction_p:
                cell_kwargs['preproc'] = (ops.FactorizedReduce, PreprocLayer)
            cell_kwargs['chn_in'] = (chn_pp, chn_p)
            cell_kwargs['edge_kwargs']['chn_in'] = (chn_cur, )
            cell_kwargs['stride'] = stride
            if shared_a:
                NASModule.add_shared_param()
            cell = cell_cls(**cell_kwargs)
            self.cells.append(cell)
            chn_out = chn_cur * config.nodes
            chn_pp, chn_p = chn_p, chn_out
            reduction_p = reduction
            if i == self.aux_pos:
                fm_size = 32
                self.aux_head = AuxiliaryHead(fm_size//4, chn_p, self.n_classes)
        
        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(chn_p, self.n_classes)

    
    def forward(self, x):
        aux_logits = None
        s0 = s1 = self.conv_first(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell((s0, s1))
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        y = self.conv_last(s1)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        if aux_logits is None:
            return y
        else:
            return y, aux_logits
    
    def build_from_genotype(self, gene, drop_path=True):
        assert len(self.cells) == len(gene.dag)
        for cell, g in zip(self.cells, gene.dag):
            cell.build_from_genotype(g, )
    
    def to_genotype(self, ops):
        assert ops[-1] == 'none' # assume last PRIMITIVE is 'none'
        gene = []
        for cell in self.cells:
            gene.append(cell.to_genotype(k=2, ops=ops)[1])
        return gene
    
    def dags(self):
        for cell in self.cells:
            yield cell
