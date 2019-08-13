# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import DAGLayer
from utils.profiling import profile_time
from profile.profiler import tprof


class BinGateNet(nn.Module):
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
    
