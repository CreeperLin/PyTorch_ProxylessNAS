# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.models import ProxylessNASNet, DARTSLikeNet, NASController
from models.layers import DAGLayer, PreprocLayer, MergeFilterLayer,\
                        DARTSMixedOp, BinGateMixedOp
from models.defs import ConcatMerger, SumMerger, LastMerger,\
                        CombinationEnumerator, LastNEnumerator,\
                        SplitAllocator, ReplicateAllocator
import genotypes as gt

from utils.profiling import profile_mem

@profile_mem
def get_proxyless_nas(config, device, dev_list):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_layers = config.layers
    n_nodes = config.nodes
    n_inputs_model = config.inputs_model
    n_inputs_layer = config.inputs_layer
    n_inputs_node = config.inputs_node
    ops = gt.PRIMITIVES_DEFAULT

    proxyless_nas_kwargs = {
        'config': config,
        'n_nodes': n_layers,
        'chn_in': (chn_in, ) * n_inputs_model,
        'shared_a': False,
        'allocator': ReplicateAllocator(),
        'merger_state': SumMerger(),
        'merger_out': LastMerger(),
        'enumerator': LastNEnumerator(),
        'preproc': None,
        'aggregate': None,
        'edge_cls': DAGLayer,
        'edge_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': (chn_cur, ) * n_inputs_layer,
            'shared_a': False,
            'allocator': ReplicateAllocator(),
            'merger_state': SumMerger(),
            'merger_out': ConcatMerger(),
            'enumerator': CombinationEnumerator(),
            'preproc': PreprocLayer,
            'aggregate': None,
            'edge_cls': BinGateMixedOp,
            'edge_kwargs': {
                'config': config,
                'chn_in': (chn, ) * n_inputs_node,
                'stride': 1,
                'ops': ops,
            },
        }
    }
    criterion = nn.CrossEntropyLoss().to(device)
    model = NASController(config, criterion, gt.PRIMITIVES_DEFAULT,
                        dev_list, ProxylessNASNet, proxyless_nas_kwargs).to(device)
    return model

def get_dartslike(config, device, dev_list):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_layers = config.layers
    n_nodes = config.nodes
    n_inputs_model = config.inputs_model
    n_inputs_layer = config.inputs_layer
    n_inputs_node = config.inputs_node
    ops = gt.PRIMITIVES_DEFAULT
    darts_kwargs = {
        'config': config,
        'n_nodes': n_layers,
        'chn_in': (chn_in, ) * n_inputs_model,
        'shared_a': True,
        'allocator': ReplicateAllocator(),
        'merger_state': SumMerger(),
        'merger_out': LastMerger(),
        'enumerator': LastNEnumerator(),
        'preproc': None,
        'aggregate': None,
        'edge_cls': DAGLayer,
        'edge_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': (chn_cur, ) * n_inputs_layer,
            'shared_a': False,
            'allocator': ReplicateAllocator(),
            'merger_state': SumMerger(),
            'merger_out': ConcatMerger(),
            'enumerator': CombinationEnumerator(),
            'preproc': PreprocLayer,
            'aggregate': None,
            'edge_cls': DARTSMixedOp,
            'edge_kwargs': {
                'config': config,
                'chn_in': (chn, ) * n_inputs_node,
                'stride': 1,
                'ops': ops,
            },
        }
    }
    criterion = nn.CrossEntropyLoss().to(device)
    model = NASController(config, criterion, gt.PRIMITIVES_DEFAULT,
                        dev_list, DARTSLikeNet, darts_kwargs).to(device)
    return model

model_creator = {
    'proxyless-nas': get_proxyless_nas,
    'darts-no-reduce': get_dartslike,
}

def get_model(config, device, dev_list):
    mtype = config.type
    if mtype in model_creator:
        return model_creator[mtype](config, device, dev_list)
    else:
        raise Exception("invalid model type")
    return model