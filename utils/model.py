# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.dagnet import BinGateNet, DARTSLikeNet
from models.pyramidnet import PyramidNet
from models.proxylessnas import ProxylessNASNet
from models.layers import DAGLayer, TreeLayer,\
                        PreprocLayer, MergeFilterLayer
from models.nas_modules import DARTSMixedOp, BinGateMixedOp,\
                                NASController
from models.defs import ConcatMerger, SumMerger, LastMerger, SumMerger,\
                        CombinationEnumerator, LastNEnumerator,\
                        TreeEnumerator, FirstNEnumerator,\
                        EvenSplitAllocator, ReplicateAllocator
from architect import DARTSArchitect, BinaryGateArchitect
import genotypes as gt

from utils.profiling import profile_mem
from utils import param_size, param_count

def get_proxylessnasnet(config, device, dev_list):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_groups = config.groups
    conv_groups = config.conv_groups
    n_blocks = config.blocks
    alpha = config.alpha
    ops = gt.PRIMITIVES_DEFAULT
    model_config = {
	    'start_planes': chn_cur,
	    'alpha': alpha,
	    'block_per_group': n_blocks,
	    'total_groups': n_groups,
	    'downsample_type': 'avg_pool',  # avg, max
	    ######################################################
	    'bottleneck': 4,
	    'ops_order': 'bn_act_weight',
	    'dropout_rate': 0,
	    ######################################################
	    'final_bn': True,
	    'no_first_relu': True,
	    'use_depth_sep_conv': False,
	    'groups_3x3': conv_groups,
	    ######################################################
	    'path_drop_rate': 0,
	    'use_zero_drop': True,
        'drop_only_add': False,
        'edge_cls': BinGateMixedOp,
        'edge_kwargs': {
            'config': config,
            'chn_in': None,
            'stride': 1,
            'ops': ops,
            'shared_a': False,
        }
    }
    criterion = nn.CrossEntropyLoss().to(device)
    net = ProxylessNASNet.set_standard_net(data_shape=(chn_in, 32, 32), n_classes=n_classes, **model_config)
    model = NASController(config, criterion, gt.PRIMITIVES_DEFAULT,
                        dev_list, net=net).to(device)
    arch = BinaryGateArchitect
    return model, arch


def get_pyramidnet(config, device, dev_list):
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

    pyramidnet_kwargs = {
        'config': config,
        'n_nodes': n_layers,
        'chn_in': (chn_cur, ) * n_inputs_model,
        'shared_a': False,
        'cell_cls': TreeLayer,
        'cell_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': (chn_cur, ) * n_inputs_layer,
            'shared_a': False,
            'allocator': ReplicateAllocator,
            'merger_out': SumMerger,
            'preproc': PreprocLayer,
            'aggregate': None,
            'edge_cls': BinGateMixedOp,
            'edge_kwargs': {
                'config': config,
                'chn_in': (chn, ) * n_inputs_node,
                'stride': 1,
                'ops': ops,
            },
            'child_cls': TreeLayer,
            'child_kwargs': {
                'config': config,
                'n_nodes': n_nodes,
                'chn_in': (chn_cur, ) * n_inputs_layer,
                'shared_a': False,
                'allocator': ReplicateAllocator,
                'merger_out': SumMerger,
                'preproc': PreprocLayer,
                'aggregate': None,
                'edge_cls': BinGateMixedOp,
                'edge_kwargs': {
                    'config': config,
                    'chn_in': (chn, ) * n_inputs_node,
                    'stride': 1,
                    'ops': ops,
                },
                'child_cls': TreeLayer,
                'child_kwargs': {
                    'config': config,
                    'n_nodes': n_nodes,
                    'chn_in': (chn_cur, ) * n_inputs_layer,
                    'shared_a': False,
                    'allocator': ReplicateAllocator,
                    'merger_out': SumMerger,
                    'preproc': PreprocLayer,
                    'aggregate': None,
                    'edge_cls': BinGateMixedOp,
                    'edge_kwargs': {
                        'config': config,
                        'chn_in': (chn, ) * n_inputs_node,
                        'stride': 1,
                        'ops': ops,
                    },
                    'child_cls': None,
                    'child_kwargs': {},
                }
            }
        }
    }
    criterion = nn.CrossEntropyLoss().to(device)
    model = NASController(config, criterion, gt.PRIMITIVES_DEFAULT,
                        dev_list, PyramidNet, pyramidnet_kwargs).to(device)
    arch = BinaryGateArchitect
    return model, arch

def get_dagnet(config, device, dev_list):
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

    dagnet_kwargs = {
        'config': config,
        'n_nodes': n_layers,
        'chn_in': (chn_cur, ) * n_inputs_model,
        'shared_a': False,
        'allocator': ReplicateAllocator,
        'merger_state': SumMerger,
        'merger_out': LastMerger,
        'enumerator': LastNEnumerator,
        'preproc': None,
        'aggregate': None,
        'edge_cls': DAGLayer,
        'edge_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': (chn_cur, ) * n_inputs_layer,
            'shared_a': False,
            'allocator': ReplicateAllocator,
            'merger_state': SumMerger,
            'merger_out': ConcatMerger,
            'enumerator': CombinationEnumerator,
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
                        dev_list, BinGateNet, dagnet_kwargs).to(device)
    arch = BinaryGateArchitect
    return model, arch

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
        'chn_in': (chn_cur, ) * n_inputs_model,
        'shared_a': True,
        'allocator': ReplicateAllocator,
        'merger_state': SumMerger,
        'merger_out': LastMerger,
        'enumerator': LastNEnumerator,
        'preproc': None,
        'aggregate': None,
        'edge_cls': DAGLayer,
        'edge_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': (chn_cur, ) * n_inputs_layer,
            'shared_a': False,
            'allocator': ReplicateAllocator,
            'merger_state': SumMerger,
            'merger_out': ConcatMerger,
            'enumerator': CombinationEnumerator,
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
    arch = DARTSArchitect
    return model, arch

model_creator = {
    'proxyless-nas': get_proxylessnasnet,
    'pyramidnet': get_pyramidnet,
    'darts-no-reduce': get_dartslike,
    'dagnet': get_dagnet,
}

# @profile_mem
def get_model(config, device, dev_list, genotype=None):
    mtype = config.type
    if mtype in model_creator:
        config.augment = not genotype is None
        model, arch = model_creator[mtype](config, device, dev_list)
        if config.augment:
            model.build_from_genotype(genotype)
            print("genotype = {}".format(model.genotype()))
        mb_params = param_size(model)
        n_params = param_count(model)
        print("Model params count: {:.3f} M, size: {:.3f} MB".format(n_params, mb_params))
        return model, arch
    else:
        raise Exception("invalid model type")