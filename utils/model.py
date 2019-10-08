# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
from models.dagnet import BinGateNet, DARTSLikeNet
from models.pyramidnet import PyramidNet, GroupConv
from models.proxylessnas import ProxylessNASNet
from models.layers import DAGLayer, TreeLayer,\
                        PreprocLayer, MergeFilterLayer
from models.nas_modules import DARTSMixedOp, BinGateMixedOp,\
                                NASController, NASModule
from models.defs import ConcatMerger, SumMerger, LastMerger, SumMerger, AvgMerger,\
                        CombinationEnumerator, LastNEnumerator,\
                        TreeEnumerator, FirstNEnumerator,\
                        EvenSplitAllocator, ReplicateAllocator
from architect import DARTSArchitect, BinaryGateArchitect
import genotypes as gt
from models.ops import configure_ops
from models.BoT import CrossEntropyLoss_LS

from utils.profiling import profile_mem
from utils import param_size, param_count, warmup_device

def get_proxylessnasnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_groups = config.groups
    conv_groups = config.conv_groups
    n_blocks = config.blocks
    alpha = config.alpha
    bneck = config.bottleneck_ratio
    ops = gt.get_primitives()
    path_drop_rate = config.path_drop_rate if config.augment else 0
    ops_order = config.pxl_ops_order
    use_avg = config.use_avg
    bn_before_add = config.bn_before_add
    model_config = {
	    'start_planes': chn_cur,
	    'alpha': alpha,
	    'block_per_group': n_blocks,
	    'total_groups': n_groups,
	    'downsample_type': 'avg_pool',  # avg, max
	    ######################################################
	    'bottleneck': bneck,
	    'ops_order': ops_order,
	    'dropout_rate': config.dropout_rate,
	    ######################################################
	    'final_bn': True,
	    'no_first_relu': True,
	    'use_depth_sep_conv': False,
	    'groups_3x3': conv_groups,
	    ######################################################
        'edge_cls': BinGateMixedOp,
        'edge_kwargs': {
            'config': config,
            'chn_in': None,
            'stride': 1,
            'ops': ops,
            'shared_a': False,
        },
        'tree_node_config': {
			'use_avg': use_avg,
			'bn_before_add': bn_before_add,
			'path_drop_rate': path_drop_rate,
			'use_zero_drop': True,
			'drop_only_add': False,
		}
    }
    net = ProxylessNASNet.set_standard_net(data_shape=(chn_in, 32, 32), n_classes=n_classes, **model_config)
    net_config = net.get_config()
    net_save_path = './net.config'
    json.dump(net_config, open(net_save_path, 'w'), indent=4)
    arch = BinaryGateArchitect
    return net, arch

def get_eas_net(config):
    net_config_json = json.load(open(config.net_config_path, 'r'))
    print('Net config:')
    for k, v in net_config_json.items():
        if k != 'blocks':
            print('\t%s: %s' % (k, v))
    net = ProxylessNASNet.set_from_config(net_config_json)
    arch = BinaryGateArchitect
    return net, arch

def get_pyramidnet_origin(config):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_groups = config.groups
    conv_groups = config.conv_groups
    n_blocks = config.blocks
    alpha = config.alpha
    ops = gt.get_primitives()
    pyramidnet_kwargs = {
        'config': config,
        'cell_cls': GroupConv,
        'cell_kwargs': {
            'chn_in': None,
            'chn_out': None,
            'stride': None,
            'kernel_size': 3,
            'padding': 1,
        }
    }
    net = PyramidNet(**pyramidnet_kwargs)
    arch = BinaryGateArchitect
    return net, arch

def get_pyramidnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_nodes = config.nodes
    conv_groups = config.conv_groups
    n_inputs_layer = config.inputs_layer
    n_inputs_node = config.inputs_node
    ops = gt.get_primitives()
    allocator = ReplicateAllocator
    merger = AvgMerger
    pyramidnet_kwargs = {
        'cell_cls': TreeLayer,
        'cell_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': (chn_cur, ) * n_inputs_layer,
            'shared_a': False,
            'allocator': allocator,
            'merger_out': merger,
            'preproc': None,
            'aggregate': None,
            'edge_cls': GroupConv,
            'edge_kwargs': {
                'chn_in': None,
                'chn_out': None,
                'stride': None,
                'groups': conv_groups,
                'kernel_size': 3,
                'padding': 1,
            },
            'child_cls': TreeLayer,
            'child_kwargs': {
                'config': config,
                'n_nodes': n_nodes,
                'chn_in': (chn_cur, ) * n_inputs_layer,
                'stride': 1,
                'shared_a': False,
                'allocator': allocator,
                'merger_out': merger,
                'preproc': None,
                'aggregate': None,
                'edge_cls': BinGateMixedOp,
                'edge_kwargs': {
                    'config': config,
                    'chn_in': (chn, ) * n_inputs_node,
                    'stride': 1,
                    'ops': ops,
                    'shared_a': None,
                },
                'child_cls': TreeLayer,
                'child_kwargs': {
                    'config': config,
                    'n_nodes': n_nodes,
                    'chn_in': (chn_cur, ) * n_inputs_layer,
                    'stride': 1,
                    'shared_a': False,
                    'allocator': allocator,
                    'merger_out': merger,
                    'preproc': None,
                    'aggregate': None,
                    'edge_cls': BinGateMixedOp,
                    'edge_kwargs': {
                        'config': config,
                        'chn_in': (chn, ) * n_inputs_node,
                        'stride': 1,
                        'ops': ops,
                        'shared_a': None,
                    },
                    'child_cls': None,
                    'child_kwargs': {},
                }
            }
        }
    }
    net = PyramidNet(**pyramidnet_kwargs)
    arch = BinaryGateArchitect
    return net, arch

def get_dagnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_layers = config.layers
    n_nodes = config.nodes
    n_inputs_model = config.inputs_model
    n_inputs_layer = config.inputs_layer
    n_inputs_node = config.inputs_node
    ops = gt.get_primitives()

    dagnet_kwargs = {
        'groups': config.groups,
        'dag_kwargs': {
            'config': config,
            'n_nodes': n_layers,
            'chn_in': (chn_cur, ) * n_inputs_model,
            'stride': 1,
            'shared_a': False,
            'allocator': ReplicateAllocator,
            'merger_state': SumMerger,
            'merger_out': SumMerger,
            'enumerator': CombinationEnumerator,
            'preproc': None,
            'aggregate': None,
            'edge_cls': DAGLayer,
            'edge_kwargs': {
                'config': config,
                'n_nodes': n_nodes,
                'chn_in': (chn_cur, ) * n_inputs_layer,
                'stride': 1,
                'shared_a': False,
                'allocator': ReplicateAllocator,
                'merger_state': SumMerger,
                'merger_out': SumMerger,
                'enumerator': CombinationEnumerator,
                'preproc': PreprocLayer,
                'aggregate': None,
                'edge_cls': BinGateMixedOp,
                'edge_kwargs': {
                    'config': config,
                    'chn_in': (chn, ) * n_inputs_node,
                    'stride': 1,
                    'ops': ops,
                    'shared_a': None,
                },
            }
        },
    }
    net = BinGateNet(**dagnet_kwargs)
    arch = BinaryGateArchitect
    return net, arch

def get_dartslike(config):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    n_layers = config.layers
    n_nodes = config.nodes
    n_inputs_model = config.inputs_model
    n_inputs_layer = config.inputs_layer
    n_inputs_node = config.inputs_node
    ops = gt.get_primitives()
    darts_kwargs = {
        'n_layers': n_layers,
        'shared_a': False,
        'cell_cls': DAGLayer,
        'cell_kwargs': {
            'config': config,
            'n_nodes': n_nodes,
            'chn_in': None,
            'shared_a': False,
            'allocator': ReplicateAllocator,
            'merger_state': SumMerger,
            'merger_out': ConcatMerger,
            'enumerator': CombinationEnumerator,
            'preproc': None,
            'aggregate': None,
            'edge_cls': DARTSMixedOp,
            'edge_kwargs': {
                'config': config,
                'chn_in': None,
                'shared_a': None,
                'stride': 1,
                'ops': ops,
            },
        },
    }
    net = DARTSLikeNet(config, **darts_kwargs)
    arch = DARTSArchitect
    return net, arch

model_creator = {
    'proxyless-nas': get_proxylessnasnet,
    'pyramidnet': get_pyramidnet,
    'pyramidnet-origin': get_pyramidnet_origin,
    'darts': get_dartslike,
    'dagnet': get_dagnet,
    'pyramidnet-eas': get_eas_net,
}

def get_net_crit(config):
    if config.label_smoothing > 0:
        crit = CrossEntropyLoss_LS(config.label_smoothing)
    else:
        crit = nn.CrossEntropyLoss()
    return crit

# @profile_mem
def get_model(config, device, dev_list, genotype=None):
    mtype = config.type
    configure_ops(config)
    if mtype in model_creator:
        config.augment = not genotype is None
        net, arch = model_creator[mtype](config)
        crit = get_net_crit(config).to(device)
        prim = gt.get_primitives()
        model = NASController(config, net, crit, prim, dev_list).to(device)
        if config.augment:
            print("genotype = {}".format(genotype))
            model.build_from_genotype(genotype)
            model.to(device=device)
        if config.verbose: print(model)
        mb_params = param_size(model)
        n_params = param_count(model)
        print("Model params count: {:.3f} M, size: {:.3f} MB".format(n_params, mb_params))
        NASModule.set_device(dev_list)
        # warmup_device(model, 32, device)
        return model, arch
    else:
        raise Exception("invalid model type")