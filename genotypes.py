""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import os
from collections import namedtuple
import torch
import torch.nn as nn
from models import ops

Genotype = namedtuple('Genotype', 'dag ops')

PRIMITIVES_DEFAULT = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    # 'sep_conv_7x7',
    'none'
]

abbr = {
    'none': 'NIL',
    'avg_pool_3x3': 'AVG',
    'max_pool_3x3': 'MAX',
    'skip_connect': 'IDT',
    'sep_conv_3x3': 'SC3',
    'sep_conv_5x5': 'SC5',
    'sep_conv_7x7': 'SC7',
    'dil_conv_3x3': 'DC3',
    'dil_conv_5x5': 'DC5',
    'conv_7x1_1x7': 'FC7',
}

deabbr = {
    'NIL': 'none',
    'AVG': 'avg_pool_3x3',
    'MAX': 'max_pool_3x3',
    'IDT': 'skip_connect',
    'SC3': 'sep_conv_3x3',
    'SC5': 'sep_conv_5x5',
    'SC7': 'sep_conv_7x7',
    'DC3': 'dil_conv_3x3',
    'DC5': 'dil_conv_5x5',
    'FC7': 'conv_7x1_1x7',
}

def pretty_print(gene):
    pass

def to_file(gene, path):
    g_str = str(gene)
    with open(path, 'w') as f:
        f.write(g_str)

def from_file(path):
    if not os.path.exists(path):
        # raise ValueError("genotype file not found")
        return Genotype(dag=None, ops=None)
    with open(path, 'r') as f:
        g_str = f.read()
    return from_str(g_str)

def from_str(s):
    """ generate genotype from string """
    genotype = eval(s)
    return genotype
