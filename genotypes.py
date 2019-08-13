""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
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

def pretty_print(gene):
    pass

def to_file(gene, path):
    g_str = str(gene)
    with open(path, 'wb') as f:
        f.write(g_str)

def from_file(path):
    with open(path, 'rb') as f:
        g_str = f.read()
    return from_str(g_str)

def from_str(s):
    """ generate genotype from string """
    genotype = eval(s)
    return genotype
